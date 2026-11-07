//===-- AMDGPUSchedBank.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lower the V_SCHEDULE_BANK_B* pseudos produced from llvm.amdgcn.sched.bank,
/// and the V_SCHEDULE_VGPR_B*, S_SCHEDULE_SGPR_B*, A_SCHEDULE_AGPR_B* pseudos
/// from llvm.amdgcn.schedule.{vgpr,sgpr,agpr}.
///
/// For each pseudo this pass:
///   1. reads the requested bank or register number from the immediate operand,
///   2. attaches an appropriate register allocation hint to the destination
///      (and source) vreg so the allocator prefers the target bank or register,
///   3. propagates the hint through COPY / REG_SEQUENCE / INSERT_SUBREG /
///      SUBREG_TO_REG so it survives coalescing, and
///   4. replaces the pseudo with a plain COPY, except for a strict AGPR pin.
///
/// A strict AGPR pseudo is left in place until expandPostRAPseudo. Replacing
/// it with a COPY drops the AGPR operand constraint, and the coalescer then
/// widens the vreg from AGPR_* to AV_*. vN and aN share a HW index, so the
/// allocator can satisfy the hint with a VGPR. Keeping the pseudo holds the
/// class at AGPR_* through coalescing.
///
/// A non-strict hint only reorders the allocation candidates, so a full bank
/// (or unavailable register) falls back to the default order. A strict hint is
/// a hard constraint.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSchedBank.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-sched-bank"

/// Classification of schedule-hint pseudos.
enum ScheduleHintKind { SHK_None, SHK_Bank, SHK_VGPR, SHK_SGPR, SHK_AGPR };

static ScheduleHintKind getScheduleHintKind(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::V_SCHEDULE_BANK_B32:
  case AMDGPU::V_SCHEDULE_BANK_B64:
  case AMDGPU::V_SCHEDULE_BANK_B128:
  case AMDGPU::V_SCHEDULE_BANK_B256:
    return SHK_Bank;
  case AMDGPU::V_SCHEDULE_VGPR_B32:
  case AMDGPU::V_SCHEDULE_VGPR_B64:
  case AMDGPU::V_SCHEDULE_VGPR_B128:
  case AMDGPU::V_SCHEDULE_VGPR_B256:
  case AMDGPU::V_SCHEDULE_VGPR_B512:
    return SHK_VGPR;
  case AMDGPU::S_SCHEDULE_SGPR_B32:
  case AMDGPU::S_SCHEDULE_SGPR_B64:
  case AMDGPU::S_SCHEDULE_SGPR_B128:
  case AMDGPU::S_SCHEDULE_SGPR_B256:
  case AMDGPU::S_SCHEDULE_SGPR_B512:
    return SHK_SGPR;
  case AMDGPU::A_SCHEDULE_AGPR_B32:
  case AMDGPU::A_SCHEDULE_AGPR_B64:
  case AMDGPU::A_SCHEDULE_AGPR_B128:
  case AMDGPU::A_SCHEDULE_AGPR_B256:
  case AMDGPU::A_SCHEDULE_AGPR_B512:
    return SHK_AGPR;
  default:
    return SHK_None;
  }
}

/// Walk the SSA use chain of \p Reg and propagate the hint through
/// value-preserving pseudos so the preference reaches the vregs that survive
/// coalescing.
static void propagateHint(MachineRegisterInfo &MRI, Register Reg,
                          unsigned HintValue, unsigned HintKind,
                          SmallDenseSet<unsigned, 32> &Visited) {
  if (!Reg.isVirtual() || !Visited.insert(Reg.id()).second)
    return;

  for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
    Register DefReg;
    switch (UseMI.getOpcode()) {
    case TargetOpcode::COPY:
    case TargetOpcode::REG_SEQUENCE:
    case TargetOpcode::INSERT_SUBREG:
    case TargetOpcode::SUBREG_TO_REG:
      DefReg = UseMI.getOperand(0).getReg();
      break;
    default:
      continue;
    }
    if (!DefReg.isVirtual())
      continue;

    // Do not clobber an existing, conflicting hint.
    std::pair<unsigned, Register> Existing = MRI.getRegAllocationHint(DefReg);
    if (Existing.first != 0 &&
        (Existing.first != HintKind || Existing.second != HintValue))
      continue;

    MRI.setRegAllocationHint(DefReg, HintKind, HintValue);
    propagateHint(MRI, DefReg, HintValue, HintKind, Visited);
  }
}

/// Give each chain of values joined by regnum -1 strict AGPR pins its own AGPR
/// index, if all chains fit in the free AGPRs. With only the file constraint
/// the allocator moves accumulators between AGPRs once they are all in use,
/// and each move is a copy; a fixed index per chain keeps every accumulator in
/// place. A chain is joined through the pin, the accumulator input of the MFMA
/// that feeds it, PHIs and COPYs, and only through values with one use that is
/// not a PHI or COPY, so its members are not live at the same time.
static void assignAnyAGPRIndices(MachineFunction &MF, const SIInstrInfo &TII,
                                 const SIRegisterInfo &TRI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  if (!MRI.reservedRegsFrozen())
    return;

  auto NumRegs = [&](Register R) {
    return TRI.getRegSizeInBits(*MRI.getRegClass(R)) / 32;
  };
  // Besides its one real use a value may be forwarded by PHIs and COPYs on
  // other paths, such as the loop back edge and exit, but not twice in one
  // block (a shared initial value of several chains).
  auto HasOneRealUse = [&](Register R) {
    unsigned N = 0;
    SmallPtrSet<const MachineBasicBlock *, 4> Blocks;
    for (const MachineInstr &U : MRI.use_nodbg_instructions(R))
      if ((!U.isPHI() && !U.isCopy() && ++N > 1) ||
          !Blocks.insert(U.getParent()).second)
        return false;
    return true;
  };
  EquivalenceClasses<unsigned> Chains;
  auto Join = [&](Register From, Register To) {
    if (From.isVirtual() && To.isVirtual() && HasOneRealUse(From) &&
        NumRegs(From) == NumRegs(To))
      Chains.unionSets(From.id(), To.id());
  };

  const TargetRegisterClass &AGPRs = AMDGPU::AGPR_32RegClass;
  BitVector Used(AGPRs.getNumRegs());
  for (MCPhysReg R : AGPRs)
    if (MRI.isReserved(R))
      Used.set(TRI.getHWRegIndex(R));

  SmallVector<MachineInstr *, 64> Pins;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isPHI()) {
        for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2)
          if (!MI.getOperand(I).getSubReg())
            Join(MI.getOperand(I).getReg(), MI.getOperand(0).getReg());
        continue;
      }
      if (MI.isCopy()) {
        if (!MI.getOperand(0).getSubReg() && !MI.getOperand(1).getSubReg())
          Join(MI.getOperand(1).getReg(), MI.getOperand(0).getReg());
        continue;
      }
      if (getScheduleHintKind(MI.getOpcode()) != SHK_AGPR)
        continue;

      Register Dst = MI.getOperand(0).getReg();
      unsigned RegNum = MI.getOperand(2).getImm();
      if (RegNum != ~0u) {
        unsigned End = std::min<unsigned>(RegNum + NumRegs(Dst), Used.size());
        if (RegNum < End)
          Used.set(RegNum, End);
        continue;
      }
      if (MI.getOperand(3).getImm() == 0 || !Dst.isVirtual())
        continue;

      Pins.push_back(&MI);
      Chains.insert(Dst.id());
      const MachineOperand &SrcMO = MI.getOperand(1);
      if (SrcMO.getSubReg() || !SrcMO.getReg().isVirtual())
        continue;
      Join(SrcMO.getReg(), Dst);
      MachineInstr *Def = MRI.getUniqueVRegDef(SrcMO.getReg());
      if (!Def || !SIInstrInfo::isMAI(*Def))
        continue;
      const MachineOperand *Src2 =
          TII.getNamedOperand(*Def, AMDGPU::OpName::src2);
      if (Src2 && Src2->isReg() && !Src2->getSubReg())
        Join(Src2->getReg(), SrcMO.getReg());
    }
  }

  // First fit, in program order; give up on all chains if one does not fit.
  DenseMap<unsigned, unsigned> ChainIdx;
  for (MachineInstr *MI : Pins) {
    unsigned Leader = Chains.getLeaderValue(MI->getOperand(0).getReg().id());
    if (ChainIdx.count(Leader))
      continue;
    unsigned Size = NumRegs(MI->getOperand(0).getReg());
    unsigned Align = std::min(Size, 4u);
    unsigned Base = 0;
    while (Base + Size <= Used.size() &&
           Used.find_first_in(Base, Base + Size) != -1)
      Base += Align;
    if (Base + Size > Used.size()) {
      LLVM_DEBUG(dbgs() << "  any-AGPR chains do not fit: " << ChainIdx.size()
                        << " placed, " << Used.count() << " AGPRs used\n");
      return;
    }
    Used.set(Base, Base + Size);
    ChainIdx[Leader] = Base;
  }

  for (MachineInstr *MI : Pins) {
    unsigned Idx =
        ChainIdx[Chains.getLeaderValue(MI->getOperand(0).getReg().id())];
    LLVM_DEBUG(dbgs() << "  any-AGPR pin "
                      << printReg(MI->getOperand(0).getReg()) << " -> a" << Idx
                      << '\n');
    MI->getOperand(2).setImm(Idx);
  }
}

// Shared implementation used by both the legacy and new-PM passes.
static bool runScheduleBank(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  assignAnyAGPRIndices(MF, *TII, *ST.getRegisterInfo());

  SmallVector<MachineInstr *, 16> ToErase;
  // Values that should share a register with a regnum -1 pin.
  SmallVector<std::pair<Register, Register>, 16> Partners;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      ScheduleHintKind Kind = getScheduleHintKind(MI.getOpcode());
      if (Kind == SHK_None)
        continue;

      Register DstReg = MI.getOperand(0).getReg();
      const MachineOperand &SrcMO = MI.getOperand(1);
      Register SrcReg = SrcMO.getReg();

      unsigned HintKind;
      unsigned HintValue;
      bool Strict = false;
      bool AnyReg = false;

      if (Kind == SHK_Bank) {
        // Bank pseudos are only valid on targets with 1024 addressable VGPRs.
        if (!ST.has1024AddressableVGPRs()) {
          // Replace with COPY and skip hint setup.
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY),
                  DstReg)
              .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());
          ToErase.push_back(&MI);
          Changed = true;
          continue;
        }

        unsigned EncodedBank = MI.getOperand(2).getImm();
        Strict = EncodedBank & 0x4;
        unsigned Bank = EncodedBank & 0x3;

        if (EncodedBank > 7) {
          Bank = 0;
          Strict = false;
        }

        HintKind = Strict ? AMDGPURI::StrictBankHint : AMDGPURI::BankHint;
        HintValue = Bank;

        LLVM_DEBUG(dbgs() << "  sched.bank: " << printReg(DstReg) << " <- "
                          << printReg(SrcReg) << " bank " << Bank
                          << " strict " << Strict << '\n');
      } else {
        // Exact register pseudos: operand 2 = regnum, operand 3 = strict.
        // regnum -1 pins only the register file: the hint carries
        // AnyRegHintIdx instead of an index. A soft pin with regnum -1 has no
        // effect.
        unsigned RegNum = MI.getOperand(2).getImm();
        Strict = MI.getOperand(3).getImm() != 0;
        AnyReg = RegNum == ~0u;

        switch (Kind) {
        case SHK_VGPR:
          HintKind = Strict ? AMDGPURI::StrictVGPRRegHint
                            : AMDGPURI::VGPRRegHint;
          break;
        case SHK_SGPR:
          HintKind = Strict ? AMDGPURI::StrictSGPRRegHint
                            : AMDGPURI::SGPRRegHint;
          break;
        case SHK_AGPR:
          HintKind = Strict ? AMDGPURI::StrictAGPRRegHint
                            : AMDGPURI::AGPRRegHint;
          break;
        default:
          llvm_unreachable("unexpected schedule hint kind");
        }
        HintValue = AnyReg ? AMDGPURI::AnyRegHintIdx : RegNum;

        LLVM_DEBUG(dbgs() << "  schedule."
                          << (Kind == SHK_VGPR   ? "vgpr"
                              : Kind == SHK_SGPR ? "sgpr"
                                                 : "agpr")
                          << ": " << printReg(DstReg) << " <- "
                          << printReg(SrcReg) << " reg " << RegNum
                          << " strict " << Strict << '\n');
      }

      // Hint both the destination and the source so the preference survives
      // whichever side coalescing keeps. Don't overwrite a strict hint with a
      // soft hint of the same category.
      bool IsStrict = (HintKind == AMDGPURI::StrictBankHint ||
                       HintKind == AMDGPURI::StrictVGPRRegHint ||
                       HintKind == AMDGPURI::StrictSGPRRegHint ||
                       HintKind == AMDGPURI::StrictAGPRRegHint);
      // The strict variant is always HintKind | 1 for each category.
      unsigned StrictVariant = HintKind | 1;

      bool NoHint = AnyReg && !Strict;
      if (!NoHint && DstReg.isVirtual() &&
          (IsStrict ||
           MRI.getRegAllocationHint(DstReg).first != StrictVariant))
        MRI.setRegAllocationHint(DstReg, HintKind, HintValue);
      if (!NoHint && SrcReg.isVirtual() &&
          (IsStrict ||
           MRI.getRegAllocationHint(SrcReg).first != StrictVariant))
        MRI.setRegAllocationHint(SrcReg, HintKind, HintValue);

      SmallDenseSet<unsigned, 32> Visited;
      if (!NoHint && DstReg.isVirtual())
        propagateHint(MRI, DstReg, HintValue, HintKind, Visited);

      // With an index every value in the chain is hinted to the same
      // register. Without one, nothing else asks RA to give the kept pseudo's
      // source and result, or an MFMA's accumulator input and result, the same
      // register, and each mismatch becomes a copy.
      if (AnyReg && Strict && DstReg.isVirtual() && SrcReg.isVirtual() &&
          !SrcMO.getSubReg()) {
        Partners.emplace_back(SrcReg, DstReg);
        MachineInstr *Def = MRI.getUniqueVRegDef(SrcReg);
        const MachineOperand *Src2 =
            Def && SIInstrInfo::isMAI(*Def)
                ? TII->getNamedOperand(*Def, AMDGPU::OpName::src2)
                : nullptr;
        if (Src2 && Src2->isReg() && Src2->getReg().isVirtual() &&
            !Src2->getSubReg())
          Partners.emplace_back(Src2->getReg(), SrcReg);
      }

      // A strict exact-register pin must keep its own def until after RA:
      // a plain COPY gets coalesced and the hint can be dropped with it. For
      // AGPR it also keeps the AGPR operand constraint. expandPostRAPseudo
      // turns the pseudo into a physreg copy.
      if (Kind != SHK_Bank && Strict) {
        Changed = true;
        continue;
      }

      // Replace the pseudo with a plain COPY.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), DstReg)
          .addReg(SrcReg, getRegState(SrcMO), SrcMO.getSubReg());

      ToErase.push_back(&MI);
      Changed = true;
    }
  }

  // After all hints are set: setRegAllocationHint drops earlier entries.
  for (auto [A, B] : Partners) {
    MRI.addRegAllocationHint(A, B);
    MRI.addRegAllocationHint(B, A);
  }

  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();

  return Changed;
}

namespace {

class AMDGPUSchedBank : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSchedBank() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return runScheduleBank(MF);
  }

  StringRef getPassName() const override { return "AMDGPU Sched Bank"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS(AMDGPUSchedBank, DEBUG_TYPE, "AMDGPU Sched Bank",
                false, false)

char AMDGPUSchedBank::ID = 0;

char &llvm::AMDGPUSchedBankID = AMDGPUSchedBank::ID;

PreservedAnalyses
AMDGPUSchedBankPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  if (!runScheduleBank(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}
