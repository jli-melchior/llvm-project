//===-- llvm/CodeGen/AllocationOrder.cpp - Allocation Order ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static cl::opt<bool> DropAntiHintedSoftHints(
    "regalloc-drop-antihinted-hints", cl::Hidden,
    cl::desc("Remove anti-hinted physical registers from soft allocation "
             "hint candidates while retaining the fallback order"),
    cl::init(false));

// Compare VirtRegMap::getRegAllocPref().
AllocationOrder AllocationOrder::create(Register VirtReg, const VirtRegMap &VRM,
                                        const RegisterClassInfo &RegClassInfo,
                                        const LiveRegMatrix *Matrix) {
  const MachineFunction &MF = VRM.getMachineFunction();
  const TargetRegisterInfo *TRI = &VRM.getTargetRegInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  auto Order = RegClassInfo.getOrder(MRI.getRegClass(VirtReg));
  SmallVector<MCPhysReg, 16> Hints;
  bool HardHints =
      TRI->getRegAllocationHints(VirtReg, Order, Hints, MF, &VRM, Matrix);

  LLVM_DEBUG({
    if (!Hints.empty()) {
      dbgs() << "hints:";
      for (MCPhysReg Hint : Hints)
        dbgs() << ' ' << printReg(Hint, TRI);
      dbgs() << '\n';
    }
  });

  SmallVector<MCPhysReg, 16> AntiHintedPhysRegs;
  MRI.getPhysRegAntiHints(VirtReg, AntiHintedPhysRegs, VRM);

  LLVM_DEBUG({
    if (!AntiHintedPhysRegs.empty()) {
      dbgs() << "anti-hints:";
      for (MCPhysReg AntiHint : AntiHintedPhysRegs)
        dbgs() << ' ' << printReg(AntiHint, TRI);
      dbgs() << '\n';
    }
  });

  SmallVector<MCPhysReg, 16> ShuffledOrder;

  if (!AntiHintedPhysRegs.empty()) {
    TRI->applyRegAllocationAntiHints(VirtReg, Order, ShuffledOrder,
                                     AntiHintedPhysRegs, MF, &VRM, Matrix);

    if (!ShuffledOrder.empty() && !Hints.empty()) {
      std::stable_partition(Hints.begin(), Hints.end(), [&](MCPhysReg Hint) {
        return llvm::none_of(AntiHintedPhysRegs, [&](MCPhysReg AntiHint) {
          return TRI->regsOverlap(Hint, AntiHint);
        });
      });

      if (DropAntiHintedSoftHints && !HardHints) {
        llvm::erase_if(Hints, [&](MCPhysReg Hint) {
          return llvm::any_of(AntiHintedPhysRegs, [&](MCPhysReg AntiHint) {
            return TRI->regsOverlap(Hint, AntiHint);
          });
        });
        LLVM_DEBUG(dbgs() << "Dropped anti-hinted soft allocation hints; "
                          << Hints.size() << " candidates remain\n");
      }

      LLVM_DEBUG({
        dbgs() << "anti-hint-aware hints:";
        for (MCPhysReg Hint : Hints)
          dbgs() << ' ' << printReg(Hint, TRI);
        dbgs() << '\n';
      });
    }
  }

  ArrayRef<MCPhysReg> FinalOrder =
      ShuffledOrder.empty() ? Order : ShuffledOrder;
  AllocationOrder AO(std::move(Hints), FinalOrder, HardHints,
                     std::move(ShuffledOrder));

  assert(all_of(AO.Hints,
                [&](MCPhysReg Hint) { return is_contained(AO.Order, Hint); }) &&
         "Target hint is outside allocation order.");
  return AO;
}
