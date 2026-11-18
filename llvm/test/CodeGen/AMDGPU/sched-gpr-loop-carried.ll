; RUN: opt -mtriple=amdgcn -mcpu=gfx950 -passes=amdgpu-codegenprepare -S < %s | FileCheck -check-prefix=IR %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck -check-prefix=GCN %s

; A strict pin of a loop's initial value covers the whole loop-carried
; variable: the updates inside the loop need no pin of their own and the
; variable keeps one register across iterations.

declare i32 @llvm.amdgcn.sched.vgpr.i32(i32, i32, i32)
declare <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float>, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; IR-LABEL: @vgpr_numbered(
; IR: %c1 = add i32 %c, %x
; IR-NEXT: call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %c1, i32 12, i32 1)
; IR: %d1 = xor i32 %d, %y
; IR-NEXT: call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %d1, i32 13, i32 1)

; GCN-LABEL: vgpr_numbered:
; GCN: .LBB0_1:
; GCN-NOT: v_mov
; GCN: v_add_u32_e32 v12, v12, v{{[0-9]+}}
; GCN-NOT: v_mov
; GCN: v_xor_b32_e32 v13, v13, v{{[0-9]+}}
; GCN-NOT: v_mov
; GCN: s_cbranch_scc1 .LBB0_1
define amdgpu_kernel void @vgpr_numbered(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %x = load volatile i32, ptr addrspace(1) %p
  %y = load volatile i32, ptr addrspace(1) %p
  %z = load volatile i32, ptr addrspace(1) %p
  %w = load volatile i32, ptr addrspace(1) %p
  %c0 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %z, i32 12, i32 1)
  %d0 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %w, i32 13, i32 1)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i1, %loop]
  %c = phi i32 [%c0, %entry], [%c1, %loop]
  %d = phi i32 [%d0, %entry], [%d1, %loop]
  %c1 = add i32 %c, %x
  %d1 = xor i32 %d, %y
  %i1 = add i32 %i, 1
  %cc = icmp slt i32 %i1, %n
  br i1 %cc, label %loop, label %exit
exit:
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %c1, ptr addrspace(1) %q
  store volatile i32 %d1, ptr addrspace(1) %q
  ret void
}

; IR-LABEL: @vgpr_any(
; IR: %c1 = add i32 %c, %x
; IR-NEXT: call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %c1, i32 -1, i32 1)

; GCN-LABEL: vgpr_any:
; GCN: .LBB1_1:
; GCN-NOT: v_mov
; GCN: v_add_u32_e32 [[C:v[0-9]+]], [[C]], v{{[0-9]+}}
; GCN-NOT: v_mov
; GCN: v_xor_b32_e32 [[D:v[0-9]+]], [[D]], v{{[0-9]+}}
; GCN-NOT: v_mov
; GCN: s_cbranch_scc1 .LBB1_1
define amdgpu_kernel void @vgpr_any(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %x = load volatile i32, ptr addrspace(1) %p
  %y = load volatile i32, ptr addrspace(1) %p
  %z = load volatile i32, ptr addrspace(1) %p
  %w = load volatile i32, ptr addrspace(1) %p
  %c0 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %z, i32 -1, i32 1)
  %d0 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %w, i32 -1, i32 1)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i1, %loop]
  %c = phi i32 [%c0, %entry], [%c1, %loop]
  %d = phi i32 [%d0, %entry], [%d1, %loop]
  %c1 = add i32 %c, %x
  %d1 = xor i32 %d, %y
  %i1 = add i32 %i, 1
  %cc = icmp slt i32 %i1, %n
  br i1 %cc, label %loop, label %exit
exit:
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %c1, ptr addrspace(1) %q
  store volatile i32 %d1, ptr addrspace(1) %q
  ret void
}

; Two MFMAs per iteration accumulate into each AGPR chain.

; IR-LABEL: @agpr_any(
; IR: %e2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(
; IR-NEXT: call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %e2, i32 -1, i32 1)
; IR: %e3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(
; IR-NEXT: call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %e3, i32 -1, i32 1)

; GCN-LABEL: agpr_any:
; GCN-NOT: v_accvgpr_mov
; GCN: .LBB2_1:
; GCN: v_mfma_f32_16x16x4_f32 a[0:3], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
; GCN: v_mfma_f32_16x16x4_f32 a[4:7], v{{[0-9]+}}, v{{[0-9]+}}, a[4:7]
; GCN-NOT: v_accvgpr_mov
; GCN: v_mfma_f32_16x16x4_f32 a[0:3], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]
; GCN: v_mfma_f32_16x16x4_f32 a[4:7], v{{[0-9]+}}, v{{[0-9]+}}, a[4:7]
; GCN-NOT: v_accvgpr_mov
; GCN: s_cbranch_scc1 .LBB2_1
define amdgpu_kernel void @agpr_any(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %a = load volatile float, ptr addrspace(1) %p
  %b = load volatile float, ptr addrspace(1) %p
  %m0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %e0 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %m0, i32 -1, i32 1)
  %n0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %b, float %a, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %f0 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %n0, i32 -1, i32 1)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i1, %loop]
  %e = phi <4 x float> [%e0, %entry], [%e3, %loop]
  %f = phi <4 x float> [%f0, %entry], [%f3, %loop]
  %e2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %e, i32 0, i32 0, i32 0)
  %f2 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %b, float %a, <4 x float> %f, i32 0, i32 0, i32 0)
  %e3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %b, float %b, <4 x float> %e2, i32 0, i32 0, i32 0)
  %f3 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %a, <4 x float> %f2, i32 0, i32 0, i32 0)
  %i1 = add i32 %i, 1
  %cc = icmp slt i32 %i1, %n
  br i1 %cc, label %loop, label %exit
exit:
  %q = getelementptr <4 x float>, ptr addrspace(1) %out, i32 %tid
  store volatile <4 x float> %e3, ptr addrspace(1) %q
  store volatile <4 x float> %f3, ptr addrspace(1) %q
  ret void
}

; Pins with different register numbers in one variable conflict: nothing is
; added.

; IR-LABEL: @conflict(
; IR: %c1 = add i32 %c, %x
; IR-NEXT: %c2 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %c1, i32 13, i32 1)
; IR-NEXT: %i1 = add i32 %i, 1
define amdgpu_kernel void @conflict(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %x = load volatile i32, ptr addrspace(1) %p
  %z = load volatile i32, ptr addrspace(1) %p
  %c0 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %z, i32 12, i32 1)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i1, %loop]
  %c = phi i32 [%c0, %entry], [%c2, %loop]
  %c1 = add i32 %c, %x
  %c2 = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %c1, i32 13, i32 1)
  %i1 = add i32 %i, 1
  %cc = icmp slt i32 %i1, %n
  br i1 %cc, label %loop, label %exit
exit:
  %q = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %c2, ptr addrspace(1) %q
  ret void
}
