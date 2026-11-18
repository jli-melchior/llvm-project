; RUN: llc -mtriple=amdgcn -mcpu=gfx950 -stop-after=amdgpu-sched-bank < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 < %s | FileCheck -check-prefix=GCN %s

; Each accumulator chain of regnum -1 strict AGPR pins gets its own AGPR index.
; Two chains that share their zero initial value stay separate.

declare <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float>, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)

; CHECK-LABEL: name: two_chains
; CHECK: A_SCHEDULE_AGPR_B128 {{.*}}%{{[0-9]+}}, 0, 1
; CHECK: A_SCHEDULE_AGPR_B128 {{.*}}%{{[0-9]+}}, 4, 1
define amdgpu_kernel void @two_chains(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %c0 = phi <4 x float> [ zeroinitializer, %entry ], [ %p0, %loop ]
  %c1 = phi <4 x float> [ zeroinitializer, %entry ], [ %p1, %loop ]
  %pa = getelementptr <4 x half>, ptr addrspace(1) %in, i32 %i
  %a = load <4 x half>, ptr addrspace(1) %pa
  %r0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %c0, i32 0, i32 0, i32 0)
  %p0 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r0, i32 -1, i32 1)
  %r1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %c1, i32 0, i32 0, i32 0)
  %p1 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r1, i32 -1, i32 1)
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %o1 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1
  store <4 x float> %p0, ptr addrspace(1) %out
  store <4 x float> %p1, ptr addrspace(1) %o1
  ret void
}

; One chain from the MFMA before the loop, through the loop PHI and the
; accumulator input of the MFMA after the loop.
; CHECK-LABEL: name: loop_chain
; CHECK-COUNT-3: A_SCHEDULE_AGPR_B128 {{.*}}%{{[0-9]+}}, 0, 1
; CHECK-NOT: A_SCHEDULE_AGPR_B128
; GCN-LABEL: {{^}}loop_chain:
; GCN-NOT: v_accvgpr_mov
; GCN: v_mfma_f32_16x16x16_f16 a[0:3], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[0:3]
; GCN-NOT: v_accvgpr_mov
; GCN: v_mfma_f32_16x16x16_f16 a[0:3], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[0:3]
; GCN-NOT: v_accvgpr_mov
; GCN: v_mfma_f32_16x16x16_f16 a[0:3], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[0:3]
; GCN-NOT: v_accvgpr_mov
; GCN: s_endpgm
define amdgpu_kernel void @loop_chain(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n, <4 x float> %init) #0 {
entry:
  %a0 = load <4 x half>, ptr addrspace(1) %in
  %r0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a0, <4 x half> %a0, <4 x float> %init, i32 0, i32 0, i32 0)
  %p0 = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r0, i32 -1, i32 1)
  br label %loop

loop:
  %i = phi i32 [ 1, %entry ], [ %i.next, %loop ]
  %c = phi <4 x float> [ %p0, %entry ], [ %p, %loop ]
  %pa = getelementptr <4 x half>, ptr addrspace(1) %in, i32 %i
  %a = load <4 x half>, ptr addrspace(1) %pa
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %c, i32 0, i32 0, i32 0)
  %p = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r, i32 -1, i32 1)
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  %b = load <4 x half>, ptr addrspace(1) %in
  %e = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %b, <4 x half> %b, <4 x float> %p, i32 0, i32 0, i32 0)
  %q = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %e, i32 -1, i32 1)
  store <4 x float> %q, ptr addrspace(1) %out
  ret void
}

; An exact pin takes a0..a3, so the chain starts at a4.
; CHECK-LABEL: name: after_exact
; CHECK: A_SCHEDULE_AGPR_B128 {{.*}}%{{[0-9]+}}, 0, 1
; CHECK: A_SCHEDULE_AGPR_B128 {{.*}}%{{[0-9]+}}, 4, 1
define amdgpu_kernel void @after_exact(ptr addrspace(1) %out, <4 x half> %a, <4 x float> %c, <4 x float> %d) #0 {
  %x = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %c, i32 0, i32 1)
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %d, i32 0, i32 0, i32 0)
  %p = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r, i32 -1, i32 1)
  %o1 = getelementptr <4 x float>, ptr addrspace(1) %out, i32 1
  store <4 x float> %x, ptr addrspace(1) %out
  store <4 x float> %p, ptr addrspace(1) %o1
  ret void
}

; A soft pin with regnum -1 gets no index.
; CHECK-LABEL: name: soft_any
; CHECK-NOT: A_SCHEDULE_AGPR
; CHECK: S_ENDPGM
define amdgpu_kernel void @soft_any(ptr addrspace(1) %out, <4 x half> %a, <4 x float> %c) #0 {
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %c, i32 0, i32 0, i32 0)
  %p = call <4 x float> @llvm.amdgcn.sched.agpr.v4f32(<4 x float> %r, i32 -1, i32 0)
  store <4 x float> %p, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" }
