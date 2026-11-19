; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -stop-after=amdgpu-sched-bank < %s | FileCheck -check-prefix=TIED %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -stop-after=amdgpu-sched-bank -amdgpu-strict-pin-tie-src-dst=0 < %s | FileCheck -check-prefix=UNTIED %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefix=GCN %s

; -amdgpu-strict-pin-tie-src-dst (default) ties a strict sched.vgpr pin's
; source to its result before RA. The defining instruction then writes the
; pinned register, and later uses read that register with no copy in between.
; -amdgpu-strict-pin-tie-src-dst=0 leaves the pseudo untied.

declare i32 @llvm.amdgcn.sched.vgpr.i32(i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; TIED: V_SCHEDULE_VGPR_B32 killed %{{[0-9]+}}(tied-def 0), 40, 1
; UNTIED: V_SCHEDULE_VGPR_B32 killed %{{[0-9]+}}, 40, 1
; UNTIED-NOT: tied-def
; GCN-LABEL: {{^}}vgpr40_tied:
; GCN: v_dual_add_nc_u32 v40,
; GCN-NOT: v_mov
; GCN: v_dual_add_nc_u32 v{{[0-9]+}}, {{[^,]+}}, v40
define amdgpu_kernel void @vgpr40_tied(ptr addrspace(1) %out, i32 %x, i32 %y) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %sum = add i32 %tid, %x
  %pinned = call i32 @llvm.amdgcn.sched.vgpr.i32(i32 %sum, i32 40, i32 1)
  %a = add i32 %pinned, %y
  %b = xor i32 %pinned, %y
  store i32 %a, ptr addrspace(1) %out
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %b, ptr addrspace(1) %out2
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "amdgpu-waves-per-eu"="1,1" }
