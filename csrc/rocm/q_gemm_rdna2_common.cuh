// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared RDNA2 (gfx1030) W4A16 device helpers used by both the dense kernel
// (q_gemm_rdna2.cu) and the MoE kernel (moe_q_gemm_rdna2.cu): the fdot2 inner
// dot, the packed 64-bit-CAS atomic add (RDNA2 has no v_global_atomic_pk_add_f16),
// and the 4-wide zero/scale loads. Dequant primitives live in qdq_4_rdna2.cuh.

#ifndef _q_gemm_rdna2_common_cuh
#define _q_gemm_rdna2_common_cuh

#include <cstdint>

#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2 {

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  // RDNA2 has v_dot2_f32_f16 (__builtin_amdgcn_fdot2): fp32 += a.x*b.x + a.y*b.y
  // in one instruction with the accumulator held in fp32 throughout. hipcc does
  // not peephole __hfma2 + cast + add into it, so we issue the builtin.
  float result = 0.0f;
  const half2* a2_ptr = (const half2*)a_ptr;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

// Packed atomic-add via CAS-loop on a 64-bit word (4 fp16 lanes per CAS).
// RDNA2 has no native v_global_atomic_pk_add_f16, so this lowers to
// global_atomic_cmpswap_b64 + retry.
__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Load 4 packed zeros (column n..n+3) from a [groups, N/8] uint32 tensor. n is a
// multiple of 4 by construction, so the 4 nibbles live within one uint32 word.
__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

__forceinline__ __device__ void load4_scales(const half* scales_row, int n,
                                             half (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gptq_rdna2
}  // namespace vllm

#endif  // _q_gemm_rdna2_common_cuh
