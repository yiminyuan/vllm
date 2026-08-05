// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared RDNA2 (gfx1030) W4A16 device helpers used by both the dense kernel
// (q_gemm_rdna2.cu) and the MoE kernel (moe_q_gemm_rdna2.cu): the fdot2 inner
// dot, the packed 64-bit-CAS atomic add (RDNA2 has no float-add atomic at any
// width), the split-K reduce, and the 4-wide zero/scale loads. Dequant
// primitives live in qdq_4_rdna2.cuh.

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

// Packed atomic-add via CAS-loop on a 64-bit word (2 fp32 lanes per CAS).
// RDNA2 has no float-add atomic of any width: the ISA's only float atomics are
// FCMPSWAP, FMIN and FMAX, so any float accumulation lowers to
// global_atomic_cmpswap_x2 + retry. Two fp32 lanes is the widest that CAS
// carries, which halves the retry traffic against one CAS per float.
//
// The accumulator is fp32 rather than the output's fp16 because the caller sums
// gridDim.z K-splits here. Rounding each partial to fp16 cost ~0.5% at K=7168,
// an order of magnitude worse than an fp16 sum of the same terms, and made the
// result depend on the order the splits happened to land in.
__forceinline__ __device__ void atomic_add_pk2_f32(float* addr, float v0,
                                                   float v1) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      float f[2];
    } cur, sum;
    cur.u = old;
    sum.f[0] = cur.f[0] + v0;
    sum.f[1] = cur.f[1] + v1;
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// Sum the per-split slices in split order and round once to the output dtype.
// Four columns per thread so both the loads and the store stay vector-width.
// A template so each translation unit gets its own instantiation; a plain
// __global__ in a header would need relocatable device code to launch across
// one. Uses no RDNA2-specific instruction, so it needs no arch guard.
template <int UNUSED>
__global__ void reduce_splits_kernel_rdna2(const float* __restrict__ partials,
                                           half* __restrict__ c,
                                           const int num_splits,
                                           const long mn) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * 4 >= mn) return;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int z = 0; z < num_splits; ++z) {
    const float4 v =
        reinterpret_cast<const float4*>(partials + (long)z * mn)[idx];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  union {
    unsigned long long u;
    half2 h[2];
  } out;
  out.h[0] = __halves2half2(__float2half_rn(acc.x), __float2half_rn(acc.y));
  out.h[1] = __halves2half2(__float2half_rn(acc.z), __float2half_rn(acc.w));
  reinterpret_cast<unsigned long long*>(c)[idx] = out.u;
}

// Combine the split slices and fold `fold` consecutive slot rows into one
// output row, over one N tile. Both loops run in a fixed order, so the sum does
// not depend on which block reached memory first -- the property a CAS epilogue
// cannot give. ``c`` is already offset to the tile's first column, so
// ``size_n`` stays the output row stride while ``partial_n`` is the scratch
// row stride.
template <int UNUSED>
__global__ void reduce_splits_fold_tile_kernel_rdna2(
    const float* __restrict__ partials, half* __restrict__ c,
    const int num_splits, const int fold, const long rows,
    const long slot_mn_tile, const int partial_n, const int n_cols,
    const int size_n) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long n4 = n_cols / 4;
  if (n4 == 0 || idx >= rows * n4) return;
  const long row = idx / n4;
  const long col = idx - row * n4;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int z = 0; z < num_splits; ++z) {
    const float* base =
        partials + (long)z * slot_mn_tile + row * (long)fold * partial_n;
    for (int f = 0; f < fold; ++f) {
      const float4 v =
          reinterpret_cast<const float4*>(base + (long)f * partial_n)[col];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  union {
    unsigned long long u;
    half2 h[2];
  } out;
  out.h[0] = __halves2half2(__float2half_rn(acc.x), __float2half_rn(acc.y));
  out.h[1] = __halves2half2(__float2half_rn(acc.z), __float2half_rn(acc.w));
  reinterpret_cast<unsigned long long*>(c + row * (long)size_n)[col] = out.u;
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
