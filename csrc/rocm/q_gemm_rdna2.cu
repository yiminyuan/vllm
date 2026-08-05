// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA2 (gfx1030 / Radeon Pro W6800 class). fp16-only
// port of q_gemm_rdna3.cu: the same exllamav2 "+1024" bit-trick dequant and
// v_dot2_f32_f16 (fdot2) inner dot, with the bf16 and WMMA paths removed —
// RDNA2 has neither v_dot2_f32_bf16 nor WMMA. The fp16 scalar path is exactly
// the one the RDNA3 kernel keeps for all M < 64 ("bit-trick dequant beats WMMA
// below M=64"); on RDNA2 it serves every M, grid-Y-tiled at M_COUNT=8.
//
// Output accumulation: gfx1030, like gfx11, has no native
// v_global_atomic_pk_add_f16, so the epilogue emulates a packed atomic add
// with a global_atomic_cmpswap_b64 CAS loop (4 fp16 lanes per CAS) directly
// into a caller-zeroed T-typed output — no FP32 scratch buffer / memset / cast
// pass. N % 8 == 0 (enforced by can_implement) makes every write 8-byte
// aligned as the b64 CAS requires.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"

namespace vllm {
namespace gptq_rdna2 {

// BLOCK_KN_SIZE (a template parameter, == threads/block) is the per-block K tile:
// each block covers BLOCK_KN_SIZE K elements and BLOCK_KN_SIZE*4 N columns, and
// gridDim.z = K / BLOCK_KN_SIZE splits K with atomic accumulation. 256 = 8 waves
// on RDNA2 wave32; the M==1 decode path bumps it to 512 to halve gridDim.z (fewer
// atomic-CAS per output) on large shapes -- see launch_gemm_q4.

// Device code below is RDNA2-only; non-RDNA2 device passes fall through to the
// empty __global__ stub at the #else for symbol parity.
// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int M_COUNT, int BLOCK_KN_SIZE>
__global__ void gemm_q4_kernel_rdna2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    half* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset, const int* __restrict__ b_q_perm) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // LDS layout: [M_COUNT][BLOCK_KN_SIZE + LDS_PAD]. PAD=8 per M row breaks the
  // 512-byte alignment that would otherwise bank-collide when a thread reads
  // block_a[0..M_COUNT-1][k] (same k, different m). The block launches with
  // BLOCK_KN_SIZE threads, so the A-stage below is a 1:1 thread->K map.
  constexpr int LDS_PAD = 8;
  __shared__ half block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A: each thread loads 1 K element per M row into LDS (with optional
  // act-order permutation), a 1:1 map since threads/block == BLOCK_KN_SIZE.
  // Slots past size_m are zero-padded (0 dot contribution); their atomic write
  // is skipped below. The fp16 inner loop indexes block_a[m][a_off]
  // unconditionally, so A is always staged (no M=1 global-read fast path).
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      half av;
      if (offset_m + m < size_m) {
        const half* a_row = a + (offset_m + m) * size_k;
        if (b_q_perm)
          av = a_row[b_q_perm[offset_k + t]];
        else
          av = a_row[offset_k + t];
      } else {
        av = __float2half_rn(0.0f);  // zero-pad invalid M rows
      }
      block_a[m][t] = av;
    }
  }
  // All THREADS_X threads reach the barrier (the load above is guarded, the
  // barrier is not), so it is non-divergent even for out-of-range n.
  __syncthreads();
  if (n >= size_n) return;

  // Group bookkeeping. We require size_k % groups == 0 (groupsize divides K).
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // qweight stride: weights are [K/8, N] uint32 with K packed at dim 0.
  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Per-column dequant constants: one (z1z16, y1y16) double-pair per column,
  // enabling the exllama upper-nibble-*16 trick.
  half2 z1z16_h[4][2], y1y16_h[4][2];

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
    const half* sc_row = b_scales + g * size_n;
    int zeros[4];
    half scales[4];
    load4_zeros(qz_row, n, zeros);
    load4_scales(sc_row, n, scales);
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scales[i],
                           z1z16_h[i], y1y16_h[i]);
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // group_size >= 32 required (mirrors exllama): the outer loop advances K by
  // 32, so a group boundary must not fall between j-iterations.
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    // Prefetch all four j-iterations' weight words up front; the dependent
    // dequant + FMA below hides the global_load latency.
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      half2 dq[4][4];
      dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0], y1y16_h[0]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1], y1y16_h[1]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2], y1y16_h[2]);
      dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3], y1y16_h[3]);

  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += dot22_8_f(dq[0], a_ptr);
        block_c[m][1] += dot22_8_f(dq[1], a_ptr);
        block_c[m][2] += dot22_8_f(dq[2], a_ptr);
        block_c[m][3] += dot22_8_f(dq[3], a_ptr);
      }
    }
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // Pack the 4 fp32 partial sums into 2 pairs and atomically add all four lanes
  // in one 64-bit CAS directly to the T-typed output (caller pre-zeros it).
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
    half* out = c + (offset_m + m) * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                               __float2half_rn(block_c[m][1]));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                               __float2half_rn(block_c[m][3]));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <int M_COUNT, int BLOCK_KN_SIZE>
__global__ void gemm_q4_kernel_rdna2(const half*, const uint32_t*,
                                     const uint32_t*, const half*, half*,
                                     const int, const int, const int, const int,
                                     const int, const int*) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <int M_COUNT, int BLOCK_KN_SIZE>
void launch_gemm_q4_for_mcount(const half* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const half* b_scales,
                               const int* b_q_perm, half* c, int size_m,
                               int size_n, int size_k, int groups,
                               int zero_offset, cudaStream_t stream) {
  dim3 block(BLOCK_KN_SIZE);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna2<M_COUNT, BLOCK_KN_SIZE><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

// Dispatch to the largest M_COUNT template that doesn't waste more than half a
// tile. Caps at 8: RDNA2 has no WMMA, so M >= 16 (and M >= 64) are handled by
// the same scalar kernel, grid-Y-tiled at M_COUNT=8 — correct at every M, just
// leaving prefill throughput on the table that WMMA would recover on RDNA3.
//   M=1   -> M_COUNT=1   M=2,3 -> 2   M=4-7 -> 4   M>=8 -> 8 (tiled over grid.y)
template <int BK>
void launch_gemm_q4_bk(const half* a, const uint32_t* b_q_weight,
                       const uint32_t* b_qzeros, const half* b_scales,
                       const int* b_q_perm, half* c, int size_m, int size_n,
                       int size_k, int groups, int zero_offset,
                       cudaStream_t stream) {
  if (size_m == 1) {
    launch_gemm_q4_for_mcount<1, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, size_m, size_n, size_k, groups,
                                     zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<2, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, size_m, size_n, size_k, groups,
                                     zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<4, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, size_m, size_n, size_k, groups,
                                     zero_offset, stream);
  } else {
    launch_gemm_q4_for_mcount<8, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, size_m, size_n, size_k, groups,
                                     zero_offset, stream);
  }
}

void launch_gemm_q4(const half* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const half* b_scales,
                    const int* b_q_perm, half* c, int size_m, int size_n,
                    int size_k, int groups, bool use_v2_format,
                    cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  // Block-K size: 512 halves gridDim.z (=> half the atomic-CAS contention per
  // output) and gains a few % GDDR6 BW on the memory-bound M==1 decode GEMV,
  // but only when the problem still yields enough blocks to fill the CUs (small
  // shapes starve occupancy, so keep 256). For M>1 the kernel is more
  // compute-bound and the larger block regresses, so 512 is gated to decode;
  // every other case is identical to the 256 path. Block count at BK=512 is
  // ceil(N/2048)*ceil(K/512); 40 is the measured knee.
  const long blocks512 =
      ((long)(size_n + 2047) / 2048) * ((long)(size_k + 511) / 512);
  if (size_m == 1 && blocks512 >= 40) {
    launch_gemm_q4_bk<512>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                           size_m, size_n, size_k, groups, zero_offset, stream);
  } else {
    launch_gemm_q4_bk<256>(a, b_q_weight, b_qzeros, b_scales, b_q_perm, c,
                           size_m, size_n, size_k, groups, zero_offset, stream);
  }
}

}  // namespace gptq_rdna2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format                bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            half

torch::Tensor gptq_gemm_rdna2(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf,
              "a must be half (RDNA2 W4A16 kernel is fp16-only)");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32, "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  vllm::gptq_rdna2::launch_gemm_q4(
      (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(), (const half*)b_scales.data_ptr(),
      g_idx_ptr, (half*)c.data_ptr(), size_m, size_n, size_k, groups,
      use_v2_format, stream);

  return c;
}
