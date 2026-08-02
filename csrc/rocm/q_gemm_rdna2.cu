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
// Output accumulation: gridDim.z splits K, so each output is a sum of several
// partials that have to meet somewhere. This kernel gives each split its own
// scratch slice, writes it with one 16-byte store, and sums the slices in split
// order in a second pass -- no atomics, so no CAS retries, and the same answer
// every run. The scratch is splits*M*N floats, so the launcher falls back to
// accumulating in place (global_atomic_cmpswap_x2, 2 fp32 lanes per CAS, which
// is as wide as RDNA2 goes: the ISA has no float-add atomic at any width, only
// FCMPSWAP/FMIN/FMAX) once that would be too large, which is prefill, where M
// and N already fill the machine and the atomics cost little.
//
// What neither path does any more is sum the splits in fp16. That needed no
// scratch and one fewer CAS per 4 columns, but rounded every partial to 11
// bits: against an exact reference it was 12x the error of a plain fp16 sum at
// K=4096 and 38x at K=7168, and since CAS order is arbitrary it also gave a
// different answer run to run -- visible as flipped argmax ties in greedy
// decoding. N % 8 == 0 keeps stores and CAS pairs aligned for both paths.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna2.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gptq_rdna2 {

// BLOCK_KN_SIZE (a template parameter, == threads/block) is the per-block K tile:
// each block covers BLOCK_KN_SIZE K elements and BLOCK_KN_SIZE*4 N columns, and
// gridDim.z = K / BLOCK_KN_SIZE splits K; the splits are combined as described
// under output accumulation above. 256 = 8 waves on RDNA2 wave32; the M==1
// decode path bumps it to 512 to halve gridDim.z (fewer splits to combine, and
// fewer CAS when it falls back to those) on large shapes -- see launch_gemm_q4.

// Device code below is RDNA2-only; non-RDNA2 device passes fall through to the
// empty __global__ stub at the #else for symbol parity.
// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

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

// Sum the per-split slices in split order and round once to the output dtype.
// Four columns per thread so both the loads and the store stay vector-width.
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

template <int M_COUNT, int BLOCK_KN_SIZE>
__global__ void gemm_q4_kernel_rdna2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    float* __restrict__ c, float* __restrict__ partials, const int size_m,
    const int size_n, const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm) {
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

  // Two ways to combine this split with the others, chosen by the launcher.
  //
  // With `partials`, each split owns a slice and writes its four sums with one
  // 16-byte store (n % 4 == 0 and size_n % 8 == 0 keep it aligned), and a second
  // pass sums the slices in split order. No atomics, so no CAS retries and the
  // same answer every run. It costs a splits*M*N fp32 buffer, which is why the
  // launcher only takes this path while that buffer stays small.
  //
  // Otherwise accumulate in place, two fp32 lanes per CAS.
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
    if (partials) {
      float* out = partials +
                   ((long)blockIdx.z * size_m + (offset_m + m)) * size_n + n;
      *reinterpret_cast<float4*>(out) =
          make_float4(block_c[m][0], block_c[m][1], block_c[m][2],
                      block_c[m][3]);
    } else {
      float* out = c + (offset_m + m) * size_n + n;
      atomic_add_pk2_f32(out, block_c[m][0], block_c[m][1]);
      atomic_add_pk2_f32(out + 2, block_c[m][2], block_c[m][3]);
    }
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <int M_COUNT, int BLOCK_KN_SIZE>
__global__ void gemm_q4_kernel_rdna2(const half*, const uint32_t*,
                                     const uint32_t*, const half*, float*,
                                     float*, const int, const int, const int,
                                     const int, const int, const int*) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <int M_COUNT, int BLOCK_KN_SIZE>
void launch_gemm_q4_for_mcount(const half* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const half* b_scales,
                               const int* b_q_perm, float* c, float* partials,
                               int size_m, int size_n, int size_k, int groups,
                               int zero_offset, cudaStream_t stream) {
  dim3 block(BLOCK_KN_SIZE);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna2<M_COUNT, BLOCK_KN_SIZE><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, partials, size_m, size_n, size_k,
      groups, zero_offset, b_q_perm);
}

// Dispatch to the largest M_COUNT template that doesn't waste more than half a
// tile. Caps at 8: RDNA2 has no WMMA, so M >= 16 (and M >= 64) are handled by
// the same scalar kernel, grid-Y-tiled at M_COUNT=8 — correct at every M, just
// leaving prefill throughput on the table that WMMA would recover on RDNA3.
//   M=1   -> M_COUNT=1   M=2,3 -> 2   M=4-7 -> 4   M>=8 -> 8 (tiled over grid.y)
template <int BK>
void launch_gemm_q4_bk(const half* a, const uint32_t* b_q_weight,
                       const uint32_t* b_qzeros, const half* b_scales,
                       const int* b_q_perm, float* c, float* partials,
                       int size_m, int size_n, int size_k, int groups,
                       int zero_offset, cudaStream_t stream) {
  if (size_m == 1) {
    launch_gemm_q4_for_mcount<1, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, partials, size_m, size_n, size_k,
                                     groups, zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<2, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, partials, size_m, size_n, size_k,
                                     groups, zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<4, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, partials, size_m, size_n, size_k,
                                     groups, zero_offset, stream);
  } else {
    launch_gemm_q4_for_mcount<8, BK>(a, b_q_weight, b_qzeros, b_scales, b_q_perm,
                                     c, partials, size_m, size_n, size_k,
                                     groups, zero_offset, stream);
  }
}

// Scratch budget for the reproducible combine. This is a tile size, not a
// threshold to give up at: rows are independent, so M is processed in passes
// that fit the budget instead of falling back to the CAS. A pass is at least
// one row, so there is no shape this cannot serve, and the answer no longer
// depends on which block reached memory first at any size.
static constexpr long SPLIT_SCRATCH_MAX_BYTES = 64L << 20;

void launch_gemm_q4(const half* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const half* b_scales,
                    const int* b_q_perm, half* out, int size_m, int size_n,
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
  const bool bk512 = (size_m == 1 && blocks512 >= 40);
  const int bk = bk512 ? 512 : 256;

  // One scratch slice per K-split lets the splits be summed in a fixed order
  // instead of raced together through CAS. With a single split each output has
  // one writer and the in-place accumulator is already order-free.
  const int num_splits = (size_k + bk - 1) / bk;
  const bool two_phase = num_splits > 1;

  // Rows are independent, so the scratch is bounded by processing M in passes
  // rather than by abandoning the reproducible combine on large shapes.
  const long row_bytes = (long)num_splits * size_n * (long)sizeof(float);
  int rows_per_pass = size_m;
  if (two_phase && row_bytes > 0) {
    const long budget = SPLIT_SCRATCH_MAX_BYTES / row_bytes;
    rows_per_pass = (int)(budget < 1 ? 1
                          : (budget < (long)size_m ? budget : (long)size_m));
  }

  auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  // Two-phase writes every slice exactly once, so it needs no zeroing; the
  // in-place path is an accumulator and does.
  at::Tensor buf =
      two_phase
          ? at::empty({(long)num_splits, (long)rows_per_pass * size_n}, f32)
          : at::zeros({(long)size_m * size_n}, f32);
  float* partials = two_phase ? (float*)buf.data_ptr() : nullptr;
  float* acc = two_phase ? nullptr : (float*)buf.data_ptr();

  constexpr int RED_THREADS = 256;
  for (int m0 = 0; m0 < size_m; m0 += rows_per_pass) {
    const int rows =
        (size_m - m0) < rows_per_pass ? (size_m - m0) : rows_per_pass;
    const half* a_pass = a + (long)m0 * size_k;
    half* out_pass = out + (long)m0 * size_n;
    float* acc_pass = acc ? acc + (long)m0 * size_n : nullptr;

    if (bk512) {
      launch_gemm_q4_bk<512>(a_pass, b_q_weight, b_qzeros, b_scales, b_q_perm,
                             acc_pass, partials, rows, size_n, size_k, groups,
                             zero_offset, stream);
    } else {
      launch_gemm_q4_bk<256>(a_pass, b_q_weight, b_qzeros, b_scales, b_q_perm,
                             acc_pass, partials, rows, size_n, size_k, groups,
                             zero_offset, stream);
    }

    // Both paths finish the same way: sum whatever slices exist (one, for the
    // in-place accumulator) and round once to the output dtype.
    const long mn = (long)rows * size_n;
    const long vec = mn / 4;  // 4 columns per thread; size_n % 8 == 0
    const int red_blocks = (int)((vec + RED_THREADS - 1) / RED_THREADS);
    reduce_splits_kernel_rdna2<<<red_blocks, RED_THREADS, 0, stream>>>(
        two_phase ? partials : acc_pass, out_pass, two_phase ? num_splits : 1,
        mn);
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
  at::Tensor c = torch::empty({size_m, size_n}, opts);

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
