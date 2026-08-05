// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 MoE kernel for RDNA2 (gfx1030). Fuses expert routing + W4A16 dequant
// (exllamav2 "+1024" bit-trick, qdq_4_rdna2.cuh) + v_dot2_f32_f16 (fdot2) dot,
// with a 64-bit-CAS packed atomic epilogue — the same fp16-only compute core as
// the dense kernel (q_gemm_rdna2.cu), extended with an expert dimension and the
// standard vLLM MoE routing (sorted_token_ids / expert_ids / block_size_m from
// moe_align_block_size). fp16-only: RDNA2 has no bf16 packed dot and no WMMA.
//
// Routing convention (matches the Triton fused_moe_kernel):
//   sid        = sorted_token_ids[block*BLOCK_M + m]   (flat token-slot id)
//   valid      = sid < num_valid   (num_valid = input_rows * top_k; pad masked)
//   input row  = sid / top_k                            (top_k op-param)
//   output row = output_topk ? sid / output_topk : sid  (>0 => fused moe_sum)
//   expert     = expert_ids[block]   (-1 => skip block)
//   result   *= topk_weights[sid]   when mul_weights is set
// Both GEMMs share one kernel: w13 (top_k=top_k, output_topk=0, mul=false) and
// w2 (top_k=1, output_topk=top_k, mul=true) — the caller pre-zeros the output.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"

namespace vllm {
namespace gptq_rdna2 {

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// BLOCK_M token-rows of a single expert per block; grid.y tiles the padded
// sorted-token space, grid.z splits K with atomic accumulation (as the dense
// kernel), grid.x tiles N at BLOCK_KN_SIZE*4 columns per block.
template <int BLOCK_M, int BLOCK_KN_SIZE>
__global__ void moe_gemm_q4_kernel_rdna2(
    const half* __restrict__ input, half* __restrict__ output,
    const uint32_t* __restrict__ b_q_weight, const half* __restrict__ b_scales,
    const uint32_t* __restrict__ b_qzeros,
    const float* __restrict__ topk_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int size_n,
    const int size_k, const int groups, const int zero_offset, const int top_k,
    const int output_topk, const int num_valid, const bool mul_weights) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int block_m_idx = blockIdx.y;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // Skip padded token-blocks and empty (expert-parallel masked) blocks.
  if (block_m_idx * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert = expert_ids[block_m_idx];
  if (expert < 0) return;

  // Resolve this block's token-slots -> input rows.
  int sids[BLOCK_M], in_rows[BLOCK_M];
  bool row_valid[BLOCK_M];
  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    int sid = sorted_token_ids[block_m_idx * BLOCK_M + m];
    sids[m] = sid;
    row_valid[m] = sid < num_valid;
    in_rows[m] = row_valid[m] ? (sid / top_k) : 0;
  }

  constexpr int LDS_PAD = 8;
  __shared__ half block_a[BLOCK_M][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A: 1 K element per row into LDS (1:1 thread->K); invalid rows zero-pad.
  if (offset_k + t < end_k) {
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
      half av = __float2half_rn(0.0f);
      if (row_valid[m]) av = input[(long)in_rows[m] * size_k + offset_k + t];
      block_a[m][t] = av;
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Per-expert weight/scale/zero bases.
  const uint32_t* qw = b_q_weight + (long)expert * ((long)(size_k / 8) * size_n);
  const half* sc = b_scales + (long)expert * ((long)groups * size_n);
  const uint32_t* qz = b_qzeros + (long)expert * ((long)groups * (size_n / 8));

  int qk = offset_k / 8;
  const uint32_t* b_ptr = qw + qk * size_n + n;

  half2 z1z16_h[4][2], y1y16_h[4][2];
  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = qz + g * (size_n / 8);
    const half* sc_row = sc + g * size_n;
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

  float block_c[BLOCK_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }
    int4 b_w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) b_w[j] = *(const int4*)(b_ptr + j * size_n);
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
      for (int m = 0; m < BLOCK_M; ++m) {
        const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
        block_c[m][0] += dot22_8_f(dq[0], a_ptr);
        block_c[m][1] += dot22_8_f(dq[1], a_ptr);
        block_c[m][2] += dot22_8_f(dq[2], a_ptr);
        block_c[m][3] += dot22_8_f(dq[3], a_ptr);
      }
    }
    k += 32;
  }

  #pragma unroll
  for (int m = 0; m < BLOCK_M; ++m) {
    if (!row_valid[m]) continue;
    const int out_row = (output_topk > 0) ? (sids[m] / output_topk) : sids[m];
    const float w = mul_weights ? topk_weights[sids[m]] : 1.0f;
    half* out = output + (long)out_row * size_n + n;
    half2 r01 = __halves2half2(__float2half_rn(block_c[m][0] * w),
                               __float2half_rn(block_c[m][1] * w));
    half2 r23 = __halves2half2(__float2half_rn(block_c[m][2] * w),
                               __float2half_rn(block_c[m][3] * w));
    atomic_add_pk4_f16(out, r01, r23);
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <int BLOCK_M, int BLOCK_KN_SIZE>
__global__ void moe_gemm_q4_kernel_rdna2(
    const half*, half*, const uint32_t*, const half*, const uint32_t*,
    const float*, const int*, const int*, const int*, const int, const int,
    const int, const int, const int, const int, const int, const bool) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int BLOCK_M, int BK>
void launch_moe_gemm_q4_bm(
    const half* input, half* output, const uint32_t* b_q_weight,
    const half* b_scales, const uint32_t* b_qzeros, const float* topk_weights,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, int em, int size_n, int size_k,
    int groups, int zero_offset, int top_k, int output_topk, int num_valid,
    bool mul_weights, cudaStream_t stream) {
  dim3 block(BK);
  dim3 grid((size_n + BK * 4 - 1) / (BK * 4), (em + BLOCK_M - 1) / BLOCK_M,
            (size_k + BK - 1) / BK);
  moe_gemm_q4_kernel_rdna2<BLOCK_M, BK><<<grid, block, 0, stream>>>(
      input, output, b_q_weight, b_scales, b_qzeros, topk_weights,
      sorted_token_ids, expert_ids, num_tokens_post_padded, size_n, size_k,
      groups, zero_offset, top_k, output_topk, num_valid, mul_weights);
}

void launch_moe_gemm_q4(const half* input, half* output,
                        const uint32_t* b_q_weight, const half* b_scales,
                        const uint32_t* b_qzeros, const float* topk_weights,
                        const int* sorted_token_ids, const int* expert_ids,
                        const int* num_tokens_post_padded, int em, int size_n,
                        int size_k, int groups, int zero_offset, int top_k,
                        int output_topk, int num_valid, bool mul_weights,
                        int block_size_m, cudaStream_t stream) {
  constexpr int BK = 256;
  if (block_size_m <= 1) {
    launch_moe_gemm_q4_bm<1, BK>(
        input, output, b_q_weight, b_scales, b_qzeros, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded, em, size_n,
        size_k, groups, zero_offset, top_k, output_topk, num_valid, mul_weights,
        stream);
  } else {
    launch_moe_gemm_q4_bm<4, BK>(
        input, output, b_q_weight, b_scales, b_qzeros, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded, em, size_n,
        size_k, groups, zero_offset, top_k, output_topk, num_valid, mul_weights,
        stream);
  }
}

}  // namespace gptq_rdna2
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
//
// Inputs:
//   input       [Min, K]          half   (Min rows; input row = sid / top_k)
//   output      [Mout, N]         half   (pre-zeroed by caller; accumulated)
//   b_q_weight  [E, K/8, N]       uint32 (per-expert, gptq_shuffle'd)
//   b_scales    [E, groups, N]    half
//   b_qzeros    [E, groups, N/8]  uint32
//   topk_weights[Min*top_k] float32 or empty (used iff mul_weights)
//   sorted_token_ids [EM]         int32  (from moe_align_block_size)
//   expert_ids  [EM/block_size_m] int32
//   num_tokens_post_padded [1]    int32
//   top_k, block_size_m, mul_weights, use_v2_format, output_topk
// ---------------------------------------------------------------------------

void moe_gptq_gemm_rdna2(
    torch::Tensor input, torch::Tensor output, torch::Tensor b_q_weight,
    torch::Tensor b_scales, torch::Tensor b_qzeros, torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded, int64_t top_k, int64_t block_size_m,
    bool mul_weights, bool use_v2_format, int64_t output_topk) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be on device");
  TORCH_CHECK(input.scalar_type() == torch::kHalf,
              "input must be half (RDNA2 W4A16 MoE kernel is fp16-only)");
  TORCH_CHECK(output.scalar_type() == torch::kHalf, "output must be half");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf, "b_scales must be half");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be [E, K/8, N]");
  TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt32,
              "sorted_token_ids must be int32");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32,
              "expert_ids must be int32");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int size_k = (int)input.size(1);
  const int size_n = (int)b_q_weight.size(2);
  const int groups = (int)b_scales.size(1);
  const int em = (int)sorted_token_ids.size(0);
  const int num_valid = (int)input.size(0) * (int)top_k;
  const int zero_offset = use_v2_format ? 0 : 1;

  TORCH_CHECK(b_q_weight.size(1) * 8 == size_k, "b_q_weight dim1 must be K/8");
  TORCH_CHECK(b_scales.size(2) == size_n, "b_scales last dim must be N");
  TORCH_CHECK(size_n % 8 == 0, "N must be a multiple of 8 (64-bit atomic CAS)");
  TORCH_CHECK(size_k % groups == 0, "K must be divisible by group count");

  const float* tw_ptr =
      (mul_weights && topk_weights.numel() > 0)
          ? (const float*)topk_weights.data_ptr()
          : nullptr;

  vllm::gptq_rdna2::launch_moe_gemm_q4(
      (const half*)input.data_ptr(), (half*)output.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(), (const half*)b_scales.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(), tw_ptr,
      (const int*)sorted_token_ids.data_ptr(),
      (const int*)expert_ids.data_ptr(),
      (const int*)num_tokens_post_padded.data_ptr(), em, size_n, size_k, groups,
      zero_offset, (int)top_k, (int)output_topk, num_valid, mul_weights,
      (int)block_size_m, stream);
}
