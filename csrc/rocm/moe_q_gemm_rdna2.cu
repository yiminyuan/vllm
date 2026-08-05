// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 MoE kernel for RDNA2 (gfx1030). Fuses expert routing + W4A16 dequant
// (exllamav2 "+1024" bit-trick, qdq_4_rdna2.cuh) + v_dot2_f32_f16 (fdot2) dot,
// with a 64-bit-CAS fp32 atomic epilogue — the same fp16-only compute core as
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
// w2 (top_k=1, output_topk=top_k, mul=true).
//
// grid.z splits K and w2 additionally sums top_k contributions into one output
// row, so a single output is combined from several partials through memory.
// Indexing the scratch by token-slot rather than by output row makes every
// (split, slot) the property of one block for both GEMMs: w13 maps a slot
// 1:1 to an output row, and w2's fused moe_sum folds output_topk consecutive
// slots into one. The reduce then walks splits and slots in index order, so
// the result does not depend on which block reached memory first. w2 used the
// fp32 CAS at first, which left its output varying by ~0.2% of RMS between
// identical launches -- enough to reorder the sparse indexer's top-k, and to
// make greedy decoding answer the same prompt differently.
//
// The scratch is zeroed, unlike the dense kernel's: padded token-blocks and
// expert-parallel masked blocks return early, so not every row is written.
// That combine accumulates in fp32 (see atomic_add_pk2_f32) into a scratch
// buffer this op casts into the caller's half output at the end: RDNA2 has no
// float-add atomic, so the CAS loop is unavoidable, but rounding each partial
// to fp16 as this kernel first did cost an order of magnitude more error than
// an fp16 sum of the same terms and made the result order-dependent.

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
// sorted-token space, grid.z splits K and the splits are combined as described
// in the header note (as the dense
// kernel), grid.x tiles N at BLOCK_KN_SIZE*4 columns per block.
template <int BLOCK_M, int BLOCK_KN_SIZE>
__global__ void moe_gemm_q4_kernel_rdna2(
    const half* __restrict__ input, float* __restrict__ output,
    float* __restrict__ partials, const int partial_rows,
    const uint32_t* __restrict__ b_q_weight, const half* __restrict__ b_scales,
    const uint32_t* __restrict__ b_qzeros,
    const float* __restrict__ topk_weights,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int size_n,
    const int n_cols, const int partial_n, const int size_k, const int groups,
    const int zero_offset, const int top_k, const int output_topk,
    const int num_valid, const bool mul_weights) {
  // Column indices are relative to this N tile; the weight and output pointers
  // are already offset to its first column, so size_n stays the row stride.
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int block_m_idx = blockIdx.y;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // Skip padded token-blocks and empty (expert-parallel masked) blocks.
  if (block_m_idx * BLOCK_M >= num_tokens_post_padded[0]) return;
  const int expert = expert_ids[block_m_idx];

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

  // An expert-parallel rank owns only some experts and skips the rest. The
  // scratch is left uninitialised, so a skipped block still owes its rows the
  // zeros they contribute -- cheaper than clearing the whole buffer, which at
  // a prefill-sized scratch costs more than the GEMM saves.
  if (expert < 0) {
    if (partials && n < size_n) {
      #pragma unroll
      for (int m = 0; m < BLOCK_M; ++m) {
        if (!row_valid[m]) continue;
        const int prow = sids[m];
        float* out =
            partials + ((long)blockIdx.z * partial_rows + prow) * partial_n + n;
        *reinterpret_cast<float4*>(out) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
    return;
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
  if (n >= n_cols) return;

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
    if (partials) {
      // Indexed by token-slot, so this row belongs to this split and this slot
      // alone -- true for w2 as well as w13 -- and it is one 16-byte store.
      const int prow = (output_topk > 0) ? sids[m] : out_row;
      float* out =
          partials + ((long)blockIdx.z * partial_rows + prow) * partial_n + n;
      *reinterpret_cast<float4*>(out) =
          make_float4(block_c[m][0] * w, block_c[m][1] * w, block_c[m][2] * w,
                      block_c[m][3] * w);
    } else {
      float* out = output + (long)out_row * size_n + n;
      atomic_add_pk2_f32(out, block_c[m][0] * w, block_c[m][1] * w);
      atomic_add_pk2_f32(out + 2, block_c[m][2] * w, block_c[m][3] * w);
    }
  }
}

#else  // non-RDNA2 device pass: empty __global__ for symbol parity.

template <int BLOCK_M, int BLOCK_KN_SIZE>
__global__ void moe_gemm_q4_kernel_rdna2(
    const half*, float*, float*, const int, const uint32_t*, const half*,
    const uint32_t*, const float*, const int*, const int*, const int*,
    const int, const int, const int, const int, const int, const int,
    const int, const int, const int, const bool) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int BLOCK_M, int BK>
void launch_moe_gemm_q4_bm(
    const half* input, float* output, float* partials, int partial_rows,
    const uint32_t* b_q_weight, const half* b_scales, const uint32_t* b_qzeros,
    const float* topk_weights, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded, int em,
    int size_n, int n_cols, int partial_n, int size_k, int groups,
    int zero_offset, int top_k, int output_topk, int num_valid,
    bool mul_weights, cudaStream_t stream) {
  dim3 block(BK);
  dim3 grid((n_cols + BK * 4 - 1) / (BK * 4), (em + BLOCK_M - 1) / BLOCK_M,
            (size_k + BK - 1) / BK);
  moe_gemm_q4_kernel_rdna2<BLOCK_M, BK><<<grid, block, 0, stream>>>(
      input, output, partials, partial_rows, b_q_weight, b_scales, b_qzeros,
      topk_weights,
      sorted_token_ids, expert_ids, num_tokens_post_padded, size_n, n_cols,
      partial_n, size_k, groups, zero_offset, top_k, output_topk, num_valid,
      mul_weights);
}

// Scratch budget for the reproducible combine. This is a tile size, not a
// threshold to give up at: columns are independent, so N is processed in passes
// that fit the budget rather than falling back to the CAS epilogue on wide
// batches. A pass is at least one 8-column group, so no shape is excluded and
// the answer never depends on which block reached memory first. The scratch is
// transient and not pre-zeroed, so the cost is traffic the reduce already pays.
static constexpr long MOE_SPLIT_SCRATCH_MAX_BYTES = 256L << 20;

void launch_moe_gemm_q4(const half* input, half* out_half,
                        const uint32_t* b_q_weight, const half* b_scales,
                        const uint32_t* b_qzeros, const float* topk_weights,
                        const int* sorted_token_ids, const int* expert_ids,
                        const int* num_tokens_post_padded, int em, int size_n,
                        int size_k, int groups, int zero_offset, int top_k,
                        int output_topk, int num_valid, bool mul_weights,
                        int block_size_m, int out_rows, cudaStream_t stream) {
  constexpr int BK = 256;

  // A row gains several writers from the K splits and, for w2, from the top_k
  // slots the fused moe_sum folds into it. Indexing the scratch by token-slot
  // gives every (split, slot) a private row, so both are combined afterwards in
  // a fixed order -- reproducible for w2 as well as w13.
  const int num_splits = (size_k + BK - 1) / BK;
  const int fold = (output_topk > 0) ? output_topk : 1;
  const long out_mn = (long)out_rows * size_n;
  const long slot_rows = (long)out_rows * fold;
  const bool two_phase = (num_splits > 1 || fold > 1);

  // Columns are independent, so the scratch is bounded by processing N in
  // passes rather than by abandoning the reproducible combine on wide batches.
  int n_per_pass = size_n;
  if (two_phase) {
    const long col_bytes = (long)num_splits * slot_rows * (long)sizeof(float);
    long budget = MOE_SPLIT_SCRATCH_MAX_BYTES / (col_bytes > 0 ? col_bytes : 1);
    budget &= ~7L;  // the epilogue stores four halves at a time
    if (budget < 8) budget = 8;
    n_per_pass = (int)(budget < (long)size_n ? budget : (long)size_n);
  }

  auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  // The CAS path accumulates, so it must start at zero. The two-phase path has
  // every row it reads written exactly once -- by the owning block, or with
  // zeros by a block its rank masked out -- so it does not.
  const long slot_mn_tile = slot_rows * n_per_pass;
  at::Tensor buf = two_phase
                       ? at::empty({(long)num_splits, slot_mn_tile}, f32)
                       : at::zeros({out_mn}, f32);
  float* partials = two_phase ? (float*)buf.data_ptr() : nullptr;
  float* acc = two_phase ? nullptr : (float*)buf.data_ptr();
  const int partial_rows = two_phase ? (int)slot_rows : out_rows;

  constexpr int RED_THREADS = 256;
  for (int n0 = 0; n0 < size_n; n0 += n_per_pass) {
    const int cols =
        (size_n - n0) < n_per_pass ? (size_n - n0) : n_per_pass;
    // The weight/output pointers move to the tile's first column; size_n stays
    // the row stride everywhere.
    const uint32_t* qw_pass = b_q_weight + n0;
    const half* sc_pass = b_scales + n0;
    const uint32_t* qz_pass = b_qzeros + n0 / 8;
    float* acc_pass = acc ? acc + n0 : nullptr;
    half* out_pass = out_half + n0;

    if (block_size_m <= 1) {
      launch_moe_gemm_q4_bm<1, BK>(
          input, acc_pass, partials, partial_rows, qw_pass, sc_pass, qz_pass,
          topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
          em, size_n, cols, n_per_pass, size_k, groups, zero_offset, top_k,
          output_topk, num_valid, mul_weights, stream);
    } else {
      launch_moe_gemm_q4_bm<4, BK>(
          input, acc_pass, partials, partial_rows, qw_pass, sc_pass, qz_pass,
          topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
          em, size_n, cols, n_per_pass, size_k, groups, zero_offset, top_k,
          output_topk, num_valid, mul_weights, stream);
    }

    const long vec = (long)out_rows * (cols / 4);
    const int red_blocks = (int)((vec + RED_THREADS - 1) / RED_THREADS);
    if (two_phase) {
      reduce_splits_fold_tile_kernel_rdna2<0>
          <<<red_blocks, RED_THREADS, 0, stream>>>(
              partials, out_pass, num_splits, fold, out_rows, slot_mn_tile,
              n_per_pass, cols, size_n);
    } else {
      reduce_splits_kernel_rdna2<0><<<red_blocks, RED_THREADS, 0, stream>>>(
          acc_pass, out_pass, 1, out_mn);
    }
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
      (int)block_size_m, (int)output.size(0), stream);
}
