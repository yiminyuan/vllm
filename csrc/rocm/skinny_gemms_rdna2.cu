// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) skinny GEMV: C[N,M] = A[N,K] @ B[M,K]^T (+bias), for small N
// (<=16) and large M -- e.g. the unquantized lm_head logits projection, which on
// RDNA2 otherwise falls back to rocBLAS at ~20% of GDDR6 bandwidth. The upstream
// wvSplitK kernels are gfx9/gfx11 only (they use global_load_lds and DPP
// row_bcast, neither on RDNA2), so this is a purpose-built RDNA2-native GEMV: B
// is streamed once (one wavefront per output row), A is staged in LDS and reused,
// the K-dot uses v_dot2_f32_f16 (fdot2) with an __shfl_xor wave reduction.
// fp16 and bf16 inputs are both accepted; RDNA2 has no bf16 packed dot, so bf16
// is upconverted to fp16 in-register before the fdot2 (no extra memory traffic).

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace skinny_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int WAVES = 8;         // 256 threads/block
static constexpr int LDS_BYTES = 64 * 1024;  // RDNA2 LDS per workgroup

// Largest N the dispatch instantiates. Each lane carries N fp32 accumulators
// across the whole K loop, so N is bounded by register pressure rather than by
// anything algorithmic; 16 covers a full max_num_seqs decode batch. Past N=8 at
// K=4096 the LDS staging no longer fits and A is streamed from global instead,
// which the dispatch already selects on its own.
static constexpr int MAX_N = 16;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// Load 8 consecutive T (one 16-byte float4) as 4 half2 for the fdot2. fp16 is a
// reinterpret; bf16 is widened to fp32 (exact) then rounded to fp16, since RDNA2
// has no bf16 packed dot.
template <typename T>
__device__ __forceinline__ void load8_as_half2(const T* p, half2 (&out)[4]);

template <>
__device__ __forceinline__ void load8_as_half2<__half>(const __half* p,
                                                       half2 (&out)[4]) {
  float4 v = *reinterpret_cast<const float4*>(p);
  const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++) out[j] = h[j];
}

template <>
__device__ __forceinline__ void load8_as_half2<__hip_bfloat16>(
    const __hip_bfloat16* p, half2 (&out)[4]) {
  float4 v = *reinterpret_cast<const float4*>(p);
  const __hip_bfloat16* b = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++)
    out[j] = __floats2half2_rn(__bfloat162float(b[2 * j]),
                               __bfloat162float(b[2 * j + 1]));
}

template <typename T>
__device__ __forceinline__ float elem_to_float(T v);
template <>
__device__ __forceinline__ float elem_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float elem_to_float<__hip_bfloat16>(
    __hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T float_to_elem(float v);
template <>
__device__ __forceinline__ __half float_to_elem<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 float_to_elem<__hip_bfloat16>(
    float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float float_to_elem<float>(float v) {
  return v;
}

// One wavefront computes one output row m for all N activation rows. Lanes split
// K into 8-element (16-byte) chunks, accumulate with fdot2, then reduce across
// the wave with __shfl_xor. lda/ldb are the A/B row strides (handles padded
// inputs). STAGE_A stages A in LDS (reused across all M rows) when it fits;
// otherwise (large K) A is read straight from global -- it is small and stays
// L2/Infinity-Cache resident, so reuse across rows stays cheap without an
// LDS-size limit.
//
// blockIdx.y selects a group, so a batch of independent [M,K]x[N,K]^T products
// is one launch instead of one per group. Callers that project per attention
// group would otherwise pay the GPU's ~4 us kernel-to-kernel dispatch latency
// once per group per layer, which at batch 1 is larger than the projection.
// The group strides are in elements; the ungrouped entry point passes zero.
template <typename T, typename TOut, int N, bool STAGE_A>
__global__ void skinny_gemv_rdna2_kernel(const T* __restrict__ B,
                                         const T* __restrict__ A,
                                         const T* __restrict__ bias,
                                         TOut* __restrict__ C, const int M,
                                         const int K, const int lda,
                                         const int ldb, const int ldc,
                                         const int bias_len, const long gsb,
                                         const long gsa, const long gsc) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* sA = reinterpret_cast<T*>(smem);  // [N][K], packed (STAGE_A only)

  const long grp = blockIdx.y;
  B += grp * gsb;
  A += grp * gsa;
  C += grp * gsc;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  if constexpr (STAGE_A) {
#pragma unroll
    for (int i = 0; i < N; i++)
      for (int k = tid; k < K; k += nthreads)
        sA[i * K + k] = A[(long)i * lda + k];
    __syncthreads();
  }

  const int lane = tid & (WARP32 - 1);
  const int wave = tid / WARP32;

  for (int m = blockIdx.x * WAVES + wave; m < M; m += gridDim.x * WAVES) {
    const T* Brow = B + (long)m * ldb;
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; i++) acc[i] = 0.f;

    for (int k = lane * 8; k < K; k += WARP32 * 8) {
      half2 bh[4];
      load8_as_half2<T>(Brow + k, bh);
#pragma unroll
      for (int i = 0; i < N; i++) {
        half2 ah[4];
        const T* Arow = STAGE_A ? (sA + i * K) : (A + (long)i * lda);
        load8_as_half2<T>(Arow + k, ah);
#pragma unroll
        for (int j = 0; j < 4; j++)
          acc[i] = __builtin_amdgcn_fdot2(ah[j], bh[j], acc[i], false);
      }
    }
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int mask = WARP32 / 2; mask >= 1; mask >>= 1)
        acc[i] += __shfl_xor(acc[i], mask);
    }
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < N; i++) {
        float b = 0.f;
        if (bias)
          b = (bias_len == N * M) ? elem_to_float<T>(bias[i * M + m])
                                  : (bias_len == M ? elem_to_float<T>(bias[m])
                                                   : 0.f);
        C[(long)i * ldc + m] = float_to_elem<TOut>(acc[i] + b);
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, typename TOut, int N, bool STAGE_A>
__global__ void skinny_gemv_rdna2_kernel(const T*, const T*, const T*, TOut*,
                                         const int, const int, const int,
                                         const int, const int, const int,
                                         const long, const long, const long) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace skinny_rdna2
}  // namespace vllm

namespace {
void skinny_launch(const at::Tensor& in_a, const at::Tensor& in_b,
                   const std::optional<at::Tensor>& in_bias, at::Tensor& out,
                   const at::ScalarType out_type, const int M, const int K,
                   const int N, const int lda, const int ldb, const int ldc,
                   const int groups, const long gsb, const long gsa,
                   const long gsc);
}  // namespace

// Requirements (caller-checked): in_a/in_b both fp16 or both bf16, N in [1,5],
// K % 8 == 0. Returns C = [N, M] in the input dtype, or in fp32 when out_dtype
// asks for it -- the accumulator is fp32 either way, so that only skips the
// narrowing on store. A is staged in LDS when it fits, else streamed from
// global; both are correct for any K, so there is no LDS-size restriction on
// the caller.
torch::Tensor wvSplitK_rdna2(const at::Tensor& in_a, const at::Tensor& in_b,
                             const std::optional<at::Tensor>& in_bias,
                             const std::optional<at::ScalarType>& out_dtype,
                             const std::optional<at::Tensor>& out_opt) {
  const int M = in_a.size(0);
  const int K = in_a.size(1);
  const int N = in_b.size(0);
  const int ldb = in_a.stride(0);  // B (weight) row stride
  const int lda = in_b.stride(0);  // A (activation) row stride

  TORCH_CHECK(in_a.dtype() == in_b.dtype(),
              "wvSplitK_rdna2 requires in_a and in_b to have the same dtype");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16,
              "wvSplitK_rdna2 supports fp16 and bf16 only");
  TORCH_CHECK(K % 8 == 0, "wvSplitK_rdna2 requires K % 8 == 0");
  TORCH_CHECK(N >= 1 && N <= vllm::skinny_rdna2::MAX_N,
              "wvSplitK_rdna2 supports N in [1, ",
              vllm::skinny_rdna2::MAX_N, "]");

  const at::ScalarType out_type =
      out_opt.has_value() ? out_opt->scalar_type()
                          : out_dtype.value_or(in_a.scalar_type());
  TORCH_CHECK(out_type == in_a.scalar_type() || out_type == torch::kFloat32,
              "wvSplitK_rdna2 writes the input dtype or fp32");
  TORCH_CHECK(!(out_opt.has_value() && out_dtype.has_value() &&
                *out_dtype != out_type),
              "wvSplitK_rdna2 got out_dtype conflicting with out's dtype");

  // Writing into a caller tensor lets a strided destination be filled directly
  // instead of through a copy; only the row stride may differ from a packed
  // result, since each row is written contiguously.
  auto out = out_opt.has_value()
                 ? *out_opt
                 : torch::empty({N, M}, torch::TensorOptions()
                                            .dtype(out_type)
                                            .device(in_a.device()));
  if (out_opt.has_value()) {
    TORCH_CHECK(out.dim() == 2 && out.size(0) == N && out.size(1) == M,
                "wvSplitK_rdna2 out must be [N, M]");
    TORCH_CHECK(out.device() == in_a.device(), "wvSplitK_rdna2 out device");
    TORCH_CHECK(out.stride(1) == 1, "wvSplitK_rdna2 out rows must be packed");
  }
  const int ldc = out.stride(0);

  skinny_launch(in_a, in_b, in_bias, out, out_type, M, K, N, lda, ldb, ldc,
                /*groups=*/1, /*gsb=*/0, /*gsa=*/0, /*gsc=*/0);
  return out;
}

namespace {

// Shared launch path. `groups` maps onto blockIdx.y and the gs* strides (in
// elements) advance each operand per group, so the batched entry point is the
// same kernel with a second grid dimension rather than a second kernel.
void skinny_launch(const at::Tensor& in_a, const at::Tensor& in_b,
                   const std::optional<at::Tensor>& in_bias, at::Tensor& out,
                   const at::ScalarType out_type, const int M, const int K,
                   const int N, const int lda, const int ldb, const int ldc,
                   const int groups, const long gsb, const long gsa,
                   const long gsc) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int block = vllm::skinny_rdna2::WAVES * vllm::skinny_rdna2::WARP32;
  const dim3 grid(
      (M + vllm::skinny_rdna2::WAVES - 1) / vllm::skinny_rdna2::WAVES, groups);

  // Stage A in LDS when it fits; otherwise stream it from global.
  const size_t lds = (size_t)N * K * in_a.element_size();
  const bool stage_a = lds <= (size_t)vllm::skinny_rdna2::LDS_BYTES;

#define VLLM_RDNA2_SKINNY_LAUNCH(T, TOUT, NN, STAGE, LDS)                      \
  vllm::skinny_rdna2::skinny_gemv_rdna2_kernel<T, TOUT, NN, STAGE>             \
      <<<grid, block, LDS, stream>>>(                                         \
          (const T*)in_a.data_ptr(), (const T*)in_b.data_ptr(),              \
          in_bias.has_value() && in_bias->numel() > 0                        \
              ? (const T*)in_bias->data_ptr()                                \
              : nullptr,                                                      \
          (TOUT*)out.data_ptr(), M, K, lda, ldb, ldc,                        \
          in_bias.has_value() ? (int)in_bias->numel() : 0, gsb, gsa, gsc)
#define VLLM_RDNA2_SKINNY_DISPATCH(T, TOUT, NN)                                \
  if (stage_a)                                                                \
    VLLM_RDNA2_SKINNY_LAUNCH(T, TOUT, NN, true, lds);                        \
  else                                                                        \
    VLLM_RDNA2_SKINNY_LAUNCH(T, TOUT, NN, false, 0)
#define VLLM_RDNA2_SKINNY_CASE(T, TOUT, NN)                                   \
  case NN: VLLM_RDNA2_SKINNY_DISPATCH(T, TOUT, NN); break;
#define VLLM_RDNA2_SKINNY_BY_N(T, TOUT)                                       \
  switch (N) {                                                                \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 1)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 2)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 3)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 4)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 5)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 6)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 7)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 8)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 9)                                        \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 10)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 11)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 12)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 13)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 14)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 15)                                       \
    VLLM_RDNA2_SKINNY_CASE(T, TOUT, 16)                                       \
  }
#define VLLM_RDNA2_SKINNY_BY_OUT(T)                                           \
  if (out_type == torch::kFloat32) {                                          \
    VLLM_RDNA2_SKINNY_BY_N(T, float);                                         \
  } else {                                                                    \
    VLLM_RDNA2_SKINNY_BY_N(T, T);                                             \
  }

  if (in_a.dtype() == torch::kFloat16) {
    VLLM_RDNA2_SKINNY_BY_OUT(__half);
  } else {
    VLLM_RDNA2_SKINNY_BY_OUT(__hip_bfloat16);
  }

#undef VLLM_RDNA2_SKINNY_BY_OUT
#undef VLLM_RDNA2_SKINNY_BY_N
#undef VLLM_RDNA2_SKINNY_CASE
#undef VLLM_RDNA2_SKINNY_DISPATCH
#undef VLLM_RDNA2_SKINNY_LAUNCH
}

}  // namespace

// Batched form of the above: contracts ``in_b [N, G, K]`` with ``in_a
// [G, M, K]`` over K, writing ``out [N, G, M]``. One launch covers every group,
// which is the point -- the per-group loop it replaces paid the GPU's
// kernel-to-kernel dispatch latency once per group per layer, and at batch 1
// that dominated the projection it was doing.
torch::Tensor wvSplitK_rdna2_grouped(const at::Tensor& in_a,
                                     const at::Tensor& in_b,
                                     const std::optional<at::Tensor>& out_opt) {
  TORCH_CHECK(in_a.dim() == 3 && in_b.dim() == 3,
              "wvSplitK_rdna2_grouped expects in_a [G, M, K] and in_b [N, G, K]");
  const int G = in_a.size(0);
  const int M = in_a.size(1);
  const int K = in_a.size(2);
  const int N = in_b.size(0);

  TORCH_CHECK(in_b.size(1) == G && in_b.size(2) == K,
              "wvSplitK_rdna2_grouped shape mismatch between in_a and in_b");
  TORCH_CHECK(in_a.dtype() == in_b.dtype(),
              "wvSplitK_rdna2_grouped requires in_a and in_b to have the same dtype");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 || in_a.dtype() == torch::kBFloat16,
              "wvSplitK_rdna2_grouped supports fp16 and bf16 only");
  TORCH_CHECK(K % 8 == 0, "wvSplitK_rdna2_grouped requires K % 8 == 0");
  TORCH_CHECK(N >= 1 && N <= vllm::skinny_rdna2::MAX_N,
              "wvSplitK_rdna2_grouped supports N in [1, ",
              vllm::skinny_rdna2::MAX_N, "]");
  // Each operand is indexed as base + group*stride, then row*ld, then element,
  // so only the innermost dimension has to be packed.
  TORCH_CHECK(in_a.stride(2) == 1 && in_b.stride(2) == 1,
              "wvSplitK_rdna2_grouped needs K-contiguous in_a and in_b");

  auto out = out_opt.has_value()
                 ? *out_opt
                 : torch::empty({N, G, M}, torch::TensorOptions()
                                               .dtype(in_a.scalar_type())
                                               .device(in_a.device()));
  TORCH_CHECK(out.dim() == 3 && out.size(0) == N && out.size(1) == G &&
                  out.size(2) == M,
              "wvSplitK_rdna2_grouped out must be [N, G, M]");
  TORCH_CHECK(out.scalar_type() == in_a.scalar_type(),
              "wvSplitK_rdna2_grouped writes the input dtype");
  TORCH_CHECK(out.stride(2) == 1,
              "wvSplitK_rdna2_grouped out rows must be packed");

  skinny_launch(in_a, in_b, std::nullopt, out, out.scalar_type(), M, K, N,
                /*lda=*/in_b.stride(0), /*ldb=*/in_a.stride(1),
                /*ldc=*/out.stride(0), G, /*gsb=*/in_a.stride(0),
                /*gsa=*/in_b.stride(1), /*gsc=*/out.stride(1));
  return out;
}
