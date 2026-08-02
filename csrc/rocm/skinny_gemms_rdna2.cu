// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) skinny GEMV: C[N,M] = A[N,K] @ B[M,K]^T (+bias), for small N
// (<=5) and large M -- e.g. the unquantized lm_head logits projection, which on
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

// One wavefront computes one output row m for all N activation rows. Lanes split
// K into 8-element (16-byte) chunks, accumulate with fdot2, then reduce across
// the wave with __shfl_xor. lda/ldb are the A/B row strides (handles padded
// inputs). STAGE_A stages A in LDS (reused across all M rows) when it fits;
// otherwise (large K) A is read straight from global -- it is small and stays
// L2/Infinity-Cache resident, so reuse across rows stays cheap without an
// LDS-size limit.
template <typename T, int N, bool STAGE_A>
__global__ void skinny_gemv_rdna2_kernel(const T* __restrict__ B,
                                         const T* __restrict__ A,
                                         const T* __restrict__ bias,
                                         T* __restrict__ C, const int M,
                                         const int K, const int lda,
                                         const int ldb, const int bias_len) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* sA = reinterpret_cast<T*>(smem);  // [N][K], packed (STAGE_A only)

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
        C[(long)i * M + m] = float_to_elem<T>(acc[i] + b);
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, int N, bool STAGE_A>
__global__ void skinny_gemv_rdna2_kernel(const T*, const T*, const T*, T*,
                                         const int, const int, const int,
                                         const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace skinny_rdna2
}  // namespace vllm

// Requirements (caller-checked): in_a/in_b both fp16 or both bf16, N in [1,5],
// K % 8 == 0. Returns C = [N, M] in the input dtype. A is staged in LDS when it
// fits, else streamed from global; both are correct for any K, so there is no
// LDS-size restriction on the caller.
torch::Tensor wvSplitK_rdna2(const at::Tensor& in_a, const at::Tensor& in_b,
                             const std::optional<at::Tensor>& in_bias) {
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
  TORCH_CHECK(N >= 1 && N <= 5, "wvSplitK_rdna2 supports N in [1, 5]");

  auto out = torch::empty(
      {N, M}, torch::TensorOptions().dtype(in_a.dtype()).device(in_a.device()));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int block = vllm::skinny_rdna2::WAVES * vllm::skinny_rdna2::WARP32;
  const int grid = (M + vllm::skinny_rdna2::WAVES - 1) / vllm::skinny_rdna2::WAVES;

  // Stage A in LDS when it fits; otherwise stream it from global.
  const size_t lds = (size_t)N * K * in_a.element_size();
  const bool stage_a = lds <= (size_t)vllm::skinny_rdna2::LDS_BYTES;

#define VLLM_RDNA2_SKINNY_LAUNCH(T, NN, STAGE, LDS)                            \
  vllm::skinny_rdna2::skinny_gemv_rdna2_kernel<T, NN, STAGE>                   \
      <<<grid, block, LDS, stream>>>(                                         \
          (const T*)in_a.data_ptr(), (const T*)in_b.data_ptr(),              \
          in_bias.has_value() && in_bias->numel() > 0                        \
              ? (const T*)in_bias->data_ptr()                                \
              : nullptr,                                                      \
          (T*)out.data_ptr(), M, K, lda, ldb,                                \
          in_bias.has_value() ? (int)in_bias->numel() : 0)
#define VLLM_RDNA2_SKINNY_DISPATCH(T, NN)                                      \
  if (stage_a)                                                                \
    VLLM_RDNA2_SKINNY_LAUNCH(T, NN, true, lds);                              \
  else                                                                        \
    VLLM_RDNA2_SKINNY_LAUNCH(T, NN, false, 0)
#define VLLM_RDNA2_SKINNY_BY_N(T)                                             \
  switch (N) {                                                                \
    case 1: VLLM_RDNA2_SKINNY_DISPATCH(T, 1); break;                         \
    case 2: VLLM_RDNA2_SKINNY_DISPATCH(T, 2); break;                         \
    case 3: VLLM_RDNA2_SKINNY_DISPATCH(T, 3); break;                         \
    case 4: VLLM_RDNA2_SKINNY_DISPATCH(T, 4); break;                         \
    case 5: VLLM_RDNA2_SKINNY_DISPATCH(T, 5); break;                         \
  }

  if (in_a.dtype() == torch::kFloat16) {
    VLLM_RDNA2_SKINNY_BY_N(__half);
  } else {
    VLLM_RDNA2_SKINNY_BY_N(__hip_bfloat16);
  }

#undef VLLM_RDNA2_SKINNY_BY_N
#undef VLLM_RDNA2_SKINNY_DISPATCH
#undef VLLM_RDNA2_SKINNY_LAUNCH

  return out;
}
