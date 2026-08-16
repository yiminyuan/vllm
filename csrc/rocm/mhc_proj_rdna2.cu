// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) mHC pre-block projection: mixes[T,N] = x[T,K] @ fn[N,K]^T in
// fp32, plus the per-row sum of squares of x, from a fp16/bf16 residual.
//
// The shape is very wide and very short -- K spans the widened residual
// (hc_mult * hidden_size) while N is 2*hc_mult + hc_mult^2 mixing coefficients,
// 24 for hc_mult=4. rocBLAS picks a 64x64x8 macro tile for it and lands at ~1.1
// TFLOP/s; the op is limited by fp32 FMA issue and LDS traffic, not bandwidth
// (a loads-only kernel over the same operand runs 7x faster), so the wins here
// come from occupancy and from reusing each staged fn value.
//
// Structure: one wave owns TPW rows of x; a workgroup walks one K chunk with
// fn[:, chunk] staged in LDS and reused by every wave. Accumulators stay in
// registers for the whole chunk, so there is one wave reduction at the end
// rather than one per K step. Lanes take CONSECUTIVE k, which keeps the global
// loads coalesced and -- with an odd LDS row stride -- puts the 32 lanes on 32
// distinct LDS banks; striding several k per lane instead collapses them onto a
// multiple of 8 and costs an 8-way conflict.
//
// The widening is exact, so folding it in here drops the fp32 copy of the
// residual that the two-pass path had to materialise and read back.
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
namespace mhc_rdna2 {

static constexpr int WAVE = 32;
static constexpr int LDS_BYTES = 64 * 1024;
// LDS row stride in floats. Odd, so consecutive lanes (consecutive k) map to
// consecutive banks; an even stride would alias lanes onto shared banks.
static constexpr int PAD_EXTRA = 1;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T, int NOUT, int KC, int WAVES, int TPW>
__global__ __launch_bounds__(WAVES* WAVE) void mhc_proj_kernel(
    const T* __restrict__ x,          // [T, K]
    const float* __restrict__ fn,     // [NOUT, K]
    float* __restrict__ partial,      // [nchunk, T, NOUT]
    float* __restrict__ sqr_partial,  // [nchunk, T]
    const int num_tokens, const int K) {
  constexpr int PAD = NOUT + PAD_EXTRA;
  static_assert(KC * PAD * sizeof(float) <= LDS_BYTES,
                "mhc_proj_rdna2: staged fn chunk exceeds the RDNA2 LDS budget; "
                "lower KC for this NOUT");
  __shared__ float fs[KC * PAD];

  const int wave = threadIdx.x / WAVE;
  const int lane = threadIdx.x % WAVE;
  const int kchunk = blockIdx.x;
  const int tbase = blockIdx.y * (WAVES * TPW) + wave * TPW;

  const int k0 = kchunk * KC;
  const int kc = min(KC, K - k0);

  // Stage fn[:, k0:k0+kc] as [kc][NOUT]; every wave in the block reuses it.
  for (int i = threadIdx.x; i < kc * NOUT; i += WAVES * WAVE) {
    const int kk = i / NOUT;
    const int n = i - kk * NOUT;
    fs[kk * PAD + n] = fn[(long)n * K + k0 + kk];
  }
  __syncthreads();

  float acc[TPW][NOUT];
  float sq[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) {
#pragma unroll
    for (int n = 0; n < NOUT; n++) acc[i][n] = 0.f;
    sq[i] = 0.f;
  }

  const int ntok = min(TPW, num_tokens - tbase);
  if (ntok > 0) {
    // k outer, tokens inner: each staged fn value feeds TPW FMAs.
    for (int k = lane; k < kc; k += WAVE) {
      float v[TPW];
#pragma unroll
      for (int i = 0; i < TPW; i++) {
        v[i] = (i < ntok) ? to_f32<T>(x[(long)(tbase + i) * K + k0 + k]) : 0.f;
        sq[i] += v[i] * v[i];
      }
      const float* f = &fs[k * PAD];
#pragma unroll
      for (int n = 0; n < NOUT; n++) {
        const float fv = f[n];
#pragma unroll
        for (int i = 0; i < TPW; i++) acc[i][n] = fmaf(v[i], fv, acc[i][n]);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < TPW; i++) {
    if (i >= ntok) break;
    const int t = tbase + i;
#pragma unroll
    for (int n = 0; n < NOUT; n++) {
      float a = acc[i][n];
#pragma unroll
      for (int m = WAVE / 2; m >= 1; m >>= 1) a += __shfl_xor(a, m, WAVE);
      if (lane == 0) partial[((long)kchunk * num_tokens + t) * NOUT + n] = a;
    }
    float s = sq[i];
#pragma unroll
    for (int m = WAVE / 2; m >= 1; m >>= 1) s += __shfl_xor(s, m, WAVE);
    if (lane == 0) sqr_partial[(long)kchunk * num_tokens + t] = s;
  }
}

// Sum the per-chunk partials. Kept as its own pass so the reduction order is
// fixed by chunk index rather than by whichever workgroup finishes first.
template <int NOUT>
__global__ void mhc_reduce_kernel(const float* __restrict__ partial,
                                  const float* __restrict__ sqr_partial,
                                  float* __restrict__ mixes,
                                  float* __restrict__ sqrsum,
                                  const int num_tokens, const int nchunk) {
  const int t = blockIdx.x;
  if (t >= num_tokens) return;
  if (threadIdx.x < NOUT) {
    float a = 0.f;
    for (int c = 0; c < nchunk; c++)
      a += partial[((long)c * num_tokens + t) * NOUT + threadIdx.x];
    mixes[(long)t * NOUT + threadIdx.x] = a;
  } else if (threadIdx.x == NOUT) {
    float s = 0.f;
    for (int c = 0; c < nchunk; c++) s += sqr_partial[(long)c * num_tokens + t];
    sqrsum[t] = s;
  }
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

template <typename T, int NOUT, int KC, int WAVES, int TPW>
__global__ void mhc_proj_kernel(const T*, const float*, float*, float*,
                                const int, const int) {}
template <int NOUT>
__global__ void mhc_reduce_kernel(const float*, const float*, float*, float*,
                                  const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace mhc_rdna2
}  // namespace vllm

// Requirements (caller-checked): x is 2-D fp16 or bf16 [T, K], fn is fp32
// [NOUT, K] with NOUT one of the instantiated widths, K % 8 == 0. Returns
// (mixes [T, NOUT] fp32, sqrsum [T] fp32).
std::tuple<torch::Tensor, torch::Tensor> mhc_proj_rdna2(
    const at::Tensor& x, const at::Tensor& fn) {
  TORCH_CHECK(x.dim() == 2 && fn.dim() == 2, "mhc_proj_rdna2 expects 2-D x/fn");
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
              "mhc_proj_rdna2 supports fp16 and bf16 x only");
  TORCH_CHECK(fn.dtype() == torch::kFloat32, "mhc_proj_rdna2 needs fp32 fn");
  TORCH_CHECK(x.size(1) == fn.size(1), "mhc_proj_rdna2 K mismatch");
  TORCH_CHECK(x.stride(1) == 1 && fn.stride(1) == 1,
              "mhc_proj_rdna2 needs K-contiguous x and fn");
  TORCH_CHECK(x.stride(0) == x.size(1) && fn.stride(0) == fn.size(1),
              "mhc_proj_rdna2 needs packed rows");

  const int num_tokens = x.size(0);
  const int K = x.size(1);
  const int nout = fn.size(0);
  TORCH_CHECK(K % 8 == 0, "mhc_proj_rdna2 requires K % 8 == 0");

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto mixes = torch::empty({num_tokens, nout}, f32);
  auto sqrsum = torch::empty({num_tokens}, f32);
  if (num_tokens == 0) return {mixes, sqrsum};

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Small batches do not fill a wide workgroup, so they take one row per wave
  // and fewer waves; wide batches amortise the fn staging over more waves.
  const bool wide = num_tokens >= 32;

#define VLLM_MHC_LAUNCH(T, NOUT, KC, WAVES, TPW)                              \
  do {                                                                        \
    const int nchunk = (K + (KC) - 1) / (KC);                                 \
    auto partial = torch::empty({nchunk, num_tokens, nout}, f32);             \
    auto sqr_partial = torch::empty({nchunk, num_tokens}, f32);               \
    const dim3 grid(nchunk,                                                   \
                    (num_tokens + (WAVES) * (TPW) - 1) / ((WAVES) * (TPW)));  \
    vllm::mhc_rdna2::mhc_proj_kernel<T, NOUT, KC, WAVES, TPW>                 \
        <<<grid, (WAVES) * vllm::mhc_rdna2::WAVE, 0, stream>>>(               \
            (const T*)x.data_ptr(), fn.data_ptr<float>(),                     \
            partial.data_ptr<float>(), sqr_partial.data_ptr<float>(),         \
            num_tokens, K);                                                   \
    vllm::mhc_rdna2::mhc_reduce_kernel<NOUT>                                  \
        <<<num_tokens, ((NOUT) + 32) & ~31, 0, stream>>>(                     \
            partial.data_ptr<float>(), sqr_partial.data_ptr<float>(),         \
            mixes.data_ptr<float>(), sqrsum.data_ptr<float>(), num_tokens,    \
            nchunk);                                                          \
  } while (0)

#define VLLM_MHC_BY_SHAPE(T, NOUT)                                            \
  do {                                                                        \
    if (wide) {                                                               \
      VLLM_MHC_LAUNCH(T, NOUT, 640, 32, 4);                                   \
    } else {                                                                  \
      VLLM_MHC_LAUNCH(T, NOUT, 640, 8, 1);                                    \
    }                                                                         \
  } while (0)

#define VLLM_MHC_BY_NOUT(T)                                                   \
  do {                                                                        \
    switch (nout) {                                                           \
      case 8: VLLM_MHC_BY_SHAPE(T, 8); break;                                 \
      case 24: VLLM_MHC_BY_SHAPE(T, 24); break;                               \
      default:                                                                \
        TORCH_CHECK(false, "mhc_proj_rdna2 has no kernel for nout=", nout);   \
    }                                                                         \
  } while (0)

  if (x.dtype() == torch::kFloat16) {
    VLLM_MHC_BY_NOUT(__half);
  } else {
    VLLM_MHC_BY_NOUT(__hip_bfloat16);
  }

#undef VLLM_MHC_BY_NOUT
#undef VLLM_MHC_BY_SHAPE
#undef VLLM_MHC_LAUNCH

  return {mixes, sqrsum};
}
