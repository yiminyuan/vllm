// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA2 (gfx1030). fp16-only sibling of
// qdq_4_rdna3.cuh — the exact same exllamav2 "+1024" bit-trick, minus the bf16
// path (RDNA2 has no bf16 packed dot). Every op here is RDNA2-supported:
// integer bit-ops, v_cvt_f32_f16, v_pk_mul_f16 (__hmul2) and v_pk_fma_f16
// (__hfma2).
//
//   (qa & 0x000F000F) | 0x64006400  ->  half2(1024+q_lo, 1024+q_hi)
//   (qa & 0x00F000F0) | 0x64006400  ->  half2(1024+q_lo*16, 1024+q_hi*16)
//
// The "*16 then divide by 16 in the FMA" trick for the upper-nibble pairs works
// in fp16 because the 10-bit mantissa is wide enough to hold a value shifted by
// 4 bits.

#ifndef _qdq_4_rdna2_cuh
#define _qdq_4_rdna2_cuh

#include <cstdint>

#include <hip/hip_fp16.h>

namespace vllm {
namespace gptq_rdna2 {

// Bit-shuffle for an int32 holding 8 sequential 4-bit weights q[0..7]:
//   in:  q[7] q[6] q[5] q[4] q[3] q[2] q[1] q[0]   (LSB first)
//   out: q[7] q[5] q[3] q[1] q[6] q[4] q[2] q[0]   (even/odd interleaved)
//
// After shuffle, q[2k]   sits at bits [4k   : 4k+3]   (lower 16)
//                q[2k+1] sits at bits [16+4k: 16+4k+3] (upper 16)
// so a single mask 0x000F000F selects the matching even/odd pair, ready to
// bitcast to half2 after OR-ing with the magic constant.
__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q) {
  uint32_t qa = q[0];
  uint32_t qb = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0F;
    uint32_t qa1 = (qa & 0xF0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

// Precompute scale-baked constants for a single zero/scale pair.
//   z1z16[0] = scale * (-1024 - zero)            (used for "low" pairs)
//   z1z16[1] = scale * (-64   - zero)            (used for "high" pairs)
//   y1y16[0] = scale * 1                          (low pairs are q + 1024)
//   y1y16[1] = scale * (1/16)                     (high pairs are q*16 + 1024)
__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  // half(-1024 - zero) via the exllamav2 bit-trick:
  //   half bits 0xE400 == -1024.0 ; ORing the zero into mantissa subtracts it.
  union {
    uint16_t u;
    half h;
  } z1u;
  z1u.u = (uint16_t)(0xE400 | zero);
  half z1 = z1u.h;
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn((int)zero));

  half2 scale2 = __half2half2(scale);
  z1z16[0] = __hmul2(scale2, __half2half2(z1));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  half y1 = __float2half_rn(1.0f);
  half y16 = __float2half_rn(1.0f / 16.0f);
  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

// Dequantize one int32 (8 shuffled 4-bit weights) into 4 half2 pairs:
//   dq[0] = (q[0], q[1]) * scale - zero*scale
//   dq[1] = (q[2], q[3]) * scale - zero*scale
//   dq[2] = (q[4], q[5]) * scale - zero*scale
//   dq[3] = (q[6], q[7]) * scale - zero*scale
__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa, half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;

  union {
    uint32_t u;
    half2 h2;
  } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;  // half2(q[0]+1024, q[1]+1024)
  q1.u = (qa & 0x00F000F0) | c0;  // half2(q[2]*16+1024, q[3]*16+1024)
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;  // half2(q[4]+1024, q[5]+1024)
  q3.u = (qa_hi & 0x00F000F0) | c0;  // half2(q[6]*16+1024, q[7]*16+1024)

  dq[0] = __hfma2(q0.h2, y1y16[0], z1z16[0]);
  dq[1] = __hfma2(q1.h2, y1y16[1], z1z16[1]);
  dq[2] = __hfma2(q2.h2, y1y16[0], z1z16[0]);
  dq[3] = __hfma2(q3.h2, y1y16[1], z1z16[1]);
}

}  // namespace gptq_rdna2
}  // namespace vllm

#endif  // _qdq_4_rdna2_cuh
