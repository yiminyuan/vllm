#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the UE8M0 block-scaled FP8 codec on RDNA2.

The DeepSeek-V4 fp8_ds_mla KV cache stores each token as 448 FP8 bytes + 64 bf16
RoPE elements, with one UE8M0 (power-of-two) exponent per 64-element block. The
decode runs in the sparse-MLA prefill gather and, per topk row, in the decode
attention kernels.

RDNA2 has no FP8 convert instruction, so ``.to(tl.float8e4nv)`` lowers to a
v_cmp/v_cndmask ladder over the special and subnormal codes. ``FP8_BITOPS``
instead widens the code to fp16 with native integer ops (v_and_b32 /
v_lshlrev_b32 / v_or_b32) and converts it with v_cvt_f32_f16. The encode
direction mirrors it: round-to-nearest-even on the dropped mantissa bits, with
the subnormal index falling out of a single fused multiply-add.

Both paths must agree bit-for-bit on every one of the 256 byte codes -- including
subnormals, which are the codes a naive shift-based decode gets wrong.

Run `pytest tests/kernels/attention/test_fp8_ue8m0.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import dequant_fp8_ue8m0, quant_fp8_to_uint8

if not current_platform.is_cuda_alike():
    pytest.skip("dequant_fp8_ue8m0 needs a GPU", allow_module_level=True)

device = "cuda"

# e4m3 NaN (0x7F / 0xFF for OCP). The bit-ops path maps these onto finite values
# instead of NaN; the KV cache never stores them because the quantizer clamps to
# +/-fp8_max. Every other code must round-trip exactly.
_OCP_NAN_CODES = (0x7F, 0xFF)


@triton.jit
def _decode_kernel(
    u8_ptr,
    scale_ptr,
    out_ptr,
    N: tl.constexpr,
    USE_FNUZ: tl.constexpr,
    FP8_BITOPS: tl.constexpr,
):
    offs = tl.arange(0, N)
    x = tl.load(u8_ptr + offs)
    enc = tl.load(scale_ptr + offs)
    tl.store(out_ptr + offs, dequant_fp8_ue8m0(x, enc, USE_FNUZ, FP8_BITOPS))


def _decode(codes: torch.Tensor, encoded: torch.Tensor, use_fnuz: bool, bitops: bool):
    out = torch.empty(codes.numel(), dtype=torch.float32, device=device)
    _decode_kernel[(1,)](
        codes, encoded, out, N=codes.numel(), USE_FNUZ=use_fnuz, FP8_BITOPS=bitops
    )
    return out


def _all_codes(encoded_scale: int):
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    encoded = torch.full((256,), encoded_scale, dtype=torch.uint8, device=device)
    return codes, encoded


# UE8M0 exponents: 127 is 2**0; the rest probe the range the cache actually
# writes, plus the extremes where the widening factor could over/underflow.
@pytest.mark.parametrize("encoded_scale", [0, 1, 64, 120, 127, 134, 200, 254])
def test_bitops_matches_convert_instruction(encoded_scale):
    """The two decode paths must be bit-identical on every non-NaN code."""
    codes, encoded = _all_codes(encoded_scale)
    ref = _decode(codes, encoded, use_fnuz=False, bitops=False)
    got = _decode(codes, encoded, use_fnuz=False, bitops=True)

    keep = torch.ones(256, dtype=torch.bool, device=device)
    keep[list(_OCP_NAN_CODES)] = False
    assert torch.equal(got[keep], ref[keep]), (
        f"encoded_scale={encoded_scale}: "
        f"{int((got[keep] != ref[keep]).sum())} codes differ"
    )


@pytest.mark.parametrize("encoded_scale", [1, 64, 127, 134, 254])
def test_bitops_matches_torch(encoded_scale):
    """Anchor against torch's own FP8 cast rather than only the other kernel.

    Results that land in the fp32 subnormal range are excluded: the AMDGPU
    backend flushes fp32 denormals, so *both* decode paths return zero there
    (asserted equal to each other by the test above). Only torch, on the host,
    keeps them. ``encoded_scale=0`` is covered separately below.
    """
    codes, encoded = _all_codes(encoded_scale)
    ref = codes.view(torch.float8_e4m3fn).to(torch.float32) * 2.0 ** (
        encoded_scale - 127
    )
    got = _decode(codes, encoded, use_fnuz=False, bitops=True)

    comparable = torch.isfinite(ref) & ((ref == 0) | (ref.abs() >= 2.0**-126))
    assert torch.equal(got[comparable], ref[comparable])


def test_smallest_ue8m0_exponent_flushes_on_both_paths():
    """``encoded_scale=0`` means 2**-127, itself a subnormal fp32.

    The AMDGPU backend flushes it to zero before it ever reaches an element, so
    the block decodes to zeros whichever decode is used. Documented here so the
    gap in the torch-anchored test above is a known property, not an oversight;
    a real UE8M0 KV block never carries an exponent this small.
    """
    codes, encoded = _all_codes(0)
    got = _decode(codes, encoded, use_fnuz=False, bitops=True)
    ref = _decode(codes, encoded, use_fnuz=False, bitops=False)

    keep = torch.ones(256, dtype=torch.bool, device=device)
    keep[list(_OCP_NAN_CODES)] = False  # NaN is not preserved by either scaling
    assert torch.equal(got[keep], ref[keep])
    assert not got[keep].any()


def test_subnormals_and_zero_are_exact():
    """Subnormal e4m3 codes (exponent 0) are what a naive shift decode breaks.

    Uses a 2**0 scale so every result stays in normal fp32 range and the decode
    is compared against torch with no flush-to-zero caveat.
    """
    codes, encoded = _all_codes(127)
    ref = codes.view(torch.float8_e4m3fn).to(torch.float32)
    got = _decode(codes, encoded, use_fnuz=False, bitops=True)

    sub = (ref != 0) & (ref.abs() < 2.0**-6)
    assert int(sub.sum()) == 14, (
        "expected 14 subnormal e4m3 codes (7 magnitudes x 2 signs)"
    )
    assert torch.equal(got[sub], ref[sub])

    zero = ref == 0
    assert torch.equal(got[zero], ref[zero])
    # Signed zero must keep its sign bit.
    assert torch.equal(
        torch.signbit(got[zero]).to(torch.uint8),
        torch.signbit(ref[zero]).to(torch.uint8),
    )


def test_scale_is_applied_per_element():
    """Each element carries its own UE8M0 exponent (one per 64-element block).

    Spans the whole exponent range, which is what caught an earlier version
    that folded the widening factor into the scale: that overflowed to inf well
    before 2**127 does.
    """
    codes = torch.full((256,), 0x38, dtype=torch.uint8, device=device)  # e4m3 1.0
    encoded = torch.arange(256, dtype=torch.uint8, device=device)
    got = _decode(codes, encoded, use_fnuz=False, bitops=True)

    expect = torch.exp2(encoded.to(torch.float32) - 127.0)
    normal = expect >= 2.0**-126  # below that the GPU flushes, torch does not
    assert torch.equal(got[normal], expect[normal])


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fnuz"), reason="torch lacks float8_e4m3fnuz"
)
@pytest.mark.parametrize("encoded_scale", [64, 127, 134])
def test_bitops_matches_torch_fnuz(encoded_scale):
    """FNUZ uses exponent bias 8 rather than 7, so it widens by a different factor."""
    codes, encoded = _all_codes(encoded_scale)
    ref = codes.view(torch.float8_e4m3fnuz).to(torch.float32) * 2.0 ** (
        encoded_scale - 127
    )
    got = _decode(codes, encoded, use_fnuz=True, bitops=True)

    # 0x80 is NaN in FNUZ (it has no negative zero); everything else is finite.
    keep = torch.isfinite(ref)
    keep[0x80] = False
    assert torch.equal(got[keep], ref[keep])


@triton.jit
def _encode_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    USE_FNUZ: tl.constexpr,
    FP8_BITOPS: tl.constexpr,
):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, quant_fp8_to_uint8(x, USE_FNUZ, FP8_BITOPS))


def _encode(x: torch.Tensor, use_fnuz: bool, bitops: bool):
    out = torch.empty(x.numel(), dtype=torch.uint8, device=device)
    _encode_kernel[(1,)](x, out, N=x.numel(), USE_FNUZ=use_fnuz, FP8_BITOPS=bitops)
    return out


def _encode_inputs():
    """Every representable value, both midpoints around each, and the subnormals."""
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    vals = codes.view(torch.float8_e4m3fn).to(torch.float32)
    vals = vals[torch.isfinite(vals)]
    mids = (vals[:-1] + vals[1:]) / 2  # exact ties -> exercises round-half-to-even
    band = torch.linspace(-0.02, 0.02, 4096, device=device)  # subnormal band
    dense = torch.linspace(-448, 448, 8192, device=device)
    x = torch.cat([vals, mids, vals * 1.0001, vals * 0.9999, band, dense])
    # tl.arange needs a power-of-two length; zeros are a valid encode input.
    pad = triton.next_power_of_2(x.numel()) - x.numel()
    return torch.cat([x, torch.zeros(pad, device=device)]).contiguous()


def test_encode_bitops_matches_torch():
    """The native encode must reproduce torch's cast bit-for-bit, ties included."""
    x = _encode_inputs()
    got = _encode(x, use_fnuz=False, bitops=True)
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(got, ref), f"{int((got != ref).sum())} of {x.numel()} differ"


def test_encode_bitops_matches_convert_instruction():
    x = _encode_inputs()
    got = _encode(x, use_fnuz=False, bitops=True)
    ref = _encode(x, use_fnuz=False, bitops=False)
    assert torch.equal(got, ref)


def test_encode_decode_round_trip():
    """Every finite code must survive decode -> encode unchanged."""
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    ref = codes.view(torch.float8_e4m3fn).to(torch.float32)
    finite = torch.isfinite(ref)
    x = torch.zeros(1024, dtype=torch.float32, device=device)
    x[:256] = ref
    x[~torch.isfinite(x)] = 0.0
    got = _encode(x, use_fnuz=False, bitops=True)[:256]
    assert torch.equal(got[finite], codes[finite])


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fnuz"), reason="torch lacks float8_e4m3fnuz"
)
def test_encode_bitops_matches_torch_fnuz():
    """FNUZ shifts the exponent bias by one; the encode tracks it."""
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    vals = codes.view(torch.float8_e4m3fnuz).to(torch.float32)
    vals = vals[torch.isfinite(vals)]
    x = torch.cat([vals, (vals[:-1] + vals[1:]) / 2])
    pad = triton.next_power_of_2(x.numel()) - x.numel()
    x = torch.cat([x, torch.zeros(pad, device=device)]).contiguous()
    got = _encode(x, use_fnuz=True, bitops=True)
    ref = x.to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert torch.equal(got, ref), f"{int((got != ref).sum())} of {x.numel()} differ"


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="KV insert path is exercised on ROCm here"
)
def test_kv_cache_insert_is_identical_either_way():
    """End-to-end: the encoder swap must not change a single cache byte.

    The helper is validated exhaustively above; this pins the wiring, since the
    surrounding kernel computes the UE8M0 exponent itself and writes the bytes.
    """
    from unittest.mock import patch

    from vllm.models.deepseek_v4.common.ops import cache_utils

    torch.manual_seed(0)
    num_tokens, block_size = 37, 64
    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device) * 3.0
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_stride = block_size * (576 + 8)
    n_blocks = num_tokens // block_size + 2

    caches = []
    for bitops in (False, True):
        cache = torch.zeros(n_blocks, block_stride, dtype=torch.uint8, device=device)
        with patch.object(
            cache_utils, "use_fp8_bitops_decode", lambda bitops=bitops: bitops
        ):
            cache_utils.quantize_and_insert_k_cache(
                k, cache, slot_mapping, block_size=block_size
            )
        caches.append(cache)

    assert torch.equal(caches[0], caches[1])
    assert caches[1].any(), "cache was never written"
