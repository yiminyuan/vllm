#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC on the non-TileLang path, in fp16 as well as bf16.

The TileLang mHC kernels are bf16-only. RDNA2 has no bf16 dot product, so
gfx1030 runs DeepSeek-V4 in fp16 and takes the torch/triton fallbacks instead
(see ``_has_tilelang_mhc``). These check the fallbacks carry fp16 through
without silently widening to bf16 or losing accuracy.

Run `pytest tests/kernels/test_mhc_fp16_fallback.py`.
"""

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

HC_MULT = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULT = 1.0
SINKHORN_REPEAT = 3


def _inputs(num_tokens, hidden_size, dtype):
    set_random_seed(0)
    hc_mult3 = HC_MULT * 2 + HC_MULT * HC_MULT
    residual = torch.randn(
        (num_tokens, HC_MULT, hidden_size), dtype=dtype, device=DEVICE
    )
    fn = (
        torch.randn(
            (hc_mult3, HC_MULT * hidden_size), dtype=torch.float32, device=DEVICE
        )
        * 1e-4
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, device=DEVICE) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float32, device=DEVICE) * 0.1
    return residual, fn, hc_scale, hc_base


def _pre(residual, fn, hc_scale, hc_base):
    return mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN_REPEAT,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_mhc_pre_preserves_dtype(dtype, num_tokens):
    """layer_input must come back in the residual's dtype, not hardcoded bf16."""
    residual, fn, hc_scale, hc_base = _inputs(num_tokens, 512, dtype)
    post_mix, comb_mix, layer_input = _pre(residual, fn, hc_scale, hc_base)

    assert layer_input.dtype == dtype
    assert layer_input.shape == (num_tokens, 512)
    # The mixes stay fp32 on every path; only the residual stream is narrowed.
    assert post_mix.dtype == torch.float32
    assert comb_mix.dtype == torch.float32
    assert torch.isfinite(layer_input).all()


def _layer_input_ref(residual_f32, fn, hc_scale, hc_base):
    """fp32 reference for the layer_input branch of mhc_pre."""
    x = residual_f32.flatten(-2)
    mixes = x @ fn.t()
    mixes = mixes * torch.rsqrt(
        x.square().sum(-1, keepdim=True) / x.shape[-1] + RMS_EPS
    )
    pre_mix = (
        torch.sigmoid(mixes[:, :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    )
    return torch.sum(pre_mix.unsqueeze(-1) * residual_f32, dim=1)


@pytest.mark.parametrize("num_tokens", [1, 8, 128])
def test_mhc_pre_fp16_no_worse_than_bf16(num_tokens):
    """fp16 must track the fp32 math at least as closely as bf16 does.

    Both runs get bit-identical inputs: the residual is rounded to bf16 first,
    and fp16 represents those values exactly (10 mantissa bits vs 8), so the
    only difference is the dtype the result is narrowed to.
    """
    residual, fn, hc_scale, hc_base = _inputs(num_tokens, 512, torch.float32)
    residual_bf16 = residual.to(torch.bfloat16)
    residual_fp16 = residual_bf16.to(torch.float16)
    assert torch.equal(residual_fp16.float(), residual_bf16.float())

    ref = _layer_input_ref(residual_bf16.float(), fn, hc_scale, hc_base)
    _, _, got_fp16 = _pre(residual_fp16, fn, hc_scale, hc_base)
    _, _, got_bf16 = _pre(residual_bf16, fn, hc_scale, hc_base)

    err_fp16 = (got_fp16.float() - ref).abs().max()
    err_bf16 = (got_bf16.float() - ref).abs().max()
    assert err_fp16 <= err_bf16, f"fp16 ({err_fp16}) worse than bf16 ({err_bf16})"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_post_preserves_dtype(dtype):
    """mhc_post already keys off residual.dtype; guard it stays that way."""
    num_tokens, hidden_size = 8, 512
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=DEVICE)
    residual = torch.randn(
        (num_tokens, HC_MULT, hidden_size), dtype=dtype, device=DEVICE
    )
    post_layer_mix = torch.randn(
        (num_tokens, HC_MULT, 1), dtype=torch.float32, device=DEVICE
    )
    comb_res_mix = torch.randn(
        (num_tokens, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE
    )

    out = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    assert out.dtype == dtype
    assert out.shape == residual.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hc_head_triton_preserves_dtype(dtype, default_vllm_config):
    """hc_head's triton fallback allocates its own output; it must not force bf16.

    This is the path DSpark/MTP take for the head reduction once TileLang is
    off, so a hardcoded bf16 buffer there would silently widen the fp16 model.
    """
    from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC, HCHeadOp

    if HAS_TILELANG_MHC:
        pytest.skip("TileLang mHC selected; this covers the triton fallback")

    num_tokens, hidden_size = 8, 512
    hidden = torch.randn((num_tokens, HC_MULT, hidden_size), dtype=dtype, device=DEVICE)
    hc_fn = (
        torch.randn(
            (HC_MULT, HC_MULT * hidden_size), dtype=torch.float32, device=DEVICE
        )
        * 1e-4
    )
    hc_scale = torch.randn((1,), dtype=torch.float32, device=DEVICE) * 0.1
    hc_base = torch.randn((HC_MULT,), dtype=torch.float32, device=DEVICE) * 0.1

    out = HCHeadOp()(hidden, hc_fn, hc_scale, hc_base, RMS_EPS, HC_EPS)

    assert out.dtype == dtype
    assert out.shape == (num_tokens, hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="TileLang mHC gating is ROCm-specific"
)
def test_tilelang_mhc_disabled_on_rdna2():
    """gfx1030 must not select the bf16-only TileLang mHC kernels."""
    from vllm.model_executor.layers.mhc import _has_tilelang_mhc
    from vllm.platforms.rocm import on_gfx1030

    if not on_gfx1030():
        pytest.skip("not gfx1030")
    assert not _has_tilelang_mhc()


def _triton_pre(residual, fn, hc_scale, hc_base):
    hc_mult, hidden_size = residual.shape[-2:]
    outer = residual.shape[:-2]
    post_mix = torch.empty(
        (*outer, hc_mult, 1), dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        (*outer, hc_mult, hc_mult), dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        (*outer, hidden_size), dtype=residual.dtype, device=residual.device
    )
    torch.ops.vllm.mhc_pre_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN_REPEAT,
        post_mix,
        comb_mix,
        layer_input,
    )
    return post_mix, comb_mix, layer_input


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128])
def test_mhc_pre_triton_matches_torch(dtype, num_tokens):
    """The triton pre block must agree with the torch reference it replaces."""
    import vllm.model_executor.kernels.mhc  # noqa: F401

    residual, fn, hc_scale, hc_base = _inputs(num_tokens, 512, dtype)
    r_post, r_comb, r_layer = _pre(residual, fn, hc_scale, hc_base)
    t_post, t_comb, t_layer = _triton_pre(residual, fn, hc_scale, hc_base)

    assert t_layer.dtype == dtype
    # post/comb come off the same GEMM, so only the reduction order differs.
    torch.testing.assert_close(t_post, r_post, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(t_comb, r_comb, rtol=1e-5, atol=1e-6)
    # layer_input is narrowed to the residual dtype, so allow one ulp of it.
    ulp = torch.finfo(dtype).eps
    torch.testing.assert_close(t_layer.float(), r_layer.float(), rtol=ulp, atol=ulp * 8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_mhc_post_triton_matches_torch(dtype, num_tokens):
    import vllm.model_executor.kernels.mhc  # noqa: F401

    hidden_size = 512
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=DEVICE)
    residual = torch.randn(
        (num_tokens, HC_MULT, hidden_size), dtype=dtype, device=DEVICE
    )
    post_layer_mix = torch.randn(
        (num_tokens, HC_MULT, 1), dtype=torch.float32, device=DEVICE
    )
    comb_res_mix = torch.randn(
        (num_tokens, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE
    )

    ref = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    out = torch.empty_like(residual)
    torch.ops.vllm.mhc_post_triton(x, residual, post_layer_mix, comb_res_mix, out)

    assert out.dtype == dtype
    ulp = torch.finfo(dtype).eps
    torch.testing.assert_close(out.float(), ref.float(), rtol=ulp, atol=ulp * 8)
