# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDNA2 fused mHC projection.

``mhc_proj_rdna2`` replaces the widen-then-project pair on gfx1030: it reads the
fp16/bf16 residual once and returns both the projection and the per-row sum of
squares, so the fp32 copy of the residual never reaches memory. These check it
against the portable path it stands in for, including the token counts where the
launch config switches and the shapes where it must decline and fall back.

Run `pytest tests/kernels/test_mhc_proj_rdna2.py`.
"""

import importlib

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

mhc_triton = importlib.import_module("vllm.model_executor.kernels.mhc.triton")

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx1030
else:

    def on_gfx1030() -> bool:
        return False


pytestmark = pytest.mark.skipif(
    not on_gfx1030(), reason="RDNA2 (gfx1030) mHC projection"
)

DTYPES = [torch.float16, torch.bfloat16]
# num_tokens: 1 and 8 are decode widths, 31/32 straddle the wide/narrow launch
# switch, 512 is a full prefill chunk, 513 leaves a partial token block.
TOKENS = [1, 8, 31, 32, 64, 512, 513]


def _inputs(num_tokens, hc_mult, hidden_size, dtype):
    set_random_seed(0)
    k = hc_mult * hidden_size
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    x = torch.randn((num_tokens, k), dtype=dtype, device="cuda")
    fn = torch.randn((hc_mult3, k), dtype=torch.float32, device="cuda") * 1e-3
    return x, fn


def _portable(x, fn):
    """The widen-then-project path the kernel replaces."""
    num_tokens, k = x.shape
    x_f32 = torch.empty((num_tokens, k), dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    mhc_triton._cast_and_sqrsum_kernel[(num_tokens,)](
        x, x_f32, sqrsum, k, BLOCK_K=1024, num_warps=4
    )
    return torch.nn.functional.linear(x_f32, fn), sqrsum


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("hc_mult", [2, 4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_portable_path(num_tokens, hc_mult, dtype):
    x, fn = _inputs(num_tokens, hc_mult, 1024, dtype)
    mixes, sqrsum = ops.mhc_proj_rdna2(x, fn)
    ref_mixes, ref_sqrsum = _portable(x, fn)

    assert mixes.shape == (num_tokens, fn.shape[0])
    assert sqrsum.shape == (num_tokens,)
    assert mixes.dtype == torch.float32 and sqrsum.dtype == torch.float32

    # Both accumulate in fp32 over the same exactly-widened values; only the
    # summation order differs, so the tolerance tracks sqrt(K) * eps.
    k = x.shape[1]
    atol = torch.finfo(torch.float32).eps * (k**0.5) * 8
    torch.testing.assert_close(mixes, ref_mixes, atol=atol, rtol=1e-3)
    torch.testing.assert_close(sqrsum, ref_sqrsum, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_widening_is_exact(dtype):
    """The kernel must see the same values a widened copy would.

    fp16 and bf16 both convert to fp32 losslessly, so widening in-register
    cannot change a product -- this pins that, since a truncating conversion
    would still look plausible on random data.
    """
    x, fn = _inputs(64, 4, 1024, dtype)
    mixes, _ = ops.mhc_proj_rdna2(x, fn)
    exact = torch.nn.functional.linear(x.float(), fn)
    k = x.shape[1]
    atol = torch.finfo(torch.float32).eps * (k**0.5) * 8
    torch.testing.assert_close(mixes, exact, atol=atol, rtol=1e-3)


def test_sqrsum_matches_direct_reduction():
    x, fn = _inputs(128, 4, 1024, torch.float16)
    _, sqrsum = ops.mhc_proj_rdna2(x, fn)
    ref = (x.float() ** 2).sum(dim=1)
    torch.testing.assert_close(sqrsum, ref, atol=1e-2, rtol=1e-3)


def test_dispatch_prefers_the_kernel_and_agrees_with_the_fallback():
    """_mhc_cast_and_proj must route here and return what the fallback would."""
    x, fn = _inputs(256, 4, 1024, torch.float16)
    got_mixes, got_sqr = mhc_triton._mhc_cast_and_proj(x, fn)
    ker_mixes, ker_sqr = ops.mhc_proj_rdna2(x, fn)
    torch.testing.assert_close(got_mixes, ker_mixes, atol=0, rtol=0)
    torch.testing.assert_close(got_sqr, ker_sqr, atol=0, rtol=0)

    ref_mixes, ref_sqr = _portable(x, fn)
    k = x.shape[1]
    atol = torch.finfo(torch.float32).eps * (k**0.5) * 8
    torch.testing.assert_close(got_mixes, ref_mixes, atol=atol, rtol=1e-3)
    torch.testing.assert_close(got_sqr, ref_sqr, atol=atol, rtol=1e-3)


def test_unsupported_width_falls_back_rather_than_raising():
    """hc_mult=3 gives nout=15, which has no instantiation."""
    x, fn = _inputs(64, 3, 1024, torch.float16)
    assert fn.shape[0] not in mhc_triton._RDNA2_MHC_PROJ_NOUT
    mixes, sqrsum = mhc_triton._mhc_cast_and_proj(x, fn)
    ref_mixes, ref_sqr = _portable(x, fn)
    k = x.shape[1]
    atol = torch.finfo(torch.float32).eps * (k**0.5) * 8
    torch.testing.assert_close(mixes, ref_mixes, atol=atol, rtol=1e-3)
    torch.testing.assert_close(sqrsum, ref_sqr, atol=atol, rtol=1e-3)


def test_rejects_unsupported_dtype_and_layout():
    x, fn = _inputs(64, 4, 1024, torch.float16)
    with pytest.raises(RuntimeError, match="fp16 and bf16"):
        ops.mhc_proj_rdna2(x.float(), fn)
    with pytest.raises(RuntimeError, match="fp32 fn"):
        ops.mhc_proj_rdna2(x, fn.half())
    with pytest.raises(RuntimeError, match="K-contiguous|packed rows"):
        ops.mhc_proj_rdna2(x[:, ::2], fn[:, ::2])


def test_empty_batch_returns_empty():
    x, fn = _inputs(0, 4, 1024, torch.float16)
    mixes, sqrsum = ops.mhc_proj_rdna2(x, fn)
    assert mixes.shape == (0, fn.shape[0])
    assert sqrsum.shape == (0,)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 512])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_portable_path_at_model_width(num_tokens, dtype):
    """DeepSeek-V4-Flash width: hidden_size 4096 with hc_mult 4, so K = 16384.

    The other equivalence test runs at hidden_size 1024 (K = 4096). Dispatch
    here is width-dependent -- there is a wide/narrow launch switch and an
    unsupported-width fallback -- so the deployed K needs its own check.
    """
    x, fn = _inputs(num_tokens, 4, 4096, dtype)
    mixes, sqrsum = ops.mhc_proj_rdna2(x, fn)
    ref_mixes, ref_sqrsum = _portable(x, fn)

    assert mixes.shape == (num_tokens, fn.shape[0])
    k = x.shape[1]
    atol = torch.finfo(torch.float32).eps * (k**0.5) * 8
    torch.testing.assert_close(mixes, ref_mixes, atol=atol, rtol=1e-3)
    torch.testing.assert_close(sqrsum, ref_sqrsum, atol=atol, rtol=1e-3)
