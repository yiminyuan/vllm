# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-indexer MQA logits on RDNA2.

RDNA2 has neither an FP8 convert instruction nor an FP8 matrix core, so the
gfx942 kernel -- which hands FP8 operands straight to ``tl.dot`` -- cannot run
here. Without a kernel the caller falls back to a dense reference that
materialises an ``[H, M, N]`` fp32 score tensor, which is quadratic in sequence
length and collapses on long prompts. These tests pin the RDNA2 kernel to that
reference, including the sequence lengths where the fallback becomes unusable.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="RDNA2 kernel requires ROCm"
)


def _inputs(num_q: int, num_kv: int, heads: int, head_dim: int, window: int | None):
    """Random FP8 operands plus a per-row [start, end) window."""
    torch.manual_seed(1234)
    device = "cuda"
    fp8 = current_platform.fp8_dtype()

    q = (torch.randn(num_q, heads, head_dim, device=device) / 4).to(fp8)
    k = (torch.randn(num_kv, head_dim, device=device) / 4).to(fp8)
    scale = torch.rand(num_kv, 1, device=device, dtype=torch.float32) + 0.5
    weights = torch.randn(num_q, heads, device=device, dtype=torch.float32)

    ends = torch.arange(1, num_q + 1, device=device, dtype=torch.int32)
    ends = torch.clamp(ends * (num_kv // max(num_q, 1)) + 1, max=num_kv)
    if window is None:
        starts = torch.zeros(num_q, device=device, dtype=torch.int32)
    else:
        starts = torch.clamp(ends - window, min=0).to(torch.int32)
    return q, k, scale, weights, starts, ends


def _run_kernel(q, k, scale, weights, starts, ends):
    from vllm.v1.attention.ops.triton_fp8_mqa_logits import fp8_mqa_logits_gfx1030

    return fp8_mqa_logits_gfx1030(q, k, scale, weights, starts, ends)


@pytest.mark.parametrize(
    "num_q,num_kv,heads,head_dim",
    [
        (16, 64, 16, 64),
        (64, 256, 16, 64),
        (128, 512, 32, 128),
        (256, 1024, 64, 128),  # DeepSeek-V4 indexer shape
    ],
)
def test_matches_dense_reference(num_q, num_kv, heads, head_dim):
    """The kernel must agree with the reference it replaces."""
    q, k, scale, weights, starts, ends = _inputs(
        num_q, num_kv, heads, head_dim, window=None
    )

    got = _run_kernel(q, k, scale, weights, starts, ends)
    want = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)

    assert got.shape == want.shape == (num_q, num_kv)
    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=2e-2, atol=2e-2)


def test_positions_outside_the_window_are_neg_inf():
    """The consumer runs top-k without masking, so excluded positions must be
    -inf rather than a small number."""
    q, k, scale, weights, starts, ends = _inputs(32, 128, 16, 64, window=8)

    got = _run_kernel(q, k, scale, weights, starts, ends)
    want = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)

    outside = ~torch.isfinite(want)
    assert outside.any(), "test window did not exclude anything"
    assert torch.all(got[outside] == float("-inf"))
    torch.testing.assert_close(
        got[~outside], want[~outside], rtol=2e-2, atol=2e-2
    )


def test_long_sequence_matches_reference():
    """The length class that makes the dense fallback collapse.

    Kept to a small head count so the reference's [H, M, N] fp32 tensor still
    fits while the sequence is long enough to be representative.
    """
    q, k, scale, weights, starts, ends = _inputs(2048, 4096, 8, 64, window=None)

    got = _run_kernel(q, k, scale, weights, starts, ends)
    want = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)

    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=2e-2, atol=2e-2)


def test_relu_is_applied_per_head_before_weighting():
    """Negative per-head scores must be clamped before the weighted sum; a
    kernel that weights first and clamps after passes shape checks but returns
    different logits wherever a head scores negative."""
    q, k, scale, weights, starts, ends = _inputs(32, 128, 16, 64, window=None)
    weights = -weights.abs()  # force every head weight negative

    got = _run_kernel(q, k, scale, weights, starts, ends)
    want = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)

    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=2e-2, atol=2e-2)
    assert (want[finite] <= 0).all(), "expected non-positive logits in this setup"


def test_smallest_fp8_codes_survive_the_fp16_widening():
    """Guard the hazard the fp16 fast path introduces.

    Mantissa-aligning e4m3 into fp16 leaves every value a fixed power of two
    too small, so the smallest codes land on fp16 subnormals. They are exactly
    representable, but a build that flushes half-precision denormals would
    silently zero them -- which this catches.
    """
    device = "cuda"
    fp8 = current_platform.fp8_dtype()
    heads, head_dim, num_kv = 16, 64, 64

    # Sweep the low end of the e4m3 encoding, including subnormal codes.
    codes = torch.arange(1, 9, device=device, dtype=torch.uint8)
    tiny = codes.view(torch.uint8).to(torch.uint8)
    q = torch.zeros(1, heads, head_dim, device=device, dtype=torch.uint8)
    q[0, :, : tiny.numel()] = tiny
    q = q.view(fp8)
    k = torch.zeros(num_kv, head_dim, device=device, dtype=torch.uint8)
    k[:, : tiny.numel()] = tiny
    k = k.view(fp8)

    scale = torch.full((num_kv, 1), 1.0, device=device, dtype=torch.float32)
    weights = torch.ones(1, heads, device=device, dtype=torch.float32)
    starts = torch.zeros(1, device=device, dtype=torch.int32)
    ends = torch.full((1,), num_kv, device=device, dtype=torch.int32)

    got = _run_kernel(q, k, scale, weights, starts, ends)
    want = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)

    assert (want > 0).any(), "reference produced no signal to compare against"
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-6)
