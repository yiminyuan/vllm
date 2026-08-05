#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 W4A16 MoE kernel (gfx1030).

fp16-only MoE sibling of the dense RDNA2 W4A16 kernel (test_rdna2_w4a16.py) and
the RDNA3 MoE kernel (PR #44075 / #44570). The fused op

    torch.ops._rocm_C.moe_gptq_gemm_rdna2(...)

does expert routing + W4A16 dequant (exllamav2 "+1024" bit-trick, qdq_4_rdna2.cuh)
+ dot product with 64-bit-CAS atomic output, one HIP launch per grouped GEMM.
Every primitive is reused from the dense kernel and ISA-verified on gfx1030
(fdot2 / v_pk_fma_f16 / the +1024 bit-trick / GLOBAL_ATOMIC_CMPSWAP_X2); no new
hardware capability is required.

Zero-point convention (verified against q_gemm_rdna2.cu:307
``zero_offset = use_v2_format ? 0 : 1``):
  - symmetric GPTQ (v1, use_v2_format=False): effective_zero = stored + 1;
    the loader synthesizes stored = uint4b8.bias - 1 = 7 -> effective 8.
  - asymmetric AWQ (v2, use_v2_format=True):  effective_zero = stored (no +1);
    the real AWQ zero-point is used directly.

Per-expert weight layout (dense layout + an expert dim):
  - qweight : packed int32 ``[E, K//8, N]``  (exllama nibble shuffle)
  - scales  : ``[E, groups, N]`` in activation dtype
  - qzeros  : packed int32 ``[E, groups, N//8]``  (stored zero-point)

Correctness is anchored three ways: an explicit per-token reference (_ref_moe),
an independent vectorized reference (_ref_moe_vec) asserted equal to it, and a
cross-check of the reference against vLLM's own ``fused_experts`` on dequantized
fp16 weights. The kernel is then compared to the reference.

Run `pytest tests/kernels/quantization/test_rdna2_moe_w4a16.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 MoE W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx1030  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4b8  # int4, bias = 8
PACK_FACTOR = 8  # 8 x 4-bit nibbles per int32

# Relative Frobenius norm, like the dense RDNA2 W4A16 test: the +1024 bit-trick
# gives large *relative* error on near-cancellation outputs, so a per-element
# tolerance is unsuitable. The dense kernel bounds one GEMM at 5e-2; a MoE forward
# chains two, so 1e-1. A separate per-token bound guards routing (a mis-routed
# token's row is ~1.0; bit-trick row noise stays under ~0.1).
_REL_L2_TOL = 1e-1
_PER_TOKEN_TOL = 0.3

gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx1030()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna2")
    ),
    reason="requires gfx1030 with the _rocm_C.moe_gptq_gemm_rdna2 op built in",
)


def _rel_l2(a, b):
    return (a.float() - b.float()).norm() / b.float().norm()


def _assert_moe_close(out, ref):
    assert _rel_l2(out, ref) < _REL_L2_TOL, f"rel-L2 {_rel_l2(out, ref):.3f}"
    denom = ref.float().norm(dim=1).clamp_min(1e-3)
    row = (out.float() - ref.float()).norm(dim=1) / denom
    assert row.max() < _PER_TOKEN_TOL, f"per-token rel-L2 max {row.max():.3f}"


# ---------------------------------------------------------------------------
# Quantized expert weights + the single source-of-truth fp32 dequant.
# ---------------------------------------------------------------------------


def _quantize_expert_weights(E, K, N, G, asym, dtype):
    """Random per-expert 4-bit weights.

    Returns (w_dq, qweight, scales, qzeros, use_v2_format):
      w_dq            : [E, K, N] fp32 dequantized weight — the source of truth,
                        used by BOTH the reference and the fused_experts oracle.
      qweight/scales/qzeros : the packed+shuffled tensors the kernel consumes.
      use_v2_format   : the flag the kernel must be called with (asym -> True).

    w_dq = (q_int4 - (stored_zero + zero_offset)) * scale, with
      zero_offset = 0 for asym AWQ (v2), 1 for sym GPTQ (v1)  [q_gemm_rdna2.cu:307]
    """
    groups = K // G
    q = torch.randint(0, 16, (E, K, N), device=device, dtype=torch.int32)
    scales = (0.05 * torch.rand((E, groups, N), device=device) + 0.01).to(dtype)

    if asym:
        stored = torch.randint(0, 16, (E, groups, N), device=device, dtype=torch.int32)
        zero_offset = 0  # v2: effective zero == stored
    else:
        stored = torch.full(
            (E, groups, N), WEIGHT_TYPE.bias - 1, device=device, dtype=torch.int32
        )  # synth 7
        zero_offset = 1  # v1: effective zero == stored + 1 == 8

    eff_zero = (stored + zero_offset).repeat_interleave(G, dim=1).float()  # [E,K,N]
    s_full = scales.float().repeat_interleave(G, dim=1)  # [E,K,N]
    w_dq = (q.float() - eff_zero) * s_full  # [E,K,N]

    # Kernel inputs: pack along K -> [E, K//8, N], then exllama-shuffle per expert.
    qweight = torch.stack(
        [
            pack_quantized_values_into_int32(q[e], WEIGHT_TYPE, packed_dim=0)
            for e in range(E)
        ]
    )
    empty_g_idx = torch.empty(0, dtype=torch.int32, device=device)
    for e in range(E):
        w_e = qweight[e].contiguous()
        ops.gptq_shuffle(w_e, empty_g_idx, 4)
        qweight[e] = w_e

    # qzeros: pack the STORED zero along N -> [E, groups, N//8].
    qzeros = torch.stack(
        [
            pack_quantized_values_into_int32(stored[e], WEIGHT_TYPE, packed_dim=1)
            for e in range(E)
        ]
    )
    return w_dq, qweight, scales, qzeros, asym


# ---------------------------------------------------------------------------
# References (independent implementations; asserted equal in test_reference_*).
# ---------------------------------------------------------------------------


def _ref_moe(x, w13_dq, w2_dq, topk_ids, topk_w):
    """Explicit per-(token, slot) reference — the MoE definition."""
    M, K = x.shape
    inter = w2_dq.shape[1]
    xf = x.float()
    out = torch.zeros(M, K, device=x.device, dtype=torch.float32)
    for m in range(M):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[m, k])
            gate_up = xf[m] @ w13_dq[e]  # [2I]  (w13_dq is [K, 2I])
            act = F.silu(gate_up[:inter]) * gate_up[inter:]  # [inter]
            out[m] += float(topk_w[m, k]) * (act @ w2_dq[e])  # [K]
    return out.to(x.dtype)


def _ref_moe_vec(x, w13_dq, w2_dq, topk_ids, topk_w):
    """Vectorized reference (different code path, same math)."""
    M, top_k = topk_ids.shape
    inter = w2_dq.shape[1]
    xf = x.float()
    e = topk_ids.reshape(-1).long()  # [M*top_k]
    xr = xf.repeat_interleave(top_k, dim=0)  # [M*top_k, K]
    gate_up = torch.bmm(xr.unsqueeze(1), w13_dq[e]).squeeze(1)  # [M*tk, 2I]
    act = F.silu(gate_up[:, :inter]) * gate_up[:, inter:]  # [M*tk, inter]
    down = torch.bmm(act.unsqueeze(1), w2_dq[e]).squeeze(1)  # [M*tk, K]
    down = down * topk_w.reshape(-1, 1).float()
    return down.view(M, top_k, -1).sum(dim=1).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel driver (runs the op exactly as the integration will).
# ---------------------------------------------------------------------------


def _run_kernel_moe(x, w13, w2, topk_ids, topk_w, use_v2):
    w13_q, w13_s, w13_z = w13
    w2_q, w2_s, w2_z = w2
    M, K = x.shape
    top_k = topk_ids.shape[1]
    E, _, N13 = w13_q.shape
    inter = N13 // 2
    total = M * top_k
    block_m = 1 if M <= 4 else 4

    sorted_ids, expert_ids, num_pad = moe_align_block_size(topk_ids, block_m, E, None)
    topk_w_f = topk_w.reshape(-1).float()
    empty = torch.empty(0, device=device)

    w1_out = torch.zeros(total, N13, dtype=x.dtype, device=device)
    ops.moe_gptq_gemm_rdna2(
        x,
        w1_out,
        w13_q,
        w13_s,
        w13_z,
        empty,
        sorted_ids,
        expert_ids,
        num_pad,
        top_k,
        block_m,
        False,
        use_v2,  # mul_topk_weights=False (applied on w2)
    )
    act = torch.empty(total, inter, dtype=x.dtype, device=device)
    torch.ops._C.silu_and_mul(act, w1_out)

    out = torch.zeros(M, K, dtype=x.dtype, device=device)
    ops.moe_gptq_gemm_rdna2(
        act,
        out,
        w2_q,
        w2_s,
        w2_z,
        topk_w_f,
        sorted_ids,
        expert_ids,
        num_pad,
        1,
        block_m,
        True,
        use_v2,  # mul_topk_weights=True
        output_topk=top_k,  # fuse moe_sum: write out[token // top_k]
    )
    return out


def _routing(M, E, top_k, dtype):
    logits = torch.randn(M, E, device=device)
    topk_w, topk_ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(dtype)
    return topk_ids.to(torch.int32), topk_w


# ---------------------------------------------------------------------------
# "Test the test": the reference is correct.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_reference_self_consistent(asym, dist_init):
    """Loop and vectorized references must agree (validates routing/indexing)."""
    set_random_seed(0)
    E, M, K, inter, top_k, G = 8, 16, 256, 256, 2, 128
    dtype = torch.float16
    x = (0.25 * torch.randn(M, K, device=device)).to(dtype)
    w13_dq, *_ = _quantize_expert_weights(E, K, 2 * inter, G, asym, dtype)
    w2_dq, *_ = _quantize_expert_weights(E, inter, K, G, asym, dtype)
    topk_ids, topk_w = _routing(M, E, top_k, dtype)

    a = _ref_moe(x, w13_dq, w2_dq, topk_ids, topk_w)
    b = _ref_moe_vec(x, w13_dq, w2_dq, topk_ids, topk_w)
    assert _rel_l2(a, b) < 1e-4, "loop vs vectorized reference disagree"


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
def test_reference_matches_fused_experts(asym, dist_init):
    """Validate the reference's MoE convention against vLLM's fused_experts
    (unquantized Triton MoE on the dequantized fp16 weights)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    set_random_seed(0)
    E, M, K, inter, top_k, G = 8, 16, 256, 256, 2, 128
    dtype = torch.float16
    x = (0.25 * torch.randn(M, K, device=device)).to(dtype)
    w13_dq, *_ = _quantize_expert_weights(E, K, 2 * inter, G, asym, dtype)
    w2_dq, *_ = _quantize_expert_weights(E, inter, K, G, asym, dtype)
    topk_ids, topk_w = _routing(M, E, top_k, dtype)

    ref = _ref_moe(x, w13_dq, w2_dq, topk_ids, topk_w)
    w1 = w13_dq.transpose(1, 2).contiguous().to(dtype)  # [E, 2I, K]
    w2 = w2_dq.transpose(1, 2).contiguous().to(dtype)  # [E, K, inter]
    got = fused_experts(
        x, w1, w2, topk_w.float(), topk_ids, activation=MoEActivation.SILU
    )
    _assert_moe_close(got, ref)


# ---------------------------------------------------------------------------
# Kernel correctness vs the validated reference.
# ---------------------------------------------------------------------------

# (E, M, K, inter, top_k, G) — decode (small M) and prefill (large M); group 32/64/128.
MOE_SHAPES = [
    (8, 1, 256, 256, 2, 128),  # decode, top_k=2
    (8, 4, 256, 512, 2, 128),  # small batch
    (16, 8, 512, 512, 4, 64),  # top_k=4, group 64
    (32, 16, 512, 768, 4, 32),  # group 32 (AWQ default here)
    (8, 64, 512, 512, 2, 128),  # prefill
    (16, 128, 768, 512, 4, 128),  # larger prefill
]


@gfx1030_only
@pytest.mark.parametrize("asym", [False, True], ids=["gptq_sym", "awq_asym"])
@pytest.mark.parametrize(
    "E,M,K,inter,top_k,G",
    MOE_SHAPES,
    ids=[f"e{e}_m{m}_k{k}_i{i}_tk{tk}_g{g}" for e, m, k, i, tk, g in MOE_SHAPES],
)
def test_rdna2_moe_w4a16_matches_reference(asym, E, M, K, inter, top_k, G, dist_init):
    set_random_seed(0)
    dtype = torch.float16
    assert K % G == 0 and inter % G == 0 and (2 * inter) % PACK_FACTOR == 0
    x = (0.25 * torch.randn(M, K, device=device)).to(dtype)

    w13_dq, w13_q, w13_s, w13_z, v2 = _quantize_expert_weights(
        E, K, 2 * inter, G, asym, dtype
    )
    w2_dq, w2_q, w2_s, w2_z, _ = _quantize_expert_weights(E, inter, K, G, asym, dtype)
    topk_ids, topk_w = _routing(M, E, top_k, dtype)

    ref = _ref_moe(x, w13_dq, w2_dq, topk_ids, topk_w)
    out = _run_kernel_moe(
        x, (w13_q, w13_s, w13_z), (w2_q, w2_s, w2_z), topk_ids, topk_w, v2
    )
    assert out.shape == (M, K) and out.dtype == dtype
    _assert_moe_close(out, ref)


# ---------------------------------------------------------------------------
# Integration: the MoeWNA16 -> RDNA2 path must match Triton's moe_wna16 (the
# kernel it replaces) on identical checkpoint-layout weights. This validates the
# layout conversion in moe_wna16_rdna2._to_kernel_layout end to end.
# ---------------------------------------------------------------------------


@gfx1030_only
@pytest.mark.parametrize("top_k,G", [(2, 128), (4, 32)])
def test_moe_wna16_rdna2_matches_triton(top_k, G, dist_init):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.quantization.moe_wna16_rdna2 import (
        _rdna2_fused_moe,
        _to_kernel_layout,
    )

    set_random_seed(0)
    E, M, K, inter = 8, 16, 256, 256
    dtype = torch.float16
    x = (0.25 * torch.randn(M, K, device=device)).to(dtype)

    def mk(N, Kin):  # moe_wna16 canonical layout for one weight matrix
        qw = torch.randint(0, 256, (E, N, Kin // 2), dtype=torch.uint8, device=device)
        sc = (0.02 * torch.rand((E, N, Kin // G), device=device) + 0.005).to(dtype)
        qz = torch.randint(
            0, 256, (E, N // 2, Kin // G), dtype=torch.uint8, device=device
        )
        return qw, sc, qz

    w13 = mk(2 * inter, K)
    w2 = mk(K, inter)
    topk_ids, topk_w = _routing(M, E, top_k, dtype)

    # Reference: Triton moe_wna16 (int4 w4a16, has_zp) on the raw layout.
    qc = int4_w4a16_moe_quant_config(
        w1_scale=w13[1],
        w2_scale=w2[1],
        w1_zp=w13[2],
        w2_zp=w2[2],
        block_shape=[0, G],
    )
    ref = fused_experts(
        x,
        w13[0],
        w2[0],
        topk_w.float(),
        topk_ids,
        activation=MoEActivation.SILU,
        quant_config=qc,
    )

    # Our path: convert + the RDNA2 fused MoE.
    c13 = _to_kernel_layout(w13[0], w13[1], w13[2], has_zp=True)
    c2 = _to_kernel_layout(w2[0], w2[1], w2[2], has_zp=True)
    layer = SimpleNamespace(
        w13_qweight=c13[0],
        w13_scales=c13[1],
        w13_qzeros=c13[2],
        w2_qweight=c2[0],
        w2_scales=c2[1],
        w2_qzeros=c2[2],
    )
    out = _rdna2_fused_moe(
        x,
        topk_w,
        topk_ids,
        layer,
        MoEActivation.SILU,
        False,
        E,
        None,
        use_v2_format=True,
    )
    _assert_moe_close(out, ref)


# ---------------------------------------------------------------------------
# Combine order: both GEMMs sum their partials in a fixed order
# ---------------------------------------------------------------------------


@gfx1030_only
def test_rdna2_moe_w13_is_deterministic():
    """The w13 GEMM must return the same bytes for the same inputs.

    grid.z splits K, so an output row is a sum of partials combined through
    memory. With ``output_topk == 0`` a token-slot maps 1:1 to an output row, so
    each split can own a scratch slice and the slices are summed in split order.
    Accumulating them with atomics instead left the answer dependent on which
    split happened to land first, which greedy decoding turns into a different
    reply to the same prompt.

    w2 is covered by ``test_rdna2_moe_w2_is_deterministic``; both GEMMs now index
    the scratch by token-slot, so neither depends on arrival order.
    """
    set_random_seed(0)
    device = "cuda"
    E, K, N, group = 4, 1024, 512, 128
    num_tokens, top_k, block_m = 8, 2, 4
    num_valid = num_tokens * top_k

    inp = torch.randn(num_tokens, K, dtype=torch.float16, device=device) * 0.1
    qw = torch.randint(0, 2**31, (E, K // 8, N), dtype=torch.int32, device=device)
    sc = torch.rand(E, K // group, N, dtype=torch.float16, device=device) * 0.02 + 0.01
    qz = torch.zeros((E, K // group, N // 8), dtype=torch.int32, device=device)
    tw = torch.rand(num_valid, dtype=torch.float32, device=device)
    sorted_ids = torch.arange(num_valid, dtype=torch.int32, device=device)
    expert_ids = (
        torch.arange(num_valid // block_m, dtype=torch.int32, device=device) % E
    )
    ntpp = torch.tensor([num_valid], dtype=torch.int32, device=device)

    first = None
    for _ in range(8):
        out = torch.zeros(num_valid, N, dtype=torch.float16, device=device)
        ops.moe_gptq_gemm_rdna2(
            inp, out, qw, sc, qz, tw, sorted_ids, expert_ids, ntpp,
            top_k, block_m, False, True, 0,
        )
        if first is None:
            first = out.clone()
        else:
            assert torch.equal(first, out), (
                "w13 returned different bytes for identical inputs"
            )


@gfx1030_only
def test_rdna2_moe_w2_is_deterministic():
    """The w2 GEMM must return the same bytes for the same inputs.

    w2 sets ``output_topk = top_k`` so the fused moe_sum folds top_k slots into
    one output row, and those slots are spread across grid.y blocks; combined
    with the grid.z K splits, a row had several writers. Accumulating them
    through the fp32 CAS made the sum depend on arrival order, measured at ~0.2%
    of the output RMS between identical launches -- around 3.5 fp16 ULP, and
    enough to reorder the sparse indexer's top-k. End to end it made greedy
    decoding return four different answers to four identical requests.
    """
    torch.manual_seed(0)
    device = "cuda"
    # DeepSeek-V4-Flash w2 under TP=4: K = moe_intermediate 2048/4, N = hidden.
    E, K, N, group = 8, 512, 4096, 32
    top_k, block_m = 6, 4

    # 512 is the deployed prefill chunk (--max-num-batched-tokens): its scratch
    # is 96 MiB, so it also pins the cap that decides between the reproducible
    # combine and the CAS fallback.
    for num_tokens in (8, 64, 512):
        num_valid = num_tokens * top_k
        inp = torch.randn(num_valid, K, dtype=torch.float16, device=device) * 0.1
        qw = torch.randint(0, 2**31, (E, K // 8, N), dtype=torch.int32, device=device)
        sc = (
            torch.rand(E, K // group, N, dtype=torch.float16, device=device) * 0.02
            + 0.01
        )
        qz = torch.zeros((E, K // group, N // 8), dtype=torch.int32, device=device)
        tw = torch.rand(num_valid, dtype=torch.float32, device=device)
        sorted_ids = torch.arange(num_valid, dtype=torch.int32, device=device)
        expert_ids = (
            torch.arange(num_valid // block_m, dtype=torch.int32, device=device) % E
        )
        ntpp = torch.tensor([num_valid], dtype=torch.int32, device=device)

        first = None
        for _ in range(8):
            out = torch.zeros(num_tokens, N, dtype=torch.float16, device=device)
            ops.moe_gptq_gemm_rdna2(
                inp, out, qw, sc, qz, tw, sorted_ids, expert_ids, ntpp,
                top_k, block_m, True, True, top_k,
            )
            if first is None:
                first = out.clone()
            else:
                assert torch.equal(first, out), (
                    f"w2 returned different bytes for identical inputs "
                    f"(num_tokens={num_tokens})"
                )


# ---------------------------------------------------------------------------
# SwiGLU clamp
# ---------------------------------------------------------------------------
#
# SwiGLU is unbounded. DeepSeek-V4 ships swiglu_limit=10.0, which clamps the
# gate above and the up projection on both sides before the product, and
# RoutedExperts carries that value for the quant method to use. The RDNA2 driver
# used to call apply_moe_activation() without it, which selects the unclamped
# silu_and_mul: a token whose w13 output ran past the limit then grew instead of
# saturating. Rare enough to look sporadic, and large enough to destroy that
# token's logits -- measured at 39x the neighbouring row's activation by layer
# 40 of DeepSeek-V4, which is what put CJK tokens in the middle of generated
# JavaScript.


def _ref_moe_clamped(x, w13_dq, w2_dq, topk_ids, topk_w, limit):
    """Reference MoE with the clamped SwiGLU the checkpoint asks for."""
    M, K = x.shape
    inter = w2_dq.shape[1]
    xf = x.float()
    out = torch.zeros(M, K, device=x.device, dtype=torch.float32)
    for m in range(M):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[m, k])
            gate_up = xf[m] @ w13_dq[e]
            gate = torch.clamp(gate_up[:inter], max=limit)
            up = torch.clamp(gate_up[inter:], min=-limit, max=limit)
            out[m] += float(topk_w[m, k]) * ((F.silu(gate) * up) @ w2_dq[e])
    return out.to(x.dtype)


@gfx1030_only
def test_rdna2_moe_applies_swiglu_limit(dist_init):
    """The driver must honour the layer's swiglu_limit.

    The input is scaled so the w13 output runs well past the limit, which is the
    only regime where the two differ: with small activations a missing clamp is
    invisible, which is why this went unnoticed. Asserting against the clamped
    reference pins the behaviour, and the unclamped comparison below confirms
    the test can actually fail.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.moe_wna16_rdna2 import (
        _rdna2_fused_moe,
    )

    set_random_seed(0)
    dtype = torch.float16
    E, M, K, inter, top_k, G, limit = 4, 8, 256, 256, 2, 128, 10.0

    # Large inputs so gate/up land outside [-limit, limit].
    x = (6.0 * torch.randn(M, K, device=device)).to(dtype)
    w13_dq, w13_q, w13_s, w13_z, v2 = _quantize_expert_weights(
        E, K, 2 * inter, G, False, dtype
    )
    w2_dq, w2_q, w2_s, w2_z, _ = _quantize_expert_weights(E, inter, K, G, False, dtype)
    topk_ids, topk_w = _routing(M, E, top_k, dtype)

    layer = SimpleNamespace(
        w13_qweight=w13_q, w13_scales=w13_s, w13_qzeros=w13_z,
        w2_qweight=w2_q, w2_scales=w2_s, w2_qzeros=w2_z,
        swiglu_limit=limit,
    )

    def run(lyr):
        return _rdna2_fused_moe(
            x, topk_w, topk_ids, lyr, MoEActivation.SILU,
            apply_router_weight_on_input=False,
            global_num_experts=E, expert_map=None, use_v2_format=v2,
        )

    clamped_ref = _ref_moe_clamped(x, w13_dq, w2_dq, topk_ids, topk_w, limit)
    got = run(layer)
    _assert_moe_close(got, clamped_ref)

    # Without the limit the driver takes the unclamped kernel, which on these
    # inputs is a different answer -- so the assertion above is load-bearing.
    unclamped = run(SimpleNamespace(**{**vars(layer), "swiglu_limit": None}))
    assert not torch.allclose(
        unclamped.float(), clamped_ref.float(), atol=1e-2, rtol=1e-2
    ), "inputs did not exceed the clamp; the test would pass either way"


@gfx1030_only
def test_rdna2_moe_multi_pass_n_tiling_matches_a_column_split():
    """Wide batches split N into passes; the seams must not move the answer.

    The reproducible combine needs one scratch row per (K-split, token-slot),
    which grows with the batch. Rather than fall back to the order-dependent
    CAS above a size, the launcher processes N in passes that fit its budget.
    Each pass offsets the weight, zero-point and output pointers while size_n
    stays the row stride, so a wrong offset would corrupt one seam and nothing
    else. Running the same problem as two independent single-pass halves pins
    that arithmetic, and the results must agree bit for bit: both sum the same
    values in the same order.
    """
    torch.manual_seed(0)
    device = "cuda"
    # 2048 tokens x top_k 6 needs 4 passes for w13 at these dimensions.
    E, K, N, group = 8, 4096, 1024, 32
    num_tokens, top_k, block_m = 2048, 6, 4
    num_valid = num_tokens * top_k

    inp = torch.randn(num_valid, K, dtype=torch.float16, device=device) * 0.1
    qw = torch.randint(0, 2**31, (E, K // 8, N), dtype=torch.int32, device=device)
    sc = torch.rand(E, K // group, N, dtype=torch.float16, device=device) * 0.02 + 0.01
    qz = torch.zeros((E, K // group, N // 8), dtype=torch.int32, device=device)
    tw = torch.rand(num_valid, dtype=torch.float32, device=device)
    sorted_ids = torch.arange(num_valid, dtype=torch.int32, device=device)
    expert_ids = (
        torch.arange(num_valid // block_m, dtype=torch.int32, device=device) % E
    )
    ntpp = torch.tensor([num_valid], dtype=torch.int32, device=device)

    def run(w, s, z, n):
        out = torch.zeros(num_valid, n, dtype=torch.float16, device=device)
        ops.moe_gptq_gemm_rdna2(
            inp, out, w, s, z, tw, sorted_ids, expert_ids, ntpp,
            top_k, block_m, False, True, 0,
        )
        return out

    full = run(qw, sc, qz, N)
    half = N // 2
    left = run(
        qw[:, :, :half].contiguous(), sc[:, :, :half].contiguous(),
        qz[:, :, : half // 8].contiguous(), half,
    )
    right = run(
        qw[:, :, half:].contiguous(), sc[:, :, half:].contiguous(),
        qz[:, :, half // 8 :].contiguous(), half,
    )
    assert torch.equal(full, torch.cat([left, right], dim=1)), (
        "multi-pass N tiling disagrees with an equivalent column split"
    )
