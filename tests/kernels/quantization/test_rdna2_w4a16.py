#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA2 W4A16 GPTQ kernel (gfx1030).

fp16-only port of the RDNA3 kernel (PR #41394). RDNA2 has v_dot2_f32_f16
(fdot2) and v_pk_fma_f16 but no bf16 packed dot and no WMMA, so this kernel
carries only the fp16 scalar path (the RDNA3 fp16 win is the bit-trick dequant,
not WMMA -- see q_gemm_rdna3.cu: "fp16 path always stays scalar"). Symmetric
GPTQ (uint4b8) only, matching the RDNA3 kernel's supported set.

The kernel is exposed via ``torch.ops._rocm_C.gptq_gemm_rdna2`` and is only
built for gfx1030; tests are skipped elsewhere.

Run `pytest tests/kernels/quantization/test_rdna2_w4a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna2_w4a16 import (  # noqa: E402
    RDNA2W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms.rocm import on_gfx1030  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

device = "cuda"

WEIGHT_TYPE = scalar_types.uint4b8  # symmetric int4, bias = 8
PACK_FACTOR = 8  # 8 x 4-bit nibbles per int32

# Skip everything in this module unless we are on the only architecture the
# kernel is built/registered for.
gfx1030_only = pytest.mark.skipif(
    not (
        on_gfx1030()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
    ),
    reason="requires gfx1030 with the _rocm_C.gptq_gemm_rdna2 op built in",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference for the RDNA2 W4A16 op.

    x_mk:       [M, K] fp16 activations.
    q_int4_kn:  [K, N] int32 raw stored nibbles in [0, 15].
    scales_gn:  [K//G, N] per-group scales (act dtype).
    zeros_gn:   [K//G, N] int32 raw stored zero points in [0, 15], or None
                for the symmetric path (kernel synthesizes stored zero = 7).
    group_size: G.

    The kernel applies the GPTQv1 "+1" zero-point quirk, so the effective
    zero is ``stored_zero + 1`` (symmetric path: 7 + 1 == bias == 8).
    """
    K, N = q_int4_kn.shape
    s_full = scales_gn.repeat_interleave(group_size, dim=0).to(torch.float32)  # [K,N]
    if zeros_gn is None:
        z_full = torch.full(
            (K, N), float(WEIGHT_TYPE.bias), device=x_mk.device, dtype=torch.float32
        )
    else:
        z_full = (zeros_gn + 1).repeat_interleave(group_size, dim=0).to(torch.float32)
    w_fp = (q_int4_kn.to(torch.float32) - z_full) * s_full  # [K, N]
    out = x_mk.to(torch.float32) @ w_fp  # [M, N]
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x_mk.dtype)


# ---------------------------------------------------------------------------
# Layer construction (GPTQ checkpoint format)
# ---------------------------------------------------------------------------


def _build_layer(
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Build a dummy layer carrying GPTQ-format params, as the loader would."""
    no_loader = lambda *args, **kwargs: None  # noqa: E731

    # qweight: int4 packed along K into int32 -> [K//8, N].
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE, packed_dim=0)

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=qweight,
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=PACK_FACTOR,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales_gn.to(dtype),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
        ),
    )
    if zeros_gn is not None:
        # qzeros: int4 packed along N into int32 -> [K//G, N//8].
        qzeros = pack_quantized_values_into_int32(zeros_gn, WEIGHT_TYPE, packed_dim=1)
        layer.register_parameter(
            "qzeros",
            PackedvLLMParameter(
                data=qzeros,
                weight_loader=no_loader,
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=PACK_FACTOR,
            ),
        )
    return layer


def _run_kernel(
    x_mk: torch.Tensor,
    q_int4_kn: torch.Tensor,
    scales_gn: torch.Tensor,
    zeros_gn: torch.Tensor | None,
    group_size: int,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    K, N = q_int4_kn.shape
    has_zp = zeros_gn is not None

    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=WEIGHT_TYPE,
        act_type=dtype,
        group_size=group_size,
        zero_points=has_zp,
        has_g_idx=False,
    )
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert ok, f"can_implement rejected a supported config: {reason}"

    layer = _build_layer(q_int4_kn, scales_gn, zeros_gn, dtype)
    kernel = RDNA2W4A16LinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros" if has_zp else None,
        w_gidx_param_name=None,
    )
    kernel.process_weights_after_loading(layer)
    return kernel.apply_weights(layer, x_mk, bias=bias)


# Relative-L2 tolerance. The fp16 path uses the exllamav2 "+1024" bit-trick
# (see qdq_4_rdna2.cuh): the dequantized weight is recovered as the fp16
# difference of two ~1024*scale magnitudes, which sheds low-order mantissa bits
# and leaves ~2-3% relative noise that accumulates over K. We compare on the
# relative Frobenius norm rather than elementwise.
_REL_L2_TOL = 5e-2


def _assert_close(out: torch.Tensor, ref: torch.Tensor):
    rel_l2 = (out.to(torch.float32) - ref.to(torch.float32)).norm() / ref.to(
        torch.float32
    ).norm()
    assert rel_l2 < _REL_L2_TOL, f"relative L2 error {rel_l2:.4f} exceeds {_REL_L2_TOL}"


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


# (M, K, N, group_size). M spans the M_COUNT=1/2/4/8 scalar dispatch variants.
# fp16 stays scalar for all M on RDNA2 (no WMMA). M=64 is the critical case:
# RDNA3 fp16 switches to WMMA at M>=64, so this exercises the port's core
# divergence (WMMA dispatch removed -> scalar for all M).
MKNG_SHAPES = [
    (1, 128, 128, 128),  # single group, decode (M_COUNT=1)
    (2, 256, 256, 128),  # two groups (M_COUNT=2)
    (8, 256, 512, 64),  # M=8 scalar, smaller group
    (16, 512, 256, 128),  # M=16, scalar-tiled
    (32, 512, 512, 64),  # larger batch
    (64, 512, 512, 128),  # M>=64: RDNA3 fp16 -> WMMA; RDNA2 must stay scalar
]


@gfx1030_only
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("has_zp", [False, True], ids=["no_zp", "with_zp"])
@pytest.mark.parametrize(
    "M,K,N,G", MKNG_SHAPES, ids=[f"m{m}_k{k}_n{n}_g{g}" for m, k, n, g in MKNG_SHAPES]
)
def test_rdna2_w4a16_matches_reference(dtype, has_zp, M, K, N, G, dist_init):
    set_random_seed(0)
    assert K % G == 0 and K % PACK_FACTOR == 0 and N % PACK_FACTOR == 0

    groups = K // G
    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    zeros_gn = (
        torch.randint(0, 16, (groups, N), device=device, dtype=torch.int32)
        if has_zp
        else None
    )

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None, dtype)
    ref = _reference(x_mk, q_int4_kn, scales_gn, zeros_gn, G, None)

    assert out.shape == (M, N) and out.dtype == dtype
    _assert_close(out, ref)


@gfx1030_only
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("M", [1, 32], ids=["decode", "batch"])
def test_rdna2_w4a16_bias(dtype, M, dist_init):
    """Bias is added on both small-M and larger-M scalar paths."""
    set_random_seed(0)
    K, N, G = 512, 256, 128
    groups = K // G

    x_mk = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(dtype)
    q_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    scales_gn = (
        0.05 * torch.rand((groups, N), device=device, dtype=torch.float32) + 0.01
    ).to(dtype)
    bias = (0.1 * torch.randn(N, device=device, dtype=torch.float32)).to(dtype)

    out = _run_kernel(x_mk, q_int4_kn, scales_gn, None, G, bias, dtype)
    ref = _reference(x_mk, q_int4_kn, scales_gn, None, G, bias)

    _assert_close(out, ref)


# ---------------------------------------------------------------------------
# fp16-only constraint (RDNA2 has no bf16 packed dot / no WMMA)
# ---------------------------------------------------------------------------


@gfx1030_only
def test_rdna2_w4a16_rejects_bf16():
    """RDNA2 lacks v_dot2_f32_bf16 and WMMA, so the kernel must refuse bf16
    activations -- unlike the RDNA3 kernel, which supports both.

    The gfx1030_only marker guarantees the op is built, so can_implement gets
    past its arch/op-presence checks and reaches the act_type gate; the
    rejection is therefore specifically the fp16-only constraint.
    """
    config = MPLinearLayerConfig(
        full_weight_shape=(512, 256),
        partition_weight_shape=(512, 256),
        weight_type=WEIGHT_TYPE,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    ok, reason = RDNA2W4A16LinearKernel.can_implement(config)
    assert not ok, "bf16 must be rejected on RDNA2 (no bf16 dot)"
    assert reason is not None and "fp16" in reason
