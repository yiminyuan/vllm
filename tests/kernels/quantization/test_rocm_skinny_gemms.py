# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950, on_gfx1030
from vllm.utils.platform_utils import num_compute_units

# Global per-test cleanup costs more than the tests themselves.
# These tests can just cleanup once at the end
pytestmark = pytest.mark.skip_global_cleanup

SEEDS = [0]

# These options are independent in the kernels, so the sets below cover every
# pair of them instead of the full product. bias_mode: 0 = none, 1 = (m,),
# 2 = (n, m).
OPTIONS_WVSPLITKRC = [
    # dtype, padded_a, bias_mode, xnorm
    (torch.float16, False, 0, False),
    (torch.float16, True, 1, True),
    (torch.float16, True, 2, False),
    (torch.bfloat16, True, 0, True),
    (torch.bfloat16, False, 1, False),
    (torch.bfloat16, False, 2, True),
]

OPTIONS_WVSPLITK = [
    # dtype, padded_a, padded_b, bias_mode, xnorm
    (torch.float16, False, False, 0, False),
    (torch.float16, False, True, 1, True),
    (torch.float16, True, False, 2, True),
    (torch.bfloat16, True, True, 0, True),
    (torch.bfloat16, True, False, 1, False),
    (torch.bfloat16, False, True, 2, False),
]

OPTIONS_WVSPLITK_FP8 = [
    # dtype, padded_a, padded_b, biased, xnorm
    (torch.float16, False, False, True, True),
    (torch.bfloat16, True, False, False, True),
    (torch.bfloat16, False, True, True, True),
    (torch.float16, True, False, True, False),
    (torch.float16, False, True, False, False),
    (torch.bfloat16, True, True, False, False),
]

DTYPES = [torch.bfloat16, torch.float16]

# Specific (N, K, M) combinations for targeted testing
NKM_FACTORS_LLMM1 = [
    # Small, medium, large cases
    (1, 8, 16),
    (1, 32, 64),
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    # Edge cases with specific K sizes
    (1, 6144, 1024),
    (1, 8192, 2048),
    # Very large case
    (1, 4096, 8192),
]

NKM_FACTORS_WVSPLITK = [
    # Different batch sizes with key dimensions
    (1, 32, 16),
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (4, 4096, 4096 + 1),
    (4, 4096 + 16, 4096),
    (4, 4096 + 16, 4096 + 1),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    (4, 16384, 8192),
    (4, 16384 * 2, 8192),
    (4, 16384 * 2, 8192 + 1),
    (4, 16384 * 2 + 16, 8192),
    (4, 16384 * 2 + 16, 8192 + 1),
    # Minimum M constraint validation (m >= 8)
    (1, 64, 8),
    (2, 128, 8),
    (4, 256, 8),
]

# N is bucketed up to its next ^2 (16/32/64/128) and the remainder is masked
N_FACTORS_WVSPLITKRC = [
    13,
    16,
    17,
    25,
    29,
    31,
    32,
    41,
    51,
    64,
    71,
    81,
    91,
    103,
    117,
    128,
]
# K shards are 512 wide, evenly divided or not, +8 for a partial 8-element load
K_FACTORS_WVSPLITKRC = [2880, 2880 + 8, 3072, 3072 + 8]
# M tiles are 64 rows, +16 for a partial tile
M_FACTORS_WVSPLITKRC = [128, 128 + 16, 256, 256 + 16, 640, 640 + 16]

NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 32, 16 + 16),
    (1, 64, 64),
    (1, 64, 64 + 16),
    (1, 64 + 16, 64),
    (1, 64 + 16, 64 + 16),
    (4, 64, 64),
    (4, 64, 64 + 16),
    (4, 64 + 16, 64),
    (4, 64 + 16, 64 + 16),
    (2, 512, 512),
    (3, 512, 512),
    (3, 512, 512 + 16),
    (4, 512, 512),
    (3, 2048, 2048),
    (3, 2048, 2048 + 16),
    (4, 2048 + 16, 2048),
    (4, 2048 + 16, 2048 + 16),
    (4, 4096, 4096),
    (4, 16400, 2048),
    (4, 16400, 2048 + 16),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
    (4, 32768 * 2, 28672),
    (4, 32768 * 2, 28672 + 16),
    (4, 32768 * 2 + 16, 28672),
    (4, 32768 * 2 + 16, 28672 + 16),
]


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_all_tests():
    yield
    cleanup_dist_env_and_memory()


def pad_fp8(weight):
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


def make_bias(bias_mode, n, m, dtype):
    if bias_mode == 0:
        return None
    shape = (m,) if bias_mode == 1 else (n, m)
    return torch.rand(shape, dtype=dtype, device="cuda") * 2 - 1


@pytest.mark.parametrize("n", N_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("k", K_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("m", M_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("dtype,padded_a,bias_mode,xnorm", OPTIONS_WVSPLITKRC)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_kernel(n, k, m, dtype, padded_a, bias_mode, xnorm, seed):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # Deterministic reduction stores one float workspace value per K shard.
    fits_wvsplitkrc = (N_p2 * m * ((k + 512 - 1) // 512)) <= 128 * 1024 * 12
    fits_wvsplitkrc &= CuNeeded <= cu_count

    if not fits_wvsplitkrc:
        pytest.skip("Too large for wvSplitKrc")

    xavier = (
        math.sqrt(2 / k) if xnorm else 1
    )  # normalize to avoid large output-bias deltas
    A = torch.randn(n, k, dtype=dtype, device="cuda") * xavier
    B = torch.randn(m, k, dtype=dtype, device="cuda") * xavier
    if padded_a:
        A = pad_fp8(A)

    BIAS = make_bias(bias_mode, n, m, dtype)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitKrc(A, B, cu_count, BIAS)

    if xnorm:
        # The O(1) bias lifts outputs to ~O(1), where one bf16 ULP (~3.9e-3, the
        # worst measured divergence) exceeds 1e-3. Bump atol to 5e-3 only for
        # biased bf16 (above that ULP, still under finfo(bf16).eps); keep 1e-3
        # otherwise.
        atol = 5e-3 if (dtype == torch.bfloat16 and BIAS is not None) else 1e-3
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-8)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm1_kernel(n, k, m, dtype, rows_per_block, seed):
    torch.manual_seed(seed)
    # TODO: Zero-centering the inputs causes errors for LLMM1!
    #      Without that the numbers quickly saturate, and may
    #      be giving false matches.
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.LLMM1(B, A, rows_per_block)

    torch.testing.assert_close(out, ref_out, atol=1e-8, rtol=1e-2)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize(
    "dtype,padded_a,padded_b,bias_mode,xnorm",
    OPTIONS_WVSPLITK,
)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
def test_rocm_wvsplitk_kernel(
    n, k, m, dtype, padded_a, padded_b, bias_mode, xnorm, seed
):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = (
        math.sqrt(2 / k) if xnorm else 1
    )  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    BIAS = make_bias(bias_mode, n, m, dtype)

    if padded_a:
        A = pad_fp8(A)
    if padded_b:
        B = pad_fp8(B)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    # Accumulation error in fp16 GEMM scales with sqrt(K)
    atol = torch.finfo(dtype).eps * math.sqrt(k)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)


# v0.28.0 folded bias into the combined OPTIONS_* tuples to cut CI cost.
# gfx1030 is the platform under active development here, so its cases keep
# the full product; they have caught real kernel bugs.
BIAS_MODES_RDNA2 = [0, 1, 2]


# (n, k, m) for the RDNA2 fp16 skinny GEMV: n <=
# GFX1030_SKINNY_GEMV_MAX_TOKENS, k%8==0. Covers the large-vocab lm_head (the
# motivating case), both dispatch paths -- A staged in LDS when it fits
# (n*k*2 <= 64 KB) and A streamed from global otherwise -- and edge cases
# (k not a multiple of the 256-elem step, odd m -> grid-stride remainder).
NKM_FACTORS_WVSPLITK_RDNA2 = [
    (1, 5120, 8192),
    (1, 5120, 124160),  # lm_head per die (vocab 248320, TP-2)
    (1, 4096, 17408),
    (2, 4096, 8192),
    (3, 2048, 4096),
    (4, 4096, 4096),
    (5, 2048, 8192),
    (1, 4104, 1024),  # k % 256 != 0
    (1, 4096, 4097),  # m not a multiple of the wave tile
    # Large K: A does not fit LDS -> streamed-from-global path.
    (1, 40960, 8192),
    (3, 20480, 4096),
    (4, 16384, 8192),
    (5, 16384, 4096),
    # Batched decode (n > 5). The LDS staging limit is n*k*2 <= 64 KB, so at
    # k=4096 n=8 is the last staged case and n=9 is the first streamed one --
    # both sides of that boundary are covered here.
    (6, 4096, 8192),
    (8, 4096, 8192),  # n*k*2 == 65536, exactly the LDS budget
    (9, 4096, 8192),  # one row past it -> streamed from global
    (12, 2048, 4096),
    (16, 4096, 8192),  # the largest instantiated n
    (16, 2048, 4097),  # largest n against the grid-stride remainder
    (16, 4104, 1024),  # largest n with k % 256 != 0
    (16, 16384, 4096),  # largest n, streamed from global
]


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_RDNA2)
@pytest.mark.parametrize("bias_mode", BIAS_MODES_RDNA2)
@pytest.mark.parametrize("padded_b", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_kernel(n, k, m, bias_mode, padded_b, dtype, seed):
    torch.manual_seed(seed)
    # fp16 and bf16 both accepted; bf16 is upconverted to fp16 in the kernel.

    xavier = math.sqrt(2 / k)  # normalize to keep fp16 accumulation well-scaled
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    if padded_b:
        B = pad_fp8(B)  # non-contiguous row stride

    BIAS = make_bias(bias_mode, n, m, dtype)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK_rdna2(B, A.view(-1, A.size(-1)), BIAS)

    atol = torch.finfo(dtype).eps * math.sqrt(k)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)


# The DeepSeek-V4 compressor/indexer kv-score shapes, which want an fp32 result
# because the compressor state cache is fp32.
NKM_FACTORS_WVSPLITK_RDNA2_F32 = [
    (1, 4096, 512),
    (1, 4096, 1024),
    (1, 4096, 2048),
    (2, 4096, 1024),
    (5, 2048, 4096),
    (1, 16384, 1024),  # A streamed from global rather than staged in LDS
    # Batched decode over the same kv-score shapes.
    (8, 4096, 2048),  # exactly the LDS budget
    (9, 4096, 2048),  # one row past it -> streamed from global
    (16, 4096, 512),
    (16, 4096, 2048),
]


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_RDNA2_F32)
@pytest.mark.parametrize("bias_mode", BIAS_MODES_RDNA2)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_out_dtype_f32(n, k, m, bias_mode, dtype, seed):
    """An fp32 result must come back as fp32, and be no worse than narrowing.

    The accumulator is fp32 either way, so asking for fp32 only skips the
    rounding on store and cannot move the result further from the reference.
    """
    torch.manual_seed(seed)
    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    BIAS = make_bias(bias_mode, n, m, dtype)

    a2 = A.view(-1, A.size(-1))
    out_f32 = ops.wvSplitK_rdna2(B, a2, BIAS, torch.float32)
    out_native = ops.wvSplitK_rdna2(B, a2, BIAS)

    assert out_f32.dtype == torch.float32
    assert out_native.dtype == dtype
    assert out_f32.shape == out_native.shape

    ref = torch.nn.functional.linear(A.float(), B.float(), None)
    if BIAS is not None:
        ref = ref + BIAS.float()

    atol = torch.finfo(dtype).eps * math.sqrt(k)
    torch.testing.assert_close(out_f32, ref, atol=atol, rtol=1e-2)
    assert (out_f32 - ref).abs().max() <= (out_native.float() - ref).abs().max()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_token_bound_matches_gate():
    """The advertised token bound must be exactly what the kernel accepts.

    Callers gate on GFX1030_SKINNY_GEMV_MAX_TOKENS and fall back to the vendor
    GEMM above it, so a Python constant that drifts from MAX_N in the .cu either
    hard-errors in production or silently leaves tokens on the vendor path.
    """
    from vllm.platforms.rocm import GFX1030_SKINNY_GEMV_MAX_TOKENS as max_n

    k, m = 2048, 1024
    B = torch.rand(m, k, dtype=torch.float16, device="cuda") * 2 - 1

    at_limit = torch.rand(max_n, k, dtype=torch.float16, device="cuda") * 2 - 1
    out = ops.wvSplitK_rdna2(B, at_limit, None, torch.float32)
    assert out.shape == (max_n, m)

    over_limit = torch.rand(max_n + 1, k, dtype=torch.float16, device="cuda") * 2 - 1
    with pytest.raises(RuntimeError, match=r"supports N in"):
        ops.wvSplitK_rdna2(B, over_limit, None, torch.float32)


# (g, n, k, m) for the grouped form. (8, 1, 512, 1024) is the DeepSeek-V4
# o_proj shape at batch 1, which is what motivated it.
GNKM_FACTORS_WVSPLITK_RDNA2_GROUPED = [
    (8, 1, 512, 1024),
    (2, 1, 4096, 256),
    (4, 3, 2048, 512),
    (8, 5, 512, 1024),
    (3, 2, 1024, 257),  # m not a multiple of the wave tile
    (16, 1, 4104, 128),  # k % 256 != 0
    (1, 1, 512, 1024),  # a single group must behave like the ungrouped op
]


@pytest.mark.parametrize("g,n,k,m", GNKM_FACTORS_WVSPLITK_RDNA2_GROUPED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_grouped_matches_per_group(g, n, k, m, dtype, seed):
    """One grouped launch must equal the per-group calls it replaces.

    Same device kernel, same arithmetic, only a second grid dimension, so this
    is exact rather than approximate -- a tolerance here would hide a group
    indexed with the wrong stride.
    """
    torch.manual_seed(seed)
    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, g, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(g, m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    grouped = ops.wvSplitK_rdna2_grouped(B, A)
    assert grouped.shape == (n, g, m)

    per_group = torch.empty_like(grouped)
    for i in range(g):
        ops.wvSplitK_rdna2(B[i], A[:, i, :], None, None, per_group[:, i, :])
    torch.testing.assert_close(grouped, per_group, atol=0, rtol=0)


@pytest.mark.parametrize("g,n,k,m", GNKM_FACTORS_WVSPLITK_RDNA2_GROUPED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_grouped_matches_einsum(g, n, k, m, dtype):
    """And it must match the einsum it stands in for in the model."""
    torch.manual_seed(0)
    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, g, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(g, m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    out = ops.wvSplitK_rdna2_grouped(B, A)
    ref = torch.einsum("tgd,grd->tgr", A, B)
    atol = torch.finfo(dtype).eps * math.sqrt(k)
    torch.testing.assert_close(out, ref, atol=atol, rtol=1e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_grouped_writes_caller_out():
    g, n, k, m = 8, 1, 512, 1024
    torch.manual_seed(0)
    A = torch.rand(n, g, k, dtype=torch.float16, device="cuda") * 2 - 1
    B = torch.rand(g, m, k, dtype=torch.float16, device="cuda") * 2 - 1
    dst = torch.empty(n, g, m, dtype=torch.float16, device="cuda")
    got = ops.wvSplitK_rdna2_grouped(B, A, dst)
    assert got.data_ptr() == dst.data_ptr()
    torch.testing.assert_close(dst, ops.wvSplitK_rdna2_grouped(B, A), atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_grouped_rejects_bad_shapes():
    g, n, k, m = 4, 1, 512, 256
    A = torch.rand(n, g, k, dtype=torch.float16, device="cuda")
    B = torch.rand(g, m, k, dtype=torch.float16, device="cuda")

    with pytest.raises(RuntimeError, match="in_a \\[G, M, K\\]"):
        ops.wvSplitK_rdna2_grouped(B[0], A)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ops.wvSplitK_rdna2_grouped(B, A[:, : g - 1, :])
    # K must stay the same size here, or the shape check fires first and this
    # would pass for the wrong reason.
    strided = torch.rand(n, g, 2 * k, dtype=torch.float16, device="cuda")[:, :, ::2]
    assert strided.shape == A.shape and strided.stride(2) != 1
    with pytest.raises(RuntimeError, match="K-contiguous"):
        ops.wvSplitK_rdna2_grouped(B, strided)
    with pytest.raises(RuntimeError, match="same dtype"):
        ops.wvSplitK_rdna2_grouped(B, A.to(torch.bfloat16))
    # the token bound is shared with the ungrouped op
    from vllm.platforms.rocm import GFX1030_SKINNY_GEMV_MAX_TOKENS as max_n

    too_many = torch.rand(max_n + 1, g, k, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="supports N in"):
        ops.wvSplitK_rdna2_grouped(B, too_many)


@pytest.mark.parametrize("n,k,m", [(1, 4096, 1024), (2, 4096, 512), (5, 2048, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rocm_wvsplitk_rdna2_writes_strided_out(n, k, m, dtype):
    """Writing into a caller tensor must match, including a strided slice.

    The strided case is the reason the parameter exists: a per-group projection
    fills one slice of a larger result, which otherwise costs a copy per group.
    """
    torch.manual_seed(0)
    xavier = math.sqrt(2 / k)
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    ref = ops.wvSplitK_rdna2(B, A, None)

    packed = torch.empty_like(ref)
    got = ops.wvSplitK_rdna2(B, A, None, None, packed)
    assert got.data_ptr() == packed.data_ptr()
    torch.testing.assert_close(packed, ref, atol=0, rtol=0)

    # A slice of a wider buffer: rows are packed, the row stride is not m.
    groups = 3
    wide = torch.zeros((n, groups, m), dtype=dtype, device="cuda")
    for g in range(groups):
        ops.wvSplitK_rdna2(B, A, None, None, wide[:, g, :])
    for g in range(groups):
        torch.testing.assert_close(wide[:, g, :], ref, atol=0, rtol=0)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize(
    "dtype,padded_a,padded_b,biased,xnorm",
    OPTIONS_WVSPLITK_FP8,
)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_kernel(
    n, k, m, dtype, padded_a, padded_b, biased, xnorm, seed
):
    torch.manual_seed(seed)

    xavier = math.sqrt(2 / k) if xnorm else 1  # normalize to avoid large deltas
    A = (torch.rand(n, k, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, device="cuda") * 2 - 1) * xavier

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if padded_b:
        B = pad_fp8(B)
    if padded_a:
        A = pad_fp8(A)

    BIAS = None if (not biased) else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b, bias=BIAS
    )
    out = ops.wvSplitKQ(B, A, dtype, scale_a, scale_b, num_compute_units(), BIAS)

    if xnorm:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-8)
    elif k >= 32 * 1024:
        # wider pytrch thresh for large-K & no xnorm
        torch.testing.assert_close(out, ref_out, atol=0.07, rtol=5e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


# Row-stride padding for the RDNA2 skinny GEMV. A contiguous [out, in] weight
# has a power-of-two row stride for every shape this model uses, which puts the
# eight rows a workgroup reads at once onto the same memory channels.
GFX1030_PAD_SHAPES = [
    (2048, 4096),  # wo_a, per group
    (4096, 2048),  # wo_b
    (1024, 4096),  # shared gate_up
    (8192, 1024),  # wq_b
    (4096, 512),  # shared down
    (1536, 4096),  # fused_wqa_wkv, replicated
]


def _linear_with_weight(w):
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
    return layer


@pytest.mark.parametrize("out_size,in_size", GFX1030_PAD_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rdna2_gemv_row_pad_is_bit_identical(out_size, in_size, dtype):
    """Padding changes only the distance between rows, never a result bit."""
    from vllm.model_executor.layers.utils import (
        GFX1030_GEMV_ROW_PAD,
        maybe_pad_rdna2_gemv_weight,
    )

    torch.manual_seed(0)
    w = (torch.rand(out_size, in_size, dtype=dtype, device="cuda") * 2 - 1) * 0.05
    a = (torch.rand(1, in_size, dtype=dtype, device="cuda") * 2 - 1) * 0.05
    before = ops.wvSplitK_rdna2(w, a, None)

    layer = _linear_with_weight(w.clone())
    maybe_pad_rdna2_gemv_weight(layer)

    padded = layer.weight
    assert padded.shape == (out_size, in_size)
    assert padded.stride(0) == in_size + GFX1030_GEMV_ROW_PAD
    assert padded.stride(1) == 1
    assert not padded.is_contiguous()
    # the visible values must be untouched
    torch.testing.assert_close(padded, w, atol=0, rtol=0)

    after = ops.wvSplitK_rdna2(padded, a, None)
    assert torch.equal(after, before), "row padding changed the GEMV result"


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rdna2_gemv_row_pad_keeps_row_alignment():
    """The pad must keep each row 16-byte aligned for the float4 loads.

    load8_as_half2 reads a row in 16-byte chunks, so the row stride in bytes
    has to stay a multiple of 16 or the vector loads become misaligned.
    """
    from vllm.model_executor.layers.utils import GFX1030_GEMV_ROW_PAD

    assert GFX1030_GEMV_ROW_PAD % 8 == 0
    for dtype in (torch.float16, torch.bfloat16):
        itemsize = torch.empty(0, dtype=dtype).element_size()
        for in_size in (512, 1024, 2048, 4096):
            assert ((in_size + GFX1030_GEMV_ROW_PAD) * itemsize) % 16 == 0


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rdna2_gemv_row_pad_declines_what_it_should():
    """Only contiguous, power-of-two-strided, kernel-eligible weights get padded."""
    from vllm.model_executor.layers.utils import maybe_pad_rdna2_gemv_weight

    def stride0_after(w):
        layer = _linear_with_weight(w)
        maybe_pad_rdna2_gemv_weight(layer)
        return layer.weight.stride(0), layer.weight.shape

    # not a power-of-two row stride: nothing to break up
    w = torch.rand(256, 4104, dtype=torch.float16, device="cuda")
    assert stride0_after(w)[0] == 4104

    # k % 8 != 0: the kernel declines these anyway
    w = torch.rand(256, 4100, dtype=torch.float16, device="cuda")
    assert stride0_after(w)[0] == 4100

    # unsupported dtype for this kernel
    w = torch.rand(256, 4096, dtype=torch.float32, device="cuda")
    assert stride0_after(w)[0] == 4096

    # 3-D (MoE expert stacks) must be left alone
    w = torch.rand(4, 256, 4096, dtype=torch.float16, device="cuda")
    layer = _linear_with_weight(w)
    maybe_pad_rdna2_gemv_weight(layer)
    assert layer.weight.shape == (4, 256, 4096)
    assert layer.weight.is_contiguous()

    # idempotent: an already-padded weight is not padded again
    w = torch.rand(256, 4096, dtype=torch.float16, device="cuda")
    layer = _linear_with_weight(w)
    maybe_pad_rdna2_gemv_weight(layer)
    first = layer.weight.stride(0)
    maybe_pad_rdna2_gemv_weight(layer)
    assert layer.weight.stride(0) == first


@pytest.mark.parametrize("tokens", [1, 5, 16, 64])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx1030(), reason="RDNA2 (gfx1030) skinny GEMV")
def test_rdna2_gemv_row_pad_through_the_dispatch(tokens):
    """The padded weight must survive the real dispatch, GEMV path and fallback.

    Token counts above the kernel's bound fall through to F.linear, which has to
    accept the strided weight rather than silently copying or erroring.
    """
    from vllm.model_executor.layers.utils import (
        maybe_pad_rdna2_gemv_weight,
        rocm_unquantized_gemm_impl,
    )

    torch.manual_seed(0)
    out_size, in_size = 2048, 4096
    w = torch.empty(out_size, in_size, dtype=torch.float16, device="cuda")
    x = torch.empty(tokens, in_size, dtype=torch.float16, device="cuda")
    w.uniform_(-0.05, 0.05)
    x.uniform_(-0.05, 0.05)

    before = rocm_unquantized_gemm_impl(x, w, None)
    layer = _linear_with_weight(w.clone())
    maybe_pad_rdna2_gemv_weight(layer)
    after = rocm_unquantized_gemm_impl(x, layer.weight, None)

    assert torch.equal(after, before), "padding changed the dispatched result"
