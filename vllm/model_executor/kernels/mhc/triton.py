# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)


@triton.jit
def _cast_and_sqrsum_kernel(
    x_ptr,  # [T, K] residual flattened over (hc_mult, hidden_size)
    out_ptr,  # [T, K] fp32
    sqrsum_ptr,  # [T] fp32
    K,
    BLOCK_K: tl.constexpr,
):
    """Widen one row to fp32 and reduce its sum of squares in the same pass.

    The torch reference casts to fp32 and then reads that copy again for the
    reduction; the projection GEMM below needs the fp32 copy either way, so
    folding the reduction in removes a full pass over it.
    """
    t = tl.program_id(0)
    row = x_ptr + t * K
    out_row = out_ptr + t * K
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        v = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + offs, v, mask=mask)
        acc += tl.sum(v * v, axis=0)
    tl.store(sqrsum_ptr + t, acc)


@triton.jit
def _mhc_proj_kernel(
    x_ptr,  # [T, K] fp32
    fn_ptr,  # [N, K] fp32
    partial_ptr,  # [SPLIT_K, T, N] fp32
    K,
    num_tokens,
    n_out,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Project the token tile onto ``fn``, one K-slice per program.

    The projection is a very wide, very short GEMM: K spans the whole widened
    residual while N is a couple of dozen mixing coefficients. Splitting K
    keeps more than one workgroup busy, and writing per-slice partials keeps
    the reduction order fixed.
    """
    pid_k = tl.program_id(0)

    offs_t = tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, BLOCK_N)
    mask_t = offs_t < num_tokens
    mask_n = offs_n < n_out

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        x = tl.load(
            x_ptr + offs_t[:, None] * K + offs_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            fn_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), allow_tf32=False)

    tl.store(
        partial_ptr
        + pid_k * num_tokens * n_out
        + offs_t[:, None] * n_out
        + offs_n[None, :],
        acc,
        mask=mask_t[:, None] & mask_n[None, :],
    )


def _mhc_proj(x_f32: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """``F.linear(x_f32, fn)`` for the short-and-wide decode shape.

    The vendor GEMM picks a large macro tile and split-K for an output only a
    couple of dozen wide, which leaves it far below memory bandwidth. Batches
    long enough for that tiling to pay off stay on the vendor path.
    """
    num_tokens, K = x_f32.shape
    n_out = fn.shape[0]
    BLOCK_T = 16
    if num_tokens > BLOCK_T or n_out > 64:
        return F.linear(x_f32, fn)

    BLOCK_K = 64
    BLOCK_N = triton.next_power_of_2(n_out)
    split_k = min(64, triton.cdiv(K, BLOCK_K))

    partials = torch.empty(
        (split_k, num_tokens, n_out), dtype=torch.float32, device=x_f32.device
    )
    _mhc_proj_kernel[(split_k,)](
        x_f32,
        fn,
        partials,
        K,
        num_tokens,
        n_out,
        BLOCK_T=BLOCK_T,
        BLOCK_N=max(BLOCK_N, 16),
        BLOCK_K=BLOCK_K,
        SPLIT_K=split_k,
        num_warps=4,
    )
    return partials.sum(0)


# nout values csrc/rocm/mhc_proj_rdna2.cu instantiates; anything else takes the
# portable path. nout is 2*hc_mult + hc_mult**2, so these are hc_mult 2 and 4.
_RDNA2_MHC_PROJ_NOUT = (8, 24)


def _mhc_cast_and_proj(x_flat: Tensor, fn: Tensor) -> tuple[Tensor, Tensor]:
    """Widen the residual, project it onto ``fn``, and reduce its squares.

    The portable path materialises the whole fp32 residual because the vendor
    GEMM needs it as an operand. On RDNA2 the fused kernel widens in-register
    instead -- exactly, since fp16 and bf16 both convert to fp32 losslessly --
    which is worth more than the projection itself: the copy is 2x the residual
    written and read back.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1030

        if (
            on_gfx1030()
            and fn.shape[0] in _RDNA2_MHC_PROJ_NOUT
            and x_flat.dtype in (torch.float16, torch.bfloat16)
            and x_flat.shape[1] % 8 == 0
            and x_flat.is_contiguous()
            and fn.is_contiguous()
        ):
            from vllm import _custom_ops as ops

            return ops.mhc_proj_rdna2(x_flat, fn)

    num_tokens, k = x_flat.shape
    x_f32 = torch.empty((num_tokens, k), dtype=torch.float32, device=x_flat.device)
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=x_flat.device)
    _cast_and_sqrsum_kernel[(num_tokens,)](
        x_flat, x_f32, sqrsum, k, BLOCK_K=1024, num_warps=4
    )
    mixes = _mhc_proj(x_f32, fn)
    del x_f32
    return mixes, sqrsum


@triton.jit
def _mhc_mix_kernel(
    mixes_ptr,  # [T, n_out] fp32, raw projection
    sqrsum_ptr,  # [T] fp32
    scale_ptr,  # [3] fp32
    base_ptr,  # [n_out] fp32
    pre_ptr,  # [T, hc_mult] fp32
    post_ptr,  # [T, hc_mult] fp32
    stride_t,
    inv_k,
    rms_eps,
    pre_eps,
    post_mult,
    HC_MULT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Turn the raw projection into the pre and post mixing weights.

    The RMS scale is applied here rather than written back over the projection,
    so the whole chain -- normalise, scale, shift, sigmoid -- is one kernel per
    token instead of a dozen elementwise launches.
    """
    t = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    mask = j < HC_MULT

    rstd = tl.rsqrt(tl.load(sqrsum_ptr + t) * inv_k + rms_eps)
    row = mixes_ptr + t * stride_t

    a = tl.load(row + j, mask=mask, other=0.0) * rstd
    b = tl.load(base_ptr + j, mask=mask, other=0.0)
    z = a * tl.load(scale_ptr + 0) + b
    tl.store(pre_ptr + t * HC_MULT + j, 1.0 / (1.0 + tl.exp(-z)) + pre_eps, mask=mask)

    a2 = tl.load(row + HC_MULT + j, mask=mask, other=0.0) * rstd
    b2 = tl.load(base_ptr + HC_MULT + j, mask=mask, other=0.0)
    z2 = a2 * tl.load(scale_ptr + 1) + b2
    tl.store(
        post_ptr + t * HC_MULT + j,
        (1.0 / (1.0 + tl.exp(-z2))) * post_mult,
        mask=mask,
    )


@triton.jit
def _sinkhorn_kernel(
    logit_ptr,  # [T, hc_mult * hc_mult] fp32, the comb slice of the projection
    base_ptr,  # [hc_mult * hc_mult] fp32
    scale_ptr,  # [3] fp32
    sqrsum_ptr,  # [T] fp32
    out_ptr,  # [T, hc_mult, hc_mult] fp32
    eps,
    stride_t,
    inv_k,
    rms_eps,
    HC_MULT: tl.constexpr,
    REPEAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Softmax the comb logits, then Sinkhorn-normalise rows and columns.

    The matrix is only ``hc_mult`` square, so every iteration is a handful of
    values; run them all in registers instead of launching a pair of reductions
    and a pair of divides per iteration.
    """
    t = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < HC_MULT) & (cols[None, :] < HC_MULT)
    offs = rows[:, None] * HC_MULT + cols[None, :]

    rstd = tl.rsqrt(tl.load(sqrsum_ptr + t) * inv_k + rms_eps)
    logits = tl.load(logit_ptr + t * stride_t + offs, mask=mask, other=0.0) * rstd
    base = tl.load(base_ptr + offs, mask=mask, other=0.0)
    z = logits * tl.load(scale_ptr + 2) + base
    z = tl.where(mask, z, float("-inf"))

    z = z - tl.max(z, axis=1)[:, None]
    e = tl.where(mask, tl.exp(z), 0.0)
    comb = tl.where(mask, e / tl.sum(e, axis=1)[:, None] + eps, 0.0)

    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in tl.static_range(REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)

    tl.store(out_ptr + t * HC_MULT * HC_MULT + offs, comb, mask=mask)


def _mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
) -> None:
    """Fill the three pre-block outputs in place. See ``mhc_pre_torch``.

    Same arithmetic as the torch reference, minus two full passes over the
    residual: the fp32 widening carries the sum of squares with it, and the
    ``pre_mix``-weighted reduction runs in one kernel instead of materialising
    another fp32 copy. The projection stays on the vendor GEMM, which beats a
    Triton dot on architectures without matrix cores, except where a native
    kernel exists -- see ``_mhc_cast_and_proj``.
    """
    hc_mult, hidden_size = residual.shape[-2:]
    x = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    k = hc_mult * hidden_size

    x_flat = x.reshape(num_tokens, k)
    mixes, sqrsum = _mhc_cast_and_proj(x_flat, fn)

    block_m = triton.next_power_of_2(hc_mult)
    inv_k = 1.0 / k
    pre_mix = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=mixes.device
    )
    _mhc_mix_kernel[(num_tokens,)](
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix.view(num_tokens, hc_mult),
        mixes.stride(0),
        inv_k,
        rms_eps,
        hc_pre_eps,
        hc_post_mult_value,
        HC_MULT=hc_mult,
        BLOCK=block_m,
        num_warps=1,
    )

    _sinkhorn_kernel[(num_tokens,)](
        mixes[:, 2 * hc_mult :],
        hc_base[2 * hc_mult :],
        hc_scale,
        sqrsum,
        comb_mix.view(num_tokens, hc_mult * hc_mult),
        hc_sinkhorn_eps,
        mixes.stride(0),
        inv_k,
        rms_eps,
        HC_MULT=hc_mult,
        REPEAT=sinkhorn_repeat,
        BLOCK=block_m,
        num_warps=1,
    )

    out = layer_input.view(num_tokens, hidden_size)
    block_h = 1024
    _hc_head_reduce_store_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        pre_mix,
        x,
        out,
        hidden_size,
        hc_mult,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


direct_register_custom_op(
    op_name="mhc_pre_triton",
    op_func=_mhc_pre_triton,
    mutates_args=["post_mix", "comb_mix", "layer_input"],
)


@triton.jit
def _mhc_post_kernel(
    x_ptr,  # [T, hidden]
    res_ptr,  # [T, hc_mult, hidden]
    post_ptr,  # [T, hc_mult] fp32
    comb_ptr,  # [T, hc_mult, hc_mult] fp32
    out_ptr,  # [T, hc_mult, hidden]
    hidden_size,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[t,j,h] = post[t,j]*x[t,h] + sum_i comb[t,i,j]*res[t,i,h]."""
    t = tl.program_id(0)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    xv = tl.load(x_ptr + t * hidden_size + offs_h, mask=mask_h, other=0.0).to(
        tl.float32
    )
    for j in tl.static_range(HC_MULT):
        acc = tl.load(post_ptr + t * HC_MULT + j).to(tl.float32) * xv
        for i in tl.static_range(HC_MULT):
            w = tl.load(comb_ptr + t * HC_MULT * HC_MULT + i * HC_MULT + j)
            r = tl.load(
                res_ptr + (t * HC_MULT + i) * hidden_size + offs_h,
                mask=mask_h,
                other=0.0,
            ).to(tl.float32)
            acc += w.to(tl.float32) * r
        tl.store(out_ptr + (t * HC_MULT + j) * hidden_size + offs_h, acc, mask=mask_h)


def _mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fill ``out`` in place. See ``mhc_post_torch``."""
    hc_mult, hidden_size = residual.shape[-2:]
    res = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = res.shape[0]
    if num_tokens == 0:
        return
    block_h = 1024
    _mhc_post_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        x.reshape(num_tokens, hidden_size),
        res,
        post_layer_mix.reshape(num_tokens, hc_mult),
        comb_res_mix.reshape(num_tokens, hc_mult, hc_mult),
        out.reshape(num_tokens, hc_mult, hidden_size),
        hidden_size,
        HC_MULT=hc_mult,
        BLOCK_H=block_h,
        num_warps=4,
    )


direct_register_custom_op(
    op_name="mhc_post_triton",
    op_func=_mhc_post_triton,
    mutates_args=["out"],
)
