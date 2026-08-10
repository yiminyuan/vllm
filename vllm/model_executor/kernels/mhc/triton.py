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
    another fp32 copy. The projection itself stays on the vendor GEMM, which
    beats a Triton dot on architectures without matrix cores.
    """
    hc_mult, hidden_size = residual.shape[-2:]
    x = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    k = hc_mult * hidden_size

    x_f32 = torch.empty((num_tokens, k), dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    _cast_and_sqrsum_kernel[(num_tokens,)](
        x.reshape(num_tokens, k), x_f32, sqrsum, k, BLOCK_K=1024, num_warps=4
    )

    mixes = F.linear(x_f32, fn)
    del x_f32
    mixes *= torch.rsqrt(sqrsum.unsqueeze(-1) / k + rms_eps)

    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    pre_mix += hc_pre_eps
    post = torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix.copy_((post * hc_post_mult_value).view(post_mix.shape))

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    comb_mix.copy_(comb.view(comb_mix.shape))

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
