# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 MoE on RDNA2 (gfx1030) via the native moe_gptq_gemm_rdna2 HIP kernel.

Drop-in for MoeWNA16Method (autoawq / autogptq MoE). The moe_wna16 loader
normalizes AWQ and GPTQ into one canonical layout:
  qweight [E, N, K/2] uint8   element(k,n) = (byte[n,k//2] >> (k%2*4)) & 0xF
  scales  [E, N, groups]
  qzeros  [E, N/2, groups] uint8 (has_zp)  element(n,g) = (byte[n//2,g] >> (n%2*4)) & 0xF
and bakes the effective zero into qzeros for both (AWQ: no +1; GPTQ: +1 at load),
so we always call the kernel with use_v2_format=True (zero_offset 0). We convert
that layout, per expert, to the kernel's format (shuffled int32 [E, K/8, N] +
scales [E, groups, N] + packed qzeros [E, groups, N/8]) and run the two grouped
GEMMs (w13 -> silu*mul -> w2) with moe_sum fused into the w2 output.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

_WT = scalar_types.uint4b8


def _to_kernel_layout(qweight_u8, scales, qzeros_u8, has_zp):
    """moe_wna16 canonical -> kernel layout, per expert.

    qweight_u8 [E, N, K/2] uint8 -> qw [E, K/8, N] int32 (exllama shuffle)
    scales     [E, N, groups]    -> [E, groups, N]
    qzeros_u8  [E, N/2, groups]  -> qz [E, groups, N/8] int32 (has_zp; else synth 8)
    """
    E, N, k_half = qweight_u8.shape
    K = k_half * 2
    dev = qweight_u8.device

    lo = qweight_u8 & 0xF
    hi = (qweight_u8 >> 4) & 0xF
    nib = torch.stack([lo, hi], dim=-1).reshape(E, N, K)  # element(k,n) at [e,n,k]
    q = nib.transpose(1, 2).contiguous().to(torch.int32)  # [E, K, N]
    qw = torch.stack(
        [pack_quantized_values_into_int32(q[e], _WT, packed_dim=0) for e in range(E)]
    )
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for e in range(E):
        w = qw[e].contiguous()
        ops.gptq_shuffle(w, empty, 4)
        qw[e] = w

    sc = scales.transpose(1, 2).contiguous()  # [E, groups, N]
    groups = sc.shape[1]

    if has_zp:
        zlo = qzeros_u8 & 0xF
        zhi = (qzeros_u8 >> 4) & 0xF
        znib = torch.stack([zlo, zhi], dim=2).reshape(E, N, groups)  # element(n,g)
        z = znib.transpose(1, 2).contiguous().to(torch.int32)  # [E, groups, N]
    else:
        # symmetric: effective zero = bias (8); use_v2_format=True => stored == 8.
        z = torch.full((E, groups, N), _WT.bias, dtype=torch.int32, device=dev)
    qz = torch.stack(
        [pack_quantized_values_into_int32(z[e], _WT, packed_dim=1) for e in range(E)]
    )
    return qw, sc.contiguous(), qz


class MoeWNA16RDNA2Method(MoeWNA16Method):
    """W4A16 MoE using the fused RDNA2 HIP kernel (gfx1030)."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        has_zp = self.quant_config.has_zp
        w13_qz = layer.w13_qzeros.data if has_zp else None
        w2_qz = layer.w2_qzeros.data if has_zp else None
        w13 = _to_kernel_layout(layer.w13_qweight.data, layer.w13_scales.data, w13_qz, has_zp)
        w2 = _to_kernel_layout(layer.w2_qweight.data, layer.w2_scales.data, w2_qz, has_zp)

        def _set(name, t):
            setattr(layer, name, torch.nn.Parameter(t, requires_grad=False))

        _set("w13_qweight", w13[0])
        _set("w13_scales", w13[1])
        _set("w13_qzeros", w13[2])
        _set("w2_qweight", w2[0])
        _set("w2_scales", w2[1])
        _set("w2_qzeros", w2[2])

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = (
            layer.activation
            if isinstance(layer.activation, MoEActivation)
            else MoEActivation.from_str(layer.activation)
        )
        return _rdna2_fused_moe(
            x,
            topk_weights,
            topk_ids,
            layer,
            activation,
            layer.apply_router_weight_on_input,
            layer.global_num_experts,
            layer.expert_map,
            use_v2_format=True,
        )


def _rdna2_fused_moe(
    x, topk_weights, topk_ids, layer, activation, apply_router_weight_on_input,
    global_num_experts, expert_map, use_v2_format,
):
    num_tokens = x.shape[0]
    top_k = topk_ids.shape[1]
    total = num_tokens * top_k
    E, _, N13 = layer.w13_qweight.shape
    hidden = layer.w2_qweight.shape[2]
    dtype, dev = x.dtype, x.device
    intermediate = N13 // 2 if activation.is_gated else N13
    if global_num_experts <= 0:
        global_num_experts = E

    block_m = 1 if num_tokens <= 4 else 4
    sorted_ids, expert_ids, num_pad = moe_align_block_size(
        topk_ids, block_m, global_num_experts, expert_map
    )
    topk_w_f = topk_weights.reshape(-1).float()
    empty = torch.empty(0, device=dev)

    w1_out = torch.zeros(total, N13, dtype=dtype, device=dev)
    ops.moe_gptq_gemm_rdna2(
        x, w1_out, layer.w13_qweight, layer.w13_scales, layer.w13_qzeros,
        topk_w_f if apply_router_weight_on_input else empty,
        sorted_ids, expert_ids, num_pad, top_k, block_m,
        apply_router_weight_on_input, use_v2_format,
    )

    act_out = torch.empty(total, intermediate, dtype=dtype, device=dev)
    apply_moe_activation(activation, act_out, w1_out)

    out = torch.zeros(num_tokens, hidden, dtype=dtype, device=dev)
    ops.moe_gptq_gemm_rdna2(
        act_out, out, layer.w2_qweight, layer.w2_scales, layer.w2_qzeros,
        topk_w_f if not apply_router_weight_on_input else empty,
        sorted_ids, expert_ids, num_pad, 1, block_m,
        not apply_router_weight_on_input, use_v2_format, output_topk=top_k,
    )
    return out
