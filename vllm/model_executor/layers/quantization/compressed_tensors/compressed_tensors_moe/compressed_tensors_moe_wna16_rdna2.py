# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A16 via the native RDNA2 (gfx1030) HIP kernel.

fp16-only sibling of compressed_tensors_moe_wna16_rdna3.py. CompressedTensors
WNA16 MoE is symmetric (uint4b8), so we synthesize stored zeros = bias-1 = 7 and
run the kernel with use_v2_format=False (the +1 zero-offset recovers bias 8).
Weights are already int32 [E, K/8, N]; we only exllama-shuffle them per expert
and reuse the shared _rdna2_fused_moe driver.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.moe_wna16_rdna2 import _rdna2_fused_moe
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

_WT = scalar_types.uint4b8


def _synth_qzeros(E, groups, N, device):
    z = torch.full((groups, N), _WT.bias - 1, dtype=torch.int32, device=device)
    packed = pack_quantized_values_into_int32(z, _WT, packed_dim=1)  # [groups, N/8]
    return packed.unsqueeze(0).expand(E, -1, -1).contiguous()


class CompressedTensorsWNA16RDNA2MoEMethod(CompressedTensorsWNA16MoEMethod):
    """W4A16 MoE using the fused RDNA2 HIP kernel (moe_gptq_gemm_rdna2)."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight_packed.device
        E = layer.w13_weight_packed.shape[0]
        empty = torch.empty(0, dtype=torch.int32, device=device)

        w13 = layer.w13_weight_packed.data
        w2 = layer.w2_weight_packed.data
        for e in range(E):
            a = w13[e].contiguous()
            ops.gptq_shuffle(a, empty, 4)
            w13[e] = a
            b = w2[e].contiguous()
            ops.gptq_shuffle(b, empty, 4)
            w2[e] = b

        act = layer.w13_weight_scale.dtype
        layer.w13_qweight = torch.nn.Parameter(w13.contiguous(), requires_grad=False)
        layer.w2_qweight = torch.nn.Parameter(w2.contiguous(), requires_grad=False)
        layer.w13_scales = torch.nn.Parameter(
            layer.w13_weight_scale.to(act).contiguous(), requires_grad=False
        )
        layer.w2_scales = torch.nn.Parameter(
            layer.w2_weight_scale.to(act).contiguous(), requires_grad=False
        )
        g13 = (layer.w13_qweight.shape[1] * 8) // self.group_size
        g2 = (layer.w2_qweight.shape[1] * 8) // self.group_size
        layer.w13_qzeros = torch.nn.Parameter(
            _synth_qzeros(E, g13, layer.w13_qweight.shape[2], device),
            requires_grad=False,
        )
        layer.w2_qzeros = torch.nn.Parameter(
            _synth_qzeros(E, g2, layer.w2_qweight.shape[2], device),
            requires_grad=False,
        )

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
            use_v2_format=False,
        )
