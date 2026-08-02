# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GPTQ/AWQ kernel for AMD RDNA2 (gfx1030) — fp16 only.

fp16-only sibling of RDNA3W4A16LinearKernel. RDNA2 has v_dot2_f32_f16 (fdot2)
and v_pk_fma_f16 but no bf16 packed dot (v_dot2_f32_bf16) and no WMMA. Handles
both symmetric GPTQ (uint4b8) and asymmetric AWQ/compressed-tensors (uint4).

Internally hybrid, mirroring the RDNA3 scalar/WMMA split with rocBLAS in place of
WMMA (which RDNA2 lacks): small M uses the fused fdot2 "+1024" bit-trick GEMV
(``csrc/rocm/q_gemm_rdna2.cu`` / ``torch.ops._rocm_C.gptq_gemm_rdna2``), which
beats Exllama at decode; large M reconstructs to fp16 and calls rocBLAS via the
shared ``gptq_gemm`` op, identical to Exllama's prefill. Weights use the same
gptq_shuffle layout as ExllamaLinearKernel, so both paths take the same tensors.

The M-dispatch lives in the ``vllm::rdna2_w4a16_gemm`` custom op, not in the
traced ``apply_weights``: vLLM compiles the forward over the prefill range with
guard checks disabled (dynamic_shapes ``evaluate_guards=False``), so a
Python-level ``if M <= threshold`` would be frozen to the traced branch (the
prefill reconstruct path) for every call, stranding decode on it. A custom op is
opaque to Dynamo, so the branch runs at real runtime per call.

Registered ahead of ExllamaLinearKernel for the ROCm-RDNA2 path; falls through
to Exllama/Triton on non-RDNA2 ROCm devices.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# M at/below this uses the fdot2 GEMV; above it reconstructs to fp16 + rocBLAS.
_RDNA2_W4A16_M_THRESHOLD = 128


def _rdna2_w4a16_gemm_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    w_g_idx: torch.Tensor,
    use_v2_format: bool,
    weight_bits: int,
) -> torch.Tensor:
    if x_2d.shape[0] <= _RDNA2_W4A16_M_THRESHOLD:
        # Decode / small batch: fused fdot2 "+1024" bit-trick GEMV.
        return ops.gptq_gemm_rdna2(x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format)
    # Prefill: reconstruct to fp16 and hand off to rocBLAS via the shared kernel.
    return ops.gptq_gemm(
        x_2d, w_q, w_zp, w_s, w_g_idx, True, use_v2_format, weight_bits
    )


def _rdna2_w4a16_gemm_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    w_g_idx: torch.Tensor,
    use_v2_format: bool,
    weight_bits: int,
) -> torch.Tensor:
    return x_2d.new_empty((x_2d.shape[0], w_q.shape[1]), dtype=x_2d.dtype)


direct_register_custom_op(
    op_name="rdna2_w4a16_gemm",
    op_func=_rdna2_w4a16_gemm_impl,
    fake_impl=_rdna2_w4a16_gemm_fake,
)


class RDNA2W4A16LinearKernel(MPLinearKernel):
    # GPTQ (uint4b8) + AWQ/compressed-tensors (uint4), 4-bit, fp16-only.
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]

    HYBRID_M_THRESHOLD = _RDNA2_W4A16_M_THRESHOLD

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx1030() in can_implement.
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA2 W4A16 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx1030

        if not on_gfx1030():
            return False, "RDNA2 W4A16 kernel requires gfx1030"

        # The HIP op is registered by the C++ extension; if a user is running
        # against a vLLM build that doesn't include it (e.g. partial rebuild),
        # fall through gracefully to the next kernel in the registry.
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna2")
        ):
            return (
                False,
                "torch.ops._rocm_C.gptq_gemm_rdna2 missing — rebuild C++ extension",
            )

        # RDNA2 has no bf16 packed dot (v_dot2_f32_bf16), so fp16 only. bf16
        # models fall through to Exllama (which casts) or Triton.
        if c.act_type != torch.float16:
            return False, "RDNA2 W4A16 kernel only supports fp16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA2 W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                "RDNA2 W4A16 kernel does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide K "
                f"({c.full_weight_shape[0]})",
            )

        # Output features must be a multiple of 8: qzeros pack 8 nibbles per
        # int32, and the kernel epilogue writes 4 columns per 64-bit atomic CAS.
        if c.partition_weight_shape[1] % 8 != 0:
            return (
                False,
                "Output features must be a multiple of 8 for the RDNA2 "
                "W4A16 kernel (qzeros packing + 64-bit atomic write)",
            )

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act-order with TP-partitioned input features is not "
                "supported by the RDNA2 W4A16 kernel",
            )

        return True, None

    # ----- Weight prep (identical layout/shuffle as ExllamaLinearKernel) -----

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        # Synthesize zero points if the checkpoint doesn't carry them.
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # GPTQv1 quirk: the kernel adds 1 to the stored zero, so we
                # encode (bias - 1) here. See exllama.py for the link to the
                # documentation of this checkpoint-format wart.
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "RDNA2 W4A16 kernel: zero-bias 4-bit quant requires "
                    "explicit zero points (GPTQv1 +1 quirk)."
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        # Act-order: convert g_idx to the inverse permutation array exllama
        # expects (kernel reads a[perm[k]] instead of using groups indirected
        # by g_idx[k]).
        if c.has_g_idx:

            def transform_w_g_idx(x):
                return torch.argsort(x).to(torch.int)

            self._transform_param(layer, self.w_gidx_name, transform_w_g_idx)  # type: ignore
        else:
            self.w_gidx_name = "g_idx"
            empty_g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int, device=device),
                requires_grad=False,
            )
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            # Same 4-bit shuffle as exllama. The RDNA2 kernel reads weights in
            # the same shuffled int32 layout and uses the (qa & 0x000F000F)
            # bit-trick on top.
            ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # AWQ/compressed-tensors ships zeros as [N//pack, K//G]; the ops want the
        # transpose. Synthesized GPTQ zeros are already in the right layout.
        if c.zero_points:

            def transform_w_zp(x):
                x.data = x.data.t().contiguous()
                return x

            self._transform_param(layer, self.w_zp_name, transform_w_zp)

    # ----- Forward --------------------------------------------------------

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        assert w_zp is not None, "Zero points are required by RDNA2 W4A16"
        assert w_g_idx is not None, "g_idx tensor (possibly empty) required"

        # v2 zero-point format for asymmetric AWQ, v1 for symmetric GPTQ. The
        # custom op holds the M-dispatch so it survives torch.compile (see the
        # module docstring).
        use_v2_format = c.zero_points
        output = torch.ops.vllm.rdna2_w4a16_gemm(
            x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format, c.weight_type.size_bits
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
