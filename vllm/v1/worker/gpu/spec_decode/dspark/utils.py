# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config

    # None re-runs backend auto-selection for the draft, which can pick a
    # different attention class than the target; fall back to the target's.
    draft_attention_backend = (
        speculative_config.attention_backend or vllm_config.attention_config.backend
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        # Under pipeline parallelism the target's embedding is a PPMissingLayer
        # on every rank but the first, so the drafter would silently embed with
        # uninitialized weights. spec_decode_needs_target_embed() gives the last
        # rank a real one; refuse rather than draft garbage if it did not.
        if isinstance(target_embed, PPMissingLayer):
            raise RuntimeError(
                "The DSpark drafter needs the target's embed_tokens, but this "
                "pipeline stage only has a placeholder. The last stage must "
                "instantiate it (see spec_decode_needs_target_embed)."
            )
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
