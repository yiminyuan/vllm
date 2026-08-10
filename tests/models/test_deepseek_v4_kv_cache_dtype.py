#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--kv-cache-dtype`` resolution for the DeepSeek-V4 MLA backends.

The fp8_ds_mla layout (UE8M0 block-scaled fp8 packed as uint8) is the only
layout the FlashMLA and ROCm sparse backends implement, so the default "auto"
has to resolve to it rather than fail. The canonical string must also be written
back onto the CacheConfig, because the KV page-size specs read it to size the
576-byte per-token slot.

Run `pytest tests/models/test_deepseek_v4_kv_cache_dtype.py`.
"""

import pytest
import torch

from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype


class _CacheConfig:
    def __init__(self, cache_dtype):
        self.cache_dtype = cache_dtype


@pytest.mark.parametrize("requested", ["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_ds_mla_layout_resolves_to_fp8_ds_mla(requested):
    cache_config = _CacheConfig(requested)
    dtype_str, torch_dtype = _resolve_dsv4_kv_cache_dtype(True, requested, cache_config)

    assert dtype_str == "fp8_ds_mla"
    assert torch_dtype is torch.uint8
    # The page-size specs read this back; leaving it as "auto" mis-sizes pages.
    assert cache_config.cache_dtype == "fp8_ds_mla"


def test_ds_mla_layout_rejects_non_fp8():
    with pytest.raises(ValueError, match="fp8_ds_mla"):
        _resolve_dsv4_kv_cache_dtype(True, "bfloat16", _CacheConfig("bfloat16"))


def test_ds_mla_layout_tolerates_missing_cache_config():
    dtype_str, torch_dtype = _resolve_dsv4_kv_cache_dtype(True, "auto", None)
    assert dtype_str == "fp8_ds_mla"
    assert torch_dtype is torch.uint8


@pytest.mark.parametrize(
    "requested,expected_dtype",
    [
        ("auto", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp8", torch.float8_e4m3fn),
        ("fp8_e4m3", torch.float8_e4m3fn),
    ],
)
def test_plain_row_layout_is_unchanged(requested, expected_dtype):
    """Backends on the plain KV row keep "auto" meaning bf16."""
    cache_config = _CacheConfig(requested)
    dtype_str, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        False, requested, cache_config
    )

    assert dtype_str == requested
    assert torch_dtype is expected_dtype
    assert cache_config.cache_dtype == requested
