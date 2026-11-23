#!/usr/bin/env python3
"""Unit tests for LoRA / Full-FT configuration helpers in chessfm.load_model."""

import pytest

from chessfm.load_model import (
    _ATTN_TARGET_MODULES,
    _TARGET_MODULES,
    load_model,
    resolve_lora_config,
)


def test_resolve_lora_config_defaults_and_alpha_scaling():
    default_cfg = resolve_lora_config()
    assert default_cfg == {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": _TARGET_MODULES,
    }

    r32_cfg = resolve_lora_config(lora_r=32)
    assert r32_cfg["r"] == 32
    assert r32_cfg["lora_alpha"] == 64
    assert r32_cfg["target_modules"] == _TARGET_MODULES

    r64_cfg = resolve_lora_config(lora_r=64, lora_alpha=96, all_linear=False)
    assert r64_cfg == {
        "r": 64,
        "lora_alpha": 96,
        "target_modules": _ATTN_TARGET_MODULES,
    }


def test_resolve_lora_config_rejects_non_positive():
    with pytest.raises(ValueError, match="lora_r must be positive"):
        resolve_lora_config(lora_r=0)
    with pytest.raises(ValueError, match="lora_alpha must be positive"):
        resolve_lora_config(lora_r=16, lora_alpha=0)


def test_load_model_rejects_full_ft_with_4bit():
    with pytest.raises(ValueError, match="full_ft=True cannot be combined with load_in_4bit=True"):
        load_model("Qwen/Qwen3-4B", load_in_4bit=True, full_ft=True)
