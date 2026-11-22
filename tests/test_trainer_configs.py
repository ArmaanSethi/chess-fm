#!/usr/bin/env python3
"""Tests that the exact TRL training kwargs used by the scripts are valid.

These construct ``GRPOConfig`` / ``SFTConfig`` with the same keyword arguments
the training entry points pass, so a TRL API rename is caught by CI instead of
on the GPU machine. They need ``trl`` (and are skipped when it is unavailable,
so the rest of the suite still runs without a GPU stack).
"""

from pathlib import Path

import pytest

pytest.importorskip("trl")


def _fake_args(output="checkpoints", **overrides):
    class _A:
        pass

    _A.output = Path(output)
    _A.steps = 500
    _A.batch_size = 4
    _A.num_generations = 8
    _A.per_device_batch_size = 8
    _A.grad_accum = 2
    _A.temperature = 1.0
    _A.beta = 0.04
    _A.lr = 1e-5
    _A.checkpoint_every = 100
    _A.seed = 42
    _A.epochs = 3

    for k, v in overrides.items():
        setattr(_A, k, v)
    return _A()


def test_sft_config_accepts_script_kwargs():
    from chessfm.training.train_sft import build_sft_config

    cfg = build_sft_config(_fake_args())
    assert cfg.output_dir == "checkpoints"
    assert cfg.num_train_epochs == 3
    assert cfg.seed == 42


def test_sft_response_template_matches_model_family():
    from chessfm.training.train_sft import get_response_template

    assert get_response_template("Qwen/Qwen2.5-3B-Instruct") == "<|im_start|>assistant\n"
    assert get_response_template("Qwen/Qwen3-4B") == "<|im_start|>assistant\n"
    assert get_response_template("unsloth/gemma-4-E4B-it") == "<|turn>model\n"


def test_grpo_config_accepts_script_kwargs():
    from chessfm.rl.train_grpo import build_grpo_config

    cfg = build_grpo_config(_fake_args())
    assert cfg.output_dir == "checkpoints"
    assert cfg.num_generations == 8
    assert cfg.per_device_train_batch_size == 8
    assert cfg.max_steps == 500
    assert cfg.temperature == 1.0
    assert cfg.beta == 0.04
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.seed == 42


def test_grpo_config_rejects_indivisible_batch_size():
    from chessfm.rl.train_grpo import build_grpo_config

    with pytest.raises(ValueError, match="divisible by num_generations"):
        build_grpo_config(_fake_args(per_device_batch_size=4, num_generations=8, world_size=1))


def test_grpo_config_allows_multi_gpu_divisible_batch_size():
    from chessfm.rl.train_grpo import build_grpo_config

    cfg = build_grpo_config(_fake_args(per_device_batch_size=4, num_generations=8, world_size=2))
    assert cfg.per_device_train_batch_size == 4
    assert cfg.num_generations == 8
