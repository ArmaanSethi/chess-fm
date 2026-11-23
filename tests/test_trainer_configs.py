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


def test_sft_config_enables_gradient_checkpointing_on_gpu(monkeypatch):
    import torch
    import trl

    from chessfm.training.train_sft import build_sft_config

    captured: dict = {}
    orig_sft_config = trl.SFTConfig

    def _spy_sft_config(**kwargs):
        captured.update(kwargs)
        # Validate TRL accepts gradient_checkpointing kwargs without requiring a CUDA driver.
        cpu_kwargs = {
            **kwargs,
            "bf16": False,
            "fp16": False,
            "use_cpu": True,
            "optim": "adamw_torch",
        }
        return orig_sft_config(**cpu_kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(trl, "SFTConfig", _spy_sft_config)

    cfg = build_sft_config(_fake_args(batch_size=2, grad_accum=4))
    assert captured["per_device_train_batch_size"] == 2
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["gradient_checkpointing"] is True
    assert captured["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert cfg.gradient_checkpointing is True
    assert cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_grpo_config_enables_gradient_checkpointing_on_gpu(monkeypatch):
    import torch
    import trl

    from chessfm.rl.train_grpo import build_grpo_config

    captured: dict = {}
    orig_grpo_config = trl.GRPOConfig

    def _spy_grpo_config(**kwargs):
        captured.update(kwargs)
        cpu_kwargs = {**kwargs, "bf16": False, "fp16": False, "use_cpu": True}
        return orig_grpo_config(**cpu_kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(trl, "GRPOConfig", _spy_grpo_config)

    cfg = build_grpo_config(_fake_args(per_device_batch_size=4, num_generations=4, grad_accum=4))
    assert captured["gradient_checkpointing"] is True
    assert captured["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert cfg.gradient_checkpointing is True
    assert cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_grpo_load_positions_passes_enable_thinking_true(tmp_path):
    import chess

    from chessfm.rl.train_grpo import load_positions

    pos_file = tmp_path / "positions.txt"
    pos_file.write_text(chess.STARTING_FEN + "\n", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _ThinkingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured_kwargs.append(kwargs)
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    rows = load_positions(
        pos_file, max_positions=1, tokenizer=_ThinkingTokenizer(), include_ascii=True
    )
    assert len(rows) == 1
    assert captured_kwargs and captured_kwargs[0].get("enable_thinking") is True
    assert "Board:\n" in rows[0]["prompt"]


def test_kv_cache_generation_mixin_unwraps_ddp_and_restores_on_error():
    from types import SimpleNamespace

    from chessfm.rl.train_grpo import _KVCacheGenerationMixin

    class _InnerModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)

    class _DDPWrapper:
        """Simulates DistributedDataParallel (no direct .config attribute)."""

        def __init__(self, module):
            self.module = module
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    class _FakeAccelerator:
        def unwrap_model(self, model):
            return model.module

    observed: list[tuple[bool, bool]] = []

    class _BaseTrainer:
        def __init__(self, model, accelerator=None, fail=False):
            self.model = model
            self.accelerator = accelerator
            self._fail = fail

        def _generate_and_score_completions(self, inputs):
            unwrapped = (
                self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
            )
            cfg = getattr(unwrapped, "config", None)
            observed.append((self.model.training, getattr(cfg, "use_cache", None)))
            if self._fail:
                raise RuntimeError("boom")
            return inputs

    class _TestTrainer(_KVCacheGenerationMixin, _BaseTrainer):
        pass

    inner = _InnerModel()
    ddp_model = _DDPWrapper(inner)
    assert not hasattr(ddp_model, "config")

    trainer = _TestTrainer(ddp_model, accelerator=_FakeAccelerator(), fail=False)
    assert trainer._generate_and_score_completions("ok") == "ok"
    assert observed == [(False, True)]
    assert ddp_model.training is True
    assert inner.config.use_cache is False

    # Exception safety: restores training=True and use_cache=False even if generation raises.
    failing_trainer = _TestTrainer(ddp_model, accelerator=_FakeAccelerator(), fail=True)
    with pytest.raises(RuntimeError, match="boom"):
        failing_trainer._generate_and_score_completions("x")
    assert ddp_model.training is True
    assert inner.config.use_cache is False

    # Config-less model without accelerator must not raise AttributeError.
    class _NoConfigModel:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    no_cfg_model = _NoConfigModel()
    no_cfg_trainer = _TestTrainer(no_cfg_model, accelerator=None, fail=False)
    assert no_cfg_trainer._generate_and_score_completions("ok") == "ok"
    assert no_cfg_model.training is True
