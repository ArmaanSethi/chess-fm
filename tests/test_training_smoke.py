#!/usr/bin/env python3
"""CPU smoke tests: one real SFT step and one real GRPO step through TRL.

These exercise the actual trainer wiring (dataset columns, config kwargs,
reward-function contract) end to end using ``sshleifer/tiny-gpt2``, so TRL API
breakage or a malformed training path fails in CI instead of on the GPU
machine. They are skipped when torch/trl are unavailable or the hub is
unreachable, so the rest of the suite still runs offline.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")

from datasets import Dataset  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _load_tiny_model():
    """Load tiny-gpt2 or skip if the hub cannot be reached."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        # eager attention avoids an MPS limitation in torch's SDPA kernel
        # (dropout during generation is unsupported there) so the smoke tests
        # run on Apple Silicon as well as Linux CI.
        model = AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", attn_implementation="eager"
        )
    except Exception as e:  # network down, hub outage, etc.
        pytest.skip(f"tiny model unavailable: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_sft_single_step_runs(tmp_path):
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = _load_tiny_model()
    dataset = Dataset.from_list(
        [
            {
                "prompt": f"<|im_start|>user\nPosition (FEN): {START_FEN}<|im_end|>\n",
                "completion": "<think>develop and control the center.</think>\ne2e4",
            },
            {
                "prompt": f"<|im_start|>user\nPosition (FEN): {START_FEN}<|im_end|>\n",
                "completion": "<think>claim the center immediately.</think>\nd2d4",
            },
        ]
    ).select_columns(["prompt", "completion"])

    config = SFTConfig(
        output_dir=str(tmp_path),
        max_steps=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        max_length=128,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()


def test_grpo_single_step_runs_with_stage0_rewards(tmp_path):
    from trl import GRPOConfig, GRPOTrainer

    from chessfm.config import build_chat_prompt
    from chessfm.rl.train_grpo import make_reward_func

    model, tokenizer = _load_tiny_model()
    dataset = Dataset.from_list([{"prompt": build_chat_prompt(START_FEN)}])

    config = GRPOConfig(
        output_dir=str(tmp_path),
        max_steps=1,
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_prompt_length=64,
        max_completion_length=32,
        temperature=1.0,
        beta=0.04,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_func(stage=0)],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
