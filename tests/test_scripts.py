#!/usr/bin/env python3
"""
Smoke tests: every CLI entry point must at least parse its arguments and start up.

These are deliberately lightweight (no GPU, no network, no Stockfish needed) so
CI catches import errors / broken CLIs fast.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Each entry is runnable via `python -m chessfm.<module> --help`.
ENTRY_POINTS = [
    "chessfm.data_generation.convert_to_training",
    "chessfm.data_generation.download_positions",
    "chessfm.data_generation.fetch_elite_data",
    "chessfm.data_generation.generate_synthetic_data",
    "chessfm.scripts.audit_tokenizer",
    "chessfm.scripts.download_models",
    "chessfm.training.train_sft",
    "chessfm.rl.train_grpo",
    "chessfm.rl.evaluate",
]


def test_all_clis_have_help():
    env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    for entry in ENTRY_POINTS:
        result = subprocess.run(
            [sys.executable, "-m", entry, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"python -m {entry} --help failed (rc={result.returncode})\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )


def test_config_importable():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from chessfm import config

    assert config.DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
    assert config.make_prompt(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ).startswith("Position (FEN):")
    # ChatML-wrapped prompt used at GRPO/eval/inference time.
    chat = config.build_chat_prompt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert chat.startswith("<|im_start|>user\nPosition (FEN):")
    assert chat.endswith("<|im_start|>assistant\n")


def test_make_prompt_color_detection():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from chessfm.config import make_prompt

    white = make_prompt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black = make_prompt("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert "Side to move: White" in white
    assert "Side to move: Black" in black
