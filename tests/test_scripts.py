#!/usr/bin/env python3
"""
Smoke tests: every CLI script must at least parse its arguments and start up.

These are deliberately lightweight (no GPU, no network, no Stockfish needed)
so CI catches import errors / broken CLIs fast.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SCRIPTS = [
    "data_generation/convert_to_training.py",
    "data_generation/download_positions.py",
    "data_generation/fetch_elite_data.py",
    "scripts/audit_tokenizer.py",
    "scripts/download_models.py",
    "training/train_sft.py",
    "rl/train_grpo.py",
    "rl/evaluate.py",
]


def test_all_clis_have_help():
    for rel in SCRIPTS:
        path = PROJECT_ROOT / rel
        assert path.exists(), f"Script missing: {rel}"
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{rel} --help failed (rc={result.returncode})\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )


def test_config_importable():
    sys.path.insert(0, str(PROJECT_ROOT))
    import config
    assert config.DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
    assert config.make_prompt(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ).startswith("Position (FEN):")


def test_make_prompt_color_detection():
    sys.path.insert(0, str(PROJECT_ROOT))
    from config import make_prompt

    white = make_prompt("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black = make_prompt("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert "Side to move: White" in white
    assert "Side to move: Black" in black
