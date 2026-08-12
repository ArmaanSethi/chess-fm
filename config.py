#!/usr/bin/env python3
"""
Central configuration for ChessFM
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data_generation"
RL_DIR = PROJECT_ROOT / "rl"
TRAINING_DIR = PROJECT_ROOT / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Standard Base Model
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Training parameters
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32

# Prompt template. Shared across SFT, GRPO, and evaluation so the model
# always sees the same instruction at train and inference time.
PROMPT_TEMPLATE = (
    "Position (FEN): {fen}\n"
    "Side to move: {color}\n\n"
    "Analyze this position and choose the best move. "
    "Reason inside <think>...</think> tags, then output the move "
    "in UCI format (e.g. e2e4)."
)


def make_prompt(fen: str) -> str:
    """Build the standard ChessFM prompt for a FEN position."""
    color = "White" if fen.split()[1] == "w" else "Black"
    return PROMPT_TEMPLATE.format(fen=fen, color=color)

# Common Stockfish paths
STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish"
]
