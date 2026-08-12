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

# Common Stockfish paths
STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish"
]
