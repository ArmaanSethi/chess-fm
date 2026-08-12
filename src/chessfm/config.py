#!/usr/bin/env python3
"""Central configuration for ChessFM."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_generation"
RL_DIR = PROJECT_ROOT / "rl"
TRAINING_DIR = PROJECT_ROOT / "training"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PYGNS_DIR = RESULTS_DIR / "pgns"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32

# Common Stockfish binary locations.
STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish",
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# The instruction is the same everywhere (SFT, GRPO, evaluation). The only
# difference is *how* it is wrapped: the raw instruction for SFT text data and
# a ChatML-wrapped version for Qwen-2.5-Instruct, which is what GRPO and
# evaluation send to the model at inference time.
PROMPT_TEMPLATE = (
    "Position (FEN): {fen}\n"
    "Side to move: {color}\n\n"
    "Analyze this position and choose the best move. "
    "Reason inside <think>...</think> tags, then output the move "
    "in UCI format (e.g. e2e4)."
)

CHATML_USER_OPEN = "<|im_start|>user\n"
CHATML_USER_CLOSE = "<|im_end|>\n"
CHATML_ASSISTANT_OPEN = "<|im_start|>assistant\n"


def make_prompt(fen: str) -> str:
    """Build the raw (unwrapped) instructional prompt for a FEN position."""
    color = "White" if fen.split()[1] == "w" else "Black"
    return PROMPT_TEMPLATE.format(fen=fen, color=color)


def build_chat_prompt(fen: str) -> str:
    """Build a ChatML-wrapped prompt for Qwen-2.5-Instruct (train/inference)."""
    return f"{CHATML_USER_OPEN}{make_prompt(fen)}{CHATML_USER_CLOSE}{CHATML_ASSISTANT_OPEN}"
