#!/usr/bin/env python3
"""Central configuration for ChessFM."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_generation"
RESULTS_DIR = PROJECT_ROOT / "results"
PGNS_DIR = RESULTS_DIR / "pgns"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------

MAX_SEQ_LENGTH = 2048
# GRPO generation budget. Completion must comfortably fit MAX_THINK_WORDS
# (250 words ~ 350 tokens) plus the move, so it is set well above the old 256.
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
# Generation budget used by the evaluation and benchmark harnesses at inference
# time. It must match the training-time completion budget so the model is never
# measured under a token budget it was not trained on.
DEFAULT_MAX_NEW_TOKENS = MAX_COMPLETION_LENGTH
LORA_R = 16
LORA_ALPHA = 32

# A move is "reasonable" if it gives up no more than this many centipawns vs
# Stockfish's best. Shared by data generation and data conversion.
MAX_CENTIPAWN_LOSS = 200

# Default Stockfish search depth for reward computation and validation.
DEFAULT_STOCKFISH_DEPTH = 10

# Fixed search depth used by the evaluation opponent (instead of a wall-clock
# time limit) so opponent strength is reproducible across runs and machines.
DEFAULT_OPPONENT_DEPTH = 6

# Hard cap on plies per game before it is adjudicated as a draw.
MAX_GAME_MOVES = 200

# GRPO defaults. Group size (completions per prompt), generation temperature,
# and KL coefficient are the levers that matter most for GRPO stability; they
# live here so every stage trains with consistent settings unless overridden.
GRPO_NUM_GENERATIONS = 8
GRPO_TEMPERATURE = 1.0
GRPO_BETA = 0.04

# Common Stockfish binary locations.
STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "stockfish",
]

# Approximate Elo for Stockfish skill levels (0-20), used only for a rough,
# ballpark comparison. These are informal community calibrations, NOT validated
# ratings. The authoritative reported metric is win rate vs the configured
# opponent.
STOCKFISH_LEVEL_ELO = {
    0: 800,
    1: 900,
    2: 1000,
    3: 1100,
    4: 1200,
    5: 1300,
    6: 1400,
    7: 1500,
    8: 1600,
    9: 1700,
    10: 1800,
    11: 1900,
    12: 2000,
    13: 2100,
    14: 2200,
    15: 2300,
    16: 2400,
    17: 2500,
    18: 2600,
    19: 2700,
    20: 2800,
}

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
# Appended to SFT completion targets. Without it the model never learns to
# emit the turn-end token, so inference runs to the token budget on every move.
CHATML_ASSISTANT_CLOSE = "<|im_end|>"


def make_prompt(fen: str) -> str:
    """Build the raw (unwrapped) instructional prompt for a FEN position."""
    color = "White" if fen.split()[1] == "w" else "Black"
    return PROMPT_TEMPLATE.format(fen=fen, color=color)


def build_chat_prompt(fen: str) -> str:
    """Build a ChatML-wrapped prompt for Qwen-2.5-Instruct (train/inference)."""
    return f"{CHATML_USER_OPEN}{make_prompt(fen)}{CHATML_USER_CLOSE}{CHATML_ASSISTANT_OPEN}"
