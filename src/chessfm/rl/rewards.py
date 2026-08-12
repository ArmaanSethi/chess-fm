#!/usr/bin/env python3
"""
Reward functions for Chess RL training.

Implements the staged reward system:

- Stage 0: Legality only (+1 legal, -1 illegal) + format/budget bonuses
- Stage 1: + Move quality vs Stockfish (chosen move's centipawn loss)
- Stage 2: + Stockfish evaluation delta (dense per-move signal)

Design notes
------------

Stage 1 and Stage 2 measure complementary things:
  - Stage 1 move quality is *relative*: how close the chosen move is to
    Stockfish's best move for the position.
  - Stage 2 eval delta is *absolute*: whether the resulting position actually
    improved the mover's evaluation.

Each stage's total reward is normalized to a comparable magnitude so training
across stages stays on the same scale (see ``combined_reward``).

Win/lose outcome rewards are not used in GRPO here: training runs on single
positions (one completion per FEN), not full rollouts, so per-completion
outcomes cannot be computed. ``reward_outcome`` is kept for full-game
evaluation (see ``chessfm.rl.evaluate``).
"""

import math

import chess
import chess.engine

from chessfm.rl.engine import best_cp, eval_after_move

# Component weights, one row per stage (legality, format, budget, quality, delta).
# Minimum/maximum reasoning word counts for the budget reward. These are WORD
# counts, not token counts (see reward_budget).
MIN_THINK_WORDS = 5
MAX_THINK_WORDS = 250

_ILLEGAL_REWARD = -1.0


def reward_legality(fen: str, move_uci: str) -> float:
    """Stage 0 reward: pure legality check (+1 legal, -1 illegal/invalid)."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return 1.0
        return _ILLEGAL_REWARD
    except Exception:
        return _ILLEGAL_REWARD


def reward_outcome(result: str, our_color: str) -> float:
    """Game-outcome reward (+1 win, +0.3 draw, -0.5 loss).

    Only used by full-game evaluation, not GRPO.
    """
    if result == "1-0":
        return 1.0 if our_color == "white" else -0.5
    if result == "0-1":
        return 1.0 if our_color == "black" else -0.5
    return 0.3


def reward_move_quality(
    fen: str,
    move_uci: str,
    engine,
    depth: int = 10,
    max_loss: float = 200.0,
) -> float:
    """How close the chosen move is to Stockfish's best (in [0, 1], or -1 illegal).

    ``max_loss`` is the centipawn threshold at which reward reaches ~0.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return _ILLEGAL_REWARD

        best = best_cp(engine, board, depth)
        chosen = eval_after_move(engine, board, move_uci, depth)
        if best is None or chosen is None:
            return 0.0

        loss = max(0.0, best - chosen)
        return max(0.0, 1.0 - (loss / max_loss))
    except Exception:
        return 0.0


def reward_stockfish_delta(
    fen_before: str,
    fen_after: str,
    engine,
    depth: int = 10,
) -> float:
    """Stockfish-eval improvement from ``fen_before`` to ``fen_after``.

    Returns tanh(delta / 100) normalized to [-1, 1]. Both evaluations are
    computed from the mover's perspective so a move that improves the mover's
    position is positive regardless of color.
    """
    try:
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        mover = board_before.turn

        before = best_cp(engine, board_before, depth)
        after = best_cp(engine, board_after, depth, for_color=mover)
        if before is None or after is None:
            return 0.0

        return math.tanh(after - before)
    except Exception:
        return 0.0


def reward_budget(output: str) -> float:
    """Penalty for reasoning that is too short or too long.

    The bounds are expressed in *word counts* (see MIN_THINK_WORDS and
    MAX_THINK_WORDS); they are a rough proxy for verbosity, not token counts.
    """
    if "<think>" not in output or "</think>" not in output:
        return 0.0
    think_text = output.split("<think>")[1].split("</think>")[0].strip()
    num_words = len(think_text.split())
    if num_words < MIN_THINK_WORDS:
        return -0.1
    if num_words > MAX_THINK_WORDS:
        return -0.05
    return 0.0


def reward_format(output: str) -> float:
    """Bonus for producing the <think>...</think> format."""
    if "<think>" in output and "</think>" in output:
        start_idx = output.find("<think>")
        end_idx = output.find("</think>")
        if start_idx < end_idx and len(output[start_idx + 7 : end_idx].strip()) > 0:
            return 0.1
    return 0.0


def combined_reward(
    fen: str,
    move_uci: str,
    output: str,
    stage: int = 0,
    engine=None,
    game_result: str | None = None,
    our_color: str = "white",
    move_quality_depth: int = 10,
) -> dict:
    """Combined reward across all stages.

    Each component already carries its intended magnitude (format +0.1,
    budget -0.1/0, quality [0, 1], delta tanh), so the total is a plain sum.
    Per-stage ranges:

    - Stage 0: [-1.1, 1.2]  (legality + format + budget)
    - Stage 1: [-1.1, 2.2]  (+ move quality)
    - Stage 2: [-1.1, 3.2]  (+ eval delta)

    GRPO normalizes rewards within a group before applying the policy update,
    so the differing absolute ranges across stages are not a correctness issue;
    they are documented here to keep the reward design explicit.

    - Illegal moves short-circuit to ``_ILLEGAL_REWARD``.
    - Stages >= 1 require an engine; if missing the run is a config error, not
      a silent downgrade.
    """
    components = {}

    r_legal = reward_legality(fen, move_uci)
    components["legality"] = r_legal
    if r_legal < 0:
        return {"total": r_legal, "components": components, "legal": False}

    if stage >= 1 and engine is None:
        raise ValueError(
            f"Stage {stage} reward requires a Stockfish engine, but engine=None. "
            "This is a configuration error, not a runnable state."
        )

    r_format = reward_format(output)
    r_budget = reward_budget(output)
    if r_format:
        components["format"] = r_format
    if r_budget:
        components["budget"] = r_budget

    total = r_legal + r_format + r_budget

    if stage >= 1 and engine:
        r_quality = reward_move_quality(fen, move_uci, engine, depth=move_quality_depth)
        components["move_quality"] = r_quality
        total += r_quality

    if game_result:
        r_outcome = reward_outcome(game_result, our_color)
        components["outcome"] = r_outcome
        total += r_outcome

    if stage >= 2 and engine:
        try:
            board = chess.Board(fen)
            fen_after = board.copy()
            fen_after.push(chess.Move.from_uci(move_uci))
            r_delta = reward_stockfish_delta(fen, fen_after.fen(), engine)
            components["eval_delta"] = r_delta
            total += r_delta
        except Exception:
            pass

    return {"total": total, "components": components, "legal": True}


if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print("Legality:", reward_legality(fen, "e2e4"), reward_legality(fen, "e2e5"))
