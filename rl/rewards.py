#!/usr/bin/env python3
"""
Reward Functions for Chess RL Training

Implements the staged reward system:
- Stage 0: Legality only (+1 legal, -1 illegal) + format/budget bonuses
- Stage 1: + Move quality vs Stockfish (chosen move's centipawn loss)
- Stage 2: + Stockfish evaluation delta (dense per-move signal)

DESIGN NOTE on Stage 1 + Stage 2: the two engine-derived rewards measure
complementary things. Stage 1's move quality is *relative*: it scores how
close the chosen move is to Stockfish's best move for the position. Stage 2's
eval delta is *absolute*: it scores whether the resulting position actually
improved the mover's evaluation. Keeping both gives the model (a) a dense
gradient toward the engine's preferred candidate, and (b) a signal toward
objectively good resulting positions. They are intentionally not ablated
against each other; the weights are empirical and can be tuned via
``move_quality_depth`` / the max-loss threshold.

NOTE on "win/lose": GRPO in this repo trains on single positions (one
completion per FEN), not full game rollouts, so outcome-based rewards
(win/draw/loss) cannot be computed per-completion. Instead Stage 1 uses
move quality (how close the chosen move is to Stockfish's best), which is a
dense, verifiable signal. reward_outcome() is kept for full-game evaluation
(see rl/evaluate.py).
"""

import chess
from typing import Optional
import math


def reward_legality(fen: str, move_uci: str) -> float:
    """
    Stage 0 reward: Pure legality check.
    
    Args:
        fen: Board position in FEN format
        move_uci: Move in UCI format (e.g., "e2e4")
        
    Returns:
        +1.0 for legal move, -1.0 for illegal move
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return 1.0
        return -1.0
    except Exception:
        return -1.0


def reward_outcome(result: str, our_color: str) -> float:
    """
    Stage 1 reward: Game outcome.
    
    Args:
        result: Game result ("1-0", "0-1", "1/2-1/2")
        our_color: Which color we're playing ("white" or "black")
        
    Returns:
        +1.0 for win, +0.3 for draw, -0.5 for loss
    """
    if result == "1-0":
        return 1.0 if our_color == "white" else -0.5
    elif result == "0-1":
        return 1.0 if our_color == "black" else -0.5
    else:  # Draw
        return 0.3


def reward_stockfish_delta(
    fen_before: str, 
    fen_after: str,
    engine,
    depth: int = 10
) -> float:
    """
    Stage 2 reward: Stockfish evaluation improvement.
    
    Args:
        fen_before: Position before move
        fen_after: Position after move
        engine: Stockfish engine instance
        depth: Search depth
        
    Returns:
        tanh(delta_eval / 100) to normalize to [-1, 1]
    """
    try:
        board_before = chess.Board(fen_before)
        board_after = chess.Board(fen_after)
        
        info_before = engine.analyse(board_before, chess.engine.Limit(depth=depth))
        info_after = engine.analyse(board_after, chess.engine.Limit(depth=depth))
        
        def get_cp_from_white(info):
            score = info["score"].white()  # Absolute evaluation from White's perspective
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return score.score()
        
        delta = get_cp_from_white(info_after) - get_cp_from_white(info_before)
        if board_before.turn == chess.BLACK:
            delta = -delta  # Black wants to decrease White's advantage
        
        # Normalize with tanh
        return math.tanh(delta / 100.0)
    except Exception:
        return 0.0


def reward_move_quality(
    fen: str,
    move_uci: str,
    engine,
    depth: int = 10,
    max_loss: float = 200.0,
) -> float:
    """
    Stage 1 reward: how close the chosen move is to Stockfish's best.

    Computes the centipawn loss of the chosen move relative to Stockfish's
    recommended move for the position. Reward is 1.0 for the best move and
    decays to ~0.0 for moves that throw away more than ``max_loss`` centipawns.

    Args:
        fen: Board position in FEN format
        move_uci: Move in UCI format (e.g., "e2e4")
        engine: Stockfish engine instance (chess.engine.SimpleEngine)
        depth: Search depth for Stockfish
        max_loss: Centipawn threshold at which reward reaches ~0

    Returns:
        Float in [0, 1] (or -1.0 for illegal moves).
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return -1.0

        mover = board.turn
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_cp = _cp_from_white(info, board.turn)

        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
        chosen_cp = _cp_from_white(info_after, mover)

        # Loss is the eval (from the mover's perspective) that the chosen
        # move gives up relative to the best move.
        loss = max(0.0, best_cp - chosen_cp)
        return max(0.0, 1.0 - (loss / max_loss))
    except Exception:
        return 0.0


def _cp_from_white(info, for_color: chess.Color) -> float:
    """Return Stockfish score in centipawns from ``for_color``'s perspective."""
    score = info["score"].white()
    cp = 10000.0 if score.is_mate() and score.mate() > 0 else -10000.0
    if not score.is_mate():
        cp = float(score.score())
    return cp if for_color == chess.WHITE else -cp


def reward_budget(output: str) -> float:
    """
    Reward penalty for too-short or too-long reasoning.
    """
    if "<think>" in output and "</think>" in output:
        think_text = output.split("<think>")[1].split("</think>")[0].strip()
        num_tokens = len(think_text.split())
        if num_tokens < 5:
            return -0.1  # Too short/trivial
        elif num_tokens > 250:
            return -0.05  # Excessive rambling
    return 0.0


def reward_format(output: str) -> float:
    """
    Bonus reward for proper output format.
    """
    if "<think>" in output and "</think>" in output:
        start_idx = output.find("<think>")
        end_idx = output.find("</think>")
        if start_idx < end_idx:
            think_content = output[start_idx + 7:end_idx].strip()
            if len(think_content) > 0:
                return 0.1
    return 0.0


def combined_reward(
    fen: str,
    move_uci: str,
    output: str,
    stage: int = 0,
    engine = None,
    game_result: Optional[str] = None,
    our_color: str = "white",
    move_quality_depth: int = 10,
) -> dict:
    """
    Combined reward function for all stages.

    Stages:
        0 - Legality + format/budget
        1 - Stage 0 + Stockfish move quality
        2 - Stage 1 + Stockfish eval delta

    ``game_result``/``our_color`` are optional and only used for full-game
    evaluation (see rl/evaluate.py); they are not used during GRPO training.
    """
    components = {}

    # Always check legality
    r_legal = reward_legality(fen, move_uci)
    components["legality"] = r_legal

    if r_legal < 0:
        # Illegal move - no other rewards
        return {
            "total": r_legal,
            "components": components,
            "legal": False
        }

    # Stages 1/2 require Stockfish; never silently degrade to Stage 0.
    if stage >= 1 and engine is None:
        raise ValueError(
            f"Stage {stage} reward requires a Stockfish engine, but engine=None. "
            "This is a configuration error, not a runnable state."
        )

    total = r_legal

    # Stage 0: Just legality + format/budget
    r_format = reward_format(output)
    r_budget = reward_budget(output)
    if r_format > 0:
        components["format"] = r_format
        total += r_format
    if r_budget != 0:
        components["budget"] = r_budget
        total += r_budget

    # Stage 1+: Add Stockfish move quality (dense, per-position signal)
    if stage >= 1 and engine:
        r_quality = reward_move_quality(fen, move_uci, engine, depth=move_quality_depth)
        components["move_quality"] = r_quality
        total += r_quality

    # Optional outcome reward (used by full-game evaluation, not GRPO)
    if game_result:
        r_outcome = reward_outcome(game_result, our_color)
        components["outcome"] = r_outcome
        total += r_outcome

    # Stage 2+: Add Stockfish eval delta
    if stage >= 2 and engine:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            fen_after = board.fen()

            r_delta = reward_stockfish_delta(fen, fen_after, engine)
            components["eval_delta"] = r_delta
            total += r_delta
        except Exception:
            pass

    return {
        "total": total,
        "components": components,
        "legal": True
    }


# Quick test
if __name__ == "__main__":
    # Test legality reward
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print("Testing reward functions:")
    print(f"  e2e4 (legal): {reward_legality(fen, 'e2e4')}")
    print(f"  e2e5 (illegal): {reward_legality(fen, 'e2e5')}")
    print(f"  xyz (invalid): {reward_legality(fen, 'xyz')}")
    
    # Test outcome reward
    print(f"  Win as white: {reward_outcome('1-0', 'white')}")
    print(f"  Win as black: {reward_outcome('0-1', 'black')}")
    print(f"  Draw: {reward_outcome('1/2-1/2', 'white')}")
    
    # Test format reward
    print(f"  With <think>: {reward_format('<think>test</think> e4')}")
    print(f"  Without: {reward_format('e4')}")
    
    print("✅ Rewards work!")
