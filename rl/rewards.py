#!/usr/bin/env python3
"""
Reward Functions for Chess RL Training

Implements the staged reward system:
- Stage 0: Legality only (+1 legal, -1 illegal)
- Stage 1: Win/Lose/Draw vs opponent
- Stage 2: Stockfish evaluation delta
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
        
        def get_cp(info):
            score = info["score"].relative
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return score.score()
        
        delta = get_cp(info_after) - get_cp(info_before)
        
        # Normalize with tanh
        return math.tanh(delta / 100.0)
    except Exception:
        return 0.0


def reward_format(output: str) -> float:
    """
    Bonus reward for proper output format.
    
    Args:
        output: Model's generated text
        
    Returns:
        +0.1 if contains <think> tags, 0.0 otherwise
    """
    if "<think>" in output and "</think>" in output:
        return 0.1
    return 0.0


def combined_reward(
    fen: str,
    move_uci: str,
    output: str,
    stage: int = 0,
    engine = None,
    game_result: Optional[str] = None,
    our_color: str = "white"
) -> dict:
    """
    Combined reward function for all stages.
    
    Args:
        fen: Current board position
        move_uci: Generated move in UCI format
        output: Full model output text
        stage: Training stage (0, 1, or 2)
        engine: Stockfish engine (required for stage 2)
        game_result: Game result if game ended
        our_color: Which color model is playing
        
    Returns:
        Dict with total reward and component breakdown
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
    
    total = r_legal
    
    # Stage 0: Just legality
    if stage == 0:
        return {
            "total": total,
            "components": components,
            "legal": True
        }
    
    # Stage 1+: Add outcome reward if game ended
    if game_result:
        r_outcome = reward_outcome(game_result, our_color)
        components["outcome"] = r_outcome
        total += r_outcome
    
    # Stage 2+: Add Stockfish delta
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
    
    # Format bonus (all stages)
    r_format = reward_format(output)
    if r_format > 0:
        components["format"] = r_format
        total += r_format
    
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
