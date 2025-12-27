#!/usr/bin/env python3
"""
Chess Environment for RL Training

Wraps python-chess to provide a gym-like interface for RL training.
Handles game state, legal move validation, and episode management.
"""

import chess
import chess.engine
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class StepResult:
    """Result of taking an action in the environment."""
    reward: float
    done: bool
    legal: bool
    info: dict


class ChessEnv:
    """
    Chess environment for reinforcement learning.
    
    Provides a simple interface:
    - reset(): Start a new game, return FEN
    - step(move_uci): Apply move, return reward and done flag
    - get_legal_moves(): Return list of legal moves in UCI format
    """
    
    def __init__(self, stockfish_path: Optional[str] = None):
        """
        Initialize the chess environment.
        
        Args:
            stockfish_path: Path to Stockfish binary. If None, tries common paths.
        """
        self.board = chess.Board()
        self.move_count = 0
        self.max_moves = 200  # Max moves before draw
        
        # Initialize Stockfish for evaluation
        self.engine = None
        if stockfish_path:
            self._init_stockfish(stockfish_path)
        else:
            # Try common paths
            for path in ["/opt/homebrew/bin/stockfish", "/usr/bin/stockfish", 
                        "/usr/local/bin/stockfish", "stockfish"]:
                try:
                    self._init_stockfish(path)
                    break
                except:
                    continue
    
    def _init_stockfish(self, path: str):
        """Initialize Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({"Threads": 1, "Hash": 16})
        except Exception as e:
            print(f"Warning: Could not init Stockfish at {path}: {e}")
            self.engine = None
    
    def reset(self, fen: Optional[str] = None) -> str:
        """
        Reset the environment to a new game.
        
        Args:
            fen: Optional starting position. If None, uses standard starting position.
            
        Returns:
            FEN string of the starting position.
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.move_count = 0
        return self.board.fen()
    
    def step(self, move_uci: str) -> StepResult:
        """
        Apply a move to the board.
        
        Args:
            move_uci: Move in UCI format (e.g., "e2e4")
            
        Returns:
            StepResult with reward, done flag, legality, and info dict.
        """
        # Try to parse the move
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            # Invalid UCI format
            return StepResult(
                reward=-1.0,
                done=True,
                legal=False,
                info={"error": "invalid_uci", "move": move_uci}
            )
        
        # Check if move is legal
        if move not in self.board.legal_moves:
            return StepResult(
                reward=-1.0,
                done=True,
                legal=False,
                info={"error": "illegal_move", "move": move_uci}
            )
        
        # Apply the move
        self.board.push(move)
        self.move_count += 1
        
        # Check for game end
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                reward = 1.0 if self.board.turn == chess.BLACK else -0.5
            elif result == "0-1":
                reward = 1.0 if self.board.turn == chess.WHITE else -0.5
            else:  # Draw
                reward = 0.3
            return StepResult(
                reward=reward,
                done=True,
                legal=True,
                info={"result": result, "moves": self.move_count}
            )
        
        # Check max moves
        if self.move_count >= self.max_moves:
            return StepResult(
                reward=0.0,
                done=True,
                legal=True,
                info={"result": "max_moves", "moves": self.move_count}
            )
        
        # Game continues - small reward for legal move
        return StepResult(
            reward=0.1,
            done=False,
            legal=True,
            info={"moves": self.move_count}
        )
    
    def get_legal_moves(self) -> list[str]:
        """Return list of legal moves in UCI format."""
        return [move.uci() for move in self.board.legal_moves]
    
    def get_fen(self) -> str:
        """Return current board position as FEN."""
        return self.board.fen()
    
    def get_turn(self) -> str:
        """Return whose turn it is ('white' or 'black')."""
        return "white" if self.board.turn == chess.WHITE else "black"
    
    def evaluate_position(self, depth: int = 10) -> Optional[float]:
        """
        Get Stockfish evaluation of current position.
        
        Args:
            depth: Search depth for Stockfish.
            
        Returns:
            Centipawn evaluation (positive = white advantage), or None if no engine.
        """
        if self.engine is None:
            return None
        
        try:
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=depth))
            score = info["score"].relative
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            return score.score() / 100.0  # Convert centipawns to pawns
        except Exception:
            return None
    
    def close(self):
        """Clean up resources."""
        if self.engine:
            self.engine.quit()
    
    def __del__(self):
        """Destructor to clean up Stockfish."""
        self.close()


# Quick test
if __name__ == "__main__":
    env = ChessEnv()
    print(f"Starting position: {env.get_fen()}")
    print(f"Legal moves: {env.get_legal_moves()[:5]}...")
    
    # Test a legal move
    result = env.step("e2e4")
    print(f"e2e4: legal={result.legal}, reward={result.reward}")
    
    # Test an illegal move
    result = env.step("e2e4")  # Can't move pawn again
    print(f"e2e4 again: legal={result.legal}, reward={result.reward}")
    
    env.close()
    print("✅ ChessEnv works!")
