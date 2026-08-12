#!/usr/bin/env python3
"""
Chess environment for RL training.

Wraps python-chess to provide a simple interface for move generation,
legality validation, and episode bookkeeping.
"""

from dataclasses import dataclass, field

import chess

from chessfm.rl.engine import best_cp, close_engine, open_engine


@dataclass
class StepResult:
    """Result of taking an action in the environment."""

    reward: float
    done: bool
    legal: bool
    info: dict = field(default_factory=dict)


class ChessEnv:
    """A single chess game with optional Stockfish evaluation.

    Usage as a context manager ensures the Stockfish engine is always closed
    (avoids relying on ``__del__``, which is nondeterministic):

        with ChessEnv() as env:
            env.reset()
            env.step("e2e4")
    """

    def __init__(self, stockfish_path: str | None = None):
        self.board = chess.Board()
        self.move_count = 0
        self.max_moves = 200

        self.engine = None
        if stockfish_path:
            self.engine = open_engine()
            if self.engine is None:
                self.engine = self._init_named(stockfish_path)
        else:
            self.engine = open_engine()

    def _init_named(self, path: str):
        import chess.engine

        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": 1, "Hash": 16})
            return engine
        except Exception as e:
            print(f"Warning: could not init Stockfish at {path}: {e}")
            return None

    def set_skill_level(self, level: int = 1):
        """Configure Stockfish skill level (0-20). Lower = weaker."""
        if self.engine is not None:
            try:
                self.engine.configure({"Skill Level": int(level)})
            except Exception:
                pass

    def reset(self, fen: str | None = None) -> str:
        """Reset to ``fen`` (or the starting position) and return the FEN."""
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_count = 0
        return self.board.fen()

    def step(self, move_uci: str) -> StepResult:
        """Apply ``move_uci`` and return a StepResult."""
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return StepResult(
                reward=-1.0, done=True, legal=False, info={"error": "invalid_uci", "move": move_uci}
            )

        if move not in self.board.legal_moves:
            return StepResult(
                reward=-1.0,
                done=True,
                legal=False,
                info={"error": "illegal_move", "move": move_uci},
            )

        mover = self.board.turn
        self.board.push(move)
        self.move_count += 1

        if self.board.is_game_over():
            result = self.board.result()
            reward = game_result_reward(result, mover)
            return StepResult(
                reward=reward,
                done=True,
                legal=True,
                info={"result": result, "moves": self.move_count},
            )

        if self.move_count >= self.max_moves:
            return StepResult(
                reward=0.0,
                done=True,
                legal=True,
                info={"result": "max_moves", "moves": self.move_count},
            )

        return StepResult(reward=0.1, done=False, legal=True, info={"moves": self.move_count})

    def get_legal_moves(self) -> list:
        return [move.uci() for move in self.board.legal_moves]

    def get_fen(self) -> str:
        return self.board.fen()

    def get_turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def evaluate_position(self, depth: int = 10) -> float | None:
        """Stockfish centipawn evaluation (positive = white advantage), or None."""
        return best_cp(self.engine, self.board, depth)

    def close(self):
        """Release the Stockfish engine."""
        close_engine(self.engine)
        self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def game_result_reward(result: str, side_just_moved: chess.Color) -> float:
    """Reward for a completed game from the side that just moved's perspective."""
    if result == "1-0":
        return 1.0 if side_just_moved == chess.WHITE else -0.5
    if result == "0-1":
        return 1.0 if side_just_moved == chess.BLACK else -0.5
    return 0.3


if __name__ == "__main__":
    with ChessEnv() as env:
        print(f"Starting position: {env.get_fen()}")
        print(f"Legal moves: {env.get_legal_moves()[:5]}...")
        result = env.step("e2e4")
        print(f"e2e4: legal={result.legal}, reward={result.reward}")
