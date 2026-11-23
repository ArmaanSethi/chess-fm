#!/usr/bin/env python3
"""
Chess environment for RL training.

Wraps python-chess to provide a simple interface for move generation,
legality validation, and episode bookkeeping.

Note: the environment is deliberately reward-free. GRPO rewards are computed
by ``chessfm.rl.rewards`` from the FEN/move/output triple, and evaluation
counts win/loss/draw itself, so ``step`` only reports legality and status.
"""

from dataclasses import dataclass, field

import chess

from chessfm.config import MAX_GAME_MOVES
from chessfm.rl.engine import best_cp, close_engine, open_engine


@dataclass
class StepResult:
    """Result of taking an action in the environment."""

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
        self.max_moves = MAX_GAME_MOVES

        self.engine = None
        if stockfish_path:
            self.engine = self._init_named(stockfish_path)
        else:
            self.engine = open_engine()

    def _init_named(self, path: str):
        engine = open_engine(stockfish_path=path)
        if engine is None:
            print(f"Warning: could not init Stockfish at {path}")
        return engine

    def set_skill_level(self, level: int = 1):
        """Configure Stockfish skill level (0-20). Lower = weaker.

        Raises if the engine rejects the option: a silently-unapplied skill
        level would make evaluation results meaningless.
        """
        self.engine.configure({"Skill Level": int(level)})

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
                done=True, legal=False, info={"error": "invalid_uci", "move": move_uci}
            )

        if move not in self.board.legal_moves:
            return StepResult(
                done=True,
                legal=False,
                info={"error": "illegal_move", "move": move_uci},
            )

        self.board.push(move)
        self.move_count += 1

        if self.board.is_game_over(claim_draw=True):
            return StepResult(
                done=True,
                legal=True,
                info={"result": self.board.result(claim_draw=True), "moves": self.move_count},
            )

        if self.move_count >= self.max_moves:
            return StepResult(
                done=True,
                legal=True,
                info={"result": "max_moves", "moves": self.move_count},
            )

        return StepResult(done=False, legal=True, info={"moves": self.move_count})

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
