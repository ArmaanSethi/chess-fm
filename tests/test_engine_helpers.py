#!/usr/bin/env python3
"""Unit tests for the shared Stockfish analysis helpers."""

import chess
import pytest

from chessfm.rl.engine import best_cp, close_engine, eval_after_move, open_engine


class _FakeScore:
    def __init__(self, cp, mate=None):
        self._cp = cp
        self._mate = mate

    def is_mate(self):
        return self._mate is not None

    def mate(self):
        return self._mate

    def white(self):
        return self

    def score(self):
        return self._cp


class _MockEngine:
    """Returns a fixed white-centric eval; the helper handles the flip."""

    def __init__(self, white_cp=50):
        self.white_cp = white_cp

    def analyse(self, board, limit):
        return {"score": _FakeScore(self.white_cp)}


class _MockEngineWithMate:
    """Reports a forced mate for White."""

    def analyse(self, board, limit):
        return {"score": _FakeScore(0, mate=2)}


def test_best_cp_white_perspective():
    board = chess.Board()  # white to move
    assert best_cp(_MockEngine(white_cp=50), board) == 50.0


def test_best_cp_black_perspective_is_flipped():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = chess.Board(fen)  # black to move
    assert best_cp(_MockEngine(white_cp=50), board) == -50.0


def test_best_cp_for_color_override():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = chess.Board(fen)
    # Ask for white's perspective explicitly despite black to move.
    assert best_cp(_MockEngine(white_cp=50), board, for_color=chess.WHITE) == 50.0


def test_eval_after_move_is_from_mover_perspective():
    board = chess.Board()  # white to move
    # After e2e4 it's black's turn, but the eval must stay from white's view.
    assert eval_after_move(_MockEngine(white_cp=50), board, "e2e4") == 50.0


def test_eval_after_move_illegal_move_returns_none():
    board = chess.Board()
    assert eval_after_move(_MockEngine(white_cp=50), board, "e2e5") is None


def test_best_cp_none_when_engine_missing():
    assert best_cp(None, chess.Board()) is None
    assert eval_after_move(None, chess.Board(), "e2e4") is None


def test_best_cp_positive_mate_uses_10000():
    assert best_cp(_MockEngineWithMate(), chess.Board()) == 10000.0


def test_best_cp_negative_mate_from_black_perspective():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert best_cp(_MockEngineWithMate(), chess.Board(fen)) == -10000.0


class _FailingEngine:
    def analyse(self, board, limit):
        # A genuine engine error (not a code bug) must map to None.
        raise chess.engine.EngineError("engine crashed")


def test_best_cp_returns_none_on_engine_error():
    assert best_cp(_FailingEngine(), chess.Board()) is None


def test_best_cp_does_not_mask_code_bugs():
    """A non-engine exception (a bug) must propagate, not become None."""

    class _BuggyEngine:
        def analyse(self, board, limit):
            raise RuntimeError("this is a bug, not an engine error")

    with pytest.raises(RuntimeError):
        best_cp(_BuggyEngine(), chess.Board())


def test_eval_after_move_returns_none_on_engine_error():
    assert eval_after_move(_FailingEngine(), chess.Board(), "e2e4") is None


class _NeverQuits:
    def quit(self):
        raise chess.engine.EngineError("refusing to quit")


def test_close_engine_handles_failure():
    close_engine(_NeverQuits())  # must not raise
    close_engine(None)  # must not raise
    close_engine(object())


def test_open_engine_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr("chessfm.rl.engine.STOCKFISH_PATHS", ["/nonexistent/stockfish"])
    assert open_engine() is None
