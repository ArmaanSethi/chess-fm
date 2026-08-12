#!/usr/bin/env python3
"""Unit tests for the chess environment."""

import unittest

from chessfm.rl.chess_env import ChessEnv, StepResult


class TestChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = ChessEnv()

    def tearDown(self):
        self.env.close()

    def test_reset_default(self):
        fen = self.env.reset()
        assert "rnbqkbnr/pppppppp" in fen
        assert "RNBQKBNR" in fen

    def test_reset_custom_fen(self):
        custom_fen = "8/8/8/8/4k3/8/4K3/8 w - - 0 1"
        fen = self.env.reset(custom_fen)
        assert "4k3" in fen.lower()

    def test_legal_move(self):
        self.env.reset()
        result = self.env.step("e2e4")
        assert result.legal is True
        assert result.done is False

    def test_illegal_move(self):
        self.env.reset()
        result = self.env.step("e2e5")  # Pawn can't move 3 squares
        assert result.legal is False
        assert result.done is True

    def test_invalid_uci(self):
        self.env.reset()
        result = self.env.step("xyz")
        assert result.legal is False
        assert result.done is True

    def test_get_legal_moves(self):
        self.env.reset()
        moves = self.env.get_legal_moves()
        assert len(moves) == 20  # 20 legal moves in starting position
        assert "e2e4" in moves
        assert "d2d4" in moves

    def test_get_turn(self):
        self.env.reset()
        assert self.env.get_turn() == "white"
        self.env.step("e2e4")
        assert self.env.get_turn() == "black"

    def test_game_ends_on_checkmate(self):
        self.env.reset()
        self.env.step("f2f3")
        self.env.step("e7e5")
        self.env.step("g2g4")
        res = self.env.step("d8h4")  # Fool's Mate

        self.assertTrue(res.done)
        self.assertTrue(res.legal)
        self.assertEqual(res.info.get("result"), "0-1")

    def test_context_manager_closes_engine(self):
        with ChessEnv() as env:
            env.reset()
            env.step("e2e4")
        # Engine should be released and not accessible afterwards.
        assert env.engine is None

    def test_get_fen(self):
        self.env.reset()
        assert self.env.get_fen() == self.env.board.fen()

    def test_evaluate_position_without_engine(self):
        # No Stockfish in the test environment, so the engine stays None and
        # evaluation returns None rather than raising.
        self.env.engine = None
        assert self.env.evaluate_position() is None

    def test_set_skill_level_without_engine_is_noop(self):
        # Must not raise when no engine is present.
        self.env.engine = None
        self.env.set_skill_level(5)

    def test_max_moves_terminates_game(self):
        self.env.max_moves = 1
        result = self.env.step("e2e4")
        assert result.done is True
        assert result.info["result"] == "max_moves"

    def test_named_stockfish_path_not_found(self):
        # Path doesn't exist, so _init_named() warns and returns None.
        env = ChessEnv(stockfish_path="/nonexistent/stockfish")
        assert env.engine is None
        env.close()

    def test_named_stockfish_path_takes_precedence(self):
        # An explicit path must be attempted even when a default-path engine
        # would have opened. A nonexistent path must never fall through to the
        # default search.
        env = ChessEnv(stockfish_path="/nonexistent/stockfish")
        assert env.engine is None
        env.close()


class TestStepResult(unittest.TestCase):
    def test_step_result_creation(self):
        result = StepResult(done=False, legal=True, info={"test": "value"})
        assert result.done is False
        assert result.legal is True
        assert result.info["test"] == "value"


if __name__ == "__main__":
    unittest.main()
