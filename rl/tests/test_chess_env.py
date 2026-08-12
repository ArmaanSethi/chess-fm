#!/usr/bin/env python3
"""Unit tests for chess environment."""

import unittest
from rl.chess_env import ChessEnv, StepResult

class TestChessEnv(unittest.TestCase):
    """Tests for the ChessEnv class."""
    
    def test_reset_default(self):
        """Reset with no FEN should use starting position."""
        env = ChessEnv()
        fen = env.reset()
        assert "rnbqkbnr/pppppppp" in fen
        assert "RNBQKBNR" in fen
        env.close()
    
    def test_reset_custom_fen(self):
        """Reset with custom FEN should use that position."""
        env = ChessEnv()
        custom_fen = "8/8/8/8/4k3/8/4K3/8 w - - 0 1"
        fen = env.reset(custom_fen)
        assert "4k3" in fen.lower()
        env.close()
    
    def test_legal_move(self):
        """Legal moves should return positive reward."""
        env = ChessEnv()
        env.reset()
        result = env.step("e2e4")
        assert result.legal == True
        assert result.reward > 0
        assert result.done == False
        env.close()
    
    def test_illegal_move(self):
        """Illegal moves should return negative reward and end episode."""
        env = ChessEnv()
        env.reset()
        result = env.step("e2e5")  # Pawn can't move 3 squares
        assert result.legal == False
        assert result.reward < 0
        assert result.done == True
        env.close()
    
    def test_invalid_uci(self):
        """Invalid UCI should return negative reward."""
        env = ChessEnv()
        env.reset()
        result = env.step("xyz")
        assert result.legal == False
        assert result.reward < 0
        env.close()
    
    def test_get_legal_moves(self):
        """Should return list of legal moves."""
        env = ChessEnv()
        env.reset()
        moves = env.get_legal_moves()
        assert len(moves) == 20  # 20 legal moves in starting position
        assert "e2e4" in moves
        assert "d2d4" in moves
        env.close()
    
    def test_get_turn(self):
        """Should return correct turn."""
        env = ChessEnv()
        env.reset()
        assert env.get_turn() == "white"
        env.step("e2e4")
        assert env.get_turn() == "black"
        env.close()
    
    def test_game_ends_on_checkmate(self):
        """Fool's Mate position should report game over and correct white loss."""
        env = ChessEnv()
        env.reset()
        env.step("f2f3")
        env.step("e7e5")
        env.step("g2g4")
        res = env.step("d8h4")  # Checkmate by Black

        self.assertTrue(res.done)
        self.assertTrue(res.legal)
        self.assertEqual(res.info.get("result"), "0-1")
        self.assertEqual(res.reward, 1.0)
        env.close()


class TestStepResult:
    """Tests for StepResult dataclass."""
    
    def test_step_result_creation(self):
        """StepResult should store all fields."""
        result = StepResult(
            reward=1.0,
            done=False,
            legal=True,
            info={"test": "value"}
        )
        assert result.reward == 1.0
        assert result.done == False
        assert result.legal == True
        assert result.info["test"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
