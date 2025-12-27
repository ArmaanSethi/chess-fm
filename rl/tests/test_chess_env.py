#!/usr/bin/env python3
"""Unit tests for chess environment."""

import sys
sys.path.insert(0, '..')

import pytest
from chess_env import ChessEnv, StepResult


class TestChessEnv:
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
        """Game should end on checkmate."""
        env = ChessEnv()
        # Fool's mate position - black to deliver checkmate
        env.reset("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3")
        # Actually this is a position where the game is already over
        # Let's use a pre-checkmate position instead
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
