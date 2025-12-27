#!/usr/bin/env python3
"""Unit tests for reward functions."""

import sys
sys.path.insert(0, '..')

import pytest
from rewards import reward_legality, reward_outcome, reward_format, combined_reward


class TestRewardLegality:
    """Tests for the legality reward function."""
    
    def test_legal_move_starting_position(self):
        """Legal moves in starting position should return +1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert reward_legality(fen, "e2e4") == 1.0
        assert reward_legality(fen, "d2d4") == 1.0
        assert reward_legality(fen, "g1f3") == 1.0
    
    def test_illegal_move(self):
        """Illegal moves should return -1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert reward_legality(fen, "e2e5") == -1.0  # Pawn can't go 3 squares
        assert reward_legality(fen, "e1e2") == -1.0  # King blocked by pawn
        assert reward_legality(fen, "a1a2") == -1.0  # Rook blocked by pawn
    
    def test_invalid_uci(self):
        """Invalid UCI format should return -1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert reward_legality(fen, "xyz") == -1.0
        assert reward_legality(fen, "") == -1.0
        assert reward_legality(fen, "ee44") == -1.0
    
    def test_capturing_move(self):
        """Capturing moves should be legal if valid."""
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        assert reward_legality(fen, "e4d5") == 1.0  # Pawn capture
    
    def test_castling(self):
        """Castling should be legal when allowed."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        assert reward_legality(fen, "e1g1") == 1.0  # Kingside castle
        assert reward_legality(fen, "e1c1") == 1.0  # Queenside castle


class TestRewardOutcome:
    """Tests for the outcome reward function."""
    
    def test_win_as_white(self):
        """Winning as white should return +1."""
        assert reward_outcome("1-0", "white") == 1.0
    
    def test_win_as_black(self):
        """Winning as black should return +1."""
        assert reward_outcome("0-1", "black") == 1.0
    
    def test_lose(self):
        """Losing should return -0.5."""
        assert reward_outcome("1-0", "black") == -0.5
        assert reward_outcome("0-1", "white") == -0.5
    
    def test_draw(self):
        """Draw should return +0.3."""
        assert reward_outcome("1/2-1/2", "white") == 0.3
        assert reward_outcome("1/2-1/2", "black") == 0.3


class TestRewardFormat:
    """Tests for the format reward function."""
    
    def test_with_think_tags(self):
        """Output with <think> tags should get bonus."""
        output = "<think>I should play e4</think>\ne2e4"
        assert reward_format(output) == 0.1
    
    def test_without_think_tags(self):
        """Output without <think> tags should get no bonus."""
        assert reward_format("e2e4") == 0.0
        assert reward_format("I play e4") == 0.0
    
    def test_partial_tags(self):
        """Partial tags should get no bonus."""
        assert reward_format("<think>reasoning") == 0.0
        assert reward_format("reasoning</think>") == 0.0


class TestCombinedReward:
    """Tests for the combined reward function."""
    
    def test_stage0_legal(self):
        """Stage 0: Legal move should get +1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = combined_reward(fen, "e2e4", "e2e4", stage=0)
        assert result["legal"] == True
        assert result["total"] == 1.0
    
    def test_stage0_illegal(self):
        """Stage 0: Illegal move should get -1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = combined_reward(fen, "e2e5", "e2e5", stage=0)
        assert result["legal"] == False
        assert result["total"] == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
