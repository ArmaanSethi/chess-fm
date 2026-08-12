#!/usr/bin/env python3
"""Unit tests for reward functions."""

import unittest
from rl.rewards import reward_legality, reward_outcome, reward_format, reward_budget, combined_reward


class TestRewardLegality(unittest.TestCase):
    """Tests for the legality reward function."""
    
    def test_legal_move_starting_position(self):
        """Legal moves in starting position should return +1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(reward_legality(fen, "e2e4"), 1.0)
        self.assertEqual(reward_legality(fen, "d2d4"), 1.0)
        self.assertEqual(reward_legality(fen, "g1f3"), 1.0)
    
    def test_illegal_move(self):
        """Illegal moves should return -1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(reward_legality(fen, "e2e5"), -1.0)
        self.assertEqual(reward_legality(fen, "e1e2"), -1.0)
        self.assertEqual(reward_legality(fen, "a1a2"), -1.0)
    
    def test_invalid_uci(self):
        """Invalid UCI format should return -1."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(reward_legality(fen, "xyz"), -1.0)
        self.assertEqual(reward_legality(fen, ""), -1.0)
        self.assertEqual(reward_legality(fen, "ee44"), -1.0)


class TestRewardOutcome(unittest.TestCase):
    """Tests for the outcome reward function."""
    
    def test_win_as_white(self):
        self.assertEqual(reward_outcome("1-0", "white"), 1.0)
    
    def test_win_as_black(self):
        self.assertEqual(reward_outcome("0-1", "black"), 1.0)
    
    def test_lose(self):
        self.assertEqual(reward_outcome("1-0", "black"), -0.5)
        self.assertEqual(reward_outcome("0-1", "white"), -0.5)
    
    def test_draw(self):
        self.assertEqual(reward_outcome("1/2-1/2", "white"), 0.3)
        self.assertEqual(reward_outcome("1/2-1/2", "black"), 0.3)


class TestRewardFormat(unittest.TestCase):
    """Tests for the format & budget reward functions."""
    
    def test_with_think_tags(self):
        output = "<think>I should play e4 because it controls the center.</think>\ne2e4"
        self.assertEqual(reward_format(output), 0.1)
        self.assertEqual(reward_budget(output), 0.0)
    
    def test_without_think_tags(self):
        self.assertEqual(reward_format("e2e4"), 0.0)
        self.assertEqual(reward_format("I play e4"), 0.0)


class TestCombinedReward(unittest.TestCase):
    """Tests for the combined reward function."""
    
    def test_stage0_legal(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = combined_reward(fen, "e2e4", "e2e4", stage=0)
        self.assertTrue(result["legal"])
        self.assertEqual(result["total"], 1.0)
    
    def test_stage0_illegal(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = combined_reward(fen, "e2e5", "e2e5", stage=0)
        self.assertFalse(result["legal"])
        self.assertEqual(result["total"], -1.0)


if __name__ == "__main__":
    unittest.main()
