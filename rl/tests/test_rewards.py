#!/usr/bin/env python3
"""Unit tests for reward functions."""

import unittest
from rl.rewards import (
    reward_legality, reward_outcome, reward_format, reward_budget,
    combined_reward, reward_move_quality,
)


class _FakeScore:
    """Minimal stand-in for a Stockfish score object."""
    def __init__(self, cp):
        self._cp = cp

    def is_mate(self):
        return False

    def score(self):
        return self._cp


class _MockEngine:
    """Minimal fake Stockfish engine. Returns the same score for every position."""
    def __init__(self, score_cp=50):
        self.score_cp = score_cp

    def analyse(self, board, limit):
        cp = self.score_cp

        class _Score:
            def white(self):
                return _FakeScore(cp)

        return {"score": _Score()}


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


class TestRewardMoveQuality(unittest.TestCase):
    """Tests for the Stockfish move-quality reward."""

    STARTING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_illegal_move_returns_negative(self):
        self.assertEqual(
            reward_move_quality(self.STARTING, "e2e5", _MockEngine(), depth=1),
            -1.0,
        )

    def test_best_move_scores_high(self):
        # Mock returns identical eval before/after, so loss=0 -> reward 1.0
        self.assertEqual(
            reward_move_quality(self.STARTING, "e2e4", _MockEngine(), depth=1),
            1.0,
        )

    def test_stage1_includes_move_quality(self):
        engine = _MockEngine()
        output = "<think>e4 controls the center and frees the bishop.</think>\ne2e4"
        result = combined_reward(self.STARTING, "e2e4", output,
                                 stage=1, engine=engine)
        self.assertTrue(result["legal"])
        self.assertIn("move_quality", result["components"])
        self.assertEqual(result["total"], 1.0 + 0.1 + 1.0)


if __name__ == "__main__":
    unittest.main()
