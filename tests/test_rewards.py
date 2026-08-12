#!/usr/bin/env python3
"""Unit tests for reward functions."""

import unittest

from chessfm.rl.rewards import (
    combined_reward,
    reward_budget,
    reward_format,
    reward_legality,
    reward_move_quality,
    reward_outcome,
    reward_stockfish_delta,
)


class _FakeScore:
    """Minimal stand-in for a Stockfish score object."""

    def __init__(self, cp):
        self._cp = cp

    def is_mate(self):
        return False

    def white(self):
        return self

    def mate(self):
        return None

    def score(self):
        return self._cp


class _MockEngine:
    """Minimal fake Stockfish engine returning the same eval for every position."""

    def __init__(self, score_cp=50):
        self.score_cp = score_cp

    def analyse(self, board, limit):
        class _Info:
            def __init__(self, cp):
                self.score = _FakeScore(cp)

        return {"score": _Info(self.score_cp).score}


class TestRewardLegality(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_legal_move_starting_position(self):
        self.assertEqual(reward_legality(self.FEN, "e2e4"), 1.0)
        self.assertEqual(reward_legality(self.FEN, "d2d4"), 1.0)
        self.assertEqual(reward_legality(self.FEN, "g1f3"), 1.0)

    def test_illegal_move(self):
        self.assertEqual(reward_legality(self.FEN, "e2e5"), -1.0)
        self.assertEqual(reward_legality(self.FEN, "e1e2"), -1.0)
        self.assertEqual(reward_legality(self.FEN, "a1a2"), -1.0)

    def test_invalid_uci(self):
        self.assertEqual(reward_legality(self.FEN, "xyz"), -1.0)
        self.assertEqual(reward_legality(self.FEN, ""), -1.0)
        self.assertEqual(reward_legality(self.FEN, "ee44"), -1.0)


class TestRewardOutcome(unittest.TestCase):
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
    def test_with_think_tags(self):
        output = "<think>I should play e4 because it controls the center.</think>\ne2e4"
        self.assertEqual(reward_format(output), 0.1)
        self.assertEqual(reward_budget(output), 0.0)

    def test_without_think_tags(self):
        self.assertEqual(reward_format("e2e4"), 0.0)
        self.assertEqual(reward_format("I play e4"), 0.0)

    def test_budget_too_short(self):
        self.assertEqual(reward_budget("<think>short</think>\ne2e4"), -0.1)

    def test_budget_too_long(self):
        long_text = "word " * 300
        self.assertEqual(reward_budget(f"<think>{long_text}</think>\ne2e4"), -0.05)


class TestCombinedReward(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_stage0_legal(self):
        result = combined_reward(self.FEN, "e2e4", "e2e4", stage=0)
        self.assertTrue(result["legal"])
        self.assertEqual(result["total"], 1.0)

    def test_stage0_illegal(self):
        result = combined_reward(self.FEN, "e2e5", "e2e5", stage=0)
        self.assertFalse(result["legal"])
        self.assertEqual(result["total"], -1.0)

    def test_stage1_includes_move_quality(self):
        engine = _MockEngine()
        output = "<think>e4 controls the center and frees the bishop.</think>\ne2e4"
        result = combined_reward(self.FEN, "e2e4", output, stage=1, engine=engine)
        self.assertTrue(result["legal"])
        self.assertIn("move_quality", result["components"])
        # 1.0 (legal) + 0.1 (format) + 1.0 (quality), no budget penalty (>= 5 words).
        self.assertEqual(result["total"], 1.0 + 0.1 + 1.0)

    def test_stage1_without_engine_raises(self):
        with self.assertRaises(ValueError):
            combined_reward(self.FEN, "e2e4", "e2e4", stage=1, engine=None)

    def test_stage2_without_engine_raises(self):
        with self.assertRaises(ValueError):
            combined_reward(self.FEN, "e2e4", "e2e4", stage=2, engine=None)


class TestRewardMoveQuality(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_illegal_move_returns_negative(self):
        self.assertEqual(reward_move_quality(self.FEN, "e2e5", _MockEngine(), depth=1), -1.0)

    def test_best_move_scores_high(self):
        # Mock returns identical eval before/after, so loss=0 -> reward 1.0.
        self.assertEqual(reward_move_quality(self.FEN, "e2e4", _MockEngine(), depth=1), 1.0)


class _BrokenEngine:
    """Raises on analyse(), exercising reward functions' defensive None paths."""

    def analyse(self, board, limit):
        raise RuntimeError("engine failed")


class TestDefensivePaths(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_move_quality_returns_zero_on_engine_error(self):
        self.assertEqual(reward_move_quality(self.FEN, "e2e4", _BrokenEngine(), depth=1), 0.0)

    def test_stockfish_delta_returns_zero_on_engine_error(self):
        self.assertEqual(reward_stockfish_delta(self.FEN, self.FEN, _BrokenEngine(), depth=1), 0.0)

    def test_stockfish_delta_improvement_is_positive(self):
        # Same eval before/after -> tanh(0) == 0.0.
        self.assertEqual(reward_stockfish_delta(self.FEN, self.FEN, _MockEngine(), depth=1), 0.0)

    def test_combined_stage2_includes_eval_delta(self):
        result = combined_reward(
            self.FEN,
            "e2e4",
            "<think>strong center for white now.</think>\ne2e4",
            stage=2,
            engine=_MockEngine(),
        )
        self.assertIn("eval_delta", result["components"])

    def test_combined_includes_outcome(self):
        result = combined_reward(
            self.FEN, "e2e4", "e2e4", stage=0, game_result="1-0", our_color="white"
        )
        self.assertIn("outcome", result["components"])
        self.assertEqual(result["components"]["outcome"], 1.0)

    def test_combined_stage0_with_budget_penalty(self):
        result = combined_reward(self.FEN, "e2e4", "<think>too short</think>\ne2e4", stage=0)
        self.assertIn("budget", result["components"])
        self.assertAlmostEqual(result["total"], 1.0 + 0.1 - 0.1)


if __name__ == "__main__":
    unittest.main()
