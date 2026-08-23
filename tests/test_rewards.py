#!/usr/bin/env python3
"""Unit tests for reward functions."""

import unittest

import chess
import chess.engine

from chessfm.rl.rewards import (
    EngineAnalysisError,
    combined_reward,
    reward_budget,
    reward_format,
    reward_legality,
    reward_move_quality,
    reward_opponent_reply_delta,
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
    """Minimal fake Stockfish engine returning the same eval for every position.

    Also reports the first legal move of the analysed position as its PV so the
    Stage-2 reply analysis has something to work with.
    """

    def __init__(self, score_cp=50):
        self.score_cp = score_cp

    def analyse(self, board, limit):
        pv = [next(iter(board.legal_moves))] if any(board.legal_moves) else []
        return {"score": _FakeScore(self.score_cp), "pv": pv}


class _BrokenEngine:
    """Raises a genuine engine error on analyse(), like a dead Stockfish."""

    def analyse(self, board, limit):
        raise chess.engine.EngineError("engine failed")


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
        self.assertEqual(reward_format("e2e4"), -0.1)
        self.assertEqual(reward_format("I play e4"), -0.1)

    def test_empty_think_block_is_penalized(self):
        self.assertEqual(reward_format("<think></think>\ne2e4"), -0.1)

    def test_inverted_tags_are_penalized(self):
        self.assertEqual(reward_format("</think>e2e4<think>"), -0.1)

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
        # 1.0 (legal) - 0.1 (no <think> format): reasoning is rewarded, not free.
        self.assertEqual(result["total"], 1.0 - 0.1)

    def test_stage0_formatted_scores_higher_than_bare_move(self):
        bare = combined_reward(self.FEN, "e2e4", "e2e4", stage=0)
        formatted = combined_reward(
            self.FEN, "e2e4", "<think>center control and development.</think>\ne2e4", stage=0
        )
        self.assertGreater(formatted["total"], bare["total"])

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

    def test_engine_failure_propagates_never_silently_zeroed(self):
        # An engine failure must reach the caller: folding it into a 0.0
        # quality score would make it indistinguishable from a blunder.
        with self.assertRaises(EngineAnalysisError):
            combined_reward(
                self.FEN,
                "e2e4",
                "<think>strong center for white now.</think>\ne2e4",
                stage=1,
                engine=_BrokenEngine(),
            )


class _ReplyPunishesEngine:
    """Eval is fine until both sides have moved (ply >= 2), then collapses.

    Simulates a move that looks acceptable immediately but is refuted by the
    opponent's reply — exactly the case Stage 2 exists to catch.
    """

    def __init__(self, good_cp=100, bad_cp=-50):
        self.good_cp = good_cp
        self.bad_cp = bad_cp

    def analyse(self, board, limit):
        cp = self.good_cp if board.ply() < 2 else self.bad_cp
        pv = [next(iter(board.legal_moves))] if any(board.legal_moves) else []
        return {"score": _FakeScore(cp), "pv": pv}


class TestRewardMoveQuality(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_illegal_move_returns_negative(self):
        self.assertEqual(reward_move_quality(self.FEN, "e2e5", _MockEngine(), depth=1), -1.0)

    def test_best_move_scores_high(self):
        # Mock returns identical eval before/after, so loss=0 -> reward 1.0.
        self.assertEqual(reward_move_quality(self.FEN, "e2e4", _MockEngine(), depth=1), 1.0)

    def test_engine_error_raises(self):
        with self.assertRaises(EngineAnalysisError):
            reward_move_quality(self.FEN, "e2e4", _BrokenEngine(), depth=1)


class TestOpponentReplyDelta(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_flat_eval_gives_zero_delta(self):
        # Identical eval before and after the reply -> tanh(0) == 0.
        self.assertEqual(
            reward_opponent_reply_delta(self.FEN, "e2e4", _MockEngine(score_cp=50), depth=1),
            0.0,
        )

    def test_punished_reply_gives_strongly_negative_delta(self):
        # The move looks fine at +100, but after the opponent's best reply the
        # eval collapses to -50: delta = tanh((-50 - 100)/100) < -0.5.
        engine = _ReplyPunishesEngine()
        delta = reward_opponent_reply_delta(self.FEN, "e2e4", engine, depth=1)
        self.assertLess(delta, -0.5)

    def test_illegal_move_returns_negative(self):
        self.assertEqual(
            reward_opponent_reply_delta(self.FEN, "e2e5", _MockEngine(), depth=1), -1.0
        )

    def test_engine_error_raises(self):
        with self.assertRaises(EngineAnalysisError):
            reward_opponent_reply_delta(self.FEN, "e2e4", _BrokenEngine(), depth=1)


class TestDefensivePaths(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_stockfish_delta_returns_zero_on_engine_error(self):
        # Generic utility: converts engine errors to a neutral 0.0 delta.
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
        self.assertIn("opponent_reply_delta", result["components"])

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
