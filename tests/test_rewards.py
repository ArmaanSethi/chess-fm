#!/usr/bin/env python3
"""Unit tests for reward functions."""

import math
import unittest
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

from chessfm.rl.rewards import (
    EngineAnalysisError,
    StockfishEvalCache,
    combined_reward,
    cp_to_expected_score,
    reward_budget,
    reward_format,
    reward_legality,
    reward_move_quality,
    reward_opponent_reply_delta,
    reward_outcome,
    reward_stockfish_delta,
    wdl_loss_reward,
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

    def test_stage2_post_reply_eval_failure_raises_not_zeroed(self):
        # Succeed on calls 1 (best_before), 2 (eval_after), 3 (best_move), then fail on call 4
        # (best_cp after reply). Must raise EngineAnalysisError instead of returning delta=0.0.
        class _FailOnNthAnalyse:
            def __init__(self, fail_on: int):
                self.calls = 0
                self.fail_on = fail_on

            def analyse(self, board, limit):
                self.calls += 1
                if self.calls == self.fail_on:
                    raise chess.engine.EngineError("engine crashed on reply eval")
                pv = [next(iter(board.legal_moves))] if any(board.legal_moves) else []
                return {"score": _FakeScore(50), "pv": pv}

        with self.assertRaises(EngineAnalysisError):
            combined_reward(
                self.FEN,
                "e2e4",
                "<think>strong center for white now.</think>\ne2e4",
                stage=2,
                engine=_FailOnNthAnalyse(fail_on=4),
            )

    def test_stage2_game_over_eval_failure_raises_not_zeroed(self):
        # Scholar's mate position before h5f7; after h5f7 the game is over, so _best_reply_eval
        # calls best_cp on call 3. Failing on call 3 must raise EngineAnalysisError.
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
            board.push_uci(uci)

        class _FailOnThirdAnalyse:
            def __init__(self):
                self.calls = 0

            def analyse(self, b, limit):
                self.calls += 1
                if self.calls == 3:
                    raise chess.engine.EngineError("engine crashed on game-over eval")
                return {"score": _FakeScore(50), "pv": []}

        with self.assertRaises(EngineAnalysisError):
            combined_reward(
                board.fen(),
                "h5f7",
                "<think>delivering checkmate on f7 immediately.</think>\nh5f7",
                stage=2,
                engine=_FailOnThirdAnalyse(),
            )


class TestExpectedScoreAndWDLReward(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_cp_to_expected_score_values_symmetry_and_clamping(self):
        self.assertAlmostEqual(cp_to_expected_score(0.0), 0.5)
        pos = cp_to_expected_score(100.0)
        neg = cp_to_expected_score(-100.0)
        self.assertGreater(pos, 0.55)
        self.assertAlmostEqual(pos + neg, 1.0, places=10)
        # Extreme mate scores and inf must not overflow math.exp
        self.assertGreater(cp_to_expected_score(10000.0), 0.9999)
        self.assertLess(cp_to_expected_score(-10000.0), 0.0001)
        self.assertEqual(cp_to_expected_score(float("inf")), 1.0)
        self.assertAlmostEqual(cp_to_expected_score(float("-inf")), 0.0)
        with self.assertRaises(ValueError):
            cp_to_expected_score(float("nan"))

    def test_wdl_loss_reward_best_move_vs_blunder_and_saturation(self):
        class _SeqEngine:
            def __init__(self, cps):
                self._cps = list(cps)

            def analyse(self, board, limit):
                cp = self._cps.pop(0)
                return {"score": _FakeScore(cp), "pv": []}

        # Best move: 0 cp loss -> wdl_loss_reward == 0.0
        self.assertAlmostEqual(wdl_loss_reward(self.FEN, "e2e4", _SeqEngine([30, 30])), 0.0)

        # Blunder from equal (0 -> -500) hurts much more than blunder when already lost (-800 -> -1200)
        blunder_equal = wdl_loss_reward(self.FEN, "e2e4", _SeqEngine([0, -500]))
        blunder_lost = wdl_loss_reward(self.FEN, "e2e4", _SeqEngine([-800, -1200]))
        self.assertLess(blunder_equal, -0.3)
        self.assertGreater(blunder_lost, -0.05)
        self.assertLess(blunder_equal, blunder_lost)

        # Illegal move -> -1.0
        self.assertEqual(wdl_loss_reward(self.FEN, "e2e5", _SeqEngine([0, 0])), -1.0)
        # Broken engine -> EngineAnalysisError
        with self.assertRaises(EngineAnalysisError):
            wdl_loss_reward(self.FEN, "e2e4", _BrokenEngine())

    def test_combined_reward_wdl_mode_and_stage3(self):
        class _SeqEngine:
            def __init__(self, cps):
                self._cps = list(cps)

            def analyse(self, board, limit):
                cp = self._cps.pop(0)
                return {"score": _FakeScore(cp), "pv": []}

        good_out = "<think>develop central pawn and claim space.</think>\ne2e4"
        best_res = combined_reward(
            self.FEN, "e2e4", good_out, stage=0, engine=_SeqEngine([50, 50]), reward_mode="wdl_loss"
        )
        self.assertTrue(best_res["legal"])
        self.assertAlmostEqual(best_res["components"]["wdl_quality"], 2.0)
        self.assertAlmostEqual(best_res["total"], 1.0 + 0.1 + 2.0)

        blunder_res = combined_reward(
            self.FEN, "e2e4", good_out, stage=3, engine=_SeqEngine([0, -600])
        )
        self.assertLess(blunder_res["total"], best_res["total"])
        self.assertAlmostEqual(sum(blunder_res["components"].values()), blunder_res["total"])

        # Missing engine in WDL mode raises ValueError
        with self.assertRaises(ValueError):
            combined_reward(
                self.FEN, "e2e4", good_out, stage=0, engine=None, reward_mode="wdl_loss"
            )

        # Invalid reward_mode or stage raises ValueError
        with self.assertRaises(ValueError):
            combined_reward(self.FEN, "e2e4", good_out, reward_mode="unknown")
        with self.assertRaises(ValueError):
            combined_reward(self.FEN, "e2e4", good_out, stage=99)


class TestStockfishEvalCache(unittest.TestCase):
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_cache_reuses_fen_and_move_evals_and_isolates_depth(self):
        class _CountingEngine:
            def __init__(self):
                self.calls = 0

            def analyse(self, board, limit):
                self.calls += 1
                pv = [next(iter(board.legal_moves))] if any(board.legal_moves) else []
                return {"score": _FakeScore(40), "pv": pv}

        engine = _CountingEngine()
        cache = StockfishEvalCache(max_size=16)
        out = "<think>control the center with e2e4 cleanly.</think>\ne2e4"

        # 8 repeated rollouts of same (fen, move) -> only 2 engine.analyse calls total (before + after)
        for _ in range(8):
            res = combined_reward(
                self.FEN, "e2e4", out, stage=1, engine=engine, move_quality_depth=8, cache=cache
            )
            self.assertAlmostEqual(res["components"]["move_quality"], 1.0)
        self.assertEqual(engine.calls, 2)

        # Different move from same FEN reuses before-eval, only 1 new engine call
        combined_reward(
            self.FEN, "d2d4", out, stage=1, engine=engine, move_quality_depth=8, cache=cache
        )
        self.assertEqual(engine.calls, 3)

        # Different depth does not collide
        combined_reward(
            self.FEN, "e2e4", out, stage=1, engine=engine, move_quality_depth=10, cache=cache
        )
        self.assertEqual(engine.calls, 5)

        st = cache.stats()
        self.assertGreater(st["fen_hits"], 0)
        self.assertGreater(st["move_hits"], 0)
        cache.clear()
        self.assertEqual(cache.stats()["fen_size"], 0)
        self.assertEqual(cache.stats()["fen_hits"], 0)

    def test_cache_does_not_cache_engine_failures_and_evicts_lru(self):
        class _FlakyEngine:
            def __init__(self):
                self.calls = 0

            def analyse(self, board, limit):
                self.calls += 1
                if self.calls == 1:
                    raise chess.engine.EngineError("transient failure")
                return {"score": _FakeScore(25), "pv": []}

        engine = _FlakyEngine()
        cache = StockfishEvalCache(max_size=2)
        board = chess.Board(self.FEN)

        cp1, _ = cache.get_best(engine, board, depth=6)
        self.assertIsNone(cp1)
        # Second call must retry because None was not cached
        cp2, _ = cache.get_best(engine, board, depth=6)
        self.assertEqual(cp2, 25.0)
        self.assertEqual(engine.calls, 2)

        # LRU eviction with max_size=2
        b1 = board.copy()
        b1.push_uci("e2e4")
        b2 = board.copy()
        b2.push_uci("d2d4")
        cache.get_best(engine, b1, depth=6)
        cache.get_best(engine, b2, depth=6)
        self.assertEqual(cache.stats()["fen_size"], 2)

    def test_cache_thread_safety(self):
        class _CountingEngine:
            def __init__(self):
                self.calls = 0

            def analyse(self, board, limit):
                self.calls += 1
                return {"score": _FakeScore(30), "pv": []}

        engine = _CountingEngine()
        cache = StockfishEvalCache(max_size=64)
        board = chess.Board(self.FEN)

        def _worker(move_uci: str):
            return cache.get_after_move_cp(engine, board, move_uci, depth=6, for_color=chess.WHITE)

        moves = ["e2e4", "d2d4", "g1f3", "c2c4"] * 8
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_worker, moves))
        self.assertTrue(all(r == 30.0 for r in results))
        self.assertFalse(any(math.isnan(r) for r in results))

    def test_cache_oserror_wrapped_and_not_cached(self):
        class _OsErrorEngine:
            def __init__(self):
                self.calls = 0

            def analyse(self, board, limit):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("io failure")
                return {"score": _FakeScore(20), "pv": []}

        engine = _OsErrorEngine()
        cache = StockfishEvalCache(max_size=16)
        out = "<think>develop the king pawn cleanly now.</think>\ne2e4"
        with self.assertRaises(EngineAnalysisError):
            combined_reward(
                self.FEN, "e2e4", out, stage=1, engine=engine, move_quality_depth=6, cache=cache
            )
        # Failure must not be cached; subsequent call succeeds cleanly.
        self.assertEqual(cache.stats()["fen_size"], 0)
        result = combined_reward(
            self.FEN, "e2e4", out, stage=1, engine=engine, move_quality_depth=6, cache=cache
        )
        self.assertGreater(result["total"], 0.0)
        self.assertGreaterEqual(engine.calls, 2)

    def test_cache_incomplete_or_missing_score_raises_and_recovers(self):
        bad_infos = [
            {},
            {"pv": [chess.Move.from_uci("e2e4")]},
            {"score": None},
            {"score": object()},
            {"score": 42},
            {"score": _FakeScore(True)},
            {"score": _FakeScore(float("nan"))},
            None,
            "bad",
        ]
        out = "<think>develop the king pawn cleanly now.</think>\ne2e4"
        for bad_info in bad_infos:
            with self.subTest(bad_info=bad_info):

                class _BadInfoOnceEngine:
                    def __init__(self, payload):
                        self._payload = payload
                        self.calls = 0

                    def analyse(self, board, limit):
                        self.calls += 1
                        if self.calls == 1:
                            return self._payload
                        pv = [next(iter(board.legal_moves))] if any(board.legal_moves) else []
                        return {"score": _FakeScore(25), "pv": pv}

                engine = _BadInfoOnceEngine(bad_info)
                cache = StockfishEvalCache(max_size=16)

                with self.assertRaises(EngineAnalysisError):
                    combined_reward(
                        self.FEN,
                        "e2e4",
                        out,
                        stage=1,
                        engine=engine,
                        move_quality_depth=6,
                        cache=cache,
                    )
                self.assertEqual(engine.calls, 1)
                self.assertEqual(cache.stats()["fen_size"], 0)
                self.assertEqual(cache.stats()["move_size"], 0)

                # Subsequent valid call must succeed cleanly and populate the cache.
                res = combined_reward(
                    self.FEN,
                    "e2e4",
                    out,
                    stage=1,
                    engine=engine,
                    move_quality_depth=6,
                    cache=cache,
                )
                self.assertAlmostEqual(res["components"]["move_quality"], 1.0)
                self.assertEqual(cache.stats()["fen_size"], 2)
                self.assertEqual(cache.stats()["move_size"], 1)

                # Also verify wdl_loss_reward raises EngineAnalysisError and caches nothing.
                wdl_engine = _BadInfoOnceEngine(bad_info)
                wdl_cache = StockfishEvalCache(max_size=16)
                with self.assertRaises(EngineAnalysisError):
                    wdl_loss_reward(self.FEN, "e2e4", wdl_engine, depth=6, cache=wdl_cache)
                self.assertEqual(wdl_engine.calls, 1)
                self.assertEqual(wdl_cache.stats()["fen_size"], 0)
                self.assertEqual(wdl_cache.stats()["move_size"], 0)
                self.assertAlmostEqual(
                    wdl_loss_reward(self.FEN, "e2e4", wdl_engine, depth=6, cache=wdl_cache),
                    0.0,
                )

    def test_cache_malformed_pv_degrades_gracefully_without_crashing(self):
        valid_score = chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE)

        class _BadUciObj:
            def uci(self):
                raise TypeError("bad uci")

        bad_pvs = [[None], 123, [_BadUciObj()]]
        out = "<think>develop the king pawn cleanly now.</think>\ne2e4"
        for bad_pv in bad_pvs:
            with self.subTest(bad_pv=bad_pv):

                class _BadPvEngine:
                    def __init__(self, pv_payload):
                        self._pv_payload = pv_payload

                    def analyse(self, board, limit):
                        return {"score": valid_score, "pv": self._pv_payload}

                engine = _BadPvEngine(bad_pv)
                cache = StockfishEvalCache(max_size=16)
                cp, pv_uci = cache.get_best(engine, chess.Board(self.FEN), depth=4)
                self.assertEqual(cp, 25.0)
                self.assertIsNone(pv_uci)

                # Stage 1 and WDL mode do not require pv and succeed cleanly.
                res = combined_reward(
                    self.FEN, "e2e4", out, stage=1, engine=engine, move_quality_depth=4, cache=cache
                )
                self.assertAlmostEqual(res["components"]["move_quality"], 1.0)
                self.assertAlmostEqual(
                    wdl_loss_reward(self.FEN, "e2e4", engine, depth=4, cache=cache), 0.0
                )

                # Stage 2 needs reply pv and raises EngineAnalysisError rather than AttributeError/TypeError.
                with self.assertRaises(EngineAnalysisError):
                    combined_reward(
                        self.FEN,
                        "e2e4",
                        out,
                        stage=2,
                        engine=engine,
                        move_quality_depth=4,
                        cache=cache,
                    )


if __name__ == "__main__":
    unittest.main()
