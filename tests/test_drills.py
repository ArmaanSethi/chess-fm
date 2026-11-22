"""Unit and live Stockfish tests for chessfm.coach.drills."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import chess
import chess.engine
import pytest

from chessfm.coach.drills import (
    SM2_DEFAULT_EASE_FACTOR,
    SM2_MIN_EASE_FACTOR,
    DrillCard,
    DrillDeck,
    grade_drill_attempt,
    sm2_update,
)
from chessfm.coach.game_review import build_weakness_profile, review_pgn_game
from chessfm.rl.engine import close_engine, open_engine

# White to move with mate-in-1 (Qxf7# = h5f7).
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# Same board state, different halfmove/fullmove counters (for FEN deduplication tests).
SCHOLARS_MATE_FEN_DIFF_CLOCK = (
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 12"
)
# Starting position where both e2e4 and d2d4 are top-tier opening moves.
START_FEN = chess.STARTING_FEN


@pytest.fixture(scope="module")
def stockfish_engine():
    engine = open_engine(threads=1, hash_mb=16)
    if engine is None:
        pytest.skip("Stockfish binary not available")
    yield engine
    close_engine(engine)


def _mock_engine_with_scores(scores_white_cp: list[int]) -> MagicMock:
    """Create a mock engine returning sequential white-perspective centipawn scores."""
    engine = MagicMock()
    side_effects = [
        {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)} for cp in scores_white_cp
    ]
    engine.analyse.side_effect = side_effects
    return engine


class TestSM2Update:
    @pytest.mark.parametrize("quality", [0, 1, 2, 3, 4, 5])
    def test_sm2_formula_all_quality_levels(self, quality: int):
        today = date(2026, 6, 1)
        card = DrillCard(
            card_id="c1",
            fen=SCHOLARS_MATE_FEN,
            best_move="h5f7",
            category="missed_mate",
            student_mistake="a2a3",
            ease_factor=2.5,
            interval_days=6,
            repetitions=2,
            due_date="2026-06-01",
            attempts=2,
            successes=2,
        )

        updated = sm2_update(card, quality, today=today)

        expected_ef = max(
            SM2_MIN_EASE_FACTOR,
            2.5 + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02),
        )
        assert updated.ease_factor == pytest.approx(expected_ef, rel=1e-9)
        assert updated.attempts == 3

        if quality < 3:
            assert updated.repetitions == 0
            assert updated.interval_days == 1
            assert updated.due_date == "2026-06-02"
            assert updated.successes == 2
        else:
            assert updated.repetitions == 3
            assert updated.interval_days == round(6 * 2.5)
            assert updated.due_date == "2026-06-16"
            assert updated.successes == 3

        # Original card is not mutated.
        assert card.attempts == 2
        assert card.repetitions == 2

    def test_sm2_interval_progression_1_to_6_to_scaled(self):
        card = DrillCard(
            card_id="c2",
            fen=START_FEN,
            best_move="e2e4",
            category="opening_inaccuracy",
            student_mistake="h2h3",
            due_date="2026-06-01",
        )
        d1 = date(2026, 6, 1)
        c1 = sm2_update(card, 5, today=d1)
        assert c1.repetitions == 1
        assert c1.interval_days == 1
        assert c1.due_date == "2026-06-02"
        assert c1.ease_factor == pytest.approx(2.6)

        d2 = date(2026, 6, 2)
        c2 = sm2_update(c1, 4, today=d2)
        assert c2.repetitions == 2
        assert c2.interval_days == 6
        assert c2.due_date == "2026-06-08"
        assert c2.ease_factor == pytest.approx(2.6)

        d3 = date(2026, 6, 8)
        c3 = sm2_update(c2, 4, today=d3)
        assert c3.repetitions == 3
        assert c3.interval_days == round(6 * 2.6)
        assert c3.due_date == "2026-06-24"

    def test_sm2_ease_factor_floor(self):
        card = DrillCard(
            card_id="c3",
            fen=START_FEN,
            best_move="e2e4",
            category="positional_drift",
            student_mistake="a2a3",
            ease_factor=1.35,
            due_date="2026-06-01",
        )
        updated = sm2_update(card, 0, today=date(2026, 6, 1))
        assert updated.ease_factor == SM2_MIN_EASE_FACTOR

    @pytest.mark.parametrize("bad_quality", [-1, 6, True, False, 3.5, "4"])
    def test_sm2_invalid_quality_raises(self, bad_quality):
        card = DrillCard(
            card_id="c4",
            fen=START_FEN,
            best_move="e2e4",
            category="opening_inaccuracy",
            student_mistake="a2a3",
        )
        with pytest.raises(ValueError, match="0..5"):
            sm2_update(card, bad_quality)  # type: ignore[arg-type]


class TestGradeDrillAttempt:
    def test_exact_match_no_engine(self):
        result = grade_drill_attempt(SCHOLARS_MATE_FEN, "H5F7 ", "h5f7", engine=None)
        assert result["quality"] == 5
        assert result["passed"] is True
        assert result["cp_loss"] == 0.0
        assert result["status"] == "exact"

    @pytest.mark.parametrize("bad_move", ["e2e5", "notamove", "", "a1h8"])
    def test_illegal_moves(self, bad_move: str):
        result = grade_drill_attempt(SCHOLARS_MATE_FEN, bad_move, "h5f7", engine=None)
        assert result["quality"] == 0
        assert result["passed"] is False
        assert result["cp_loss"] is None
        assert result["status"] == "illegal"

    def test_legal_non_best_without_engine(self):
        result = grade_drill_attempt(SCHOLARS_MATE_FEN, "a2a3", "h5f7", engine=None)
        assert result["quality"] == 1
        assert result["passed"] is False
        assert result["cp_loss"] is None
        assert result["status"] == "missed"

    def test_legal_non_best_with_mock_engine_tiers(self):
        # Acceptable: cp_loss = 25 <= 30 -> quality=4, passed=True
        eng_ok = _mock_engine_with_scores([50, 25])
        res_ok = grade_drill_attempt(START_FEN, "d2d4", "e2e4", engine=eng_ok)
        assert res_ok["quality"] == 4
        assert res_ok["passed"] is True
        assert res_ok["cp_loss"] == 25.0
        assert res_ok["status"] == "acceptable"

        # Inaccurate: cp_loss = 65 (30 < cp_loss <= 100) -> quality=2, passed=False
        eng_inacc = _mock_engine_with_scores([50, -15])
        res_inacc = grade_drill_attempt(START_FEN, "a2a3", "e2e4", engine=eng_inacc)
        assert res_inacc["quality"] == 2
        assert res_inacc["passed"] is False
        assert res_inacc["cp_loss"] == 65.0
        assert res_inacc["status"] == "inaccurate"

        # Missed: cp_loss = 180 > 100 -> quality=1, passed=False
        eng_miss = _mock_engine_with_scores([50, -130])
        res_miss = grade_drill_attempt(START_FEN, "g2g4", "e2e4", engine=eng_miss)
        assert res_miss["quality"] == 1
        assert res_miss["passed"] is False
        assert res_miss["cp_loss"] == 180.0
        assert res_miss["status"] == "missed"

    def test_engine_error_falls_back_gracefully(self):
        broken_engine = MagicMock()
        broken_engine.analyse.side_effect = chess.engine.EngineError("crash")
        res = grade_drill_attempt(START_FEN, "d2d4", "e2e4", engine=broken_engine)
        assert res["quality"] == 1
        assert res["passed"] is False
        assert res["cp_loss"] is None
        assert res["status"] == "missed"

    def test_live_stockfish_grading(self, stockfish_engine):
        # Exact mate-in-1
        exact = grade_drill_attempt(SCHOLARS_MATE_FEN, "h5f7", "h5f7", engine=stockfish_engine)
        assert exact["quality"] == 5
        assert exact["passed"] is True
        assert exact["cp_loss"] == 0.0

        # Missing mate-in-1 with a2a3 incurs huge centipawn loss -> quality=1
        missed = grade_drill_attempt(
            SCHOLARS_MATE_FEN, "a2a3", "h5f7", engine=stockfish_engine, depth=6
        )
        assert missed["quality"] == 1
        assert missed["passed"] is False
        assert missed["cp_loss"] is not None and missed["cp_loss"] > 100.0

        # Opening alternative d2d4 vs e2e4 should have small cp_loss (<= 30 cp)
        alt = grade_drill_attempt(START_FEN, "d2d4", "e2e4", engine=stockfish_engine, depth=6)
        assert alt["cp_loss"] is not None
        assert alt["quality"] in (2, 4)
        if alt["cp_loss"] <= 30.0:
            assert alt["quality"] == 4
            assert alt["passed"] is True


class TestDrillDeck:
    def test_deduplication_by_normalized_fen(self):
        deck = DrillDeck()
        c1, added1 = deck.add_card(
            fen=SCHOLARS_MATE_FEN,
            best_move="h5f7",
            category="missed_mate",
            student_mistake="a2a3",
            today=date(2026, 6, 1),
        )
        c2, added2 = deck.add_card(
            fen=SCHOLARS_MATE_FEN_DIFF_CLOCK,
            best_move="h5f7",
            category="missed_mate",
            student_mistake="b2b3",
            today=date(2026, 6, 2),
        )
        assert added1 is True
        assert added2 is False
        assert c1.card_id == c2.card_id
        assert len(deck) == 1
        assert deck.get_card(c1.card_id) == c1
        assert deck.get_card("nonexistent") is None

        # Constructor also deduplicates cards sharing a normalized FEN.
        dup_deck = DrillDeck(
            cards=[
                DrillCard("id1", SCHOLARS_MATE_FEN, "h5f7", "missed_mate", "a2a3"),
                DrillCard("id2", SCHOLARS_MATE_FEN_DIFF_CLOCK, "h5f7", "missed_mate", "b2b3"),
            ]
        )
        assert len(dup_deck) == 1

    def test_add_from_weakness_profile_and_game_review_integration(self):
        pgn = (
            '[Event "Test"]\n[White "Student"]\n[Black "Opp"]\n[Result "*"]\n\n'
            "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. a3 Nxe4 *\n"
        )
        review = review_pgn_game(pgn, engine=None, student_color="white")
        profile = build_weakness_profile([review])

        # Also add invalid/incomplete entries to verify defensive skipping.
        profile_with_noise = {
            **profile,
            "drills": [
                *profile["drills"],
                {"fen": "", "best_move": "e2e4", "category": "opening_inaccuracy"},
                {"fen": "not-a-valid-fen", "best_move": "e2e4", "category": "opening_inaccuracy"},
                {
                    "fen": SCHOLARS_MATE_FEN,
                    "best_move": "h5f7",
                    "category": "missed_mate",
                    "student_mistake": "a2a3",
                },
                {
                    "fen": SCHOLARS_MATE_FEN_DIFF_CLOCK,
                    "best_move": "h5f7",
                    "category": "missed_mate",
                    "student_mistake": "a2a3",
                },
            ],
        }

        deck = DrillDeck()
        added_first = deck.add_from_weakness_profile(profile_with_noise, today=date(2026, 6, 1))
        assert added_first >= 1
        assert len(deck) == added_first

        # Re-importing the same profile adds 0 duplicates.
        added_second = deck.add_from_weakness_profile(profile_with_noise, today=date(2026, 6, 2))
        assert added_second == 0
        assert len(deck) == added_first

    def test_due_cards_record_attempt_and_stats(self):
        deck = DrillDeck()
        assert deck.stats(today=date(2026, 6, 1)) == {
            "total_cards": 0,
            "due_cards": 0,
            "mastered_cards": 0,
            "total_attempts": 0,
            "total_successes": 0,
            "success_rate": 0.0,
            "average_ease_factor": 0.0,
            "by_category": {},
        }

        c1, _ = deck.add_card(
            fen=SCHOLARS_MATE_FEN,
            best_move="h5f7",
            category="missed_mate",
            student_mistake="a2a3",
            today=date(2026, 6, 1),
        )
        c2, _ = deck.add_card(
            fen=START_FEN,
            best_move="e2e4",
            category="opening_inaccuracy",
            student_mistake="h2h3",
            today=date(2026, 6, 3),
        )

        # On June 1 only c1 is due.
        due_june_1 = deck.due_cards(today=date(2026, 6, 1))
        assert [c.card_id for c in due_june_1] == [c1.card_id]

        # On June 3 both are due; limit=1 returns the earlier due card.
        due_june_3 = deck.due_cards(today=date(2026, 6, 3))
        assert [c.card_id for c in due_june_3] == [c1.card_id, c2.card_id]
        assert len(deck.due_cards(today=date(2026, 6, 3), limit=1)) == 1

        # Record 3 consecutive successes on c1 -> reaches mastery (reps=3, interval>=14).
        r1 = deck.record_attempt(c1.card_id, "h5f7", today=date(2026, 6, 1))
        assert r1["passed"] is True
        assert r1["card"]["due_date"] == "2026-06-02"

        r2 = deck.record_attempt(c1.card_id, "h5f7", today=date(2026, 6, 2))
        assert r2["card"]["interval_days"] == 6
        assert r2["card"]["due_date"] == "2026-06-08"

        r3 = deck.record_attempt(c1.card_id, "h5f7", today=date(2026, 6, 8))
        assert r3["card"]["repetitions"] == 3
        assert r3["card"]["interval_days"] >= 14

        # Record a failure on c2.
        r_fail = deck.record_attempt(c2.card_id, "a2a3", today=date(2026, 6, 3))
        assert r_fail["passed"] is False
        assert r_fail["quality"] == 1
        assert r_fail["card"]["repetitions"] == 0
        assert r_fail["card"]["interval_days"] == 1

        stats = deck.stats(today=date(2026, 6, 4))
        assert stats["total_cards"] == 2
        assert stats["due_cards"] == 1  # c2 is due June 4; c1 is due later in June
        assert stats["mastered_cards"] == 1
        assert stats["total_attempts"] == 4
        assert stats["total_successes"] == 3
        assert stats["success_rate"] == 0.75
        assert stats["by_category"] == {"missed_mate": 1, "opening_inaccuracy": 1}

        with pytest.raises(KeyError, match="Unknown drill card_id"):
            deck.record_attempt("missing_id", "e2e4")

    def test_save_and_load_round_trip(self, tmp_path: Path):
        deck = DrillDeck()
        c1, _ = deck.add_card(
            fen=SCHOLARS_MATE_FEN,
            best_move="h5f7",
            category="missed_mate",
            student_mistake="a2a3",
            today=date(2026, 6, 1),
        )
        deck.record_attempt(c1.card_id, "h5f7", today=date(2026, 6, 1))

        save_path = tmp_path / "nested" / "drills.json"
        written = deck.save(save_path)
        assert written == save_path
        assert save_path.is_file()

        loaded = DrillDeck.load(save_path)
        assert len(loaded) == 1
        loaded_card = loaded.get_card(c1.card_id)
        assert loaded_card is not None
        assert loaded_card == deck.get_card(c1.card_id)
        assert loaded_card.ease_factor == pytest.approx(SM2_DEFAULT_EASE_FACTOR + 0.1)
        assert loaded_card.attempts == 1
        assert loaded_card.successes == 1
