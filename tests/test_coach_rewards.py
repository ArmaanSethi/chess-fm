#!/usr/bin/env python3
"""Unit tests for Stockfish-FREE Coach-GRPO rewards (coach_v2, coach_v1, anti-slop, eval summary)."""

from __future__ import annotations

import json

import chess
import pytest

from chessfm.coach.coach_rewards import (
    CrossFenSlopTracker,
    make_coach_reward_func,
    score_coach_completion,
    summarize_coach_eval,
    verify_claims_v1,
)
from chessfm.coach.tactical_facts import extract_tactical_facts
from chessfm.coach.verifier import verify_claims

# Exact verbatim 4 M4 Goodhart slop cases from held-out audit (#0, #5, #185, #240).
CASE_0_FEN = "3r1r1k/p1nb2R1/1p1p1n2/2pPpN2/2P1P3/P1NB3P/1P5K/6R1 b - - 1 28"
CASE_0_STUDENT = (
    "By playing **b5**, you missed a critical opportunity to challenge the center and "
    "pressure the kingside. The best move is **Nce8**, which forces the white king to "
    "move and prepares to challenge the center. This sequence allows you to maintain "
    "control and prepare for a decisive central breakthrough."
)
CASE_0_TEACHER = (
    "Your move `b5` is a mistake because it ignores the immediate tactical tension "
    "on the kingside. The best response is `Nce8`, which repositions the knight to "
    "challenge the white rook on g7 and defend the kingside."
)

CASE_5_FEN = "2b3k1/1p2rp1p/r3p1p1/p7/8/2P3P1/PP1R1PBP/3R2K1 w - - 0 21"
CASE_5_STUDENT = (
    "Look closely at the tension in the center and the placement of your rook; "
    "is there a way to use a check to force a sequence that forces your opponent's "
    "king into a vulnerable position?"
)

CASE_185_FEN = "r1r3k1/q4ppp/8/p2pb3/N2p4/PP5P/2P2QP1/2BR2K1 w - - 0 23"
CASE_185_STUDENT = (
    "By playing Qf6, you have allowed Black to maintain their central pressure and "
    "central control. Look at the tension in the center: if Black plays Nb2, they "
    "force your knight to move, which allows them to create a powerful attacking sequence."
)

CASE_240_FEN = "r1bq1rk1/pp4pp/2nbpn2/2pp4/2P5/3P1NP1/PP1N1PBP/R1BQR1K1 w - - 1 12"
CASE_240_STUDENT = (
    "By playing Qa4, you have allowed Black to exploit the tension in the center. "
    "Look for a way to use your knight to create a powerful attacking sequence that "
    "challenges the central pawns and forces Black to make a decision. If you play Ng5, "
    "you immediately challenge the e8-pawn and prepare to use your heavy pieces to "
    "create a decisive advantage."
)


class TestM4GoodhartShowdownSeparation:
    """ALL 4 verbatim M4 Goodhart slop cases MUST pass coach_v1 (>0) and FAIL coach_v2 (<0)."""

    @pytest.mark.parametrize(
        ("label", "fen", "text", "task_type", "best_move"),
        [
            ("case_0_fake_check", CASE_0_FEN, CASE_0_STUDENT, "postmortem", "c7e8"),
            ("case_5_center_tension_slop", CASE_5_FEN, CASE_5_STUDENT, "socratic_l2", "d2d8"),
            ("case_185_reversed_polarity", CASE_185_FEN, CASE_185_STUDENT, "why_not", "a4b2"),
            ("case_240_impossible_e8_pawn", CASE_240_FEN, CASE_240_STUDENT, "why_not", "f3g5"),
        ],
    )
    def test_all_four_m4_slop_cases_pass_v1_and_fail_v2(
        self, label: str, fen: str, text: str, task_type: str, best_move: str
    ):
        # 1. Verify Verifier V1 passes while Strict Verifier V2 fails.
        v1 = verify_claims_v1(fen, text)
        v2 = verify_claims(fen, text, engine=None, check_boilerplate=True)
        assert v1["passed"] is True, f"{label} unexpectedly failed V1: {v1['violations']}"
        assert v2["passed"] is False, f"{label} unexpectedly passed Strict V2"

        # 2. Verify coach_v1 reward > 0 while coach_v2 reward < 0.
        tf = extract_tactical_facts(fen, best_move=best_move)
        r_v1 = score_coach_completion(
            fen,
            text,
            task_type=task_type,
            best_move_uci=best_move,
            tactical_facts=tf,
            mode="coach_v1",
        )
        r_v2 = score_coach_completion(
            fen,
            text,
            task_type=task_type,
            best_move_uci=best_move,
            tactical_facts=tf,
            mode="coach_v2",
        )
        assert r_v1["total"] > 0.0, f"{label}: coach_v1 should be > 0, got {r_v1}"
        assert r_v2["total"] < 0.0, f"{label}: coach_v2 should be < 0, got {r_v2}"

    def test_verify_claims_v1_still_catches_illegal_move_and_wrong_piece(self):
        # V1 DOES catch genuine illegal moves and non-hyphenated wrong piece-on-square.
        res_illegal = verify_claims_v1(chess.STARTING_FEN, "Play Qh5 to win right away.")
        assert res_illegal["passed"] is False
        assert res_illegal["violations"][0]["type"] == "illegal_move"

        res_piece = verify_claims_v1(chess.STARTING_FEN, "The knight on e4 dominates.")
        assert res_piece["passed"] is False
        assert res_piece["violations"][0]["type"] == "piece_on_square"

        res_side = verify_claims_v1(chess.STARTING_FEN, "Black is to move here.")
        assert res_side["passed"] is False
        assert res_side["violations"][0]["type"] == "side_to_move"


class TestCrossFenSlopTracker:
    def test_same_fen_g8_rollouts_not_falsely_penalized(self):
        tracker = CrossFenSlopTracker(maxlen=256, max_doc_freq_ratio=0.20, min_doc_count=3)
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        good_text = "Play Ng5 to attack the black pawn on f7 immediately."
        # 8 rollouts on the exact same FEN must NOT trigger cross-FEN anti-slop!
        flags = [tracker.check_and_update(fen, good_text) for _ in range(8)]
        assert flags == [False] * 8

    def test_cross_fen_repeated_4grams_penalized(self):
        tracker = CrossFenSlopTracker(
            maxlen=256, max_doc_freq_ratio=0.20, min_doc_count=3, min_distinct_fens=8
        )
        # Feed 10 distinct FENs all repeating a novel non-blocklisted boilerplate 4-gram.
        boiler = "Always remember the secret golden rule of harmony in this position."
        fens = [
            f"4k3/8/8/8/8/8/{idx}P{7 - idx}/4K3 w - - 0 1"
            if 0 < idx < 7
            else f"4k3/8/8/8/8/{idx}P{7 - idx}/8/4K3 w - - 0 1"
            for idx in range(1, 7)
        ] + [
            "4k3/8/8/8/P7/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/1P6/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/2P5/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/3P4/8/8/4K3 w - - 0 1",
        ]
        results = [tracker.check_and_update(f, boiler) for f in fens]
        # Early distinct FENs (<8) are not flagged; once >=8 distinct FENs exist, flagged True!
        assert any(results[-2:]) is True

    def test_known_slop_blocklist_immediately_flagged(self):
        tracker = CrossFenSlopTracker()
        assert (
            tracker.check_and_update(
                chess.STARTING_FEN,
                "Look closely at the tension in the center to create a powerful attacking sequence.",
            )
            is True
        )


class TestCoachRewardScoringAndFactory:
    def test_degenerate_short_completion_penalized_in_both_modes(self):
        for mode in ("coach_v2", "coach_v1"):
            res = score_coach_completion(chess.STARTING_FEN, "Play e4.", mode=mode)
            assert res["total"] == pytest.approx(-1.5)
            assert res["degenerate"] is True

    def test_spoiler_penalized_on_l1_and_l2(self):
        tf = extract_tactical_facts(chess.STARTING_FEN, best_move="e2e4")
        spoiled = "Advance your pawn with e4 to fight for central space immediately."
        r_v2 = score_coach_completion(
            chess.STARTING_FEN,
            spoiled,
            task_type="socratic_l1",
            best_move_uci="e2e4",
            tactical_facts=tf,
            mode="coach_v2",
        )
        assert r_v2["spoiler_free"] is False
        assert r_v2["coverage"] == 0.0
        # +1.0 (V2 pass) + 0.0 (cov) - 1.0 (spoiler) = 0.0 (vs +1.7 for clean high-coverage)
        assert r_v2["total"] == pytest.approx(0.0)

        clean = "Focus on pushing a central pawn to claim space in the center early."
        r_clean = score_coach_completion(
            chess.STARTING_FEN,
            clean,
            task_type="socratic_l1",
            best_move_uci="e2e4",
            tactical_facts=tf,
            mode="coach_v2",
        )
        assert r_clean["spoiler_free"] is True
        assert r_clean["total"] > 1.2

    def test_make_coach_reward_func_trl_kwargs_and_fallback(self):
        tf = extract_tactical_facts(CASE_0_FEN, student_move="b6b5", best_move="c7e8", cp_loss=180)
        fn_v2 = make_coach_reward_func(mode="coach_v2")
        fn_v1 = make_coach_reward_func(mode="coach_v1")

        rewards_v2 = fn_v2(
            prompts=["p1", "p2"],
            completions=[CASE_0_TEACHER, CASE_0_STUDENT],
            fen=[CASE_0_FEN, CASE_0_FEN],
            task_type=["postmortem", "postmortem"],
            best_move_uci=["c7e8", "c7e8"],
            tactical_facts_json=[json.dumps(tf), json.dumps(tf)],
        )
        assert rewards_v2[0] > 1.0
        assert rewards_v2[1] < 0.0

        rewards_v1 = fn_v1(
            prompts=["p1", "p2"],
            completions=[CASE_0_TEACHER, CASE_0_STUDENT],
            fen=[CASE_0_FEN, CASE_0_FEN],
            task_type=["postmortem", "postmortem"],
            best_move_uci=["c7e8", "c7e8"],
        )
        assert rewards_v1[0] == pytest.approx(1.2)
        assert rewards_v1[1] == pytest.approx(1.2)

        # Fallback FEN extraction from prompt when fen kwarg omitted, and missing FEN -> 0.0.
        fb = fn_v2(
            prompts=[f'Engine Facts (JSON):\n{{"fen": "{chess.STARTING_FEN}"}}', "no fen prompt"],
            completions=[
                "Developing centrally with e4 claims space and opens bishop diagonals.",
                "Some valid length text without any FEN in prompt.",
            ],
        )
        assert fb[0] > 0.0
        assert fb[1] == 0.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported coach reward mode"):
            score_coach_completion(chess.STARTING_FEN, "Valid coaching text here.", mode="bad")
        with pytest.raises(ValueError, match="Unsupported coach reward mode"):
            make_coach_reward_func(mode="bad")

    def test_summarize_coach_eval_metrics(self):
        rows = [
            {
                "fen": CASE_0_FEN,
                "task_type": "postmortem",
                "best_move": "c7e8",
                "student_move": "b6b5",
                "response": CASE_0_TEACHER,
            },
            {
                "fen": CASE_0_FEN,
                "task_type": "postmortem",
                "best_move": "c7e8",
                "student_move": "b6b5",
                "response": CASE_0_STUDENT,
            },
            {
                "fen": chess.STARTING_FEN,
                "task_type": "socratic_l1",
                "best_move": "e2e4",
                "response": "Focus on claiming central space and opening diagonals for your minor pieces.",
            },
        ]
        enriched, summary = summarize_coach_eval(rows)
        assert len(enriched) == 3
        assert summary["total_cases"] == 3
        assert summary["v1_passed"] == 3
        assert summary["v2_passed"] == 2
        assert summary["socratic_l12_spoiler_free_rate"] == 1.0
        assert "fake_check" in summary["violation_breakdown"]
        assert "postmortem" in summary["per_task_type"]
        assert "socratic_l1" in summary["per_task_type"]

    def test_summarize_coach_eval_preserves_pv_and_is_idempotent(self):
        fork_fen = "6k1/5p1p/4p3/4q3/3n4/2Q3P1/PP1N1P1P/6K1 b - - 3 37"
        rows = [
            {
                "fen": CASE_0_FEN,
                "task_type": "postmortem",
                "best_move": "c7e8",
                "student_move": "b6b5",
                "pv": ["c7e8", "g7g6"],
                "response": CASE_0_TEACHER,
            },
            {
                "fen": fork_fen,
                "task_type": "socratic_l3",
                "best_move": "d4e2",
                "pv": ["d4e2", "g1f1", "e2c3"],
                # Mentions Ne2+, fork, c3, and PV reply Kf1 -> full 4/4 coverage.
                "response": (
                    "Play Ne2+ to fork the king and queen; after Kf1 you win the white queen on c3."
                ),
            },
            {
                "fen": fork_fen,
                "task_type": "socratic_l3",
                "best_move": "d4e2",
                "pv": ["d4e2", "g1f1", "e2c3"],
                # Omits PV reply Kf1 -> 3/4 = 0.75 coverage ONLY if pv is preserved (otherwise 3/3 = 1.0!).
                "response": (
                    "Play Ne2+ to fork the white king on g1 and the white queen on c3 right away."
                ),
            },
        ]
        enriched, summary = summarize_coach_eval(rows)
        assert enriched[0]["pv"] == ["c7e8", "g7g6"]
        assert enriched[1]["pv"] == ["d4e2", "g1f1", "e2c3"]
        assert enriched[1]["tactical_fact_coverage"] == pytest.approx(1.0)
        assert enriched[2]["tactical_fact_coverage"] == pytest.approx(0.75)

        # Standalone re-scoring on enriched output MUST be strictly idempotent.
        enriched_again, summary_again = summarize_coach_eval(enriched)
        assert summary_again == summary
        assert enriched_again == enriched
