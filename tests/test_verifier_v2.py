#!/usr/bin/env python3
"""Comprehensive unit and regression tests for Strict Verifier V2."""

from __future__ import annotations

import chess
import pytest

from chessfm.coach.verifier import (
    CoTVerificationResult,
    _has_center_tension,
    detect_boilerplate_slop,
    extract_claims,
    verify_claims,
    verify_think_trace,
)

START_FEN = chess.STARTING_FEN
ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
CASTLE_ROOK_ATTACK_FEN = "r3k2r/pppp1ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1"
DISCOVERED_ATTACK_FEN = "4k3/4q3/8/8/4N3/8/8/4R1K1 w - - 0 1"

# Exact FENs and texts from the 4 real held-out coaching audit cases (#0, #5, #185, #240).
CASE_0_FEN = "3r1r1k/p1nb2R1/1p1p1n2/2pPpN2/2P1P3/P1NB3P/1P5K/6R1 b - - 1 28"
CASE_0_TEACHER = (
    "Your move `b5` is a mistake because it ignores the immediate tactical tension "
    "on the kingside. The best response is `Nce8`, which reposition the knight to "
    "challenge your dominant rook on g7 and prepare to defend the kingside. By failing "
    "to address the pressure from your pieces, you allow Black to neutralize your attack "
    "and gain a significant positional advantage."
)
CASE_0_STUDENT = (
    "By playing **b5**, you missed a critical opportunity to challenge the center and "
    "pressure the kingside. The best move is **Nce8**, which forces the white king to "
    "move and prepares to challenge the center. This sequence allows you to maintain "
    "control and prepare for a decisive central breakthrough."
)

CASE_5_FEN = "2b3k1/1p2rp1p/r3p1p1/p7/8/2P3P1/PP1R1PBP/3R2K1 w - - 0 21"
CASE_5_TEACHER = (
    "Look for a tactical way to exploit the back rank and create a decisive fork. "
    "Can you find a way to deliver a check that simultaneously attacks two key pieces?"
)
CASE_5_STUDENT = (
    "Look closely at the tension in the center and the placement of your rook; "
    "is there a way to use a check to force a sequence that forces your opponent's "
    "king into a vulnerable position?"
)

CASE_185_FEN = "r1r3k1/q4ppp/8/p2pb3/N2p4/PP5P/2P2QP1/2BR2K1 w - - 0 23"
CASE_185_TEACHER = (
    "By playing Qf6, you have placed your Queen on a square where it can be captured "
    "immediately by the bishop on e5. Instead of losing a full queen for nothing, you "
    "should retreat your knight to b2 to defend the c4 square and prepare to reposition."
)
CASE_185_STUDENT = (
    "By playing Qf6, you have allowed Black to maintain their central pressure and "
    "central control. Look at the tension in the center: if Black plays Nb2, they "
    "force your knight to move, which allows them to create a powerful attacking sequence."
)

CASE_240_FEN = "r1bq1rk1/pp4pp/2nbpn2/2pp4/2P5/3P1NP1/PP1N1PBP/R1BQR1K1 w - - 1 12"
CASE_240_TEACHER = (
    "By playing Qa4, you are focusing on a pin, but you are missing a more direct way "
    "to exploit the tension in the center. Look for a way to reposition your knight to "
    "attack the e6-square, which is currently defended only by the queen and bishop. "
    "If you play Ng5, you immediately pressure the e6-pawn and force Black to make a "
    "difficult defensive decision."
)
CASE_240_STUDENT = (
    "By playing Qa4, you have allowed Black to exploit the tension in the center. "
    "Look for a way to use your knight to create a powerful attacking sequence that "
    "challenges the central pawns and forces Black to make a decision. If you play Ng5, "
    "you immediately challenge the e8-pawn and prepare to use your heavy pieces to "
    "create a decisive advantage."
)


# ---------------------------------------------------------------------------
# 1. Hyphenated & Determiner Square-First Piece-Square Parsing
# ---------------------------------------------------------------------------


def test_hyphenated_and_determiner_piece_square_parsing():
    text = (
        "White's c4-bishop eyes the f7-pawn, while the f6 knight and enemy a8-rook remain active."
    )
    claims = extract_claims(text)
    pos = [c for c in claims if c["type"] == "piece_on_square"]
    assert [(c["color"], c["piece"], c["square"]) for c in pos] == [
        ("white", "bishop", "c4"),
        (None, "pawn", "f7"),
        (None, "knight", "f6"),
        (None, "rook", "a8"),
    ]
    res = verify_claims(ITALIAN_FEN, text)
    assert res["passed"] is True
    assert res["violations"] == []


def test_hyphenated_square_reference_not_treated_as_piece():
    claims = extract_claims("Pressure the e6-square and control the d5-square.")
    pos = [c for c in claims if c["type"] == "piece_on_square"]
    assert pos == []


def test_hyphenated_piece_wrong_square_flagged():
    res = verify_claims(ITALIAN_FEN, "The e4-knight dominates the center.")
    assert res["passed"] is False
    assert any(v["type"] == "piece_on_square" for v in res["violations"])


def test_piece_on_square_post_move_grace_in_same_sentence():
    res = verify_claims(START_FEN, "After playing e4, the e4-pawn controls central squares.")
    assert res["passed"] is True
    assert res["violations"] == []


# ---------------------------------------------------------------------------
# 2. Impossible Rank-1/8 Pawn Detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sq", ["a1", "e1", "h1", "a8", "e8", "h8"])
def test_impossible_rank_1_and_8_pawns_hyphenated_and_prose(sq: str):
    res_hyph = verify_claims(START_FEN, f"Target the {sq}-pawn immediately.")
    assert res_hyph["passed"] is False
    assert any(v["type"] == "impossible_pawn_rank" for v in res_hyph["violations"])

    res_prose = verify_claims(START_FEN, f"The pawn on {sq} is vulnerable.")
    assert res_prose["passed"] is False
    assert any(v["type"] == "impossible_pawn_rank" for v in res_prose["violations"])


def test_impossible_pawn_rank_not_excused_by_post_move():
    # Even after a legal move in the same sentence, rank-8 pawn is never excused.
    res = verify_claims(START_FEN, "After playing e4, watch the e8-pawn.")
    assert res["passed"] is False
    assert any(v["type"] == "impossible_pawn_rank" for v in res["violations"])


# ---------------------------------------------------------------------------
# 3. Attack / Ray Geometry Verification
# ---------------------------------------------------------------------------


def test_attack_geometry_direct_knight_bishop_rook_queen_pawn():
    # Knight: Ng5 attacks f7 in Italian Game
    assert verify_claims(ITALIAN_FEN, "Play Ng5 to attack the f7-pawn.")["passed"] is True
    # False knight attack: Ng5 does not attack d8
    res_bad_n = verify_claims(ITALIAN_FEN, "Play Ng5 to attack d8.")
    assert res_bad_n["passed"] is False
    assert any(v["type"] == "invalid_attack_claim" for v in res_bad_n["violations"])

    # Bishop: Bb5 attacks c6
    assert verify_claims(ITALIAN_FEN, "Play Bb5, attacking the knight on c6.")["passed"] is True
    # False bishop attack: Bb5 does not attack f7
    res_bad_b = verify_claims(ITALIAN_FEN, "Play Bb5, attacking f7.")
    assert res_bad_b["passed"] is False
    assert any(v["type"] == "invalid_attack_claim" for v in res_bad_b["violations"])

    # Pawn: d4 attacks e5
    assert verify_claims(ITALIAN_FEN, "Push d4 to challenge the e5-pawn.")["passed"] is True
    # False pawn attack: a3 does not attack e5
    res_bad_p = verify_claims(ITALIAN_FEN, "Play a3 to challenge the e5-pawn.")
    assert res_bad_p["passed"] is False
    assert any(v["type"] == "invalid_attack_claim" for v in res_bad_p["violations"])


def test_attack_geometry_discovered_attack_and_castling():
    # Moving Ne4 to c5 discovers Re1's attack on Black's queen on e7.
    res_disc = verify_claims(DISCOVERED_ATTACK_FEN, "Play Nc5 to attack the queen on e7.")
    assert res_disc["passed"] is True

    # Queenside castling brings the rook to d1, adding pressure to d7.
    res_castle = verify_claims(CASTLE_ROOK_ATTACK_FEN, "Castle O-O-O to pressure the d7-pawn.")
    assert res_castle["passed"] is True


def test_attack_geometry_static_piece_and_hedges():
    # Static piece on root board: c4 bishop attacks f7 -> valid
    assert verify_claims(ITALIAN_FEN, "The bishop on c4 targets f7.")["passed"] is True
    # Static piece on root board: c4 bishop does NOT attack h8 -> invalid
    res_static_bad = verify_claims(ITALIAN_FEN, "The bishop on c4 targets h8.")
    assert res_static_bad["passed"] is False
    assert any(v["type"] == "invalid_attack_claim" for v in res_static_bad["violations"])

    # Negated and future-hedged attack claims must NOT trigger false positives.
    assert verify_claims(ITALIAN_FEN, "Play a3, which does not attack f7.")["passed"] is True
    assert (
        verify_claims(ITALIAN_FEN, "Play a3, preparing to eventually target f7.")["passed"] is True
    )
    assert verify_claims(ITALIAN_FEN, "Play a3; it prepares to attack d8.")["passed"] is True


# ---------------------------------------------------------------------------
# 4. Check & Forcing-King-Move Verification
# ---------------------------------------------------------------------------


def test_check_and_forcing_king_move_claims():
    # Valid check claims
    assert verify_claims(ITALIAN_FEN, "Play Bxf7+, which puts the black king in check.")["passed"]
    assert verify_claims(ITALIAN_FEN, "Play Bxf7+, forcing the black king to move.")["passed"]
    assert verify_claims(ITALIAN_FEN, "Play Bxf7+, which checks the black king.")["passed"]

    # False check claims
    res_puts = verify_claims(ITALIAN_FEN, "Play O-O, which puts the king in check.")
    assert res_puts["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_puts["violations"])

    res_forces = verify_claims(ITALIAN_FEN, "Play d3, which forces the black king to retreat.")
    assert res_forces["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_forces["violations"])

    # Wrong king color: White plays Bxf7+ (checks Black's king, NOT White's king)
    res_wrong_color = verify_claims(ITALIAN_FEN, "Play Bxf7+, which forces the white king to move.")
    assert res_wrong_color["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_wrong_color["violations"])


def test_why_not_preceding_sentence_check_context():
    # Mirrors Teacher #91: check claim in sentence 2 refers to the checking move in sentence 1.
    text = (
        "**Why not Bxf7+?** While capturing the pawn with check feels aggressive, "
        "it gives up the bishop pair."
    )
    res = verify_claims(ITALIAN_FEN, text)
    assert res["passed"] is True
    assert res["violations"] == []


# ---------------------------------------------------------------------------
# 5. Side-to-Move / Actor / Pronoun Polarity Checks
# ---------------------------------------------------------------------------


def test_color_actor_and_side_to_move_polarity():
    # Matching actor passes
    assert verify_claims(START_FEN, "White plays e4 to control the center.")["passed"] is True
    assert verify_claims(START_FEN, "White's best move is e4.")["passed"] is True
    assert verify_claims(START_FEN, "White is to move. Play e4.")["passed"] is True

    # Mismatched actor fails
    res_actor = verify_claims(START_FEN, "Black plays e4 to control the center.")
    assert res_actor["passed"] is False
    assert any(v["type"] == "color_polarity_violation" for v in res_actor["violations"])

    res_best = verify_claims(START_FEN, "Black's best move is Nf3.")
    assert res_best["passed"] is False
    assert any(v["type"] == "color_polarity_violation" for v in res_best["violations"])

    # Mismatched explicit side-to-move fails
    res_stm = verify_claims(START_FEN, "Black is to move here.")
    assert res_stm["passed"] is False
    assert any(v["type"] == "color_polarity_violation" for v in res_stm["violations"])

    # Root-turn opponent pronoun polarity fails
    res_pronoun = verify_claims(START_FEN, "If the opponent plays Nf3, you lose tempo.")
    assert res_pronoun["passed"] is False
    assert any(v["type"] == "color_polarity_violation" for v in res_pronoun["violations"])

    # Valid ply-2 opponent reply in the same sentence MUST pass
    res_ply2 = verify_claims(START_FEN, "After 1. e4, if Black plays e5, continue with Nf3.")
    assert res_ply2["passed"] is True


# ---------------------------------------------------------------------------
# 6. Boilerplate / Anti-Slop Checks
# ---------------------------------------------------------------------------


def test_center_tension_helper_and_per_row_boilerplate():
    assert _has_center_tension(chess.Board(START_FEN)) is False
    assert _has_center_tension(chess.Board(CASE_5_FEN)) is False
    assert _has_center_tension(chess.Board(ITALIAN_FEN)) is True
    assert _has_center_tension(chess.Board(CASE_240_FEN)) is True

    # Empty center + tension claim -> hollow_boilerplate violation
    res_bad = verify_claims(CASE_5_FEN, "Look closely at the tension in the center.")
    assert res_bad["passed"] is False
    assert any(v["type"] == "hollow_boilerplate" for v in res_bad["violations"])

    # Opt-out via check_boilerplate=False
    res_opt_out = verify_claims(
        CASE_5_FEN, "Look closely at the tension in the center.", check_boilerplate=False
    )
    assert res_opt_out["passed"] is True

    # Contested center + tension claim -> passes
    res_ok = verify_claims(ITALIAN_FEN, "Look closely at the tension in the center.")
    assert res_ok["passed"] is True


def test_detect_boilerplate_slop_diverse_vs_repetitive_and_empty():
    empty = detect_boilerplate_slop([])
    assert empty["total_texts"] == 0
    assert empty["distinct_4gram_ratio"] == 1.0
    assert empty["is_slop_corpus"] is False

    diverse = [
        "Advance the kingside pawns to crack open the castled king shelter.<|im_end|>",
        "Retreat the bishop along the long diagonal before trapping the rook.<|im_end|>",
        "Trade queens immediately to enter a winning opposite-colored bishop endgame.",
        "Push the passed a-pawn while the enemy knight remains pinned on f6.",
        "Activate the king toward the queenside before the pawn majority rolls.",
    ]
    div_report = detect_boilerplate_slop(diverse)
    assert div_report["is_slop_corpus"] is False
    assert div_report["distinct_4gram_ratio"] > 0.9
    assert div_report["flagged_count"] == 0

    sloppy = [
        "Look closely at the tension in the center and find a tactical sequence.",
        "Look closely at the tension in the center and find a tactical sequence.",
        "Look closely at the tension in the center and improve piece coordination.",
        "Look closely at the tension in the center and defend your king.",
        "Look closely at the tension in the center to seize the initiative.",
    ]
    slop_report = detect_boilerplate_slop(sloppy)
    assert slop_report["is_slop_corpus"] is True
    assert slop_report["flagged_count"] == 5
    assert slop_report["flagged_rate"] == pytest.approx(1.0)
    assert slop_report["exact_duplicate_rate"] == pytest.approx(0.2)
    assert slop_report["top_repeated_4grams"][0]["doc_count"] == 5


# ---------------------------------------------------------------------------
# 7. CoT Faithfulness Mode (verify_think_trace + CoTVerificationResult)
# ---------------------------------------------------------------------------


def test_verify_think_trace_clean_hallucinated_and_raw():
    clean_trace = (
        "<think>White has a bishop on c4 and a knight on f3. "
        "Playing O-O safeguards the king while Ng5 attacks the f7-pawn.</think>\n"
        "f3g5"
    )
    res_clean = verify_think_trace(ITALIAN_FEN, clean_trace)
    assert isinstance(res_clean, CoTVerificationResult)
    assert res_clean.zero_violation is True
    assert res_clean.claims_total >= 4
    assert res_clean.claims_verified == res_clean.claims_total
    assert res_clean.claim_accuracy == pytest.approx(1.0)
    d = res_clean.to_dict()
    assert d["zero_violation"] is True
    assert d["violations"] == []

    bad_trace = (
        "<think>If Black plays Ng5, it challenges the e8-pawn and forces the black king to move."
        "</think>\nf3g5"
    )
    res_bad = verify_think_trace(ITALIAN_FEN, bad_trace)
    assert res_bad.zero_violation is False
    vtypes = {v["type"] for v in res_bad.violations}
    assert "color_polarity_violation" in vtypes
    assert "impossible_pawn_rank" in vtypes
    assert "fake_check" in vtypes

    # Raw text without <think> tags also works directly.
    raw_res = verify_think_trace(START_FEN, "White plays e4, which gives check.")
    assert raw_res.zero_violation is False
    assert raw_res.violations[0]["type"] == "fake_check"


# ---------------------------------------------------------------------------
# 8. Real M4 Audit Regressions (#0, #5, #185, #240)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "fen", "teacher_text", "student_text", "expected_student_violations"),
    [
        (0, CASE_0_FEN, CASE_0_TEACHER, CASE_0_STUDENT, {"fake_check"}),
        (5, CASE_5_FEN, CASE_5_TEACHER, CASE_5_STUDENT, {"hollow_boilerplate"}),
        (185, CASE_185_FEN, CASE_185_TEACHER, CASE_185_STUDENT, {"color_polarity_violation"}),
        (
            240,
            CASE_240_FEN,
            CASE_240_TEACHER,
            CASE_240_STUDENT,
            {"impossible_pawn_rank", "invalid_attack_claim"},
        ),
    ],
)
def test_m4_four_case_audit_regressions(
    case_id: int,
    fen: str,
    teacher_text: str,
    student_text: str,
    expected_student_violations: set[str],
):
    teacher_res = verify_claims(fen, teacher_text)
    assert teacher_res["passed"] is True, (
        f"Case #{case_id} Teacher 26B unexpectedly failed: {teacher_res['violations']}"
    )
    assert teacher_res["violations"] == []

    student_res = verify_claims(fen, student_text)
    assert student_res["passed"] is False, f"Case #{case_id} SFT 0.6B unexpectedly passed V2!"
    actual_types = {v["type"] for v in student_res["violations"]}
    assert expected_student_violations.issubset(actual_types), (
        f"Case #{case_id}: expected {expected_student_violations}, got {actual_types}"
    )


# ---------------------------------------------------------------------------
# 9. Iteration 2 Surgical Fix Regressions (Fixes 1-8)
# ---------------------------------------------------------------------------


def test_contraction_and_cannot_negations():
    rook_blocked_fen = "4k3/8/8/8/4P3/8/8/4R1K1 w - - 0 1"
    empty_center_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    assert verify_claims(rook_blocked_fen, "White plays Re2, which doesn't attack e8.")["passed"]
    assert verify_claims(rook_blocked_fen, "White plays Re2, which cannot attack e8.")["passed"]
    assert verify_claims(START_FEN, "Playing Nf3 isn't challenging the rook on a8.")["passed"]
    assert verify_claims(START_FEN, "Playing Nf3 can't force the black king to move.")["passed"]
    assert verify_claims(START_FEN, "Playing Nf3 cannot force the black king to move.")["passed"]
    assert verify_claims(empty_center_fen, "There isn't any tension in the center.")["passed"]
    assert verify_claims(START_FEN, "It isn't Black to move.")["passed"]


def test_participle_check_and_checkmate_verbs():
    rd8_fen = "4k3/8/8/8/8/8/8/3R2K1 w - - 0 1"

    for phrase in (
        "White plays Nf3, forcing the black king to move.",
        "White plays Nf3, putting the black king in check.",
        "White plays Nf3, placing the black king in check.",
        "White plays Nf3, leaving the black king in check.",
        "White plays Nf3, giving check.",
        "White plays Nf3, delivering check.",
        "White plays Nf3, checking the black king.",
    ):
        res = verify_claims(START_FEN, phrase)
        assert res["passed"] is False, f"Expected fake_check for: {phrase}"
        assert any(v["type"] == "fake_check" for v in res["violations"])

    # Own-king mismatch with participle forcing verb.
    res_own = verify_claims(rd8_fen, "White plays Rd8+, forcing the white king to move.")
    assert res_own["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_own["violations"])

    # Valid check with participle passes.
    assert verify_claims(rd8_fen, "White plays Rd8+, forcing the black king to move.")["passed"]

    # Fake checkmate participle fails.
    res_mate = verify_claims(rd8_fen, "White plays Rd8+, delivering checkmate.")
    assert res_mate["passed"] is False
    assert any(v["type"] == "fake_checkmate" for v in res_mate["violations"])


def test_unlinked_check_claim_king_color_validation():
    rd8_fen = "4k3/8/8/8/8/8/8/3R2K1 w - - 0 1"
    white_in_check_fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"

    # White to move, not in check: cannot force white's own king to move.
    res_own = verify_claims(rd8_fen, "This tactic forces the white king to move.")
    assert res_own["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_own["violations"])

    # White to move, Rd8+ is legal: forcing black king to move passes.
    assert verify_claims(rd8_fen, "This tactic forces the black king to move.")["passed"] is True

    # White IS currently in check: describing white king in check passes,
    # whereas claiming black king is forced to move (no counter-check) fails.
    assert verify_claims(white_in_check_fen, "This leaves the white king in check.")["passed"]
    res_no_counter = verify_claims(white_in_check_fen, "This forces the black king to move.")
    assert res_no_counter["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_no_counter["violations"])


def test_same_sentence_attack_claim_binds_to_closest_subject():
    # Bishop on c4 is closer to "attacks f7" than move d3 -> binds to c4 (valid).
    res_ok = verify_claims(ITALIAN_FEN, "White plays d3, while the bishop on c4 attacks f7.")
    assert res_ok["passed"] is True
    assert res_ok["violations"] == []

    # Bishop on c4 is closer to "attacks h8" than move d3 -> binds to c4 (invalid).
    res_bad = verify_claims(ITALIAN_FEN, "White plays d3, while the bishop on c4 attacks h8.")
    assert res_bad["passed"] is False
    assert any(
        v["type"] == "invalid_attack_claim" and "Piece on c4" in v["reason"]
        for v in res_bad["violations"]
    )

    # Move Ng5 is closer to "attack f7" than earlier piece mention -> binds to Ng5 (valid).
    res_move_closer = verify_claims(
        ITALIAN_FEN, "With the knight on f3 active, White plays Ng5 to attack f7."
    )
    assert res_move_closer["passed"] is True


def test_relocation_infinitive_guard_teacher_row_3():
    # Exact Teacher 26B Row #3 prose: repositioning knight on c7 to challenge g7.
    teacher_row_3 = (
        "Look for a way to reposition the knight on c7 to challenge your dominant "
        "rook on g7 or to defend the critical d6 square."
    )
    res = verify_claims(CASE_0_FEN, teacher_row_3)
    assert res["passed"] is True
    assert res["violations"] == []

    # Static non-relocation attack claim on c4 bishop targeting h8 still fails.
    res_static = verify_claims(ITALIAN_FEN, "The bishop on c4 targets h8.")
    assert res_static["passed"] is False
    assert any(v["type"] == "invalid_attack_claim" for v in res_static["violations"])


def test_missed_opportunity_check_lookback_guard_teacher_row_201():
    # Exact Teacher 26B Row #201 (why_not, best_move=e5d4).
    row_201_fen = "5R2/7p/p1p1k1p1/2p1b3/2r5/6P1/P6P/6K1 b - - 1 32"
    teacher_row_201 = (
        "Why not **Re4**? While this move attempts to centralize the rook, it misses "
        "a powerful tactical opportunity to improve your position with a check. "
        "Instead, **Bd4+** is much stronger."
    )
    res = verify_claims(row_201_fen, teacher_row_201)
    assert res["passed"] is True
    assert res["violations"] == []

    # Non-missed 1-sentence lookback still binds to prior non-checking move.
    res_still_caught = verify_claims(
        ITALIAN_FEN, "Why not **O-O**? Capturing with check wins material."
    )
    assert res_still_caught["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_still_caught["violations"])


def test_verify_think_trace_unclosed_and_multi_block():
    # Unclosed <think> ignores hallucinated prefix outside <think>.
    unclosed = verify_think_trace(START_FEN, "Hallucinated prefix Qh8# <think>White plays Nf3")
    assert unclosed.zero_violation is True
    assert unclosed.violations == []

    # Multiple <think> blocks: violations in the second block MUST be caught.
    multi = verify_think_trace(
        START_FEN,
        "<think>Nf3 is legal.</think> some text "
        "<think>White has a queen on h8 and plays Qh8#.</think>",
    )
    assert multi.zero_violation is False
    vtypes = {v["type"] for v in multi.violations}
    assert "piece_on_square" in vtypes
    assert "illegal_move" in vtypes


def test_detect_boilerplate_slop_short_text_exact_duplicates():
    short_clones = ["Look at center"] * 20
    report = detect_boilerplate_slop(short_clones)
    assert report["total_4grams"] == 0
    assert report["flagged_count"] == 20
    assert report["flagged_rate"] == pytest.approx(1.0)
    assert report["is_slop_corpus"] is True

    below_min = ["Look at center"] * 2
    report_small = detect_boilerplate_slop(below_min)
    assert report_small["flagged_count"] == 0
    assert report_small["is_slop_corpus"] is False


# ---------------------------------------------------------------------------
# 10. Iteration 3 Surgical Fix Regressions (Fixes 1-5)
# ---------------------------------------------------------------------------


def test_attack_claim_object_piece_and_conjunction_participle_binding():
    fork_fen = "4k3/7p/4b3/8/8/5N2/8/4K3 w - - 0 1"
    ignore_fen = "4k3/5p2/4b3/8/8/5N2/8/4K3 w - - 0 1"

    # Valid fork: "bishop on e6" is the object of "attacking", so "threatening h7"
    # must bind to Ng5 (which attacks h7), not to the enemy bishop on e6.
    res_fork = verify_claims(
        fork_fen, "White plays Ng5, attacking the bishop on e6 and threatening h7."
    )
    assert res_fork["passed"] is True
    assert res_fork["violations"] == []

    # Hallucinated move attack: "bishop on e6" is the object of "ignoring", so
    # "attacking f7" must bind to Nd2 (which does NOT attack f7), not Be6.
    res_ignore = verify_claims(
        ignore_fen, "White plays Nd2, ignoring the bishop on e6 and attacking f7."
    )
    assert res_ignore["passed"] is False
    assert any(
        v["type"] == "invalid_attack_claim" and "Move Nd2" in v["reason"]
        for v in res_ignore["violations"]
    )

    # Non-listed verb preceding a piece + coordinating conjunction + participle
    # still falls back to the move when both exist in the same sentence.
    res_conj = verify_claims(
        ignore_fen, "White plays Nd2, bypassing the bishop on e6 and attacking f7."
    )
    assert res_conj["passed"] is False
    assert any(
        v["type"] == "invalid_attack_claim" and "Move Nd2" in v["reason"]
        for v in res_conj["violations"]
    )

    # Independent subject piece in a subordinate clause still binds to the piece.
    res_subj = verify_claims(ITALIAN_FEN, "White plays d3, while the bishop on c4 attacks f7.")
    assert res_subj["passed"] is True
    assert res_subj["violations"] == []


def test_possessive_kings_safety_not_flagged_as_fake_check():
    for text in (
        "Before advancing, try checking the king's safety on e1.",
        "White plays Nf3, checking the king's safety before castling.",
        "White plays Nf3, checking the king\u2019s safety before castling.",
    ):
        res = verify_claims(START_FEN, text)
        assert res["passed"] is True, f"Unexpected failure for: {text} -> {res['violations']}"
        assert res["violations"] == []

    # Non-possessive "checking the black king" must still be caught.
    res_real = verify_claims(START_FEN, "White plays Nf3, checking the black king.")
    assert res_real["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_real["violations"])


def test_single_sentence_missed_opportunity_check_guard():
    row_201_fen = "5R2/7p/p1p1k1p1/2p1b3/2r5/6P1/P6P/6K1 b - - 1 32"
    for text in (
        "Re4 misses a powerful tactical opportunity to improve your position with a check.",
        "Re4 centralizes the rook instead of improving your position with a check.",
    ):
        res = verify_claims(row_201_fen, text)
        assert res["passed"] is True, f"Unexpected failure for: {text} -> {res['violations']}"
        assert res["violations"] == []

    # Direct false check in the same sentence without missed-opportunity phrasing still fails.
    res_bad = verify_claims(row_201_fen, "Re4 improves your position with a check.")
    assert res_bad["passed"] is False
    assert any(v["type"] == "fake_check" for v in res_bad["violations"])


def test_relocation_infinitive_noun_move_not_bypassed():
    # Noun phrases like "this move" / "the best move" must NOT suppress static
    # piece_on_square attack verification.
    for text in (
        "After this move, the bishop on c4 tries to attack h8.",
        "The best move lets the bishop on c4 continue to attack h8.",
        "Your move allows the bishop on c4 to target h8.",
    ):
        res = verify_claims(ITALIAN_FEN, text)
        assert res["passed"] is False, f"Expected invalid_attack_claim for: {text}"
        assert any(v["type"] == "invalid_attack_claim" for v in res["violations"])

    # Genuine verb-form relocation (Teacher Row #3 and imperative "Move ...") still passes.
    teacher_row_3 = (
        "Look for a way to reposition the knight on c7 to challenge your dominant "
        "rook on g7 or to defend the critical d6 square."
    )
    assert verify_claims(CASE_0_FEN, teacher_row_3)["passed"] is True
    assert (
        verify_claims(CASE_0_FEN, "Move the knight on c7 to challenge your dominant rook on g7.")[
            "passed"
        ]
        is True
    )


def test_curly_apostrophe_contraction_negations():
    rook_blocked_fen = "4k3/8/8/8/4P3/8/8/4R1K1 w - - 0 1"
    empty_center_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    assert verify_claims(rook_blocked_fen, "White plays Re2, which doesn\u2019t attack e8.")[
        "passed"
    ]
    assert verify_claims(START_FEN, "Playing Nf3 isn\u2019t challenging the rook on a8.")["passed"]
    assert verify_claims(START_FEN, "Playing Nf3 can\u2019t force the black king to move.")[
        "passed"
    ]
    assert verify_claims(empty_center_fen, "There isn\u2019t any tension in the center.")["passed"]
