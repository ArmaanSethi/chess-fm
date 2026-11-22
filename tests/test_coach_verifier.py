#!/usr/bin/env python3
"""Unit tests for the deterministic Zero-Hallucination Claim Verifier."""

from __future__ import annotations

import chess
import chess.engine
import pytest

from chessfm.coach.verifier import extract_claims, verify_and_repair, verify_claims

START_FEN = chess.STARTING_FEN
# After 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 — Qxf7# is checkmate.
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# Italian game after 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6
ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
# Ambiguous rook move position: White rooks on a1 and f1, both can play to e1.
AMBIGUOUS_ROOK_FEN = "4k3/8/8/8/8/8/8/R3KR2 w - - 0 1"
# Position where White king is on e1, rook on h1, castling rights K.
CASTLE_FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
# Position where White has pawn on e7 ready to promote.
PROMO_FEN = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
# Already checkmate position (Fool's mate).
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


class _MockEngine:
    """Minimal deterministic Stockfish double returning configured cp scores."""

    def __init__(self, scores_by_fen: dict[str, int], default_cp: int = 0) -> None:
        self._scores = scores_by_fen
        self._default = default_cp

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict:
        cp = self._scores.get(board.fen(), self._default)
        # PovScore expects score from White's perspective when using chess.WHITE.
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}


# ---------------------------------------------------------------------------
# extract_claims tests
# ---------------------------------------------------------------------------


def test_extract_claims_empty_and_conceptual_prose():
    assert extract_claims("") == []
    prose = "Develop your pieces, control e4 and d5, and safeguard your king."
    assert extract_claims(prose) == []


def test_extract_claims_moves_uci_and_san():
    text = (
        "Play 1. e4 e5 2. Nf3 Nc6 3. Bc4 or try e2e4 and e7e8q. "
        "Also consider Bxc6+, O-O, Qxf7#, and exd5."
    )
    claims = extract_claims(text)
    moves = [c["move"] for c in claims if c["type"] == "move"]
    assert moves == [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bc4",
        "e2e4",
        "e7e8q",
        "Bxc6+",
        "O-O",
        "Qxf7#",
        "exd5",
    ]


def test_extract_claims_piece_on_square():
    text = (
        "The knight on f6 defends the center, while the white bishop on c4 "
        "and black rook on a8 eye the flanks, and the queen at d1 is active."
    )
    claims = extract_claims(text)
    pos_claims = [c for c in claims if c["type"] == "piece_on_square"]
    assert len(pos_claims) == 4
    assert pos_claims[0]["piece"] == "knight"
    assert pos_claims[0]["square"] == "f6"
    assert pos_claims[0]["color"] is None
    assert pos_claims[1]["color"] == "white"
    assert pos_claims[1]["piece"] == "bishop"
    assert pos_claims[1]["square"] == "c4"
    assert pos_claims[2]["color"] == "black"
    assert pos_claims[2]["piece"] == "rook"
    assert pos_claims[2]["square"] == "a8"
    assert pos_claims[3]["piece"] == "queen"
    assert pos_claims[3]["square"] == "d1"


def test_extract_claims_check_and_negation():
    text = (
        "Qh5 gives check, Qxf7 delivers checkmate, and checkmate on f7 ends it. "
        "Note that Nf3 does not give check and avoids checkmate."
    )
    claims = extract_claims(text)
    check_claims = [c for c in claims if c["type"] == "check_claim"]
    assert len(check_claims) == 3
    assert check_claims[0]["kind"] == "check"
    assert check_claims[1]["kind"] == "checkmate"
    assert check_claims[2]["kind"] == "checkmate"
    assert check_claims[2]["square"] == "f7"


# ---------------------------------------------------------------------------
# verify_claims tests — valid claims
# ---------------------------------------------------------------------------


def test_verify_claims_valid_variation_and_pieces():
    text = (
        "White has a white bishop on c4 and a knight on f3, while Black has "
        "a black knight on f6 and a black rook on a8. "
        "Play 1. O-O Nxe4 2. Re1 to fight for the initiative."
    )
    res = verify_claims(ITALIAN_FEN, text)
    assert res["passed"] is True
    assert res["factuality_score"] == pytest.approx(1.0)
    assert res["total_claims"] == 7
    assert res["verified_claims"] == 7
    assert res["violations"] == []


def test_verify_claims_valid_alternatives_from_root():
    text = "Consider playing Nf3 or d4 or e2e4 from the starting position."
    res = verify_claims(START_FEN, text)
    assert res["passed"] is True
    assert res["total_claims"] == 3
    assert res["verified_claims"] == 3


def test_verify_claims_valid_checkmate_and_promotion():
    res_mate = verify_claims(
        SCHOLARS_MATE_FEN,
        "The white queen on h5 plays Qxf7#, which delivers checkmate on f7!",
    )
    assert res_mate["passed"] is True
    assert res_mate["violations"] == []

    res_promo = verify_claims(
        PROMO_FEN,
        "Play e7e8q or e8=Q to promote the pawn on e7.",
    )
    assert res_promo["passed"] is True

    res_already_mate = verify_claims(CHECKMATE_FEN, "This position is checkmate.")
    assert res_already_mate["passed"] is True


def test_verify_claims_no_claims_passes():
    res = verify_claims(START_FEN, "Focus on piece activity and center control.")
    assert res["passed"] is True
    assert res["factuality_score"] == 1.0
    assert res["total_claims"] == 0
    assert res["verified_claims"] == 0


# ---------------------------------------------------------------------------
# Synthetic Hallucination Suite (must achieve 100% catch rate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fen", "hallucinated_text", "expected_violation_type"),
    [
        # Illegal UCI moves
        (START_FEN, "Play e2e5 to seize the center.", "illegal_move"),
        (START_FEN, "Move a1a8 to dominate the file.", "illegal_move"),
        (START_FEN, "Play e1e1 to pass the turn.", "illegal_move"),
        # Illegal SAN moves
        (START_FEN, "Play Nf6 immediately.", "illegal_move"),
        (START_FEN, "Castle with O-O on move one.", "illegal_move"),
        (ITALIAN_FEN, "Play Bxh7+ to win a pawn.", "illegal_move"),
        # Ambiguous SAN move
        (AMBIGUOUS_ROOK_FEN, "Play Re1 to centralize.", "illegal_move"),
        # Wrong piece on square (empty, wrong type, wrong color)
        (START_FEN, "The knight on f6 is very strong.", "piece_on_square"),
        (START_FEN, "White's queen on e1 is safe.", "piece_on_square"),
        (ITALIAN_FEN, "The black bishop on c4 targets f7.", "piece_on_square"),
        (ITALIAN_FEN, "The white knight on f6 attacks e4.", "piece_on_square"),
        # Fake check annotations & prose check claims
        (START_FEN, "Play Nf3+ to seize the initiative.", "invalid_check_annotation"),
        (START_FEN, "Play Nf3, which gives check.", "fake_check"),
        (START_FEN, "White has a check on f7.", "fake_check"),
        (START_FEN, "White delivers check right now.", "fake_check"),
        # Fake checkmate annotations & prose checkmate claims
        (ITALIAN_FEN, "Play Bxf7# for an instant win.", "invalid_check_annotation"),
        (ITALIAN_FEN, "Play Bxf7+, which delivers checkmate.", "fake_checkmate"),
        (ITALIAN_FEN, "There is checkmate on f7.", "fake_checkmate"),
        (START_FEN, "White delivers checkmate.", "fake_checkmate"),
        (
            SCHOLARS_MATE_FEN,
            "Play Qxf7#, which delivers checkmate on h7.",
            "fake_checkmate",
        ),
    ],
)
def test_synthetic_hallucinations_caught_100_percent(
    fen: str, hallucinated_text: str, expected_violation_type: str
):
    res = verify_claims(fen, hallucinated_text)
    assert res["passed"] is False, f"Missed hallucination: {hallucinated_text}"
    assert res["factuality_score"] < 1.0
    violation_types = [v["type"] for v in res["violations"]]
    assert expected_violation_type in violation_types


# ---------------------------------------------------------------------------
# Optional Engine Blunder Verification
# ---------------------------------------------------------------------------


def test_verify_claims_with_engine_catches_blunder():
    board_after_blunder = chess.Board(START_FEN)
    board_after_blunder.push_san("f3")
    board_after_good = chess.Board(START_FEN)
    board_after_good.push_san("e4")

    engine = _MockEngine(
        scores_by_fen={
            START_FEN: 30,
            board_after_blunder.fen(): -350,  # 380 cp loss (> 200)
            board_after_good.fen(): 25,  # 5 cp loss
        }
    )

    bad_res = verify_claims(START_FEN, "Play f3 to open the diagonal.", engine=engine)
    assert bad_res["passed"] is False
    assert bad_res["violations"][0]["type"] == "blunder"

    # Avoidance warnings should NOT be flagged as recommended blunders.
    warn_res = verify_claims(START_FEN, "Avoid f3 because it weakens the kingside.", engine=engine)
    assert warn_res["passed"] is True

    good_res = verify_claims(START_FEN, "Play e4 to claim the center.", engine=engine)
    assert good_res["passed"] is True


# ---------------------------------------------------------------------------
# verify_and_repair tests
# ---------------------------------------------------------------------------


def test_verify_and_repair_no_retry_needed():
    called = False

    def repair_fn(text: str, violations: list[dict]) -> str:
        nonlocal called
        called = True
        return text

    result = verify_and_repair(ITALIAN_FEN, "The white bishop on c4 supports O-O.", repair_fn)
    assert called is False
    assert result["retries_used"] == 0
    assert result["verification"]["passed"] is True


def test_verify_and_repair_converges_in_one_retry():
    def repair_fn(text: str, violations: list[dict]) -> str:
        assert len(violations) >= 1
        return "Play Nf3 to develop the knight on g1."

    result = verify_and_repair(
        START_FEN,
        "Play Nf6 with the knight on f6, which gives check.",
        repair_fn,
        max_retries=2,
    )
    assert result["retries_used"] == 1
    assert result["verification"]["passed"] is True
    assert result["verification"]["factuality_score"] == 1.0
    assert "Nf3" in result["text"]


def test_verify_and_repair_converges_in_two_retries():
    attempts = [
        "Play Nf3+ to develop.",  # Still has fake check annotation
        "Play Nf3 to develop the knight on g1.",  # Clean
    ]

    def repair_fn(text: str, violations: list[dict]) -> str:
        return attempts.pop(0)

    result = verify_and_repair(START_FEN, "Play e2e5 to attack.", repair_fn, max_retries=2)
    assert result["retries_used"] == 2
    assert result["verification"]["passed"] is True


def test_verify_and_repair_exhausts_retries():
    def stubborn_repair(text: str, violations: list[dict]) -> str:
        return "Play e2e5 again."

    result = verify_and_repair(START_FEN, "Play Nf6.", stubborn_repair, max_retries=2)
    assert result["retries_used"] == 2
    assert result["verification"]["passed"] is False


def test_verify_and_repair_invalid_max_retries():
    with pytest.raises(ValueError, match="max_retries"):
        verify_and_repair(START_FEN, "Play e4.", lambda t, v: t, max_retries=-1)
