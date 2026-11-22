#!/usr/bin/env python3
"""Unit tests for the 3-Level Socratic Hint Engine (0%-spoiler guarantee)."""

from __future__ import annotations

import re

import chess
import pytest

from chessfm.coach.socratic import contains_move_spoiler, generate_socratic_hint

# ---------------------------------------------------------------------------
# Diverse 22-position benchmark covering openings, tactics, promotions,
# castling, en passant, discovered checks, forks, and checkmates.
# ---------------------------------------------------------------------------

DIVERSE_POSITIONS: list[tuple[str, str, str, list[str] | None]] = [
    # 1. Opening central pawn push
    ("opening_e4", chess.STARTING_FEN, "e2e4", ["e2e4", "e7e5", "g1f3"]),
    # 2. Opening d4 pawn push
    ("opening_d4", chess.STARTING_FEN, "d2d4", ["d2d4", "d7d5"]),
    # 3. Opening knight development
    ("opening_nf3", chess.STARTING_FEN, "g1f3", ["g1f3", "d7d5"]),
    # 4. Black knight development
    (
        "black_nc6",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "b8c6",
        ["b8c6", "f1b5"],
    ),
    # 5. Italian bishop development
    (
        "italian_bc4",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "f1c4",
        ["f1c4", "f8c5"],
    ),
    # 6. White kingside castling
    (
        "white_castle_kingside",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "e1g1",
        ["e1g1", "e8g8"],
    ),
    # 7. White queenside castling
    (
        "white_castle_queenside",
        "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 7",
        "e1c1",
        ["e1c1", "e8g8"],
    ),
    # 8. Black kingside castling
    (
        "black_castle_kingside",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
        "e8g8",
        ["e8g8", "d2d3"],
    ),
    # 9. Queen promotion
    ("promo_queen", "7k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q", ["e7e8q", "h8h7"]),
    # 10. Knight underpromotion with check
    (
        "promo_knight_check",
        "6k1/4P3/8/8/8/8/8/4K3 w - - 0 1",
        "e7e8n",
        ["e7e8n", "g8f7"],
    ),
    # 11. Promotion with capture
    (
        "promo_capture",
        "2r4k/3P4/8/8/8/8/8/4K3 w - - 0 1",
        "d7c8q",
        ["d7c8q", "h8g7"],
    ),
    # 12. Scholar's mate
    (
        "scholars_mate",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "h5f7",
        ["h5f7"],
    ),
    # 13. Back-rank mate
    (
        "back_rank_mate",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "d1d8",
        ["d1d8"],
    ),
    # 14. Smothered mate with knight
    (
        "smothered_mate",
        "6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1",
        "h6f7",
        ["h6f7"],
    ),
    # 15. Royal knight fork (king + queen)
    (
        "knight_fork",
        "4k3/8/3q4/8/5N2/8/8/4K3 w - - 0 1",
        "f4e6",
        None,
    ),
    # 16. Discovered check
    (
        "discovered_check",
        "4k3/8/8/8/4B3/8/8/4R1K1 w - - 0 1",
        "e4c6",
        ["e4c6", "e8f7"],
    ),
    # 17. En passant capture
    (
        "en_passant",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "e5d6",
        ["e5d6", "d8d6"],
    ),
    # 18. Central pawn capture
    (
        "pawn_capture",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "e4d5",
        ["e4d5", "d8d5"],
    ),
    # 19. Queen captures undefended rook
    (
        "queen_capture",
        "4k3/8/8/3r4/8/8/3Q4/4K3 w - - 0 1",
        "d2d5",
        ["d2d5", "e8e7"],
    ),
    # 20. Rook invasion on the 7th rank
    (
        "rook_7th_rank",
        "6k1/pp3ppp/8/8/8/8/PP3PPP/4R1K1 w - - 0 1",
        "e1e7",
        ["e1e7", "g8f8"],
    ),
    # 21. Forcing bishop check
    (
        "bishop_check",
        "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
        "f1b5",
        ["f1b5", "c7c6"],
    ),
    # 22. Pawn check
    (
        "pawn_check",
        "4k3/8/5P2/8/8/8/8/4K3 w - - 0 1",
        "f6f7",
        ["f6f7", "e8f7"],
    ),
]


# ---------------------------------------------------------------------------
# contains_move_spoiler tests
# ---------------------------------------------------------------------------


def test_contains_move_spoiler_uci_variants():
    fen = chess.STARTING_FEN
    assert contains_move_spoiler("Play e2e4 now.", "e2e4", fen) is True
    assert contains_move_spoiler("Try E2E4.", "e2e4", fen) is True
    assert contains_move_spoiler("Move e2-e4.", "e2e4", fen) is True
    assert contains_move_spoiler("Play e2->e4.", "e2e4", fen) is True
    assert contains_move_spoiler("Play e2\u2192e4.", "e2e4", fen) is True


def test_contains_move_spoiler_san_and_aliases():
    scholars = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    assert contains_move_spoiler("Play Qxf7#!", "h5f7", scholars) is True
    assert contains_move_spoiler("Consider Qxf7.", "h5f7", scholars) is True
    assert contains_move_spoiler("Target the f7 square.", "h5f7", scholars) is False

    castle_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    assert contains_move_spoiler("Castle with O-O.", "e1g1", castle_fen) is True
    assert contains_move_spoiler("Castle with 0-0.", "e1g1", castle_fen) is True
    assert contains_move_spoiler("Think about king safety.", "e1g1", castle_fen) is False

    qcastle_fen = "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 7"
    assert contains_move_spoiler("Play O-O-O.", "e1c1", qcastle_fen) is True
    assert contains_move_spoiler("Play 0-0-0.", "e1c1", qcastle_fen) is True


def test_contains_move_spoiler_origin_and_dest_pair():
    fen = chess.STARTING_FEN
    assert contains_move_spoiler("Move the knight from g1 to f3.", "g1f3", fen) is True
    assert contains_move_spoiler("Focus on the f3 square.", "g1f3", fen) is False
    assert contains_move_spoiler("Focus on your knight.", "g1f3", fen) is False
    assert contains_move_spoiler("", "g1f3", fen) is False


def test_contains_move_spoiler_promotion():
    promo_fen = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
    assert contains_move_spoiler("Play e7e8q.", "e7e8q", promo_fen) is True
    assert contains_move_spoiler("Play e7e8.", "e7e8q", promo_fen) is True
    assert contains_move_spoiler("Promote with e8=Q.", "e7e8q", promo_fen) is True
    assert (
        contains_move_spoiler("Push your passed pawn on the e-file.", "e7e8q", promo_fen) is False
    )


def test_contains_move_spoiler_illegal_move_raises():
    with pytest.raises(ValueError, match="not legal"):
        contains_move_spoiler("Play e2e5.", "e2e5", chess.STARTING_FEN)


# ---------------------------------------------------------------------------
# generate_socratic_hint tests across all 22 diverse positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fen", "best_uci", "pv"),
    DIVERSE_POSITIONS,
    ids=[p[0] for p in DIVERSE_POSITIONS],
)
def test_socratic_hints_zero_spoilers_across_diverse_positions(
    name: str, fen: str, best_uci: str, pv: list[str] | None
):
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_uci)
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    expected_san = board.san(move)

    # Level 1: strict 0% spoiler AND neither origin nor destination square mentioned.
    h1 = generate_socratic_hint(fen, best_uci, level=1, pv_ucis=pv)
    assert h1["level"] == 1
    assert h1["spoiler_free"] is True
    assert not contains_move_spoiler(h1["hint"], best_uci, fen)
    assert not re.search(rf"\b{from_sq}\b", h1["hint"], re.IGNORECASE)
    assert not re.search(rf"\b{to_sq}\b", h1["hint"], re.IGNORECASE)
    assert h1["move_san"] is None
    assert h1["move_uci"] is None
    assert h1["continuation_san"] == []

    # Level 2: strict 0% spoiler (no exact move, no origin+dest pair).
    h2 = generate_socratic_hint(fen, best_uci, level=2, pv_ucis=pv)
    assert h2["level"] == 2
    assert h2["spoiler_free"] is True
    assert not contains_move_spoiler(h2["hint"], best_uci, fen)
    assert h2["move_san"] is None
    assert h2["move_uci"] is None
    assert h2["continuation_san"] == []

    # Level 3: full verified move (SAN + UCI) and verified continuation.
    h3 = generate_socratic_hint(fen, best_uci, level=3, pv_ucis=pv)
    assert h3["level"] == 3
    assert h3["move_san"] == expected_san
    assert h3["move_uci"] == best_uci
    assert expected_san in h3["hint"]
    assert best_uci in h3["hint"]
    assert contains_move_spoiler(h3["hint"], best_uci, fen) is True
    if pv and len(pv) > 1:
        assert len(h3["continuation_san"]) >= 2
        assert h3["continuation_san"][0] == expected_san


def test_socratic_sanitizes_spoiler_themes():
    fen = chess.STARTING_FEN
    # Caller injects spoilers inside themes; Levels 1 & 2 must still be 0% spoiler.
    h1 = generate_socratic_hint(fen, "g1f3", level=1, themes=["play Nf3", "use g1", "development"])
    assert h1["spoiler_free"] is True
    assert not contains_move_spoiler(h1["hint"], "g1f3", fen)
    assert "Nf3" not in h1["hint"]
    assert "g1" not in h1["hint"]

    h2 = generate_socratic_hint(fen, "g1f3", level=2, themes=["g1f3", "Nf3", "center control"])
    assert h2["spoiler_free"] is True
    assert not contains_move_spoiler(h2["hint"], "g1f3", fen)


def test_socratic_level_3_pv_reply_only_and_invalid_tail():
    fen = chess.STARTING_FEN
    # pv_ucis without best_move_uci as first element, plus a bogus move at end.
    h3 = generate_socratic_hint(fen, "e2e4", level=3, pv_ucis=["e7e5", "g1f3", "notamove", "a1a8"])
    assert h3["continuation_san"] == ["e4", "e5", "Nf3"]
    assert "Verified continuation: e4 e5 Nf3." in h3["hint"]


def test_socratic_invalid_inputs():
    with pytest.raises(ValueError, match="level must be"):
        generate_socratic_hint(chess.STARTING_FEN, "e2e4", level=0)
    with pytest.raises(ValueError, match="level must be"):
        generate_socratic_hint(chess.STARTING_FEN, "e2e4", level=4)
    with pytest.raises(ValueError, match="not legal"):
        generate_socratic_hint(chess.STARTING_FEN, "e2e5", level=1)
