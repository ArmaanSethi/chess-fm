"""Unit tests for the transposition-aware ECO opening book and lookup helpers."""

from __future__ import annotations

import chess
import pytest

from chessfm.coach import openings
from chessfm.coach.openings import (
    OPENING_BOOK,
    OpeningEntry,
    _fen_index,
    identify_opening,
    list_openings,
)
from chessfm.data_generation.phase_sampler import normalize_board_fen


def test_opening_book_size_and_legal_unique_fens() -> None:
    """Every opening entry replays legally from start and yields a unique 4-field FEN."""
    assert len(OPENING_BOOK) >= 35

    seen_fens: dict[str, str] = {}
    for entry in OPENING_BOOK:
        assert entry.eco and len(entry.eco) == 3
        assert entry.name
        assert entry.moves
        assert entry.white_plan
        assert entry.black_plan
        assert entry.common_trap

        board = chess.Board()
        for san in entry.moves:
            board.push_san(san)

        key = normalize_board_fen(board.fen())
        assert key not in seen_fens, f"FEN collision between {entry.name} and {seen_fens[key]}"
        seen_fens[key] = entry.name

        by_fen = identify_opening(fen=board.fen())
        assert by_fen is not None
        assert by_fen["name"] == entry.name
        assert by_fen["eco"] == entry.eco
        assert by_fen["moves"] == list(entry.moves)

        by_moves = identify_opening(moves_san=entry.moves)
        assert by_moves is not None
        assert by_moves["name"] == entry.name

    assert len(_fen_index()) == len(OPENING_BOOK)


def test_transposition_matching_qgd_and_scholars_mate() -> None:
    """Different move orders reaching the same board state resolve to the same opening."""
    # Queen's Gambit Declined via 1. c4 e6 2. d4 d5 and 1. d4 e6 2. c4 d5
    for order in (["c4", "e6", "d4", "d5"], ["d4", "e6", "c4", "d5"]):
        board = chess.Board()
        for san in order:
            board.push_san(san)

        by_fen = identify_opening(fen=board.fen())
        by_moves = identify_opening(moves_san=order)
        assert by_fen is not None and by_fen["name"] == "Queen's Gambit Declined"
        assert by_moves is not None and by_moves["name"] == "Queen's Gambit Declined"

    # Scholar's Mate Trap via 1. e4 e5 2. Qh5 Nc6 3. Bc4
    transposed_scholar = ["e4", "e5", "Qh5", "Nc6", "Bc4"]
    result = identify_opening(moves_san=transposed_scholar)
    assert result is not None
    assert result["name"] == "Scholar's Mate Trap"


def test_four_field_and_pseudo_legal_ep_fen_matching() -> None:
    """FEN lookup accepts 4-field FENs and FENs with pseudo-legal en passant squares."""
    # 1. c4 (English Opening) with pseudo-legal en passant square c3
    pseudo_ep_fen = "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1"
    result = identify_opening(fen=pseudo_ep_fen)
    assert result is not None
    assert result["name"] == "English Opening"

    # 4-field FEN (no halfmove/fullmove counters)
    four_field = "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq -"
    result_4 = identify_opening(fen=four_field)
    assert result_4 is not None
    assert result_4["eco"] == "A10"


def test_deepest_match_prefix_lookup() -> None:
    """Walking moves_san returns the deepest book entry along the game prefix."""
    najdorf_game = [
        "e4",
        "c5",
        "Nf3",
        "d6",
        "d4",
        "cxd4",
        "Nxd4",
        "Nf6",
        "Nc3",
        "a6",
        "Be3",
        "e5",
        "Nb3",
    ]
    result = identify_opening(moves_san=najdorf_game)
    assert result is not None
    assert result["name"] == "Sicilian Defense: Najdorf Variation"
    assert result["eco"] == "B90"

    fried_liver_game = [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bc4",
        "Nf6",
        "Ng5",
        "d5",
        "exd5",
        "Nxd5",
        "Nxf7",
        "Kxf7",
        "Qf3+",
    ]
    fl_result = identify_opening(moves_san=fried_liver_game)
    assert fl_result is not None
    assert fl_result["name"] == "Fried Liver Attack"


def test_invalid_move_in_history_returns_deepest_valid_match() -> None:
    """If an illegal/invalid SAN appears later in moves_san, return the deepest match so far."""
    result = identify_opening(moves_san=["e4", "c5", "c3", "NotAMove"])
    assert result is not None
    assert result["name"] == "Sicilian Defense: Alapin Variation"


def test_moves_san_fallback_to_fen() -> None:
    """When moves_san yields no book hit, identify_opening falls back to fen."""
    board = chess.Board()
    for san in ("d4", "f5"):
        board.push_san(san)

    result = identify_opening(fen=board.fen(), moves_san=["a3", "h6"])
    assert result is not None
    assert result["name"] == "Dutch Defense"


def test_unknown_and_invalid_inputs_return_none() -> None:
    """Unknown positions, empty args, and malformed FENs return None cleanly."""
    assert identify_opening() is None
    assert identify_opening(fen=chess.STARTING_FEN) is None
    assert identify_opening(fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1") is None
    assert identify_opening(fen="not-a-valid-fen") is None
    assert identify_opening(moves_san=["a3", "h6"]) is None


def test_list_openings_and_eco_filtering() -> None:
    """list_openings returns all openings or filters case-insensitively by ECO prefix."""
    all_openings = list_openings()
    assert len(all_openings) == len(OPENING_BOOK)
    required_keys = {"eco", "name", "moves", "white_plan", "black_plan", "common_trap"}
    for item in all_openings:
        assert set(item.keys()) == required_keys

    b_openings = list_openings("b")
    assert b_openings
    assert all(item["eco"].startswith("B") for item in b_openings)

    c5_openings = list_openings(" c5 ")
    assert c5_openings
    assert all(item["eco"].startswith("C5") for item in c5_openings)

    assert list_openings("Z") == []


def test_fen_index_rejects_duplicate_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fen_index raises ValueError if two book entries resolve to the same FEN."""
    dup_a = OpeningEntry(
        eco="A10",
        name="Dup A",
        moves=("c4",),
        white_plan="p",
        black_plan="p",
        common_trap="t",
    )
    dup_b = OpeningEntry(
        eco="A10",
        name="Dup B",
        moves=("c4",),
        white_plan="p",
        black_plan="p",
        common_trap="t",
    )
    monkeypatch.setattr(openings, "OPENING_BOOK", (dup_a, dup_b))
    _fen_index.cache_clear()
    try:
        with pytest.raises(ValueError, match="Duplicate opening FEN key"):
            _fen_index()
    finally:
        _fen_index.cache_clear()
