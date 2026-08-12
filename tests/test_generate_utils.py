#!/usr/bin/env python3
"""Unit tests for synthetic-data helpers (pure, no API/Stockfish required)."""

import json

import chess

from chessfm.data_generation.download_positions import generate_positions
from chessfm.data_generation.generate_synthetic_data import (
    extract_move,
    is_uci,
    load_positions,
    maker_prompt,
    resume_from,
    validate_with_stockfish,
)


class TestIsUci:
    def test_valid(self):
        for s in ["e2e4", "g1f3", "e7e8q", "a7a8n"]:
            assert is_uci(s), s

    def test_invalid(self):
        for s in ["Nf3", "e4", "O-O", "exd5", "1.e4", "e2e4q6", None, 42]:
            assert not is_uci(s), s


class TestExtractMove:
    def test_after_think(self):
        assert extract_move("<think>Good center.</think>\ne2e4") == "e2e4"

    def test_last_token_wins(self):
        assert extract_move("e2e4 d2d4") == "d2d4"

    def test_strips_punctuation(self):
        assert extract_move("e2e4.") == "e2e4"
        assert extract_move("1. e4 e5") is None  # "e4" is SAN, not UCI

    def test_no_move(self):
        assert extract_move("<think>Unknown.</think>") is None


class TestPrompts:
    def test_maker_prompt_color(self):
        assert "Side to move: White" in maker_prompt("8/8/8/8/8/8/8/8 w - - 0 1")
        assert "Side to move: Black" in maker_prompt("8/8/8/8/8/8/8/8 b - - 0 1")


class TestLoadResume:
    def test_load_positions_respects_limit(self, tmp_path):
        p = tmp_path / "fens.txt"
        p.write_text("a\nb\nc\n\n")
        assert load_positions(p, limit=2) == ["a", "b"]
        assert load_positions(p, limit=None) == ["a", "b", "c"]

    def test_resume_from_returns_done_fens(self, tmp_path):
        out = tmp_path / "data.jsonl"
        out.write_text(
            json.dumps({"fen": "X"}) + "\n" + "garbage\n" + json.dumps({"fen": "Y"}) + "\n"
        )
        assert resume_from(out) == {"X", "Y"}
        assert resume_from(tmp_path / "missing.jsonl") == set()


class _FakeScore:
    def __init__(self, cp):
        self._cp = cp

    def is_mate(self):
        return False

    def white(self):
        return self

    def score(self):
        return self._cp


class _MockEngine:
    """White-centric eval that changes after the move is played.

    The initial board is white to move and yields ``best``; the board after the
    move is black to move and yields ``chosen``. Both are expressed from
    White's perspective, matching what a real Stockfish info dict provides.
    """

    def __init__(self, best=50, chosen=50):
        self._best = best
        self._chosen = chosen

    def analyse(self, board, limit):
        white_cp = self._best if board.turn == chess.WHITE else self._chosen
        return {"score": _FakeScore(white_cp)}


class TestValidateWithStockfish:
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_good_move(self):
        # e2e4 keeps White at +50 before and after -> zero centipawn loss.
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(), self.FEN, "e2e4")
        assert is_legal is True
        assert loss == 0.0

    def test_lossy_move(self):
        # A move that drops from +50 to -100 is clearly a blunder.
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(chosen=-100), self.FEN, "e2e4")
        assert is_legal is True
        assert loss == 150.0

    def test_illegal_move(self):
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(), self.FEN, "e2e5")
        assert is_legal is False

    def test_garbage_move(self):
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(), self.FEN, "not-a-move")
        assert is_legal is False


class TestGeneratePositions:
    def test_count_respected(self):
        positions = generate_positions(10)
        assert len(positions) == 10

    def test_all_valid_fens(self):
        for fen in generate_positions(30):
            chess.Board(fen)  # raises ValueError if the FEN is malformed
            assert len(fen.split()) == 6

    def test_unique_within_small_request(self):
        positions = generate_positions(5)
        assert len(set(positions)) == 5

    def test_cycles_when_count_exceeds_curated(self):
        # Fewer than ~100 curated positions; a request for 1000 must cycle.
        positions = generate_positions(1000)
        assert len(positions) == 1000
        assert len(set(positions)) < len(positions)
