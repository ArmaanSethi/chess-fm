#!/usr/bin/env python3
"""Unit tests for synthetic-data helpers (pure, no API/Stockfish required)."""

import json

import chess
import pytest

from chessfm.data_generation.download_positions import generate_positions
from chessfm.data_generation.generate_synthetic_data import (
    ProviderAPI,
    gen_batch,
    load_positions,
    resume_from,
    validate_with_stockfish,
)


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


class _MockEngineWithPV(_MockEngine):
    """Like ``_MockEngine`` but also reports a principal variation."""

    def __init__(self, best=50, chosen=50, pv="e2e4"):
        super().__init__(best, chosen)
        self._pv = pv

    def analyse(self, board, limit):
        info = super().analyse(board, limit)
        info["pv"] = [chess.Move.from_uci(self._pv)]
        return info


class TestValidateWithStockfish:
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_good_move(self):
        # e2e4 keeps White at +50 before and after -> zero centipawn loss.
        is_legal, loss, best = validate_with_stockfish(_MockEngine(), self.FEN, "e2e4")
        assert is_legal is True
        assert loss == 0.0
        assert best is None  # no PV reported by this mock

    def test_best_move_reported_when_pv_present(self):
        is_legal, loss, best = validate_with_stockfish(
            _MockEngineWithPV(pv="e2e4"), self.FEN, "e2e4"
        )
        assert is_legal is True
        assert loss == 0.0
        assert best == "e2e4"

    def test_lossy_move(self):
        # A move that drops from +50 to -100 is clearly a blunder.
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(chosen=-100), self.FEN, "e2e4")
        assert is_legal is True
        assert loss == 150.0

    def test_loss_never_negative(self):
        # A "better-than-best" move must still be clamped to 0, never negative.
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(chosen=200), self.FEN, "e2e4")
        assert is_legal is True
        assert loss == 0.0

    def test_illegal_move(self):
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(), self.FEN, "e2e5")
        assert is_legal is False

    def test_garbage_move(self):
        is_legal, loss, _ = validate_with_stockfish(_MockEngine(), self.FEN, "not-a-move")
        assert is_legal is False

    def test_engine_none_returns_invalid(self):
        is_legal, loss, best = validate_with_stockfish(None, self.FEN, "e2e4")
        assert is_legal is False
        assert loss is None
        assert best is None


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


class _FakeAPI:
    """Returns canned teacher responses, one per call."""

    def __init__(self, responses, provider="openai", model="gpt-test"):
        self._responses = list(responses)
        self.provider = provider
        self.model = model

    def complete(self, prompt):
        return self._responses.pop(0)


class TestGenBatch:
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_writes_validated_rows(self, tmp_path):
        out = tmp_path / "data.jsonl"
        api = _FakeAPI(["<think>center control.</think>\ne2e4"])
        engine = _MockEngineWithPV(pv="e2e4")
        gen_batch(api, engine, [self.FEN], out, depth=1, delay=0)

        rows = [json.loads(line) for line in out.open() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["move"] == "e2e4"
        assert rows[0]["stockfish_best"] == "e2e4"
        assert rows[0]["is_reasonable"] is True
        assert rows[0]["model"] == "openai:gpt-test"

    def test_resume_skips_done_fens(self, tmp_path):
        out = tmp_path / "data.jsonl"
        out.write_text(json.dumps({"fen": self.FEN}) + "\n")
        api = _FakeAPI(["<think>x</think>\ne2e4"])
        engine = _MockEngineWithPV(pv="e2e4")
        gen_batch(api, engine, [self.FEN], out, depth=1, delay=0)
        # Nothing new appended: the FEN was already processed.
        assert sum(1 for _ in out.open() if _.strip()) == 1

    def test_unparseable_response_is_skipped(self, tmp_path):
        out = tmp_path / "data.jsonl"
        api = _FakeAPI(["no move here at all"])
        engine = _MockEngineWithPV()
        gen_batch(api, engine, [self.FEN], out, depth=1, delay=0)
        assert sum(1 for _ in out.open() if _.strip()) == 0

    def test_illegal_move_is_skipped(self, tmp_path):
        out = tmp_path / "data.jsonl"
        api = _FakeAPI(["<think>push pawn three squares.</think>\ne2e5"])
        engine = _MockEngineWithPV(pv="e2e4")
        gen_batch(api, engine, [self.FEN], out, depth=1, delay=0)
        assert sum(1 for _ in out.open() if _.strip()) == 0

    def test_api_error_is_caught_and_batch_continues(self, tmp_path, capsys):
        class _FlakyAPI:
            provider = "openai"
            model = "gpt-test"

            def __init__(self):
                self.calls = 0

            def complete(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient 429")
                return "<think>solid opening.</think>\ne2e4"

        out = tmp_path / "data.jsonl"
        gen_batch(_FlakyAPI(), _MockEngineWithPV(pv="e2e4"), [self.FEN, self.FEN], out, delay=0)
        rows = [json.loads(line) for line in out.open() if line.strip()]
        assert len(rows) == 1
        assert "API error on 0: transient 429" in capsys.readouterr().out

    def test_progress_logged_every_ten_samples(self, tmp_path, capsys):
        out = tmp_path / "data.jsonl"
        api = _FakeAPI(["<think>ok.</think>\ne2e4"] * 10)
        gen_batch(api, _MockEngineWithPV(pv="e2e4"), [self.FEN] * 10, out, depth=1, delay=0)
        stdout = capsys.readouterr().out
        assert "10/10 processed, 10 kept, 10 legal, 0 skipped" in stdout


class TestProviderAPI:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            ProviderAPI("bogus", "m", "k").complete("p")

    def test_openai_path(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "hello"}}]}

        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
        api = ProviderAPI("openai", "gpt-test", "key")
        assert api.complete("p") == "hello"

    def test_anthropic_path(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": [{"text": "hi"}]}

        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
        api = ProviderAPI("anthropic", "claude-test", "key")
        assert api.complete("p") == "hi"

    def test_gemini_path(self, monkeypatch):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "yo"}]}}]}

        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
        api = ProviderAPI("gemini", "gemini-test", "key")
        assert api.complete("p") == "yo"


class TestMainCLI:
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_missing_api_key_exits(self, tmp_path, monkeypatch):
        from chessfm.data_generation import generate_synthetic_data as gsd

        pos = tmp_path / "positions.txt"
        pos.write_text(self.FEN + "\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--provider", "openai", "--model", "gpt-4o", "--positions", str(pos)],
        )
        with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
            gsd.main()

    def test_missing_stockfish_exits(self, tmp_path, monkeypatch):
        from chessfm.data_generation import generate_synthetic_data as gsd

        pos = tmp_path / "positions.txt"
        pos.write_text(self.FEN + "\n")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(gsd, "open_engine", lambda: None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--provider",
                "anthropic",
                "--model",
                "claude-test",
                "--positions",
                str(pos),
            ],
        )
        with pytest.raises(SystemExit, match="Stockfish not found"):
            gsd.main()

    def test_main_runs_batch_and_closes_engine(self, tmp_path, monkeypatch):
        from chessfm.data_generation import generate_synthetic_data as gsd

        pos = tmp_path / "positions.txt"
        pos.write_text(self.FEN + "\n")
        out = tmp_path / "nested" / "out.jsonl"

        fake_engine = _MockEngineWithPV(pv="e2e4")
        closed = []

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(gsd, "open_engine", lambda: fake_engine)
        monkeypatch.setattr(gsd, "close_engine", lambda eng: closed.append(eng))
        monkeypatch.setattr(
            gsd.ProviderAPI,
            "complete",
            lambda self, prompt: "<think>develop center.</think>\ne2e4",
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--provider",
                "gemini",
                "--model",
                "gemini-test",
                "--positions",
                str(pos),
                "--output",
                str(out),
                "--limit",
                "1",
                "--depth",
                "1",
                "--delay",
                "0",
            ],
        )

        gsd.main()

        assert closed == [fake_engine]
        rows = [json.loads(line) for line in out.open() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["move"] == "e2e4"
        assert rows[0]["model"] == "gemini:gemini-test"
