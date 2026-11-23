#!/usr/bin/env python3
"""
Integration tests for the evaluation harness and Stockfish-dependent rewards.

These run only when Stockfish is available (it is installed in CI). They verify
that the evaluator advances the FEN every ply and that the move-quality reward
produces sane values with a real engine.
"""

import json
import math
import shutil
import unittest

from chessfm.config import STOCKFISH_PATHS

STOCKFISH_OK = any(shutil.which(p) for p in STOCKFISH_PATHS)


@unittest.skipUnless(STOCKFISH_OK, "Stockfish not available")
class TestStockfishRewards(unittest.TestCase):
    """End-to-end checks of the move-quality reward with a real engine."""

    @classmethod
    def setUpClass(cls):
        from chessfm.rl.engine import open_engine

        cls.engine = open_engine(threads=1, hash_mb=16)
        assert cls.engine is not None

    @classmethod
    def tearDownClass(cls):
        from chessfm.rl.engine import close_engine

        close_engine(cls.engine)

    def test_best_move_scores_high(self):
        from chessfm.rl.rewards import reward_move_quality

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        r = reward_move_quality(fen, "e2e4", self.engine, depth=5)
        self.assertGreater(r, 0.5)
        self.assertLessEqual(r, 1.0)

    def test_blunder_scores_low(self):
        from chessfm.rl.rewards import reward_move_quality

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        r_bad = reward_move_quality(fen, "f2f3", self.engine, depth=5)
        r_good = reward_move_quality(fen, "e2e4", self.engine, depth=5)
        self.assertLess(r_bad, r_good)


class _FakeTensor:
    """Torch-free stand-in for a 2D batch tensor (shape, slicing, indexing)."""

    def __init__(self, values):
        self._values = values

    @property
    def shape(self):
        return [1, len(self._values)]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self._values
        if isinstance(item, slice):
            return self._values[item]
        if isinstance(item, int):
            return self._values
        return self._values[item]

    def __iter__(self):
        return iter(self._values)


class _FakeInputs:
    """Fake tokenizer output supporting .input_ids, .to(), and **-unpacking."""

    def __init__(self, ids):
        self.input_ids = _FakeTensor(ids)

    def to(self, device):
        return self

    def keys(self):
        return ["input_ids"]

    def __getitem__(self, key):
        return getattr(self, key)


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        ids = list(range(len(prompt.split()))) + [99, 100]
        return _FakeInputs(ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=None):
        return "decoded:" + ",".join(str(int(i)) for i in ids)


class _MoveTokenizer(_FakeTokenizer):
    """A tokenizer whose decode always yields a legal-looking UCI move."""

    def decode(self, ids, skip_special_tokens=None):
        return "<think>develop the knight.</think>\ne2e4"


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        out = input_ids._values + [101, 102, 103]
        return _FakeTensor(out)


class TestEvaluatorFenAdvancement(unittest.TestCase):
    """The evaluator must refresh the FEN after every move."""

    def test_generate_completion_strips_prompt(self):
        from chessfm.rl.evaluate import generate_completion

        tok = _FakeTokenizer()
        model = _FakeModel()
        completion = generate_completion(model, tok, "some prompt here")
        # Only the 3 generated tokens (101,102,103) should be decoded.
        self.assertEqual(completion, "decoded:101,102,103")


class TestCountResult(unittest.TestCase):
    """Win/loss tallies must be from the model's perspective, not White's."""

    def _stats(self):
        return {
            "wins": 0,
            "losses": 0,
            "draws": 0,
        }

    def test_white_win_counts_for_white_model(self):
        from chessfm.rl.evaluate import _count_result

        stats = self._stats()
        _count_result(stats, "1-0", "white")
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 0)

    def test_white_win_is_loss_for_black_model(self):
        from chessfm.rl.evaluate import _count_result

        stats = self._stats()
        _count_result(stats, "1-0", "black")
        self.assertEqual(stats["wins"], 0)
        self.assertEqual(stats["losses"], 1)

    def test_black_win_is_loss_for_white_model(self):
        from chessfm.rl.evaluate import _count_result

        stats = self._stats()
        _count_result(stats, "0-1", "white")
        self.assertEqual(stats["wins"], 0)
        self.assertEqual(stats["losses"], 1)

    def test_black_win_counts_for_black_model(self):
        from chessfm.rl.evaluate import _count_result

        stats = self._stats()
        _count_result(stats, "0-1", "black")
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 0)

    def test_draw_counts_as_draw(self):
        from chessfm.rl.evaluate import _count_result

        stats = self._stats()
        _count_result(stats, "1/2-1/2", "black")
        self.assertEqual(stats["draws"], 1)


class TestPlayGame(unittest.TestCase):
    """The game loop must terminate and tally correctly against a random bot."""

    def test_white_model_forfeits_on_illegal_move(self):
        from chessfm.rl.chess_env import ChessEnv
        from chessfm.rl.evaluate import play_game

        env = ChessEnv()
        model = _FakeModel()
        tokenizer = _MoveTokenizer()

        game, stats = play_game(model, tokenizer, env, "white", use_random_opponent=True)

        self.assertEqual(stats["result"], "0-1")
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["wins"], 0)
        self.assertGreaterEqual(stats["illegal_by_model"], 1)
        self.assertGreaterEqual(stats["total_moves"], 1)
        self.assertGreaterEqual(stats["legal_moves"], 1)
        # The first e2e4 is legal and the opponent's reply is legal; the second
        # e2e4 (pawn no longer on e2) forfeits. Both legal moves are in the PGN.
        self.assertEqual(len(list(game.mainline_moves())), 2)
        env.close()

    def test_black_model_illegal_first_move_forfeits(self):
        from chessfm.rl.chess_env import ChessEnv
        from chessfm.rl.evaluate import play_game

        env = ChessEnv()
        model = _FakeModel()
        tokenizer = _MoveTokenizer()

        game, stats = play_game(model, tokenizer, env, "black", use_random_opponent=True)

        self.assertEqual(stats["result"], "1-0")
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["illegal_by_model"], 1)
        # White's random opener was legal; black's e2e4 was not.
        self.assertEqual(len(list(game.mainline_moves())), 1)
        env.close()


class TestEvalHelpers(unittest.TestCase):
    def test_estimated_elo_known_score(self):
        from chessfm.rl.evaluate import estimated_elo

        # 5 wins + 5 draws out of 10 -> score 0.75 vs a 900-Elo opponent.
        self.assertAlmostEqual(estimated_elo(5, 0, 5, 900), 900 + 400 * math.log10(3), places=2)
        self.assertEqual(estimated_elo(0, 0, 0, 900), 0.0)

    def test_estimated_elo_clamps_perfect_score(self):
        from chessfm.rl.evaluate import estimated_elo

        self.assertLess(estimated_elo(10, 0, 0, 900), 1e6)

    def test_git_revision_returns_string(self):
        from chessfm.rl.evaluate import git_revision

        rev = git_revision()
        assert isinstance(rev, str)
        assert len(rev) > 0


def test_load_openings(tmp_path):
    from chessfm.rl.evaluate import load_openings

    assert load_openings(None) == []
    assert load_openings(tmp_path / "missing.txt") == []
    p = tmp_path / "openings.txt"
    p.write_text("fen-a\n\nfen-b\n")
    assert load_openings(p) == ["fen-a", "fen-b"]


def test_write_pgn(tmp_path):
    import chess
    import chess.pgn

    from chessfm.rl.evaluate import write_pgn

    game = chess.pgn.Game()
    game.headers["Result"] = "1-0"
    path = tmp_path / "games.pgn"
    write_pgn([game], path)
    assert path.exists()
    content = path.read_text()
    assert "1-0" in content


def test_write_report(tmp_path):
    from chessfm.rl.evaluate import write_report

    path = tmp_path / "report.json"
    write_report({"wins": 3}, path)
    assert json.loads(path.read_text()) == {"wins": 3}


def test_evaluate_model_end_to_end_with_fakes(tmp_path, monkeypatch):
    """The full evaluation loop (games -> PGN -> JSON report) runs with fakes."""
    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    model = _FakeModel()
    tokenizer = _MoveTokenizer()

    evaluate.evaluate_model(
        model,
        tokenizer,
        "fake-checkpoint",
        num_games=2,
        use_random_opponent=True,
        seed=1,
    )

    reports = list(tmp_path.glob("eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["model"] == "fake-checkpoint"
    assert report["num_games"] == 2
    assert report["opponent"] == "Random bot"
    assert report["git_revision"]
    assert report["alternate_colors"] is True
    pgns = list((tmp_path / "pgns").glob("*.pgn"))
    assert len(pgns) == 1
    # Two games were alternated white/black; both headers recorded.
    import chess.pgn

    game_headers = []
    with open(pgns[0]) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_headers.append(game.headers.get("ModelColor"))
    assert sorted(game_headers) == ["black", "white"]


def test_alternate_colors_disabled_keeps_one_color(tmp_path, monkeypatch):
    import chess.pgn

    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    evaluate.evaluate_model(
        _FakeModel(),
        _MoveTokenizer(),
        "fake-checkpoint",
        num_games=2,
        use_random_opponent=True,
        alternate_colors=False,
        model_color="white",
        seed=3,
    )

    pgn = list((tmp_path / "pgns").glob("*.pgn"))[0]
    colors = []
    with open(pgn) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            colors.append(game.headers.get("ModelColor"))
    assert colors == ["white", "white"]


def test_wilson_ci_bounds():
    from chessfm.rl.evaluate import wilson_ci

    assert wilson_ci(0, 0, 0) == (0.0, 0.0)

    low, high = wilson_ci(5, 5, 0)
    assert 0.0 < low < 0.5 < high < 1.0
    assert math.isclose(low + high, 1.0, rel_tol=1e-6)

    low_100, high_100 = wilson_ci(10, 0, 0)
    assert 0.0 < low_100 < 1.0
    assert math.isclose(high_100, 1.0, abs_tol=1e-9)

    low_0, high_0 = wilson_ci(0, 10, 0)
    assert math.isclose(low_0, 0.0, abs_tol=1e-9)
    assert 0.0 < high_0 < 1.0

    # Larger sample size narrows the interval.
    lo_small, hi_small = wilson_ci(5, 5, 0)
    lo_large, hi_large = wilson_ci(50, 50, 0)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_evaluate_model_with_custom_openings_writes_valid_pgn(tmp_path, monkeypatch):
    import chess.pgn

    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    openings_file = tmp_path / "openings.txt"
    fen_1 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_2 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    openings_file.write_text(f"{fen_1}\n{fen_2}\n")

    class _KnightTokenizer(_FakeTokenizer):
        def decode(self, ids, skip_special_tokens=None):
            return "<think>develop knight to f3.</think>\ng1f3"

    evaluate.evaluate_model(
        _FakeModel(),
        _KnightTokenizer(),
        "fake-checkpoint",
        num_games=4,
        use_random_opponent=True,
        openings=openings_file,
        seed=7,
    )

    reports = list(tmp_path.glob("eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert "score_ci_95" in report
    assert len(report["score_ci_95"]) == 2
    assert report["decoding_mode"] == "strict"
    assert report["color_balanced"] is True
    assert report["white"]["games"] == 2
    assert report["black"]["games"] == 2

    pgns = list((tmp_path / "pgns").glob("*.pgn"))
    assert len(pgns) == 1
    fens_in_pgn = []
    colors_in_pgn = []
    with open(pgns[0]) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            assert game.headers.get("SetUp") == "1"
            fens_in_pgn.append(game.headers.get("FEN"))
            colors_in_pgn.append(game.headers.get("ModelColor"))
            assert len(list(game.mainline_moves())) >= 1
    # Each opening is played as both White and Black (paired opening indexing).
    assert fens_in_pgn == [fen_1, fen_1, fen_2, fen_2]
    assert colors_in_pgn == ["white", "black", "white", "black"]


def test_evaluate_model_writes_json_report_before_pgn(tmp_path, monkeypatch):
    import pytest

    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    def _boom(games, path):
        raise RuntimeError("simulated PGN serialization failure")

    monkeypatch.setattr(evaluate, "write_pgn", _boom)

    with pytest.raises(RuntimeError, match="simulated PGN"):
        evaluate.evaluate_model(
            _FakeModel(),
            _MoveTokenizer(),
            "fake-checkpoint",
            num_games=1,
            use_random_opponent=True,
            seed=1,
        )

    # JSON report must already exist on disk even though PGN writing failed.
    reports = list(tmp_path.glob("eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["num_games"] == 1
    assert report["decoding_mode"] == "strict"
    assert len(report["score_ci_95"]) == 2


def test_generate_completion_strict_omits_logits_processor():
    import chess

    from chessfm.rl.evaluate import generate_completion

    model = _FakeModel()
    tok = _FakeTokenizer()
    generate_completion(
        model, tok, "some prompt here", decoding_mode="strict", fen=chess.STARTING_FEN
    )

    assert len(model.generate_calls) == 1
    assert "logits_processor" not in model.generate_calls[0]


def test_generate_completion_constrained_attaches_logits_processor():
    import chess
    import pytest

    from chessfm.rl.constrained_decoding import LegalMoveLogitsProcessor
    from chessfm.rl.evaluate import generate_completion

    model = _FakeModel()
    tok = _FakeTokenizer()
    prompt = "some prompt here"
    generate_completion(model, tok, prompt, decoding_mode="constrained", fen=chess.STARTING_FEN)

    assert len(model.generate_calls) == 1
    call_kwargs = model.generate_calls[0]
    assert "logits_processor" in call_kwargs
    processors = call_kwargs["logits_processor"]
    assert len(processors) == 1
    proc = processors[0]
    assert isinstance(proc, LegalMoveLogitsProcessor)
    assert proc.fen == chess.STARTING_FEN
    assert proc.prompt_length == len(prompt.split()) + 2

    with pytest.raises(ValueError, match="fen must be provided"):
        generate_completion(model, tok, prompt, decoding_mode="constrained")

    with pytest.raises(ValueError, match="Unsupported decoding_mode"):
        generate_completion(model, tok, prompt, decoding_mode="beam")


def test_evaluate_model_records_constrained_decoding_mode(tmp_path, monkeypatch):
    import pytest

    import chessfm.rl.evaluate as evaluate
    from chessfm.rl.constrained_decoding import LegalMoveLogitsProcessor

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    model = _FakeModel()
    tokenizer = _MoveTokenizer()

    evaluate.evaluate_model(
        model,
        tokenizer,
        "fake-checkpoint",
        num_games=1,
        use_random_opponent=True,
        seed=1,
        decoding_mode="constrained",
    )

    reports = list(tmp_path.glob("eval_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["decoding_mode"] == "constrained"
    assert len(model.generate_calls) >= 1
    for call_kwargs in model.generate_calls:
        assert "logits_processor" in call_kwargs
        assert isinstance(call_kwargs["logits_processor"][0], LegalMoveLogitsProcessor)

    with pytest.raises(ValueError, match="Unsupported decoding_mode"):
        evaluate.evaluate_model(
            _FakeModel(),
            _MoveTokenizer(),
            "fake-checkpoint",
            num_games=1,
            use_random_opponent=True,
            decoding_mode="invalid",
        )


def test_generate_completion_passes_require_think_close_false():
    import chess

    from chessfm.rl.constrained_decoding import LegalMoveLogitsProcessor
    from chessfm.rl.evaluate import generate_completion

    model = _FakeModel()
    tok = _FakeTokenizer()
    generate_completion(
        model,
        tok,
        "prompt",
        decoding_mode="constrained",
        fen=chess.STARTING_FEN,
        require_think_close=False,
    )
    proc = model.generate_calls[0]["logits_processor"][0]
    assert isinstance(proc, LegalMoveLogitsProcessor)
    assert proc.require_think_close is False


def test_evaluate_model_maia_opponent_prompt_parity_and_manifest(tmp_path, monkeypatch):
    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    template_calls: list[dict] = []

    class _ChatMoveTokenizer(_MoveTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            template_calls.append({"messages": messages, "kwargs": kwargs})
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    manifest_path = tmp_path / "manifests" / "eval_manifest.json"
    results = evaluate.evaluate_model(
        _FakeModel(),
        _ChatMoveTokenizer(),
        "Qwen/Qwen3-4B",
        num_games=2,
        maia_elo=1200,
        include_ascii=True,
        enable_thinking=False,
        seed=11,
        manifest_output=manifest_path,
    )

    assert results["opponent_backend"] in ("calibrated_fallback", "maia2")
    assert results["maia_elo"] == 1200
    assert results["include_ascii"] is True
    assert results["enable_thinking"] is False
    assert results["color_balanced"] is True
    assert results["white"]["games"] == 1
    assert results["black"]["games"] == 1
    assert results["timestamp_utc"].endswith("Z")
    assert results["git_sha"]
    assert template_calls
    assert template_calls[0]["kwargs"]["enable_thinking"] is False
    assert "Board:\n" in template_calls[0]["messages"][0]["content"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["model_id"] == "Qwen/Qwen3-4B"
    assert manifest["enable_thinking"] is False
    assert manifest["include_ascii"] is True
    assert manifest["timestamp_utc"].endswith("Z")
    assert manifest["tiers"]["tier4_match"]["opponent_elo"] == 1200
    assert manifest["tiers"]["tier4_match"]["color_balanced"] is True


def test_evaluate_model_rejects_non_positive_maia_elo(tmp_path, monkeypatch):
    import pytest

    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    for bad_elo in (0, -100):
        with pytest.raises(ValueError, match="maia_elo must be a positive integer"):
            evaluate.evaluate_model(
                _FakeModel(),
                _MoveTokenizer(),
                "fake",
                num_games=1,
                maia_elo=bad_elo,
            )

    monkeypatch.setattr("sys.argv", ["evaluate", "--maia-elo", "0"])
    with pytest.raises(SystemExit) as exc_info:
        evaluate.main()
    assert exc_info.value.code == 2


def test_evaluate_model_closes_env_on_exception(tmp_path, monkeypatch):
    import pytest

    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    closed = {"called": False}

    class _TrackingEnv:
        engine = None

        def reset(self, fen=None):
            return fen or ""

        def close(self):
            closed["called"] = True

    monkeypatch.setattr(evaluate, "ChessEnv", _TrackingEnv)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated play_game failure")

    monkeypatch.setattr(evaluate, "play_game", _boom)

    with pytest.raises(RuntimeError, match="simulated play_game failure"):
        evaluate.evaluate_model(
            _FakeModel(),
            _MoveTokenizer(),
            "fake",
            num_games=1,
            use_random_opponent=True,
        )
    assert closed["called"] is True


def test_evaluate_model_alternate_colors_starts_with_black(tmp_path, monkeypatch):
    import chessfm.rl.evaluate as evaluate

    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    seen_colors: list[str] = []
    orig_play_game = evaluate.play_game

    def _spy_play_game(model, tokenizer, env, model_color, *args, **kwargs):
        seen_colors.append(model_color)
        return orig_play_game(model, tokenizer, env, model_color, *args, **kwargs)

    monkeypatch.setattr(evaluate, "play_game", _spy_play_game)

    results = evaluate.evaluate_model(
        _FakeModel(),
        _MoveTokenizer(),
        "fake",
        num_games=4,
        model_color="black",
        alternate_colors=True,
        use_random_opponent=True,
        seed=7,
    )
    assert seen_colors == ["black", "white", "black", "white"]
    assert results["color_balanced"] is True
    assert results["white"]["games"] == 2
    assert results["black"]["games"] == 2


if __name__ == "__main__":
    unittest.main()
