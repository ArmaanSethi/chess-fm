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
        import chess.engine

        path = next(p for p in STOCKFISH_PATHS if shutil.which(p))
        cls.engine = chess.engine.SimpleEngine.popen_uci(path)
        cls.engine.configure({"Threads": 1, "Hash": 16})

    @classmethod
    def tearDownClass(cls):
        cls.engine.quit()

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
    def __call__(self, prompt, return_tensors=None):
        ids = list(range(len(prompt.split()))) + [99, 100]
        return _FakeInputs(ids)

    def decode(self, ids, skip_special_tokens=None):
        return "decoded:" + ",".join(str(int(i)) for i in ids)


class _MoveTokenizer(_FakeTokenizer):
    """A tokenizer whose decode always yields a legal-looking UCI move."""

    def decode(self, ids, skip_special_tokens=None):
        return "<think>develop the knight.</think>\ne2e4"


class _FakeModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):
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


if __name__ == "__main__":
    unittest.main()
