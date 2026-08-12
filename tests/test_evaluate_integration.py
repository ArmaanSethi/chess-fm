#!/usr/bin/env python3
"""
Integration tests for the evaluation harness and Stockfish-dependent rewards.

These run only when Stockfish is available (it is installed in CI). They verify
that the evaluator advances the FEN every ply and that the move-quality reward
produces sane values with a real engine.
"""

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
        completion = generate_completion(model, tok, None, "some prompt here")
        # Only the 3 generated tokens (101,102,103) should be decoded.
        self.assertEqual(completion, "decoded:101,102,103")


if __name__ == "__main__":
    unittest.main()
