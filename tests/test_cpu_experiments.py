"""Fast deterministic unit tests for scripts/run_cpu_experiments.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import chess
import pytest

from chessfm.fens import read_jsonl
from chessfm.rl.engine import close_engine, open_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data_generation"


def _load_experiments_module():
    path = SCRIPTS_DIR / "run_cpu_experiments.py"
    spec = importlib.util.spec_from_file_location("run_cpu_experiments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_exp = _load_experiments_module()


class _FakeScore:
    def __init__(self, cp: int) -> None:
        self._cp = cp

    def white(self) -> _FakeScore:
        return self

    def is_mate(self) -> bool:
        return False

    def score(self) -> int:
        return self._cp


class _DeterministicStubEngine:
    """Fast deterministic engine stand-in when Stockfish is unavailable."""

    def analyse(self, board: chess.Board, limit) -> dict:
        del limit
        # Score by simple material + center pawn bonus so best vs random moves differ.
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        cp = 0
        for sq, piece in board.piece_map().items():
            val = piece_values[piece.piece_type]
            # Small deterministic positional term so different moves yield distinct scores.
            pos_bonus = (chess.square_rank(sq) + chess.square_file(sq)) * 3
            cp += (val + pos_bonus) if piece.color == chess.WHITE else -(val + pos_bonus)
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        return {
            "score": _FakeScore(cp),
            "pv": [legal[0]] if legal else [],
        }


@pytest.fixture(scope="module")
def sample_benchmark_rows() -> list[dict]:
    return read_jsonl(DATA_DIR / "eval_benchmark.jsonl")[:2]


@pytest.fixture(scope="module")
def sample_sft_rows() -> list[dict]:
    return read_jsonl(DATA_DIR / "sft_train.jsonl")[:4]


class TestExpConstrainedVsStrict:
    def test_constrained_enforces_100_percent_legality_on_mock_lm(self, sample_benchmark_rows):
        res = cpu_exp.exp_constrained_vs_strict(
            sample_benchmark_rows,
            max_positions=2,
            max_new_tokens=10,
            use_mock=True,
        )
        assert res["used_mock_fallback"] is True
        assert res["positions_evaluated"] == 2
        assert res["strict_legal_rate"] == 0.0
        assert res["constrained_legal_rate"] == 1.0
        assert res["constrained_legal_count"] == 2
        assert res["strict_mean_latency_ms"] >= 0.0
        assert res["constrained_mean_latency_ms"] >= 0.0
        assert res["trie_build_mean_ms"] >= 0.0
        assert len(res["sample_outputs"]) == 2
        for sample in res["sample_outputs"]:
            assert sample["strict_legal"] is False
            assert sample["constrained_legal"] is True
            board = chess.Board(sample["fen"])
            assert chess.Move.from_uci(sample["constrained_move"]) in board.legal_moves


class TestExpGrpoRewardVariance:
    def test_candidate_group_has_8_expected_slots(self, sample_benchmark_rows):
        import random

        group = cpu_exp._build_grpo_candidate_group(sample_benchmark_rows[0], random.Random(42))
        assert len(group) == 8
        roles = [c["role"] for c in group]
        assert roles == [
            "best",
            "top_2",
            "top_3",
            "random_1",
            "random_2",
            "random_3",
            "illegal_uci",
            "malformed",
        ]
        board = chess.Board(sample_benchmark_rows[0]["fen"])
        for cand in group[:6]:
            assert chess.Move.from_uci(cand["move"]) in board.legal_moves
        assert group[6]["move"] not in {m.uci() for m in board.legal_moves}
        assert group[7]["move"] == ""

    def test_grpo_reward_variance_across_stages(self, sample_benchmark_rows):
        engine = open_engine()
        owns_real_engine = engine is not None
        if engine is None:
            engine = _DeterministicStubEngine()
        try:
            res = cpu_exp.exp_grpo_reward_variance(
                sample_benchmark_rows,
                engine=engine,
                max_positions=2,
                depth=4,
                seed=42,
            )
        finally:
            if owns_real_engine:
                close_engine(engine)

        assert res["positions_evaluated"] == 2
        assert res["group_size"] == 8
        stages = res["stages"]
        assert set(stages.keys()) == {"stage_0", "stage_1", "stage_2"}

        # Stage 0 assigns identical reward to all well-formed legal moves.
        s0 = stages["stage_0"]
        assert s0["legal_only_mean_group_std"] == 0.0
        assert s0["legal_nonzero_variance_group_rate"] == 0.0
        assert s0["best_vs_worst_legal_separation"] == 0.0
        assert s0["nonzero_variance_group_rate"] == 1.0
        assert s0["illegal_mean_reward"] == -1.0

        # Stage 1 and Stage 2 introduce positive legal-move variance and separation.
        for stage_key in ("stage_1", "stage_2"):
            st = stages[stage_key]
            assert st["nonzero_variance_group_rate"] == 1.0
            assert st["legal_only_mean_group_std"] > 0.0
            assert st["best_vs_worst_legal_separation"] >= 0.0
            assert st["illegal_mean_reward"] == -1.0


class TestExpSftCompletionMasking:
    def test_completion_collator_masks_prompt_and_keeps_completion(self, sample_sft_rows):
        res = cpu_exp.exp_sft_completion_masking(sample_sft_rows, use_mock=True)
        assert res["rows_evaluated"] == 4
        assert res["response_template"] == "<|im_start|>assistant\n"

        std = res["standard_prompt"]
        assert std["rows_evaluated"] == 4
        assert std["zero_active_rows"] == 0
        assert std["mean_masked_prompt_tokens"] > 0
        assert std["mean_active_completion_tokens"] > 0
        assert 0.0 < std["mean_masked_fraction"] < 1.0

        asc = res["ascii_board_prompt"]
        assert asc is not None
        assert asc["zero_active_rows"] == 0
        assert asc["mean_masked_prompt_tokens"] > std["mean_masked_prompt_tokens"]
        assert asc["mean_active_completion_tokens"] == pytest.approx(
            std["mean_active_completion_tokens"], abs=1e-2
        )


def test_cli_help_succeeds():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_cpu_experiments.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--positions" in proc.stdout
    assert "--depth" in proc.stdout
    assert "--output" in proc.stdout
