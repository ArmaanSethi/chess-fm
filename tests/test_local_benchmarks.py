"""Unit and smoke tests for scripts/run_local_benchmarks.py."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import chess
import pytest

from chessfm.coach import MaiaSidecar
from chessfm.config import STOCKFISH_PATHS
from chessfm.rl.move_utils import parse_move_from_output

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
STOCKFISH_OK = any(shutil.which(p) for p in STOCKFISH_PATHS)


def _load_script_module(name: str, filename: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_local_benchmarks = _load_script_module("run_local_benchmarks", "run_local_benchmarks.py")

STARTING_FEN = chess.STARTING_FEN
SICILIAN_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


class TestLocalBenchmarkHelpers:
    def test_random_legal_predict_fn_is_seeded_and_legal(self):
        predict_a = run_local_benchmarks._random_legal_predict_fn(seed=123)
        predict_b = run_local_benchmarks._random_legal_predict_fn(seed=123)

        out_a = predict_a(STARTING_FEN)
        out_b = predict_b(STARTING_FEN)
        assert out_a == out_b
        assert "<think>" in out_a and "</think>" in out_a

        move_uci = parse_move_from_output(out_a)
        assert move_uci is not None
        board = chess.Board(STARTING_FEN)
        assert chess.Move.from_uci(move_uci) in board.legal_moves

    def test_random_legal_predict_fn_terminal_position_returns_null_move(self):
        predict = run_local_benchmarks._random_legal_predict_fn(seed=7)
        output = predict(CHECKMATE_FEN)
        assert output.endswith("\n0000")

    def test_maia_predict_fn_returns_calibrated_legal_move(self):
        sidecar = MaiaSidecar(use_neural=False, engine=None)
        predict = run_local_benchmarks._maia_predict_fn(sidecar, elo=1500)
        output = predict(SICILIAN_FEN)

        assert "<think>" in output and "</think>" in output
        assert "1500" in output
        move_uci = parse_move_from_output(output)
        assert move_uci is not None
        board = chess.Board(SICILIAN_FEN)
        assert chess.Move.from_uci(move_uci) in board.legal_moves

    def test_oracle_probe_answer(self):
        assert run_local_benchmarks._oracle_probe_answer({"expected": "white"}) == "white"
        assert run_local_benchmarks._oracle_probe_answer({"expected": 20}) == "20"


@pytest.mark.skipif(not STOCKFISH_OK, reason="Stockfish binary not available")
def test_run_benchmarks_smoke_schema(tmp_path: Path):
    benchmark_path = tmp_path / "tiny_benchmark.jsonl"
    rows = [
        {"fen": STARTING_FEN, "best_move": "e2e4", "phase": "opening"},
        {"fen": SICILIAN_FEN, "best_move": "g1f3", "phase": "opening"},
    ]
    benchmark_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = run_local_benchmarks.run_benchmarks(
        benchmark_path,
        depth=2,
        match_games=2,
    )

    required_keys = {
        "timestamp",
        "benchmark_file",
        "positions_evaluated",
        "stockfish_depth",
        "elapsed_seconds",
        "tier1_static",
        "tier2_probing_oracle",
        "tier3_coaching_faithfulness",
        "tier4_paired_match",
        "pgn_review_smoke",
    }
    assert required_keys.issubset(report.keys())
    assert report["positions_evaluated"] == 2
    assert report["stockfish_depth"] == 2

    # Tier 1: static engine metrics across all three baselines
    assert set(report["tier1_static"].keys()) == {
        "random_legal",
        "maia_fallback_1200",
        "maia_fallback_2000",
    }
    for metrics in report["tier1_static"].values():
        assert metrics["total_positions"] == 2
        assert metrics["legal_rate"] == pytest.approx(1.0)

    # Tier 2: oracle probing
    assert report["tier2_probing_oracle"]["accuracy"] == pytest.approx(1.0)

    # Tier 3: ACT-Eval coaching faithfulness + Socratic 0%-spoiler check
    tier3 = report["tier3_coaching_faithfulness"]
    assert set(tier3.keys()) == {"coach_explanations", "socratic_l1_l2"}
    assert "mean_factuality_score" in tier3["coach_explanations"]
    assert "pass_rate" in tier3["coach_explanations"]
    assert tier3["socratic_l1_l2"]["spoiler_rate"] == pytest.approx(0.0)

    # Tier 4: paired color-balanced match
    assert report["tier4_paired_match"]["overall"]["games"] == 2
    assert report["tier4_paired_match"]["color_balanced"] is True

    # PGN review smoke output
    assert len(report["pgn_review_smoke"]) == len(run_local_benchmarks.SAMPLE_PGNS)
    for item in report["pgn_review_smoke"]:
        assert {"accuracy", "counts", "critical_moments"}.issubset(item.keys())
