"""Unit and live Stockfish integration tests for ``chessfm.eval.suite``."""

from __future__ import annotations

import json
import math
import shutil
from typing import Any

import chess
import chess.engine
import pytest

from chessfm.coach.classifier import win_prob_from_cp
from chessfm.coach.socratic import generate_socratic_hint
from chessfm.config import STOCKFISH_PATHS
from chessfm.eval import (
    evaluate_board_probes,
    evaluate_coaching_faithfulness,
    evaluate_static_positions,
    generate_board_probes,
    run_demo_suite,
    summarize_paired_match,
)
from chessfm.eval.suite import main

STOCKFISH_PATH = next((p for p in STOCKFISH_PATHS if shutil.which(p)), None)

SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
STALEMATE_IN_1_FEN = "7k/5Q2/5K2/8/8/8/8/8 w - - 0 1"


class _FakeEngine:
    """Minimal deterministic engine stub compatible with ``chess.engine.SimpleEngine``."""

    def __init__(self, scores_by_fen: dict[str, tuple[int, list[str]]]):
        self._scores = scores_by_fen

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict[str, Any]:
        key = board.fen()
        cp_white, pv_ucis = self._scores.get(key, (0, []))
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(cp_white), chess.WHITE),
            "pv": [chess.Move.from_uci(u) for u in pv_ucis],
        }


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------


def test_public_exports():
    assert callable(evaluate_static_positions)
    assert callable(generate_board_probes)
    assert callable(evaluate_board_probes)
    assert callable(evaluate_coaching_faithfulness)
    assert callable(summarize_paired_match)
    assert callable(run_demo_suite)


# ---------------------------------------------------------------------------
# Tier 1: Static Engine Metrics
# ---------------------------------------------------------------------------


def test_tier1_empty_positions():
    res = evaluate_static_positions([], lambda fen: "")
    assert res["total_positions"] == 0
    assert res["legal_rate"] == 0.0
    assert res["format_adherence"] == 0.0
    assert res["acpl"] is None
    assert res["mar"] is None
    assert res["mean_wdl_loss"] is None
    assert res["puzzle_accuracy"] is None
    assert res["details"] == []


def test_tier1_offline_with_and_without_best_move():
    positions = [
        {"fen": chess.STARTING_FEN, "best_move": "e2e4"},
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7"},
        {"fen": chess.STARTING_FEN, "best_move": "d2d4"},
        {"fen": chess.STARTING_FEN, "best_move": "g1f3"},
    ]
    outputs = iter(
        [
            "<think>Control the center.</think> e2e4",  # legal, formatted, exact match
            "<think>Play a developing move.</think> g1f3",  # legal, formatted, not best
            "e2e5",  # illegal, unformatted
            "</think> broken <think> notamove",  # unparseable, bad tag order
        ]
    )
    res = evaluate_static_positions(positions, lambda _fen: next(outputs), engine=None)

    assert res["total_positions"] == 4
    assert res["legal_moves"] == 2
    assert res["legal_rate"] == 0.5
    assert res["format_adherence"] == 0.5
    assert res["acpl"] is None
    assert res["mean_wdl_loss"] is None
    assert res["puzzle_accuracy"] == 0.25
    # 1 of 2 legal moves matched best_move.
    assert res["mar"] == 0.5

    # Without best_move and without engine, MAR and puzzle_accuracy are None.
    res_no_gt = evaluate_static_positions(
        [{"fen": chess.STARTING_FEN}],
        lambda _fen: "<think>Open.</think> e2e4",
        engine=None,
    )
    assert res_no_gt["legal_rate"] == 1.0
    assert res_no_gt["mar"] is None
    assert res_no_gt["puzzle_accuracy"] is None


def test_tier1_with_fake_engine_cp_wdl_mate_and_stalemate():
    b_start = chess.Board(chess.STARTING_FEN)
    b_after_g1f3 = b_start.copy()
    b_after_g1f3.push_uci("g1f3")
    b_after_f2f3 = b_start.copy()
    b_after_f2f3.push_uci("f2f3")

    fake_engine = _FakeEngine(
        {
            chess.STARTING_FEN: (40, ["e2e4"]),
            b_after_g1f3.fen(): (10, ["d7d5"]),  # 30 cp loss (<= 50 -> accurate)
            b_after_f2f3.fen(): (-160, ["e7e5"]),  # 200 cp loss (> 50 -> inaccurate)
            SCHOLARS_MATE_FEN: (10000, ["h5f7"]),
            STALEMATE_IN_1_FEN: (900, ["f7g7"]),
        }
    )

    positions = [
        {"fen": chess.STARTING_FEN, "best_move": "e2e4"},  # engine best -> 0 cp loss
        {"fen": chess.STARTING_FEN},  # g1f3 -> 30 cp loss (accurate)
        {"fen": chess.STARTING_FEN},  # f2f3 -> 200 cp loss (blunder)
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7"},  # checkmate -> 0 cp loss
        {"fen": STALEMATE_IN_1_FEN},  # f6g6 delivers stalemate -> 900 cp loss
    ]
    preds = iter(
        [
            "<think>Best.</think> e2e4",
            "<think>Solid.</think> g1f3",
            "<think>Oops.</think> f2f3",
            "<think>Mate.</think> h5f7",
            "<think>Stalemate oops.</think> f6g6",
        ]
    )

    res = evaluate_static_positions(positions, lambda _fen: next(preds), engine=fake_engine)
    assert res["legal_rate"] == 1.0
    assert res["format_adherence"] == 1.0
    assert res["puzzle_accuracy"] == 1.0
    # Losses: 0, 30, 200, 0, 900 -> mean = 226.0
    assert math.isclose(res["acpl"], 226.0, abs_tol=1e-2)
    # 3 of 5 legal moves have <= 50 cp loss (e2e4, g1f3, h5f7)
    assert res["mar"] == 0.6

    expected_wdl = (
        0.0
        + (win_prob_from_cp(40) - win_prob_from_cp(10))
        + (win_prob_from_cp(40) - win_prob_from_cp(-160))
        + 0.0
        + (win_prob_from_cp(900) - win_prob_from_cp(0))
    ) / 5.0
    assert math.isclose(res["mean_wdl_loss"], expected_wdl, abs_tol=1e-3)


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="Stockfish binary not installed")
def test_tier1_live_stockfish():
    assert STOCKFISH_PATH is not None
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        positions = [
            {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7"},
            {"fen": chess.STARTING_FEN},
        ]
        preds = {
            SCHOLARS_MATE_FEN: "<think>Qxf7 is checkmate.</think> h5f7",
            chess.STARTING_FEN: "<think>Open with e4.</think> e2e4",
        }
        res = evaluate_static_positions(positions, lambda fen: preds[fen], engine=engine, depth=6)
        assert res["legal_rate"] == 1.0
        assert res["format_adherence"] == 1.0
        assert res["puzzle_accuracy"] == 1.0
        assert res["acpl"] is not None and res["acpl"] <= 50.0
        assert res["mar"] == 1.0
        assert res["mean_wdl_loss"] is not None and res["mean_wdl_loss"] >= 0.0


# ---------------------------------------------------------------------------
# Tier 2: Board-State Probing / ChessQA
# ---------------------------------------------------------------------------


def test_tier2_generate_board_probes_deterministic_and_accurate():
    probes_a = generate_board_probes(chess.STARTING_FEN)
    probes_b = generate_board_probes(chess.STARTING_FEN)
    assert probes_a == probes_b

    categories = [p["category"] for p in probes_a]
    assert categories == [
        "side_to_move",
        "piece_on_square",
        "move_legality",
        "move_legality",
        "legal_move_count",
    ]
    assert probes_a[0]["expected"] == "white"
    assert probes_a[2]["expected"] == "yes"
    assert probes_a[3]["expected"] == "no"
    assert probes_a[4]["expected"] == "20"

    # Stalemate position has 0 legal moves: no "yes" move_legality probe, count == 0.
    stale_probes = generate_board_probes(STALEMATE_FEN)
    assert [p["category"] for p in stale_probes] == [
        "side_to_move",
        "piece_on_square",
        "move_legality",
        "legal_move_count",
    ]
    assert stale_probes[0]["expected"] == "black"
    assert stale_probes[2]["expected"] == "no"
    assert stale_probes[3]["expected"] == "0"


def test_tier2_evaluate_board_probes_normalization_and_breakdown():
    fens = [chess.STARTING_FEN, STALEMATE_FEN]

    def _noisy_oracle(probe: dict[str, Any]) -> Any:
        cat = probe["category"]
        exp = probe["expected"]
        if cat == "side_to_move":
            return "W." if exp == "white" else "B!"
        if cat == "move_legality":
            return "True" if exp == "yes" else "illegal"
        if cat == "legal_move_count":
            return f"There are {exp} legal moves."
        # Deliberately miss piece_on_square on the stalemate FEN.
        if probe["fen"] == STALEMATE_FEN:
            return "white rook"
        return f"  {exp.upper()}! "

    res = evaluate_board_probes(fens, _noisy_oracle)
    assert res["total_probes"] == 9
    assert res["correct"] == 8
    assert res["accuracy"] == round(8 / 9, 4)
    assert res["per_category"]["side_to_move"]["accuracy"] == 1.0
    assert res["per_category"]["move_legality"]["accuracy"] == 1.0
    assert res["per_category"]["legal_move_count"]["accuracy"] == 1.0
    assert res["per_category"]["piece_on_square"]["accuracy"] == 0.5

    empty = evaluate_board_probes([], lambda _p: "")
    assert empty["total_probes"] == 0
    assert empty["accuracy"] == 0.0
    assert empty["per_category"] == {}


# ---------------------------------------------------------------------------
# Tier 3: Coaching Faithfulness / ACT-Eval
# ---------------------------------------------------------------------------


def test_tier3_empty_and_socratic_faithfulness():
    empty = evaluate_coaching_faithfulness([], lambda _c: "")
    assert empty["total_cases"] == 0
    assert empty["mean_factuality_score"] == 0.0
    assert empty["pass_rate"] == 0.0
    assert empty["spoiler_rate"] == 0.0

    cases = [
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7", "level": 1},
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7", "level": 2},
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7", "level": 3},
    ]

    res = evaluate_coaching_faithfulness(
        cases,
        lambda c: generate_socratic_hint(c["fen"], c["best_move"], level=c["level"]),
    )
    assert res["total_cases"] == 3
    assert res["mean_factuality_score"] == 1.0
    assert res["pass_rate"] == 1.0
    assert res["spoiler_rate"] == 0.0
    assert res["cases"][0]["spoiler"] is False
    assert res["cases"][1]["spoiler"] is False
    assert res["cases"][2]["spoiler"] is None


def test_tier3_detects_hallucinations_and_spoilers():
    cases = [
        {"fen": SCHOLARS_MATE_FEN, "best_move": "h5f7", "level": 1},
        {"fen": chess.STARTING_FEN, "best_move": "e2e4", "level": 2},
        {"fen": chess.STARTING_FEN, "level": 3},
    ]
    responses = iter(
        [
            "Play Qxf7# right away!",  # factual move, but spoils Level 1
            "Your rook on c5 can play e2e5!",  # hallucinated piece + illegal move, no e2e4 spoiler
            {"text": "Develop cleanly with Nf3, controlling the center."},  # valid
        ]
    )
    res = evaluate_coaching_faithfulness(cases, lambda _c: next(responses))
    assert res["total_cases"] == 3
    assert res["spoiler_rate"] == 0.5  # 1 of 2 Level 1-2 cases spoiled
    assert res["pass_rate"] == round(1 / 3, 4)  # only case 3 passed
    assert res["cases"][0]["spoiler"] is True
    assert res["cases"][0]["passed"] is False
    assert res["cases"][1]["passed"] is False
    assert len(res["cases"][1]["violations"]) >= 2
    assert res["cases"][2]["passed"] is True


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="Stockfish binary not installed")
def test_tier3_live_stockfish_blunder_detection():
    assert STOCKFISH_PATH is not None
    # In Scholar's mate position, playing Qxf7# is mate while Qh5-h4 misses mate-in-1.
    cases = [{"fen": SCHOLARS_MATE_FEN, "level": 3}]
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        res = evaluate_coaching_faithfulness(
            cases,
            lambda _c: "Retreat the queen with Qh4 to stay safe.",
            engine=engine,
        )
        assert res["pass_rate"] == 0.0
        assert any(v["type"] == "blunder" for v in res["cases"][0]["violations"])


# ---------------------------------------------------------------------------
# Tier 4: Paired Color-Balanced Match Summary
# ---------------------------------------------------------------------------


def test_tier4_summarize_paired_match():
    records = [
        {"color": "white", "result": "1-0"},  # W win
        {"color": "black", "result": "0-1"},  # B win
        {"color": "white", "result": "1/2-1/2"},  # W draw
        {"color": "black", "outcome": "loss"},  # B loss
        {"model_color": "white", "wins": 1},  # W win
        {"model_color": "black", "draws": 1},  # B draw
    ]
    summary = summarize_paired_match(
        records,
        opponent_name="Stockfish L2",
        opponent_elo=1000,
        model_id="chessfm-3b-v1",
        decoding_mode="constrained",
    )
    assert summary["opponent_name"] == "Stockfish L2"
    assert summary["opponent_elo"] == 1000
    assert summary["overall"] == {
        "games": 6,
        "wins": 3,
        "draws": 2,
        "losses": 1,
        "score": round(4.0 / 6.0, 4),
    }
    assert summary["white"] == {
        "games": 3,
        "wins": 2,
        "draws": 1,
        "losses": 0,
        "score": round(2.5 / 3.0, 4),
    }
    assert summary["black"] == {
        "games": 3,
        "wins": 1,
        "draws": 1,
        "losses": 1,
        "score": 0.5,
    }
    assert summary["color_balanced"] is True
    ci_low, ci_high = summary["wilson_ci_95"]
    assert 0.0 <= ci_low < summary["overall"]["score"] < ci_high <= 1.0
    assert summary["estimated_elo"] is not None and summary["estimated_elo"] > 1000.0
    assert summary["metadata"]["model_id"] == "chessfm-3b-v1"
    assert summary["metadata"]["decoding_mode"] == "constrained"
    assert isinstance(summary["metadata"]["git_sha"], str)
    assert isinstance(summary["metadata"]["timestamp"], str)


def test_tier4_empty_unbalanced_and_validation_errors():
    empty = summarize_paired_match([], opponent_name="SF", opponent_elo=900)
    assert empty["overall"]["games"] == 0
    assert empty["color_balanced"] is False
    assert empty["estimated_elo"] is None
    assert empty["wilson_ci_95"] == [0.0, 0.0]

    unbalanced = summarize_paired_match(
        [{"color": "white", "losses": 1}],
        opponent_name="SF",
        opponent_elo=900,
    )
    assert unbalanced["color_balanced"] is False
    assert unbalanced["white"]["losses"] == 1

    with pytest.raises(ValueError, match="color"):
        summarize_paired_match([{"outcome": "win"}], opponent_name="SF", opponent_elo=900)

    with pytest.raises(ValueError, match="outcome"):
        summarize_paired_match([{"color": "white"}], opponent_name="SF", opponent_elo=900)


# ---------------------------------------------------------------------------
# CLI & Demo
# ---------------------------------------------------------------------------


def test_cli_demo_stdout_and_file_output(capsys, tmp_path):
    out_file = tmp_path / "eval_demo.json"
    exit_code = main(["--demo", "--output", str(out_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {
        "tier1_static_engine",
        "tier2_board_probes",
        "tier3_coaching_faithfulness",
        "tier4_paired_match",
    }
    assert payload["tier1_static_engine"]["legal_rate"] == 1.0
    assert payload["tier2_board_probes"]["accuracy"] == 1.0
    assert payload["tier3_coaching_faithfulness"]["spoiler_rate"] == 0.0
    assert payload["tier4_paired_match"]["color_balanced"] is True

    file_payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert file_payload["tier1_static_engine"] == payload["tier1_static_engine"]


def test_cli_requires_demo_flag():
    with pytest.raises(SystemExit):
        main([])
