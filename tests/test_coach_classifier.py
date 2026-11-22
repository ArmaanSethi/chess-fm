"""Unit and Stockfish integration tests for ``chessfm.coach.classifier``."""

from __future__ import annotations

import json
import math
import shutil

import chess
import chess.engine
import pytest

from chessfm.coach import (
    MaiaSidecar,
    build_partial_strategy_tree,
    classify_move,
    win_prob_from_cp,
)
from chessfm.config import STOCKFISH_PATHS

STOCKFISH_PATH = next((p for p in STOCKFISH_PATHS if shutil.which(p)), None)
START_FEN = chess.STARTING_FEN


def test_public_exports():
    assert callable(win_prob_from_cp)
    assert callable(classify_move)
    assert callable(build_partial_strategy_tree)
    assert MaiaSidecar is not None


def test_win_prob_from_cp_values_and_bounds():
    assert math.isclose(win_prob_from_cp(0.0), 50.0, abs_tol=1e-9)

    # Symmetry around 50%.
    for cp in (50.0, 150.0, 350.0, 1000.0):
        assert math.isclose(win_prob_from_cp(cp) + win_prob_from_cp(-cp), 100.0, abs_tol=1e-9)

    # Monotonicity.
    assert win_prob_from_cp(-200) < win_prob_from_cp(0) < win_prob_from_cp(200)

    # Exact Lichess logistic formula check at +100 cp.
    expected_100 = 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * 100.0)) - 1.0)
    assert math.isclose(win_prob_from_cp(100.0), expected_100, abs_tol=1e-9)

    # Extreme values clamp cleanly without OverflowError.
    assert win_prob_from_cp(1e9) == 100.0
    assert win_prob_from_cp(-1e9) == 0.0


def test_classify_move_all_labels():
    # Best move.
    best = classify_move(50.0, 50.0, is_best_move=True)
    assert best["label"] == "Best"
    assert best["cp_loss"] == 0.0
    assert best["win_prob_drop"] == 0.0
    assert best["is_best_move"] is True

    # Improved eval clamps loss to 0 -> Excellent when not flagged best.
    exc = classify_move(20.0, 35.0, is_best_move=False)
    assert exc["label"] == "Excellent"
    assert exc["cp_loss"] == 0.0
    assert exc["win_prob_drop"] == 0.0

    # Good: ~30 cp loss (~2.7% win prob drop).
    good = classify_move(30.0, 0.0, is_best_move=False)
    assert good["label"] == "Good"
    assert good["cp_loss"] == 30.0

    # Inaccuracy: ~70 cp loss (~6.4% win prob drop).
    inacc = classify_move(70.0, 0.0, is_best_move=False)
    assert inacc["label"] == "Inaccuracy"

    # Mistake: ~140 cp loss (~12.7% win prob drop).
    mistake = classify_move(140.0, 0.0, is_best_move=False)
    assert mistake["label"] == "Mistake"

    # Blunder: 0 -> -350 cp (drops from 50% to ~21.6%).
    blunder = classify_move(0.0, -350.0, is_best_move=False)
    assert blunder["label"] == "Blunder"

    # Missed Win: +500 cp (~86.3%) -> 0 cp (50.0%) throws away a decisive win into a draw.
    missed = classify_move(500.0, 0.0, is_best_move=False)
    assert missed["label"] == "Missed Win"

    # Turning a decisive win into a lost position (+500 -> -500) is a Blunder, not just Missed Win.
    collapse = classify_move(500.0, -500.0, is_best_move=False)
    assert collapse["label"] == "Blunder"


class _FakeEngine:
    """Deterministic fake engine supporting MultiPV and single-position analysis."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def analyse(self, board, limit, multipv=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def test_build_partial_strategy_tree_with_fake_engine():
    multipv_resp = [
        {
            "pv": [
                chess.Move.from_uci("e2e4"),
                chess.Move.from_uci("e7e5"),
                chess.Move.from_uci("g1f3"),
                chess.Move.from_uci("b8c6"),
                chess.Move.from_uci("f1b5"),
            ],
            "score": chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE),
        },
        {
            "pv": [
                chess.Move.from_uci("d2d4"),
                chess.Move.from_uci("d7d5"),
            ],
            "score": chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE),
        },
    ]
    student_resp = {
        "pv": [chess.Move.from_uci("e7e5"), chess.Move.from_uci("g2g4")],
        "score": chess.engine.PovScore(chess.engine.Cp(-90), chess.WHITE),
    }

    engine = _FakeEngine([multipv_resp, student_resp])
    tree = build_partial_strategy_tree(
        START_FEN,
        engine,
        depth=8,
        multipv=2,
        pv_plies=4,
        student_move="f2f3",
    )

    # Must be JSON-serializable.
    serialized = json.dumps(tree)
    assert isinstance(serialized, str)

    assert tree["fen"] == START_FEN
    assert tree["side_to_move"] == "white"
    assert tree["best_move"] == "e2e4"
    assert len(tree["top_lines"]) == 2
    assert tree["top_lines"][0]["san"] == "e4"
    assert tree["top_lines"][0]["pv_uci"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert tree["top_lines"][0]["pv_san"] == ["e4", "e5", "Nf3", "Nc6"]

    branch = tree["student_branch"]
    assert branch is not None
    assert branch["move"] == "f2f3"
    assert branch["san"] == "f3"
    assert branch["eval_cp"] == -90
    assert branch["pv_uci"] == ["f2f3", "e7e5", "g2g4"]
    assert branch["classification"]["label"] == "Mistake"


def test_build_partial_strategy_tree_student_matches_top_line_and_single_dict():
    single_resp = {
        "pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("c7c5")],
        "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
    }
    engine = _FakeEngine([single_resp])
    tree = build_partial_strategy_tree(
        START_FEN,
        engine,
        depth=6,
        multipv=1,
        pv_plies=2,
        student_move="e2e4",
    )
    # Reused top line without a second engine call.
    assert engine.calls == 1
    assert tree["student_branch"]["move"] == "e2e4"
    assert tree["student_branch"]["classification"]["label"] == "Best"


def test_build_partial_strategy_tree_edge_cases():
    # Terminal position (checkmate).
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    engine = _FakeEngine([])
    tree = build_partial_strategy_tree(mate_fen, engine, depth=4)
    assert tree["best_move"] is None
    assert tree["top_lines"] == []
    assert tree["student_branch"] is None
    assert engine.calls == 0

    # Invalid UCI / illegal student move raises ValueError.
    base_resp = [
        {
            "pv": [chess.Move.from_uci("e2e4")],
            "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
        }
    ]
    with pytest.raises(ValueError, match="Invalid UCI"):
        build_partial_strategy_tree(START_FEN, _FakeEngine([base_resp]), student_move="not_uci")
    with pytest.raises(ValueError, match="Illegal move"):
        build_partial_strategy_tree(START_FEN, _FakeEngine([base_resp]), student_move="e1e8")

    # Student move delivering checkmate (not in top_lines).
    mate1_fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    alt_resp = [
        {
            "pv": [chess.Move.from_uci("f7g7")],
            "score": chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE),
        }
    ]
    tree_mate = build_partial_strategy_tree(
        mate1_fen,
        _FakeEngine([alt_resp]),
        depth=4,
        multipv=1,
        student_move="f7f8",
    )
    assert tree_mate["student_branch"]["eval_cp"] == 10000
    assert tree_mate["student_branch"]["pv_uci"] == ["f7f8"]

    # Student move causing stalemate (not in top_lines).
    stale_fen = "7k/5Q2/5K2/8/8/8/8/8 w - - 0 1"
    stale_resp = [
        {
            "pv": [chess.Move.from_uci("f7g7")],
            "score": chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE),
        }
    ]
    tree_stale = build_partial_strategy_tree(
        stale_fen,
        _FakeEngine([stale_resp]),
        depth=4,
        multipv=1,
        student_move="f6g6",
    )
    assert tree_stale["student_branch"]["eval_cp"] == 0
    assert tree_stale["student_branch"]["pv_uci"] == ["f6g6"]


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="Stockfish not available")
def test_build_partial_strategy_tree_live_stockfish():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        engine.configure({"Threads": 1, "Hash": 16})
        tree = build_partial_strategy_tree(
            START_FEN,
            engine,
            depth=6,
            multipv=3,
            pv_plies=4,
            student_move="f2f3",
        )

        # JSON serializable check.
        parsed = json.loads(json.dumps(tree))
        assert parsed["side_to_move"] == "white"
        assert parsed["best_move"] is not None
        assert len(parsed["top_lines"]) == 3
        for line in parsed["top_lines"]:
            assert 1 <= len(line["pv_uci"]) <= 4
            assert len(line["pv_uci"]) == len(line["pv_san"])

        branch = parsed["student_branch"]
        assert branch is not None
        assert branch["move"] == "f2f3"
        assert branch["eval_cp"] < parsed["top_lines"][0]["eval_cp"]
        assert branch["classification"]["label"] in {
            "Good",
            "Inaccuracy",
            "Mistake",
            "Blunder",
        }
    finally:
        engine.quit()
