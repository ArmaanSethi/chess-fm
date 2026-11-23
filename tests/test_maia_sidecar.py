"""Unit and Stockfish integration tests for ``chessfm.coach.maia_sidecar``."""

from __future__ import annotations

import math
import shutil
import sys
import types

import chess
import chess.engine
import pytest

from chessfm.coach import MaiaSidecar
from chessfm.coach.maia_sidecar import ELO_BRACKETS, snap_elo_bracket
from chessfm.config import STOCKFISH_PATHS
from chessfm.rl.engine import close_engine, open_engine

STOCKFISH_PATH = next((p for p in STOCKFISH_PATHS if shutil.which(p)), None)
START_FEN = chess.STARTING_FEN


def _entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0.0)


def test_elo_brackets_and_snapping():
    assert ELO_BRACKETS == (600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600)
    assert snap_elo_bracket(300) == 600
    assert snap_elo_bracket(1200) == 1200
    assert snap_elo_bracket(1290) == 1200
    assert snap_elo_bracket(1310) == 1400
    assert snap_elo_bracket(3100) == 2600


def test_sidecar_explicit_fallback_backend():
    sidecar = MaiaSidecar(use_neural=False)
    assert sidecar.backend == "calibrated_fallback"


def test_sidecar_falls_back_when_maia2_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "maia2", None)
    sidecar = MaiaSidecar(use_neural=True)
    assert sidecar.backend == "calibrated_fallback"


def test_sidecar_uses_mocked_maia2_backend(monkeypatch):
    fake_model_mod = types.ModuleType("maia2.model")
    fake_model_mod.from_pretrained = lambda type, device: {"type": type, "device": device}

    fake_inf_mod = types.ModuleType("maia2.inference")
    fake_inf_mod.prepare = lambda: "prepared"
    fake_inf_mod.inference_each = lambda mdl, prep, fen, elo_self, elo_oppo: (
        {"e2e4": 0.6, "d2d4": 0.2, "a1a8": 0.99},  # a1a8 is illegal
        0.52,
    )

    fake_pkg = types.ModuleType("maia2")
    fake_pkg.model = fake_model_mod
    fake_pkg.inference = fake_inf_mod

    monkeypatch.setitem(sys.modules, "maia2", fake_pkg)
    monkeypatch.setitem(sys.modules, "maia2.model", fake_model_mod)
    monkeypatch.setitem(sys.modules, "maia2.inference", fake_inf_mod)

    sidecar = MaiaSidecar(use_neural=True)
    assert sidecar.backend == "maia2"

    res = sidecar.predict_human_moves(START_FEN, student_elo=1400, opponent_elo=1500, top_k=2)
    assert res["backend"] == "maia2"
    assert res["opponent_elo"] == 1500
    assert "a1a8" not in res["move_probs"]
    assert math.isclose(sum(res["move_probs"].values()), 1.0, rel_tol=1e-9)
    assert math.isclose(res["move_probs"]["e2e4"], 0.75, rel_tol=1e-6)
    assert res["top_moves"][0][0] == "e2e4"
    assert len(res["top_moves"]) == 2

    # When maia2 returns zero probability mass for legal moves, fallback kicks in.
    fake_inf_mod.inference_each = lambda mdl, prep, fen, elo_self, elo_oppo: ({"a1a8": 1.0}, 0.5)
    fallback_res = sidecar.predict_human_moves(START_FEN, student_elo=1400)
    assert math.isclose(sum(fallback_res["move_probs"].values()), 1.0, rel_tol=1e-9)
    assert len(fallback_res["move_probs"]) == 20


def test_predict_human_moves_properties_and_determinism():
    sidecar = MaiaSidecar(use_neural=False)
    r1 = sidecar.predict_human_moves(START_FEN, student_elo=1200, top_k=5)
    r2 = sidecar.predict_human_moves(START_FEN, student_elo=1200, top_k=5)

    assert r1 == r2
    assert r1["fen"] == START_FEN
    assert r1["student_elo"] == 1200
    assert r1["backend"] == "calibrated_fallback"

    board = chess.Board(START_FEN)
    legal_ucis = {m.uci() for m in board.legal_moves}
    assert set(r1["move_probs"].keys()) == legal_ucis
    assert math.isclose(sum(r1["move_probs"].values()), 1.0, rel_tol=1e-9)
    assert len(r1["top_moves"]) == 5
    probs = [p for _, p in r1["top_moves"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_human_moves_terminal_position():
    # Fool's mate final position (White is checkmated).
    mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    sidecar = MaiaSidecar(use_neural=False)
    res = sidecar.predict_human_moves(mate_fen, student_elo=1200)
    assert res["move_probs"] == {}
    assert res["top_moves"] == []


def test_elo_calibration_sharpens_distribution_and_avoids_blunders():
    sidecar = MaiaSidecar(use_neural=False)
    low = sidecar.predict_human_moves(START_FEN, student_elo=600)
    high = sidecar.predict_human_moves(START_FEN, student_elo=2400)

    # Higher-rated distribution has lower entropy (sharper preference).
    assert _entropy(high["move_probs"]) < _entropy(low["move_probs"])

    # Position where White can play a queen blunder onto an attacked square (Qd1-g4,
    # attacked by Black's c8 bishop and undefended).
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    low_q = sidecar.predict_human_moves(fen, student_elo=600)["move_probs"]["d1g4"]
    high_q = sidecar.predict_human_moves(fen, student_elo=2400)["move_probs"]["d1g4"]
    assert high_q < low_q


def test_special_moves_en_passant_castling_promotion_checkmate():
    sidecar = MaiaSidecar(use_neural=False)

    # En passant + castling available.
    fen_ep_castle = "r3k2r/ppp2ppp/8/3pP3/8/8/PPP2PPP/R3K2R w KQkq d6 0 1"
    res = sidecar.predict_human_moves(fen_ep_castle, student_elo=1600)
    assert "e5d6" in res["move_probs"]
    assert "e1g1" in res["move_probs"]
    assert math.isclose(sum(res["move_probs"].values()), 1.0, rel_tol=1e-9)

    # Promotion position: White pawn on a7 promotes to queen.
    fen_promo = "7k/P5pp/8/8/8/8/8/6K1 w - - 0 1"
    promo_res = sidecar.predict_human_moves(fen_promo, student_elo=1800)
    assert promo_res["top_moves"][0][0] == "a7a8q"

    # Legal mate-in-1: White queen on f7 with king on g6 has several mate-in-1 moves.
    fen_mate1 = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    mate_res = sidecar.predict_human_moves(fen_mate1, student_elo=1600)
    mate_board = chess.Board(fen_mate1)
    mate_board.push_uci(mate_res["top_moves"][0][0])
    assert mate_board.is_checkmate()


def test_detect_human_trap_categories(monkeypatch):
    sidecar = MaiaSidecar(use_neural=False)

    # Force a known distribution to verify exact threshold boundaries.
    monkeypatch.setattr(
        sidecar,
        "predict_human_moves",
        lambda fen, student_elo=1200, opponent_elo=None, top_k=5: {
            "fen": fen,
            "student_elo": student_elo,
            "backend": "calibrated_fallback",
            "move_probs": {"e2e4": 0.40, "f2f3": 0.20, "a2a3": 0.05},
            "top_moves": [("e2e4", 0.40), ("f2f3", 0.20), ("a2a3", 0.05)],
        },
    )

    trap = sidecar.detect_human_trap(
        START_FEN, played_move="f2f3", best_move="e2e4", cp_loss=150.0, student_elo=1000
    )
    assert trap["is_common_human_trap"] is True
    assert trap["category"] == "common_human_trap"
    assert trap["played_move_prob"] == 0.2
    assert "f2f3" in trap["coaching_context"]
    assert "e2e4" in trap["coaching_context"]

    oversight = sidecar.detect_human_trap(
        START_FEN, played_move="a2a3", best_move="e2e4", cp_loss=180.0, student_elo=1000
    )
    assert oversight["is_common_human_trap"] is False
    assert oversight["category"] == "unusual_oversight"
    assert "Unusual oversight" in oversight["coaching_context"]

    sound = sidecar.detect_human_trap(
        START_FEN, played_move="e2e4", best_move="e2e4", cp_loss=0.0, student_elo=1000
    )
    assert sound["is_common_human_trap"] is False
    assert sound["category"] == "sound_move"
    assert "Sound choice" in sound["coaching_context"]


@pytest.mark.skipif(STOCKFISH_PATH is None, reason="Stockfish not available")
def test_sidecar_with_live_stockfish():
    engine = open_engine(threads=1, hash_mb=16, stockfish_path=STOCKFISH_PATH)
    assert engine is not None
    try:
        sidecar = MaiaSidecar(use_neural=False, engine=engine, engine_depth=5)
        res_low = sidecar.predict_human_moves(START_FEN, student_elo=800, top_k=5)
        res_high = sidecar.predict_human_moves(START_FEN, student_elo=2400, top_k=5)

        assert res_low["backend"] == "calibrated_fallback"
        assert math.isclose(sum(res_low["move_probs"].values()), 1.0, rel_tol=1e-9)
        assert math.isclose(sum(res_high["move_probs"].values()), 1.0, rel_tol=1e-9)
        assert _entropy(res_high["move_probs"]) < _entropy(res_low["move_probs"])
    finally:
        close_engine(engine)
