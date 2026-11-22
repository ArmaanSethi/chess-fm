"""Unit and live Stockfish tests for chessfm.coach.game_review."""

from __future__ import annotations

import json
import subprocess
import sys

import chess
import chess.engine
import pytest

from chessfm.coach.game_review import (
    WEAKNESS_CATEGORIES,
    _classify_move,
    _is_endgame,
    _tag_weakness,
    build_weakness_profile,
    cp_to_win_prob,
    format_review_report,
    main,
    move_accuracy,
    review_pgn_game,
)
from chessfm.rl.engine import close_engine, open_engine

SCHOLARS_MATE_PGN = """\
[Event "Casual Game"]
[White "Student"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

HANGING_QUEEN_PGN = """\
[Event "Blunder Test"]
[White "Student"]
[Black "Opponent"]
[Result "0-1"]

1. e4 e5 2. Qh5 Nc6 3. Qxe5+ Nxe5 4. d4 Ng6 0-1
"""

MISSED_MATE_PGN = """\
[Event "Missed Mate"]
[White "Student"]
[Black "Opponent"]
[Result "*"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. a3 Nxe4 *
"""

EMPTY_MOVES_PGN = """\
[Event "Empty"]
[White "A"]
[Black "B"]
[Result "*"]

*
"""


@pytest.fixture(scope="module")
def stockfish_engine():
    engine = open_engine(threads=1, hash_mb=16)
    if engine is None:
        pytest.skip("Stockfish binary not available")
    yield engine
    close_engine(engine)


class TestScoringHelpers:
    def test_cp_to_win_prob_monotonic_and_bounded(self):
        assert cp_to_win_prob(0.0) == pytest.approx(50.0, abs=1e-6)
        assert cp_to_win_prob(500.0) > cp_to_win_prob(100.0) > 50.0
        assert cp_to_win_prob(-500.0) < cp_to_win_prob(-100.0) < 50.0
        assert 0.0 <= cp_to_win_prob(-10000.0) < 1.0
        assert 99.0 < cp_to_win_prob(10000.0) <= 100.0

    def test_move_accuracy_range(self):
        assert move_accuracy(0.0) == 100.0
        assert 70.0 < move_accuracy(5.0) < 95.0
        assert move_accuracy(30.0) < move_accuracy(10.0)
        assert move_accuracy(200.0) == 0.0

    def test_classify_move_thresholds(self):
        assert _classify_move("e2e4", "e2e4", 0.0, 0.0, False, False) == "Best"
        assert _classify_move("d2d4", "e2e4", 10.0, 1.0, False, False) == "Best"
        assert _classify_move("d2d4", "e2e4", 30.0, 3.0, False, False) == "Good"
        assert _classify_move("d2d4", "e2e4", 75.0, 6.0, False, False) == "Inaccuracy"
        assert _classify_move("d2d4", "e2e4", 150.0, 12.0, False, False) == "Mistake"
        assert _classify_move("d2d4", "e2e4", 250.0, 25.0, False, False) == "Blunder"
        assert _classify_move("a2a3", "h5f7", 50.0, 5.0, True, False) == "Blunder"
        assert _classify_move("g2g4", "e2e4", 50.0, 5.0, False, True) == "Blunder"

    def test_is_endgame_detection(self):
        assert not _is_endgame(chess.Board())
        assert _is_endgame(chess.Board("8/5k2/8/8/3R4/8/4P3/4K3 w - - 0 40"))


class TestWeaknessTagging:
    def test_tag_missed_mate(self):
        board_before = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        )
        board_after = board_before.copy()
        board_after.push_uci("a2a3")
        cat = _tag_weakness(
            board_before,
            board_after,
            chess.WHITE,
            best_uci="h5f7",
            cp_before=10000.0,
            cp_after=0.0,
            cp_loss=10000.0,
            mate_before=1,
            ply=7,
        )
        assert cat == "missed_mate"

    def test_tag_hanging_piece(self):
        board_before = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3"
        )
        board_after = board_before.copy()
        board_after.push_uci("h5e5")
        cat = _tag_weakness(
            board_before,
            board_after,
            chess.WHITE,
            best_uci="f1c4",
            cp_before=20.0,
            cp_after=-800.0,
            cp_loss=820.0,
            mate_before=None,
            ply=5,
        )
        assert cat == "hanging_piece"

    def test_tag_missed_tactic(self):
        # White has Nxe5 winning a free pawn/piece, plays a3 instead.
        board_before = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
        board_after = board_before.copy()
        board_after.push_uci("a2a3")
        cat = _tag_weakness(
            board_before,
            board_after,
            chess.WHITE,
            best_uci="f3e5",
            cp_before=200.0,
            cp_after=20.0,
            cp_loss=180.0,
            mate_before=None,
            ply=18,
        )
        assert cat == "missed_tactic"

    def test_tag_opening_endgame_and_positional(self):
        start = chess.Board()
        after = start.copy()
        after.push_uci("a2a3")
        assert (
            _tag_weakness(start, after, chess.WHITE, "e2e4", 30.0, -40.0, 70.0, None, ply=1)
            == "opening_inaccuracy"
        )

        endgame = chess.Board("8/5k2/8/8/3R4/8/4P3/4K3 w - - 0 40")
        eg_after = endgame.copy()
        eg_after.push_uci("e1f1")
        assert (
            _tag_weakness(endgame, eg_after, chess.WHITE, "d4d7", 400.0, 310.0, 90.0, None, ply=79)
            == "endgame_conversion"
        )

        middlegame = chess.Board(
            "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9"
        )
        mg_after = middlegame.copy()
        mg_after.push_uci("h2h3")
        assert (
            _tag_weakness(
                middlegame, mg_after, chess.WHITE, "c1e3", 40.0, -30.0, 70.0, None, ply=20
            )
            == "positional_drift"
        )


class TestReviewPgnGameHeuristic:
    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            review_pgn_game("   ")
        with pytest.raises(ValueError, match="student_color"):
            review_pgn_game(SCHOLARS_MATE_PGN, student_color="green")

    def test_empty_moves_pgn(self):
        review = review_pgn_game(EMPTY_MOVES_PGN, engine=None)
        assert review["total_plies"] == 0
        assert review["student_moves"] == 0
        assert review["accuracy"] == 100.0
        assert review["critical_moments"] == []

    def test_hanging_queen_detected_without_engine(self):
        review = review_pgn_game(
            HANGING_QUEEN_PGN,
            engine=None,
            student_color="white",
            student_elo=1000,
            max_critical=3,
        )
        assert review["engine_used"] == "heuristic"
        assert review["student_color"] == "white"
        assert review["student_elo"] == 1000
        assert review["counts"]["Blunder"] >= 1
        assert review["accuracy"] < 90.0
        assert len(review["critical_moments"]) >= 1
        top = review["critical_moments"][0]
        assert top["played_move"] == "h5e5"
        assert top["classification"] == "Blunder"
        assert top["category"] == "hanging_piece"

    def test_missed_mate_detected_without_engine(self):
        review = review_pgn_game(MISSED_MATE_PGN, engine=None, student_color="w")
        assert review["student_color"] == "white"
        missed = [m for m in review["moves"] if m["played_move"] == "a2a3"]
        assert len(missed) == 1
        assert missed[0]["best_move"] == "h5f7"
        assert missed[0]["classification"] == "Blunder"
        assert missed[0]["category"] == "missed_mate"

    def test_black_and_both_student_colors(self):
        black_rev = review_pgn_game(SCHOLARS_MATE_PGN, engine=None, student_color="black")
        assert black_rev["student_color"] == "black"
        assert black_rev["student_moves"] == 3
        # Black's 3...Nf6 allows Qxf7# mate-in-1 -> Blunder
        assert black_rev["counts"]["Blunder"] >= 1

        both_rev = review_pgn_game(SCHOLARS_MATE_PGN, engine=None, student_color="both")
        assert both_rev["student_color"] == "both"
        assert both_rev["student_moves"] == 7

    def test_engine_error_falls_back_to_heuristic(self):
        class BrokenEngine:
            def analyse(self, *args, **kwargs):
                raise chess.engine.EngineError("crashed")

        review = review_pgn_game(HANGING_QUEEN_PGN, engine=BrokenEngine())
        assert review["total_plies"] == 8
        assert review["counts"]["Blunder"] >= 1


class TestReviewPgnGameLiveStockfish:
    def test_live_stockfish_review_hanging_queen(self, stockfish_engine):
        review = review_pgn_game(
            HANGING_QUEEN_PGN,
            engine=stockfish_engine,
            student_color="white",
            depth=6,
            max_critical=3,
        )
        assert review["engine_used"] == "stockfish"
        assert review["counts"]["Blunder"] >= 1
        assert review["critical_moments"][0]["played_move"] == "h5e5"
        assert review["critical_moments"][0]["cp_loss"] > 400.0

    def test_live_stockfish_review_missed_mate(self, stockfish_engine):
        review = review_pgn_game(
            MISSED_MATE_PGN,
            engine=stockfish_engine,
            student_color="white",
            depth=6,
        )
        a3_move = [m for m in review["moves"] if m["played_move"] == "a2a3"][0]
        assert a3_move["best_move"] == "h5f7"
        assert a3_move["category"] == "missed_mate"


class TestBuildWeaknessProfile:
    def test_empty_reviews(self):
        profile = build_weakness_profile([])
        assert profile["games_analyzed"] == 0
        assert profile["top_priorities"] == []
        assert profile["drills"] == []

    def test_aggregates_multiple_reviews_and_deduplicates_drills(self):
        rev1 = review_pgn_game(HANGING_QUEEN_PGN, engine=None, student_color="white")
        rev2 = review_pgn_game(MISSED_MATE_PGN, engine=None, student_color="white")
        # Duplicate rev1 to verify deduplication of identical (fen, best_move) drills
        profile = build_weakness_profile([rev1, rev2, rev1])

        assert profile["games_analyzed"] == 3
        assert profile["total_mistakes"] >= 2
        assert 0.0 <= profile["average_accuracy"] <= 100.0
        assert 1 <= len(profile["top_priorities"]) <= 3
        for prio in profile["top_priorities"]:
            assert prio["category"] in WEAKNESS_CATEGORIES
            assert prio["count"] >= 1
            assert prio["recommendation"]

        assert len(profile["drills"]) >= 2
        seen_keys = set()
        for drill in profile["drills"]:
            assert set(drill.keys()) == {"fen", "best_move", "category", "student_mistake"}
            key = (drill["fen"], drill["best_move"])
            assert key not in seen_keys
            seen_keys.add(key)

    def test_fallback_to_critical_moments_when_moves_missing(self):
        synthetic_reviews = [
            {
                "accuracy": 72.5,
                "critical_moments": [
                    {
                        "fen": chess.STARTING_FEN,
                        "best_move": "e2e4",
                        "played_move": "h2h4",
                        "category": "opening_inaccuracy",
                        "cp_loss": 80.0,
                        "wdl_swing": 7.5,
                    },
                    {
                        "fen": "8/5k2/8/8/3R4/8/4P3/4K3 w - - 0 40",
                        "best_move": "d4d7",
                        "played_move": "e1f1",
                        "category": "endgame_conversion",
                        "cp_loss": 160.0,
                        "wdl_swing": 14.0,
                    },
                    {
                        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                        "best_move": "f1b5",
                        "played_move": "f3e5",
                        "category": "hanging_piece",
                        "cp_loss": 310.0,
                        "wdl_swing": 25.0,
                    },
                    {
                        "fen": "r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 9",
                        "best_move": "c1e3",
                        "played_move": "h2h3",
                        "category": "positional_drift",
                        "cp_loss": 65.0,
                        "wdl_swing": 6.0,
                    },
                ],
            }
        ]
        profile = build_weakness_profile(synthetic_reviews)
        assert profile["games_analyzed"] == 1
        assert profile["average_accuracy"] == 72.5
        assert len(profile["top_priorities"]) == 3
        assert profile["top_priorities"][0]["category"] == "hanging_piece"
        assert len(profile["drills"]) == 4


class TestGameReviewCLI:
    def test_cli_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "chessfm.coach.game_review", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "--pgn" in proc.stdout
        assert "--color" in proc.stdout
        assert "--elo" in proc.stdout
        assert "--depth" in proc.stdout
        assert "--json" in proc.stdout

    def test_cli_review_pgn_json(self, tmp_path):
        pgn_file = tmp_path / "sample.pgn"
        pgn_file.write_text(HANGING_QUEEN_PGN, encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "chessfm.coach.game_review",
                "--pgn",
                str(pgn_file),
                "--color",
                "white",
                "--elo",
                "1200",
                "--depth",
                "6",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["student_color"] == "white"
        assert payload["student_elo"] == 1200
        assert payload["engine_used"] in ("stockfish", "heuristic")
        assert payload["counts"]["Blunder"] >= 1
        assert payload["critical_moments"][0]["played_move"] == "h5e5"

    def test_main_human_readable_report_and_fallback(self, tmp_path, monkeypatch, capsys):
        pgn_file = tmp_path / "sample.pgn"
        pgn_file.write_text(HANGING_QUEEN_PGN, encoding="utf-8")
        monkeypatch.setattr("chessfm.coach.game_review.open_engine", lambda **_: None)

        rc = main(
            [
                "--pgn",
                str(pgn_file),
                "--color",
                "white",
                "--elo",
                "1350",
                "--depth",
                "6",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "ChessFM Game Review" in out
        assert "Engine: heuristic" in out
        assert "Elo 1350" in out
        assert "Critical Moments:" in out
        assert "Qxe5+" in out

    def test_main_missing_file_returns_nonzero(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.pgn"
        rc = main(["--pgn", str(missing)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "PGN file not found" in err

    def test_format_review_report_no_critical_moments(self):
        review = review_pgn_game(EMPTY_MOVES_PGN, engine=None)
        text = format_review_report(review)
        assert "Critical Moments: none" in text
