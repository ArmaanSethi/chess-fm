"""Unit tests for scripts/build_phase2_data.py Phase II benchmark & corpus builders."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pytest

from chessfm.data_generation.phase_sampler import normalize_board_fen

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_phase2_data.py"
_SPEC = importlib.util.spec_from_file_location("build_phase2_data", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
b2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(b2)


# ---------------------------------------------------------------------------
# 1. Diverse ECO Openings
# ---------------------------------------------------------------------------


def test_build_diverse_eco_openings_spans_all_five_eco_letters() -> None:
    openings = b2.build_diverse_eco_openings(count=20)
    assert len(openings) == 20
    keys = [normalize_board_fen(o["fen"]) for o in openings]
    assert len(set(keys)) == 20
    letters = {o["eco"][0] for o in openings}
    assert letters == {"A", "B", "C", "D", "E"}
    for o in openings:
        board = chess.Board(o["fen"])
        assert not board.is_game_over()


# ---------------------------------------------------------------------------
# 2. Movetext + [%eval] Extraction & Position Judgment
# ---------------------------------------------------------------------------


def test_parse_eval_to_cp_and_position_judgment() -> None:
    assert b2.parse_eval_to_cp("0.25") == pytest.approx(25.0)
    assert b2.parse_eval_to_cp("-2.10") == pytest.approx(-210.0)
    assert b2.parse_eval_to_cp("#3") == pytest.approx(10000.0)
    assert b2.parse_eval_to_cp("#-2") == pytest.approx(-10000.0)
    assert b2.parse_eval_to_cp("invalid") is None

    assert b2.classify_position_judgment(200.0) == "White"
    assert b2.classify_position_judgment(-180.0) == "Black"
    assert b2.classify_position_judgment(0.0) == "Roughly Equal"
    assert b2.classify_position_judgment(100.0) is None
    assert b2.classify_position_judgment(-100.0) is None


def test_extract_positions_from_movetext() -> None:
    movetext = (
        "1. e4 { [%eval 0.20] } 1... e5 { [%eval 0.25] } "
        "2. Nf3 { [%eval 0.18] } 2... Nc6 { [%eval 0.22] } "
        "3. Bb5 { [%eval 0.30] } 3... a6 { [%eval 0.28] } "
        "4. Ba4 { [%eval 0.25] } 4... Nf6 { [%eval 0.15] } 1/2-1/2"
    )
    positions = b2.extract_positions_from_movetext(
        movetext, eco="C78", opening_name="Ruy Lopez", min_fullmove=3, max_fullmove=10, stride=1
    )
    assert len(positions) >= 3
    for pos in positions:
        board = chess.Board(pos["fen"])
        assert board.fullmove_number >= 3
        assert pos["phase"] == "opening"
        assert pos["eval_cp_white"] is not None
        assert pos["eco"] == "C78"


# ---------------------------------------------------------------------------
# 3. All 5 ChessQA Categories
# ---------------------------------------------------------------------------


def test_structural_chessqa_all_four_subtypes() -> None:
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    item_piece = b2.build_structural_chessqa_item(fen, "piece_lookup")
    assert item_piece["category"] == "Structural"
    assert item_piece["subcategory"] == "piece_lookup"
    assert any(color in item_piece["answer"] for color in ("white ", "black "))

    item_yes = b2.build_structural_chessqa_item(fen, "legal_moves", force_yes=True)
    assert item_yes["answer"] == "yes"
    item_no = b2.build_structural_chessqa_item(fen, "legal_moves", force_yes=False)
    assert item_no["answer"] == "no"

    item_att = b2.build_structural_chessqa_item(fen, "attacks")
    assert item_att["subcategory"] == "attacks"
    assert isinstance(item_att["answer"], str) and len(item_att["answer"]) > 0

    check_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    item_chk = b2.build_structural_chessqa_item(check_fen, "check_status")
    assert item_chk["answer"] == "yes"
    item_nochk = b2.build_structural_chessqa_item(fen, "check_status")
    assert item_nochk["answer"] == "no"


def test_motifs_short_tactics_judgment_and_semantic_chessqa() -> None:
    puzzle = {
        "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25",
        "best_move": "e6e7",
        "line": ["e6e7", "b2b1", "b3c1"],
        "rating": 1811,
        "themes": ["crushing", "hangingPiece", "long", "middlegame"],
        "phase": "middlegame",
    }
    motif_item = b2.build_motifs_chessqa_item(puzzle)
    assert motif_item["category"] == "Motifs"
    assert "capture" in motif_item["motifs"]

    tactic_item = b2.build_short_tactics_chessqa_item(puzzle)
    assert tactic_item["category"] == "Short Tactics"
    assert tactic_item["answer"] == "e6e7"

    judg_item = b2.build_position_judgment_chessqa_item(puzzle["fen"], 350.0)
    assert judg_item["category"] == "Position Judgment"
    assert judg_item["answer"] == "White"
    with pytest.raises(ValueError, match="Ambiguous"):
        b2.build_position_judgment_chessqa_item(puzzle["fen"], 80.0)

    fen_open = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    for sub, expected in (
        ("game_phase", "opening"),
        ("castling_rights", "KQkq"),
        ("side_to_move", "white"),
        ("material_balance", "Equal"),
    ):
        sem = b2.build_semantic_chessqa_item(fen_open, sub)
        assert sem["category"] == "Semantic"
        assert sem["answer"] == expected


# ---------------------------------------------------------------------------
# 4. Puzzle SFT Builder & C1-4B Stratification
# ---------------------------------------------------------------------------


def test_build_puzzle_sft_example_valid_think_tags() -> None:
    puzzle = {
        "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25",
        "best_move": "e6e7",
        "line": ["e6e7", "b2b1", "b3c1"],
        "rating": 1811,
        "themes": ["crushing", "hangingPiece"],
        "phase": "middlegame",
    }
    row = b2.build_puzzle_sft_example(puzzle, include_ascii=False)
    assert row["source"] == "lichess_puzzle"
    assert row["task_type"] == "chess_move"
    assert row["best_move"] == "e6e7"
    assert "<think>" in row["completion"] and "</think>\ne6e7" in row["completion"]


def test_stratify_puzzles_c1_disjoint_and_exact_counts() -> None:
    # Build synthetic unique valid FENs across themes and difficulty bands.
    base_board = chess.Board()
    synthetic_puzzles: list[dict[str, Any]] = []
    # Generate distinct positions by making different 2-ply sequences from starting FEN.
    fens: list[str] = []
    for m1 in list(base_board.legal_moves)[:20]:
        b1 = base_board.copy()
        b1.push(m1)
        for m2 in list(b1.legal_moves)[:10]:
            b2_board = b1.copy()
            b2_board.push(m2)
            fens.append(b2_board.fen())

    themes = ("fork", "pin")
    ratings = (900, 1400, 1800, 2200)
    for idx, fen in enumerate(fens[:40]):
        b = chess.Board(fen)
        mv = next(iter(b.legal_moves)).uci()
        synthetic_puzzles.append(
            {
                "fen": fen,
                "best_move": mv,
                "line": [mv],
                "rating": ratings[idx % len(ratings)],
                "themes": [themes[idx % len(themes)]],
                "phase": "opening",
            }
        )

    selected = b2.stratify_puzzles_c1(synthetic_puzzles, themes=themes, per_theme=3, per_band=2)
    assert len(selected) == 3 * 2 + 2 * 4
    keys = [normalize_board_fen(p["fen"]) for p in selected]
    assert len(set(keys)) == len(selected)


# ---------------------------------------------------------------------------
# 5. Stratified 20K Subset & Strict Leakage Audit
# ---------------------------------------------------------------------------


def test_build_stratified_20k_subset_and_leakage_audit() -> None:
    # Build tiny synthetic 80K-like pool with distinct FENs.
    fens: list[str] = []
    start = chess.Board()
    for m1 in list(start.legal_moves):
        b1 = start.copy()
        b1.push(m1)
        for m2 in list(b1.legal_moves)[:6]:
            b2_board = b1.copy()
            b2_board.push(m2)
            fens.append(b2_board.fen())

    rows_80k: list[dict[str, Any]] = []
    idx = 0
    for phase in ("opening", "opening", "middlegame", "middlegame", "endgame", "endgame"):
        rows_80k.append({"fen": fens[idx], "source": "engine_best_line", "phase": phase})
        idx += 1
    for _ in range(10):
        rows_80k.append({"fen": fens[idx], "source": "lichess_puzzle", "phase": "middlegame"})
        idx += 1
    for cat in b2.CHESSQA_CATEGORIES:
        for _ in range(2):
            rows_80k.append({"fen": fens[idx], "source": "chessqa", "category": cat})
            idx += 1

    subset = b2.build_stratified_20k_subset(
        rows_80k, game_count=4, puzzle_count=5, chessqa_count=5, seed=7
    )
    assert len(subset) == 14

    rl_fens = fens[idx : idx + 5]
    test_fens = fens[idx + 5 : idx + 10]

    audit = b2.compute_phase2_leakage_audit(
        {
            "sft_phase2_80k": [r["fen"] for r in rows_80k],
            "sft_phase2_20k": [r["fen"] for r in subset],
            "rl_positions_40k": rl_fens,
            "test_fens_1000": test_fens,
        },
        strict=True,
    )
    assert audit["zero_leakage"] is True
    assert audit["subset_checks"]["sft_phase2_20k__in__sft_phase2_80k"]["is_strict_subset"] is True

    # Inject a 1-FEN leak between RL and test -> must raise ValueError under strict=True.
    with pytest.raises(ValueError, match="leakage audit failed"):
        b2.compute_phase2_leakage_audit(
            {
                "sft_phase2_80k": [r["fen"] for r in rows_80k],
                "sft_phase2_20k": [r["fen"] for r in subset],
                "rl_positions_40k": rl_fens + [test_fens[0]],
                "test_fens_1000": test_fens,
            },
            strict=True,
        )


# ---------------------------------------------------------------------------
# 6. Stockfish SIGFPE Auto-Restart Helper
# ---------------------------------------------------------------------------


def test_safe_analyse_auto_restarts_on_engine_terminated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _FlakyEngine:
        def analyse(self, board: chess.Board, limit: Any, **kwargs: Any) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise chess.engine.EngineTerminatedError("simulated SIGFPE -8")
            return {
                "pv": [next(iter(board.legal_moves))],
                "score": chess.engine.PovScore(chess.engine.Cp(25), chess.WHITE),
            }

        def quit(self) -> None:
            pass

    monkeypatch.setattr(b2, "open_engine", lambda **kw: _FlakyEngine())
    monkeypatch.setattr(b2, "close_engine", lambda eng: None)

    box: list[Any] = [_FlakyEngine()]
    info = b2._safe_analyse(box, chess.Board(), depth=4, multipv=1, max_retries=3)
    assert info is not None
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 7. Regression Tests: Chess960 Filter, ChessQA Messages & min_legal_moves=2
# ---------------------------------------------------------------------------


def test_chess960_fen_rejected_by_loader_and_audit(tmp_path: Path) -> None:
    chess960_fen = "nnbrkqrb/pp1ppp1p/6p1/2p5/2PP4/8/PP2PPPP/NNBRKQRB w KQkq - 0 3"
    valid_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    assert b2.is_valid_standard_fen(chess960_fen) is False
    assert b2.is_valid_standard_fen(valid_fen) is True

    fen_file = tmp_path / "fens.txt"
    fen_file.write_text(f"{chess960_fen}\n{valid_fen}\n", encoding="utf-8")
    loaded = b2._load_local_fens([fen_file])
    assert loaded == [valid_fen]

    with pytest.raises(ValueError, match="invalid_standard_boards=1"):
        b2.compute_phase2_leakage_audit({"test_fens_1000": [chess960_fen]}, strict=True)


def test_chessqa_messages_preserved_by_train_sft_format_example() -> None:
    from chessfm.training.train_sft import format_sft_example

    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    puzzle = {
        "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2b1/PqP3PP/7K w - - 0 25",
        "best_move": "e6e7",
        "line": ["e6e7", "b2b1", "b3c1"],
        "rating": 1811,
        "themes": ["crushing", "hangingPiece"],
        "phase": "middlegame",
    }
    rows = [
        b2.build_structural_chessqa_item(fen, "piece_lookup"),
        b2.build_motifs_chessqa_item(puzzle),
        b2.build_short_tactics_chessqa_item(puzzle),
        b2.build_position_judgment_chessqa_item(puzzle["fen"], 350.0),
        b2.build_semantic_chessqa_item(fen, "castling_rights"),
    ]
    for row in rows:
        assert isinstance(row.get("messages"), list) and len(row["messages"]) == 3
        formatted = format_sft_example(row, None, include_piece_list=True)
        text = formatted["text"]
        assert row["question"] in text
        assert row["answer"] in text
        assert "Analyze this position and choose the best move." not in text


def test_pop_phase_balanced_min_legal_moves_skips_forced_one_move_positions() -> None:
    import random

    # Forced single legal move: White king on h1 must capture undefended queen on g2.
    forced_fen = "4k3/8/8/8/8/8/6q1/7K w - - 0 1"
    assert len(list(chess.Board(forced_fen).legal_moves)) == 1
    assert b2.is_valid_standard_fen(forced_fen) is True

    normal_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    assert len(list(chess.Board(normal_fen).legal_moves)) >= 2

    by_phase = {
        "opening": [],
        "middlegame": [],
        "endgame": [normal_fen, forced_fen],  # forced_fen popped first
    }
    used_keys: set[str] = set()
    picked = b2._pop_phase_balanced(
        by_phase,
        used_keys,
        opening=0,
        middlegame=0,
        endgame=1,
        rng=random.Random(42),
        min_legal_moves=2,
    )
    assert picked == [normal_fen]
    assert forced_fen in by_phase["endgame"]
    assert normalize_board_fen(forced_fen) not in used_keys
