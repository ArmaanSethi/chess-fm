#!/usr/bin/env python3
"""Unit and live Stockfish tests for chessfm.data_generation.build_corpus and Kaggle notebook."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine
import pytest

from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.verifier import verify_claims
from chessfm.config import CHATML_ASSISTANT_CLOSE, MAX_CENTIPAWN_LOSS, PROJECT_ROOT, STOCKFISH_PATHS
from chessfm.data_generation.build_corpus import (
    COACHING_MODES,
    CURATED_SEED_FENS,
    ENDGAME_SEED_FENS,
    MIDDLEGAME_SEED_FENS,
    OPENING_SEED_FENS,
    _ensure_spoiler_free,
    _extract_reasoning_from_completion,
    build_verified_coaching_corpus,
    clean_and_verify_sft_file,
    main,
    parse_args,
)
from chessfm.data_generation.phase_sampler import (
    classify_game_phase,
    normalize_board_fen,
    phase_distribution,
)
from chessfm.fens import read_jsonl
from chessfm.rl.engine import close_engine, open_engine

STOCKFISH_OK = any(shutil.which(p) for p in STOCKFISH_PATHS)
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "chessfm_kaggle_training.ipynb"


class _MockEngine:
    """Fast deterministic stand-in for chess.engine.SimpleEngine."""

    def __init__(self, cp_overrides: dict[str, int] | None = None) -> None:
        self.cp_overrides = cp_overrides or {}

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int | None = None,
    ):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            score = chess.engine.PovScore(chess.engine.Cp(0), board.turn)
            return [] if multipv is not None else {"pv": [], "score": score}

        if multipv is None:
            best = legal[0]
            cp = self.cp_overrides.get(normalize_board_fen(board.fen()), 25)
            score = chess.engine.PovScore(chess.engine.Cp(cp), board.turn)
            return {"pv": [best], "score": score}

        infos = []
        for idx, move in enumerate(legal[:multipv]):
            sim = board.copy()
            sim.push(move)
            reply = sorted(sim.legal_moves, key=lambda m: m.uci())
            pv = [move, reply[0]] if reply else [move]
            cp = 40 - idx * 35
            infos.append(
                {
                    "pv": pv,
                    "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
                }
            )
        return infos

    def quit(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Curated Seed FENs & Phase Balance Tests
# ---------------------------------------------------------------------------


def test_curated_seed_fens_count_uniqueness_and_phase_balance() -> None:
    assert len(CURATED_SEED_FENS) >= 36

    normalized = [normalize_board_fen(f) for f in CURATED_SEED_FENS]
    assert len(set(normalized)) == len(CURATED_SEED_FENS)

    for fen in CURATED_SEED_FENS:
        board = chess.Board(fen)
        assert board.is_valid(), f"Invalid seed FEN: {fen}"
        assert not board.is_game_over()
        assert board.legal_moves.count() >= 2

    dist = phase_distribution(CURATED_SEED_FENS)
    assert dist["opening"] >= 0.25
    assert dist["middlegame"] >= 0.50
    assert dist["endgame"] >= 0.25

    assert all(classify_game_phase(f) == "opening" for f in OPENING_SEED_FENS)
    assert all(classify_game_phase(f) == "middlegame" for f in MIDDLEGAME_SEED_FENS)
    assert all(classify_game_phase(f) == "endgame" for f in ENDGAME_SEED_FENS)


# ---------------------------------------------------------------------------
# build_verified_coaching_corpus Unit Tests
# ---------------------------------------------------------------------------


def test_build_verified_coaching_corpus_with_mock_engine() -> None:
    sample_fens = [
        OPENING_SEED_FENS[0],
        MIDDLEGAME_SEED_FENS[0],
        ENDGAME_SEED_FENS[0],
        # Duplicate board FEN (should be deduplicated)
        OPENING_SEED_FENS[0],
        # Invalid FEN and checkmate position (should be skipped cleanly)
        "not a valid fen",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    ]
    result = build_verified_coaching_corpus(
        fens=sample_fens,
        engine=_MockEngine(),
        depth=4,
        include_ascii=True,
    )

    assert result["total_fens"] == 3
    assert result["total_rows"] == 3 * (1 + len(COACHING_MODES))
    assert result["passed"] is True
    assert result["audit"]["passed"] is True
    assert result["audit"]["illegal_moves"] == 0
    assert result["audit"]["blunders_labeled_reasonable"] == 0
    assert result["spoiler_violations"] == 0
    assert result["claim_violations"] == 0

    assert result["mode_counts"]["best_line"] == 3
    for mode in COACHING_MODES:
        assert result["mode_counts"][mode] == 3

    assert len(result["eval_benchmark"]) == 3
    for entry in result["eval_benchmark"]:
        assert set(entry.keys()) == {"fen", "phase", "best_move", "top_moves", "eval_cp"}
        assert entry["phase"] in {"opening", "middlegame", "endgame"}
        assert len(entry["top_moves"]) >= 1

    # Verify ASCII board is included in best_line prompts
    best_line_rows = [r for r in result["rows"] if "mode" not in r]
    assert all("Board:\n8 " in r["prompt"] for r in best_line_rows)

    # Verify zero spoilers on Socratic L1/L2 and 100% claim verification on coaching rows
    for row in result["rows"]:
        mode = row.get("mode")
        if mode in {"socratic_l1", "socratic_l2"}:
            assert not contains_move_spoiler(
                row["messages"][-1]["content"], row["best_move"], row["fen"]
            )
        if mode is not None:
            vc = verify_claims(row["fen"], row["messages"][-1]["content"])
            assert vc["passed"] is True


def test_build_verified_coaching_corpus_single_legal_move() -> None:
    # Position where White has only a single legal move (Kh1 -> g1).
    single_move_fen = "6k1/8/8/8/8/8/5r2/7K w - - 0 50"
    board = chess.Board(single_move_fen)
    assert board.legal_moves.count() == 1

    result = build_verified_coaching_corpus(
        fens=[single_move_fen],
        engine=_MockEngine(),
        depth=4,
        include_ascii=False,
    )
    assert result["total_fens"] == 1
    assert result["mode_counts"]["best_line"] == 1
    assert result["mode_counts"]["postmortem"] == 0
    assert result["passed"] is True


def test_ensure_spoiler_free_replaces_leaking_hint() -> None:
    fen = OPENING_SEED_FENS[0]
    row = {
        "messages": [
            {"role": "user", "content": "Give a hint."},
            {"role": "assistant", "content": "Try playing c2c3 or c3 right away!"},
        ],
        "completion": f"Try playing c2c3 or c3 right away!{CHATML_ASSISTANT_CLOSE}",
    }
    cleaned = _ensure_spoiler_free(row, fen, best_move_uci="c2c3", level=1)
    assert not contains_move_spoiler(cleaned["messages"][-1]["content"], "c2c3", fen)
    assert cleaned["completion"].endswith(CHATML_ASSISTANT_CLOSE)


# ---------------------------------------------------------------------------
# clean_and_verify_sft_file Unit Tests
# ---------------------------------------------------------------------------


def test_extract_reasoning_repairs_nested_tags_and_duplicates() -> None:
    malformed = f"<think><think>\nthink\nGood center move.\ne2e4\n</think></think>\ne2e4 e2e4{CHATML_ASSISTANT_CLOSE}"
    reasoning, fixed_think, fixed_trailing = _extract_reasoning_from_completion(malformed, "e2e4")
    assert fixed_think is True
    assert fixed_trailing is True
    assert reasoning.count("<think>") == 1
    assert reasoning.count("</think>") == 1
    assert not reasoning.endswith("e2e4")
    assert "\nthink\n" not in reasoning

    bare = "Just a plain explanation without tags."
    reasoning2, fixed_think2, _ = _extract_reasoning_from_completion(bare, "e2e4")
    assert fixed_think2 is True
    assert reasoning2.startswith("<think>\n") and reasoning2.endswith("\n</think>")


def test_clean_and_verify_sft_file_without_engine(tmp_path: Path) -> None:
    fen1 = OPENING_SEED_FENS[0]
    fen2 = OPENING_SEED_FENS[1]
    in_path = tmp_path / "dirty_sft.jsonl"
    out_path = tmp_path / "clean_sft.jsonl"

    dirty_rows = [
        # Valid row with nested <think> tags and duplicate trailing move
        {
            "fen": fen1,
            "move": "c2c3",
            "completion": f"<think><think>\nPrepare d4.\nc2c3\n</think></think>\nc2c3 c2c3{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 15,
        },
        # Duplicate FEN with worse cp_loss (should be dropped in favor of row 0)
        {
            "fen": fen1,
            "move": "e1g1",
            "completion": f"<think>\nCastle.\n</think>\ne1g1{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 40,
        },
        # Blunder (> MAX_CENTIPAWN_LOSS) labeled reasonable -> must be dropped
        {
            "fen": fen2,
            "move": "b5a4",
            "completion": f"<think>\nRetreat.\n</think>\nb5a4{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": MAX_CENTIPAWN_LOSS + 50,
        },
        # Illegal move -> must be dropped
        {
            "fen": fen2,
            "move": "e1e8",
            "completion": f"<think>\nIllegal.\n</think>\ne1e8{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 0,
        },
        # Missing FEN -> must be dropped
        {"move": "e2e4", "completion": "<think>x</think>\ne2e4"},
        # Marked unreasonable -> must be dropped
        {
            "fen": OPENING_SEED_FENS[2],
            "move": "f1e2",
            "completion": f"<think>\nDevelop.\n</think>\nf1e2{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": False,
            "centipawn_loss": 10,
        },
    ]

    with open(in_path, "w", encoding="utf-8") as fout:
        for r in dirty_rows:
            fout.write(json.dumps(r) + "\n")

    summary = clean_and_verify_sft_file(in_path, out_path, engine=None)
    assert summary["passed"] is True
    assert summary["total_input"] == 6
    assert summary["total_output"] == 1
    assert summary["dropped_invalid"] == 2
    assert summary["dropped_blunders"] == 2
    assert summary["dropped_duplicates"] == 1
    assert summary["repaired_think_tags"] >= 1
    assert summary["repaired_trailing_moves"] >= 1
    assert summary["engine_verified"] is False

    cleaned = read_jsonl(out_path)
    assert len(cleaned) == 1
    assert cleaned[0]["move"] == "c2c3"
    assert cleaned[0]["completion"].count("<think>") == 1
    assert cleaned[0]["completion"].count("</think>") == 1


# ---------------------------------------------------------------------------
# CLI Entry Point Tests
# ---------------------------------------------------------------------------


def test_cli_main_builds_files_and_cleans_sft(tmp_path: Path) -> None:
    sft_in = tmp_path / "sft_train.jsonl"
    sft_row = {
        "fen": OPENING_SEED_FENS[0],
        "move": "c2c3",
        "completion": f"<think>\nPrepare d4.\n</think>\nc2c3{CHATML_ASSISTANT_CLOSE}",
        "is_reasonable": True,
        "centipawn_loss": 10,
    }
    sft_in.write_text(json.dumps(sft_row) + "\n", encoding="utf-8")

    with patch("chessfm.data_generation.build_corpus.open_engine", return_value=_MockEngine()):
        rc = main(
            [
                "--output-dir",
                str(tmp_path),
                "--depth",
                "4",
                "--max-fens",
                "3",
                "--clean-sft",
            ]
        )
    assert rc == 0
    assert (tmp_path / "coaching_sample.jsonl").exists()
    assert (tmp_path / "eval_benchmark.jsonl").exists()
    assert len(read_jsonl(tmp_path / "eval_benchmark.jsonl")) == 3


def test_cli_main_raises_when_stockfish_missing() -> None:
    args = parse_args(["--output-dir", "/tmp/out", "--depth", "6", "--no-include-ascii"])
    assert args.depth == 6
    assert args.include_ascii is False

    with (
        patch("chessfm.data_generation.build_corpus.open_engine", return_value=None),
        pytest.raises(RuntimeError, match="Stockfish binary not found"),
    ):
        main(["--output-dir", "/tmp/out"])


# ---------------------------------------------------------------------------
# Live Stockfish Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not STOCKFISH_OK, reason="Stockfish binary not installed")
def test_live_stockfish_corpus_and_sft_verification(tmp_path: Path) -> None:
    engine = open_engine(threads=1, hash_mb=16)
    assert engine is not None
    try:
        # 2 opening, 4 middlegame, 2 endgame (exact 25/50/25 slice)
        live_fens = [
            *OPENING_SEED_FENS[:2],
            *MIDDLEGAME_SEED_FENS[:4],
            *ENDGAME_SEED_FENS[:2],
        ]
        result = build_verified_coaching_corpus(
            fens=live_fens,
            engine=engine,
            depth=6,
            include_ascii=True,
        )
        assert result["passed"] is True
        assert result["total_fens"] == 8
        assert result["audit"]["illegal_moves"] == 0
        assert result["audit"]["blunders_labeled_reasonable"] == 0
        assert result["spoiler_violations"] == 0
        assert result["claim_violations"] == 0

        # Verify live Stockfish catches an unmistakable blunder (hanging Queen with Qd7+)
        blunder_fen = "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 40"
        in_path = tmp_path / "sft_live.jsonl"
        out_path = tmp_path / "sft_live_clean.jsonl"
        rows = [
            {
                "fen": blunder_fen,
                "move": "d2d7",
                "completion": f"<think>\nPlay Qd7+.\n</think>\nd2d7{CHATML_ASSISTANT_CLOSE}",
                "is_reasonable": True,
                "centipawn_loss": 0,
            },
            {
                "fen": OPENING_SEED_FENS[0],
                "move": "c2c3",
                "completion": f"<think>\nPrepare d4.\n</think>\nc2c3{CHATML_ASSISTANT_CLOSE}",
                "is_reasonable": True,
                "centipawn_loss": 0,
            },
        ]
        in_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        summary = clean_and_verify_sft_file(in_path, out_path, engine=engine, depth=6)
        assert summary["passed"] is True
        assert summary["engine_verified"] is True
        assert summary["dropped_blunders"] == 1
        assert summary["total_output"] == 1
        cleaned = read_jsonl(out_path)
        assert cleaned[0]["fen"] == OPENING_SEED_FENS[0]
        assert cleaned[0]["stockfish_best"] is not None
    finally:
        close_engine(engine)


# ---------------------------------------------------------------------------
# Kaggle Training Notebook Validation Tests
# ---------------------------------------------------------------------------


def test_kaggle_training_notebook_valid_nbformat_4_5() -> None:
    assert NOTEBOOK_PATH.is_file(), f"Missing notebook at {NOTEBOOK_PATH}"
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert "kernelspec" in nb["metadata"]
    assert "language_info" in nb["metadata"]

    cells = nb["cells"]
    assert len(cells) >= 6

    seen_ids: set[str] = set()
    all_source_parts: list[str] = []

    for cell in cells:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell.get("id"), str) and cell["id"]
        assert cell["id"] not in seen_ids
        seen_ids.add(cell["id"])
        assert isinstance(cell.get("metadata"), dict)
        assert isinstance(cell.get("source"), list)
        src = "".join(cell["source"])
        all_source_parts.append(src)

        if cell["cell_type"] == "code":
            assert "execution_count" in cell
            assert isinstance(cell.get("outputs"), list)
            # Every non-magic/non-shell Python line in code cells must parse cleanly.
            py_lines = [
                line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
            ]
            ast.parse("\n".join(py_lines))

    full_text = "\n".join(all_source_parts)
    for required_token in (
        "apt-get",
        "stockfish",
        "pip install",
        "TRACK",
        "INCLUDE_ASCII",
        "GRPO_STAGE",
        "STEPS",
        "scripts/kaggle_train.py",
        "--task sft",
        "--task grpo",
        "--resume",
        "--task eval",
        "chessfm.eval.suite --demo",
        "scripts/export_gguf.py",
    ):
        assert required_token in full_text, f"Notebook missing required token: {required_token!r}"
