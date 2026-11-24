#!/usr/bin/env python3
"""Unit tests for Phase 4 (E5) Coach data prep, merging, leakage auditing, and blind audit packet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import chess

from chessfm.data_generation.distill_open_teacher import VALID_TASK_TYPES
from chessfm.data_generation.phase_sampler import normalize_board_fen

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_e5 = _load_script("build_coach_e5_data", "build_coach_e5_data.py")
blind_audit = _load_script("build_blind_audit_e5", "build_blind_audit_e5.py")

TEST_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
]


def _make_row(fen: str, task_type: str, response: str, severity: str = "mistake") -> dict:
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    best = moves[0].uci()
    student = moves[1].uci() if len(moves) > 1 else best
    return {
        "fen": fen,
        "mode": task_type,
        "task_type": task_type,
        "best_move": best,
        "best_move_uci": best,
        "student_move": student,
        "pv": [best],
        "cp_loss": 120.0,
        "severity": severity,
        "student_elo": 1200,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": response},
        ],
        "response": response,
    }


class TestBuildCoachE5Data:
    def test_enrich_filter_excludes_test_overlap_and_v2_failures(self):
        good_fen = TEST_FENS[0]
        overlap_fen = TEST_FENS[1]
        excluded = {normalize_board_fen(overlap_fen)}

        distilled = [
            _make_row(
                good_fen,
                "postmortem",
                "Develop your pieces actively toward the center to contest key central squares.",
            ),
            _make_row(
                overlap_fen,
                "postmortem",
                "Develop your pieces actively toward the center to contest key central squares.",
            ),
            _make_row(
                good_fen,
                "why_not",
                "Play Qh5 immediately to force checkmate on the black king.",  # illegal move -> V2 fail
            ),
            _make_row(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "socratic_l1",
                "Play Ng5 immediately to attack f7.",  # spoiler on L1!
            ),
        ]
        # Force best_move=f3g5 on row 3 so spoiler check fires.
        distilled[3]["best_move"] = "f3g5"
        distilled[3]["best_move_uci"] = "f3g5"

        kept, stats = build_e5.enrich_and_filter_existing_traces(distilled, distilled, excluded)
        assert len(kept) == 1
        assert kept[0]["fen"] == good_fen
        assert "Tactical Facts (Verified):" in kept[0]["messages"][1]["content"]
        assert "tactical_facts" in kept[0]
        assert stats["dropped_test_overlap"] == 1
        assert stats["dropped_v2_fail"] == 1
        assert stats["dropped_spoiler"] == 1

    def test_stratified_coach_slice_is_deterministic_subset(self):
        rows = []
        for tt in VALID_TASK_TYPES:
            for idx in range(4):
                rows.append(
                    _make_row(TEST_FENS[idx], tt, f"Clean unique hint {tt} {idx} for board.")
                )

        s1 = build_e5.build_stratified_coach_slice(rows, per_task=2, seed=42)
        s2 = build_e5.build_stratified_coach_slice(rows, per_task=2, seed=42)
        assert len(s1) == 10
        assert s1 == s2
        for tt in VALID_TASK_TYPES:
            assert sum(1 for r in s1 if r["task_type"] == tt) == 2

    def test_build_coach_grpo_prompts_respects_zero_leakage(self):
        excluded = {normalize_board_fen(TEST_FENS[0])}
        candidates = [
            _make_row(TEST_FENS[i], tt, f"Hint {i} {tt} with concrete piece activity.")
            for i in range(4)
            for tt in VALID_TASK_TYPES
        ]
        prompts = build_e5.build_coach_grpo_prompts(candidates, excluded, per_task=2, seed=42)
        assert len(prompts) == 10
        for row in prompts:
            assert normalize_board_fen(row["fen"]) not in excluded
            assert "tactical_facts_json" in row
            assert "Tactical Facts (Verified):" in row["messages"][1]["content"]

    def test_merge_and_audit_coach_sft_enforces_zero_leakage_and_non_slop(self):
        test_fens_by_set = {"test_coach_300.jsonl": {normalize_board_fen(TEST_FENS[5])}}
        existing = [
            _make_row(
                TEST_FENS[0],
                "postmortem",
                "Advancing a central pawn opens lines for both your queen and light-squared bishop cleanly.",
            ),
            _make_row(
                TEST_FENS[1],
                "socratic_l1",
                "Think about how your minor pieces can coordinate rapidly toward central squares.",
            ),
        ]
        fresh = [
            _make_row(
                TEST_FENS[2],
                "why_not",
                "Passive wing moves cede initiative while active development challenges your opponent immediately.",
            ),
            _make_row(
                TEST_FENS[5],  # overlaps frozen test set -> MUST be dropped!
                "postmortem",
                "This overlapping test position must be filtered out automatically during merge.",
            ),
        ]
        merged, audit = build_e5.merge_and_audit_coach_sft(
            existing, fresh, test_fens_by_set, seed=42
        )
        assert len(merged) == 3
        assert audit["zero_leakage_verified"] is True
        assert audit["test_overlap_counts"] == {"test_coach_300.jsonl": 0}
        assert audit["dropped_counts"]["test_overlap"] == 1
        assert audit["v2_pass_rate"] == 1.0
        assert audit["l12_spoiler_rate"] == 0.0
        assert audit["slop_audit"]["is_slop_corpus"] is False


class TestBlindAuditE5:
    def test_blind_audit_zero_model_name_leakage_and_deterministic_key(self, tmp_path: Path):
        test_rows = []
        for idx, fen in enumerate(TEST_FENS[:3]):
            for tt in VALID_TASK_TYPES:
                sev = ("blunder", "mistake", "inaccuracy")[idx % 3]
                test_rows.append(_make_row(fen, tt, "placeholder", severity=sev))

        secret_models = {
            "arm_a_sft_grpo_v2": [
                {
                    "fen": r["fen"],
                    "task_type": r["task_type"],
                    "response": f"Concrete coaching advice for position {i} focusing on piece activity.",
                }
                for i, r in enumerate(test_rows)
            ],
            "arm_b_e3b_transfer": [
                {
                    "fen": r["fen"],
                    "task_type": r["task_type"],
                    "response": f"Alternative grounded guidance for position {i} emphasizing king safety.",
                }
                for i, r in enumerate(test_rows)
            ],
            "gemma_26b_teacher": [
                {
                    "fen": r["fen"],
                    "task_type": r["task_type"],
                    "response": f"Teacher reference explanation for position {i} highlighting tactical control.",
                }
                for i, r in enumerate(test_rows)
            ],
        }

        md1, key1 = blind_audit.build_blind_audit(secret_models, test_rows, per_task=2, seed=2026)
        md2, key2 = blind_audit.build_blind_audit(secret_models, test_rows, per_task=2, seed=2026)
        assert md1 == md2
        assert key1 == key2
        assert key1["total_positions"] == 10

        # CRITICAL: ZERO real model-name leakage in Markdown output!
        for real_name in secret_models:
            assert real_name not in md1
        assert "Model X" in md1 and "Model Y" in md1 and "Model Z" in md1

        # Verify CLI end-to-end writing files.
        bench_path = tmp_path / "bench.jsonl"
        build_e5.write_jsonl(bench_path, test_rows)
        specs = []
        for name, rows in secret_models.items():
            p = tmp_path / f"{name}.jsonl"
            build_e5.write_jsonl(p, rows)
            specs.append(f"{name}={p}")

        out_md = tmp_path / "blind.md"
        out_key = tmp_path / "blind_key.json"
        rc = blind_audit.main(
            [
                "--inputs",
                *specs,
                "--benchmark",
                str(bench_path),
                "--output-md",
                str(out_md),
                "--output-key",
                str(out_key),
                "--per-task",
                "2",
                "--seed",
                "2026",
            ]
        )
        assert rc == 0
        assert out_md.is_file() and out_key.is_file()
        loaded_key = json.loads(out_key.read_text(encoding="utf-8"))
        assert loaded_key["total_positions"] == 10
