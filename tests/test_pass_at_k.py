"""Unit tests for stratified pass@k, Shannon move entropy, and CoT V2 faithfulness."""

from __future__ import annotations

import json
import math
from pathlib import Path

import chess
import pytest
import torch

from chessfm.eval.pass_at_k import (
    INVALID_MOVE_BUCKET,
    bootstrap_mean_ci,
    compare_pass_at_k_reports,
    evaluate_pass_at_k_rollouts,
    parse_args,
    run_pass_at_k_evaluation,
    sample_completions_batched,
    shannon_entropy_bits,
    stratified_puzzle_sample,
    unbiased_pass_at_k,
)


def _make_synthetic_puzzle_corpus() -> list[dict]:
    """Build a 90-puzzle synthetic corpus mimicking test_puzzles_900.jsonl structure."""
    rows: list[dict] = []
    # 50 theme puzzles (5 themes x 10)
    for t_idx, theme in enumerate(["fork", "pin", "mateIn1", "skewer", "backRankMate"]):
        for j in range(10):
            rows.append(
                {
                    "fen": chess.STARTING_FEN,
                    "best_move": "e2e4",
                    "rating": 1200 + t_idx * 100 + j,
                    "split_group": "theme",
                    "primary_theme": theme,
                }
            )
    # 40 difficulty puzzles (4 bands x 10)
    for band, rating in [
        ("<1200", 1000),
        ("1200-1599", 1400),
        ("1600-1999", 1800),
        (">=2000", 2100),
    ]:
        for _ in range(10):
            rows.append(
                {
                    "fen": chess.STARTING_FEN,
                    "best_move": "e2e4",
                    "rating": rating,
                    "split_group": "difficulty",
                    "difficulty_band": band,
                }
            )
    return rows


def test_stratified_puzzle_sample_proportions_and_determinism():
    corpus = _make_synthetic_puzzle_corpus()
    s1 = stratified_puzzle_sample(corpus, n=27, seed=42)
    s2 = stratified_puzzle_sample(corpus, n=27, seed=42)
    assert len(s1) == 27
    assert [r["original_index"] for r in s1] == [r["original_index"] for r in s2]
    assert [r["puzzle_id"] for r in s1] == [r["puzzle_id"] for r in s2]
    # 50/90 * 27 = 15 theme puzzles (3 per theme), 40/90 * 27 = 12 difficulty puzzles (3 per band).
    theme_rows = [r for r in s1 if r["split_group"] == "theme"]
    diff_rows = [r for r in s1 if r["split_group"] == "difficulty"]
    assert len(theme_rows) == 15
    assert len(diff_rows) == 12
    # All 5 themes and all 4 difficulty bands represented equally.
    assert len({r["primary_theme"] for r in theme_rows}) == 5
    assert len({r["difficulty_band"] for r in diff_rows}) == 4

    assert stratified_puzzle_sample(corpus, n=0) == []
    assert len(stratified_puzzle_sample(corpus, n=500)) == len(corpus)
    with pytest.raises(ValueError, match="n must be >= 0"):
        stratified_puzzle_sample(corpus, n=-1)


def test_unbiased_pass_at_k_exact_values_and_validation():
    assert unbiased_pass_at_k(8, 0, 1) == 0.0
    assert unbiased_pass_at_k(8, 0, 8) == 0.0
    assert unbiased_pass_at_k(8, 8, 1) == 1.0
    assert unbiased_pass_at_k(8, 1, 8) == 1.0
    assert unbiased_pass_at_k(8, 2, 1) == pytest.approx(0.25)
    # 1 - C(6, 2) / C(8, 2) = 1 - 15/28 = 13/28
    assert unbiased_pass_at_k(8, 2, 2) == pytest.approx(13.0 / 28.0)
    # 1 - C(6, 4) / C(8, 4) = 1 - 15/70 = 55/70
    assert unbiased_pass_at_k(8, 2, 4) == pytest.approx(55.0 / 70.0)

    with pytest.raises(ValueError):
        unbiased_pass_at_k(0, 0, 1)
    with pytest.raises(ValueError):
        unbiased_pass_at_k(8, 0, 9)
    with pytest.raises(ValueError):
        unbiased_pass_at_k(8, -1, 1)


def test_shannon_entropy_bits():
    assert shannon_entropy_bits([]) == 0.0
    assert shannon_entropy_bits(["e2e4"] * 8) == pytest.approx(0.0)
    assert shannon_entropy_bits(["e2e4", "d2d4"] * 4) == pytest.approx(1.0)
    assert shannon_entropy_bits(["a", "b", "c", "d", "e", "f", "g", "h"]) == pytest.approx(3.0)


def test_bootstrap_mean_ci_edge_cases():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)
    assert bootstrap_mean_ci([0.5]) == (0.5, 0.5)
    lo, hi = bootstrap_mean_ci([0.0, 1.0, 1.0, 0.0, 1.0], n_bootstrap=500, seed=42)
    assert 0.0 <= lo <= 0.6 <= hi <= 1.0
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.0, 1.0], n_bootstrap=0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.0, 1.0], ci_level=1.5)


def test_evaluate_pass_at_k_rollouts_metrics_and_cot_v2():
    # Mate-in-1 Scholar's Mate position (White queen h5, bishop c4, target h5f7).
    scholar_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    puzzles = [
        {
            "puzzle_id": "p0",
            "original_index": 0,
            "fen": scholar_fen,
            "best_move": "h5f7",
            "split_group": "theme",
            "primary_theme": "mateIn1",
            "rating": 1000,
        },
        {
            "puzzle_id": "p1",
            "original_index": 1,
            "fen": chess.STARTING_FEN,
            "best_move": "e2e4",
            "split_group": "difficulty",
            "difficulty_band": "<1200",
            "rating": 1000,
        },
    ]
    # Puzzle 0: 2/4 correct (h5f7), 1 legal wrong (c4f7), 1 illegal (z9z9)
    # Puzzle 1: 0/4 correct, all 4 legal identical (d2d4) -> 0 bits entropy
    rollouts = [
        [
            "<think>The white queen on h5 and bishop on c4 target f7.</think>\nh5f7",
            "<think>The white queen on h5 delivers mate.</think>\nh5f7",
            "<think>The white rook on h5 attacks f7.</think>\nc4f7",  # false V2 claim (no rook on h5!)
            "<think>Invalid move.</think>\nz9z9",
        ],
        [
            "<think>Develop center pawn.</think>\nd2d4",
            "<think>Develop center pawn.</think>\nd2d4",
            "<think>Develop center pawn.</think>\nd2d4",
            "<think>Develop center pawn.</think>\nd2d4",
        ],
    ]

    summary, records = evaluate_pass_at_k_rollouts(
        puzzles, rollouts, k_values=(1, 2, 4), n_bootstrap=200, seed=42
    )
    assert summary["num_puzzles"] == 2
    assert summary["k"] == 4
    assert summary["total_rollouts"] == 8
    assert summary["entropy_unit"] == "bits"
    # Puzzle 0 pass@1 = 0.5, Puzzle 1 pass@1 = 0.0 -> mean = 0.25
    assert summary["pass_at_k"]["pass@1"]["mean"] == pytest.approx(0.25)
    # Puzzle 0 pass@4 = 1.0, Puzzle 1 pass@4 = 0.0 -> mean = 0.5, solved_any = 1
    assert summary["pass_at_k"]["pass@4"]["mean"] == pytest.approx(0.5)
    assert summary["pass_at_k"]["pass@4"]["solved_any"] == 1
    assert len(summary["pass_at_k"]["pass@4"]["wilson_ci_95"]) == 2
    # Strict legal: 7 / 8
    assert summary["strict_legal_rollouts"] == 7
    assert summary["strict_legal_rate"] == pytest.approx(7 / 8)
    # Puzzle 0 all-buckets = [h5f7, h5f7, c4f7, <invalid>] -> H = 1.5 bits; Puzzle 1 -> 0.0 bits; mean = 0.75
    assert records[0]["move_entropy_all_bits"] == pytest.approx(1.5)
    assert records[1]["move_entropy_all_bits"] == pytest.approx(0.0)
    assert summary["mean_move_entropy_all_bits"] == pytest.approx(0.75)
    # Puzzle 0 legal-only = [h5f7, h5f7, c4f7] -> H = -(2/3 log2(2/3) + 1/3 log2(1/3)) ~ 0.9183
    expected_p0_legal_h = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert records[0]["move_entropy_legal_bits"] == pytest.approx(expected_p0_legal_h, abs=1e-4)
    # CoT V2 detected the hallucinated rook on h5!
    assert summary["cot_v2"]["traces_evaluated"] == 8
    assert summary["cot_v2"]["traces_with_claims"] >= 3
    assert summary["cot_v2"]["zero_violation_rate_with_claims"] < 1.0
    assert INVALID_MOVE_BUCKET == "<invalid>"


def test_sample_completions_batched_restores_tokenizer_and_runs(tmp_path: Path):
    class _ToyTok:
        def __init__(self):
            self.padding_side = "right"
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token_id = 0
            self.eos_token = "<eos>"

        def __call__(self, prompts, return_tensors="pt", padding=True):
            assert self.padding_side == "left"
            batch = torch.ones((len(prompts), 3), dtype=torch.long)
            return {"input_ids": batch}

        def decode(self, ids, skip_special_tokens=True):
            return "<think>ok</think>\ne2e4"

    class _ToyModel:
        device = None

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is True
            assert kwargs["temperature"] == pytest.approx(0.7)
            assert kwargs["top_p"] == pytest.approx(0.95)
            inp = kwargs["input_ids"]
            extra = torch.full((inp.shape[0], 2), 2, dtype=torch.long)
            return torch.cat([inp, extra], dim=1)

    tok = _ToyTok()
    outs = sample_completions_batched(
        _ToyModel(),
        tok,
        ["p1", "p2", "p3"],
        temperature=0.7,
        top_p=0.95,
        batch_size=2,
        seed=42,
    )
    assert len(outs) == 3
    assert tok.padding_side == "right"
    assert tok.pad_token_id is None
    assert tok.pad_token is None

    # End-to-end run_pass_at_k_evaluation + comparison helper
    puzzles_file = tmp_path / "puzzles.jsonl"
    corpus = _make_synthetic_puzzle_corpus()[:10]
    puzzles_file.write_text("\n".join(json.dumps(r) for r in corpus) + "\n", encoding="utf-8")

    rep_base, _ = run_pass_at_k_evaluation(
        _ToyModel(),
        tok,
        model_id="E2a",
        puzzles_path=puzzles_file,
        num_puzzles=4,
        k=4,
        n_bootstrap=100,
        seed=42,
        sample_fn=lambda *a, **kw: ["<think>go</think>\nd2d4"] * len(a[2]),
    )
    rep_cand, _ = run_pass_at_k_evaluation(
        _ToyModel(),
        tok,
        model_id="E3a",
        puzzles_path=puzzles_file,
        num_puzzles=4,
        k=4,
        n_bootstrap=100,
        seed=42,
        sample_fn=lambda *a, **kw: ["<think>go</think>\ne2e4"] * len(a[2]),
    )
    cmp_rep = compare_pass_at_k_reports(rep_base, rep_cand)
    assert cmp_rep["puzzle_ids_match"] is True
    assert cmp_rep["pass_at_k_comparison"]["pass@1"]["delta_pp"] == pytest.approx(100.0)
    assert cmp_rep["pass_at_k_comparison"]["pass@4"]["delta_pp"] == pytest.approx(100.0)

    args = parse_args(
        ["--model", "m", "--puzzles", str(puzzles_file), "--output", str(tmp_path / "o.json")]
    )
    assert args.num_puzzles == 200
    assert args.k == 8
    assert args.temperature == pytest.approx(0.7)
    assert args.top_p == pytest.approx(0.95)
