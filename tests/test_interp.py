"""Unit tests for chessfm.interp (probes, LoRA SVD, logit lens, attention circuits)."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import torch

from chessfm.interp import (
    PIECE_TO_CLASS,
    analyze_lora_adapter,
    build_corrupted_think_prefill,
    build_random_control_vector,
    build_square_steering_vector,
    compare_lora_profiles,
    compute_exact_lora_svd,
    fen_to_board_targets,
    find_square_token_indices,
    parse_lora_module_key,
    rank_piece_tracker_heads,
    sample_layer_matched_random_heads,
    sample_think_checkpoints,
    score_attention_heads,
    summarize_logit_lens,
    train_board_linear_probe,
)


def test_fen_to_board_targets_startpos():
    targets = fen_to_board_targets(chess.STARTING_FEN)
    assert len(targets) == 64
    assert targets[chess.A1] == PIECE_TO_CLASS["R"]
    assert targets[chess.E1] == PIECE_TO_CLASS["K"]
    assert targets[chess.E2] == PIECE_TO_CLASS["P"]
    assert targets[chess.E4] == 0
    assert targets[chess.E8] == PIECE_TO_CLASS["k"]


def test_train_board_linear_probe_recovers_synthetic_signal():
    torch.manual_seed(0)
    n_train, n_test, d = 160, 40, 64
    y_all = torch.zeros((n_train + n_test, 64), dtype=torch.long)
    # Put a linearly separable signal on squares 0..7
    x_all = torch.randn((n_train + n_test, d)) * 0.05
    for sq in range(8):
        labels = torch.randint(0, 3, (n_train + n_test,))
        y_all[:, sq] = labels
        for cls in range(3):
            x_all[labels == cls, sq * 3 + cls] += 4.0

    res = train_board_linear_probe(
        x_all[:n_train],
        y_all[:n_train],
        x_all[n_train:],
        y_all[n_train:],
        epochs=120,
        lr=5e-2,
    )
    assert res["overall_accuracy"] > 0.95
    assert res["occupied_accuracy"] > 0.85
    assert res["empty_accuracy"] > 0.95
    assert res["weight"].shape == (64, 13, d)


def test_train_board_linear_probe_validates_shapes():
    with pytest.raises(ValueError, match="2D tensors"):
        train_board_linear_probe(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 64, dtype=torch.long),
            torch.zeros(2, 4),
            torch.zeros(2, 64, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="shape"):
        train_board_linear_probe(
            torch.zeros(4, 8),
            torch.zeros(4, 10, dtype=torch.long),
            torch.zeros(2, 8),
            torch.zeros(2, 64, dtype=torch.long),
        )


def test_steering_and_orthogonal_random_control_vector():
    w = torch.randn(64, 13, 32)
    steer = build_square_steering_vector(w, chess.D8, PIECE_TO_CLASS["q"], 0)
    assert steer.shape == (32,)
    assert pytest.approx(float(steer.norm()), rel=1e-4) == 1.0

    ctrl = build_random_control_vector(steer * 3.5, seed=7)
    assert pytest.approx(float(ctrl.norm()), rel=1e-4) == 3.5
    # Must be orthogonal to steer
    assert abs(float(torch.dot(steer, ctrl))) < 1e-4

    with pytest.raises(ValueError):
        build_square_steering_vector(w, 64, 1, 0)
    with pytest.raises(ValueError):
        build_square_steering_vector(w, 0, 15, 0)


def test_parse_lora_module_key_and_exact_svd():
    key = "base_model.model.model.layers.17.mlp.down_proj.lora_B.weight"
    assert parse_lora_module_key(key) == (17, "mlp", "down_proj", "lora_B")
    assert parse_lora_module_key("lm_head.weight") is None

    torch.manual_seed(1)
    a = torch.randn(4, 32)
    b = torch.randn(48, 4)
    scaling = 2.0
    stats = compute_exact_lora_svd(a, b, scaling=scaling)
    full_dw = (b @ a) * scaling
    exact_svals = torch.linalg.svdvals(full_dw)
    assert pytest.approx(stats["frobenius_norm"], rel=1e-4) == float(full_dw.norm())
    assert pytest.approx(stats["spectral_norm"], rel=1e-4) == float(exact_svals[0])
    assert 1.0 <= stats["effective_rank"] <= 4.0

    with pytest.raises(ValueError, match="Incompatible LoRA shapes"):
        compute_exact_lora_svd(torch.zeros(4, 8), torch.zeros(16, 3))


def test_analyze_and_compare_lora_adapter(tmp_path: Path):
    from safetensors.torch import save_file

    def _make_adapter(subdir: Path, scale_mlp: float) -> Path:
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8"
        )
        torch.manual_seed(42)
        tensors = {}
        for layer in range(3):
            tensors[f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"] = (
                torch.randn(4, 16) * 0.1
            )
            tensors[f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight"] = (
                torch.randn(16, 4) * 0.1
            )
            tensors[f"base_model.model.model.layers.{layer}.mlp.down_proj.lora_A.weight"] = (
                torch.randn(4, 16) * 0.1
            )
            tensors[f"base_model.model.model.layers.{layer}.mlp.down_proj.lora_B.weight"] = (
                torch.randn(16, 4) * (0.1 * scale_mlp)
            )
        save_file(tensors, str(subdir / "adapter_model.safetensors"))
        return subdir

    p_a = analyze_lora_adapter(_make_adapter(tmp_path / "a", 1.0))
    p_b = analyze_lora_adapter(_make_adapter(tmp_path / "b", 3.0))
    assert p_a["num_layers"] == 3
    assert p_a["num_modules"] == 6
    assert p_b["mlp_energy_share"] > p_a["mlp_energy_share"]

    cmp_res = compare_lora_profiles(p_a, p_b)
    assert cmp_res["cosine_similarity"] > 0.9
    assert cmp_res["norm_ratio_b_over_a"] > 1.5
    assert len(cmp_res["top_diff_layers"]) == 3


def test_think_checkpoints_corrupted_prefill_and_logit_lens():
    assert sample_think_checkpoints(21) == [0, 5, 10, 15, 20]
    assert sample_think_checkpoints(0) == [0, 0, 0, 0, 0]

    prefill, distractor = build_corrupted_think_prefill(chess.STARTING_FEN, "e2e4")
    assert "<think>" in prefill and "</think>" in prefill
    assert distractor != "e2e4"
    assert chess.Move.from_uci(distractor) in chess.Board(chess.STARTING_FEN).legal_moves

    summary = summarize_logit_lens(
        [
            {
                "prob_by_fraction": {"0.0": 0.1, "0.5": 0.4, "1.0": 0.8},
                "top1_by_fraction": {"0.0": False, "0.5": False, "1.0": True},
            },
            {
                "prob_by_fraction": {"0.0": 0.2, "0.5": 0.6, "1.0": 0.9},
                "top1_by_fraction": {"0.0": False, "0.5": True, "1.0": True},
            },
        ],
        fractions=(0.0, 0.5, 1.0),
    )
    assert summary["num_puzzles"] == 2
    assert summary["mean_prob_by_fraction"]["1.0"] == 0.85
    assert summary["top1_match_by_fraction"]["1.0"] == 1.0
    assert summary["mean_prob_gain_0_to_100"] == 0.7
    assert summarize_logit_lens([])["num_puzzles"] == 0


def test_attention_circuit_helpers():
    toks = ["Pieces:", "White:", "Kf1,", "Ra2,", "Black:", "Rb3,", "Kf7"]
    tgt, bg = find_square_token_indices(toks, {"b3", "f1"})
    assert tgt == [2, 5]
    assert bg == [3, 6]

    # Synthetic attention: layer 1 head 2 strongly attends to target indices [2, 5]
    seq_len = 8
    attn = []
    for layer in range(2):
        t = torch.full((1, 4, seq_len, seq_len), 0.05)
        if layer == 1:
            t[0, 2, 7, 2] = 0.45
            t[0, 2, 7, 5] = 0.40
        attn.append(t)

    scores = score_attention_heads(attn, query_pos=7, target_indices=tgt, background_indices=bg)
    top = rank_piece_tracker_heads([scores], top_k=2)
    assert top[0]["layer"] == 1 and top[0]["head"] == 2
    assert top[0]["mean_selectivity"] > 0.3

    ctrls = sample_layer_matched_random_heads(top, num_heads_per_layer=4, seed=42)
    assert len(ctrls) == 2
    assert ctrls[0]["layer"] == 1 and ctrls[0]["head"] != 2

    # Multi-token BPE splits (e.g., Qwen3 splitting digits into separate tokens)
    bpe_toks = ["White:", " K", "f", "1", ",", " R", "b", "3", ",", " pawns", " e", "4"]
    bpe_tgt, bpe_bg = find_square_token_indices(bpe_toks, {"b3"})
    assert bpe_tgt == [5, 6, 7]
    assert bpe_bg == [1, 2, 3, 10, 11]
