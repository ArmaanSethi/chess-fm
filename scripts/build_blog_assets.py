"""Deterministic 0-diff compiler for ChessFM blog data and publication SVGs.

Reads raw verified manifests under ``results/`` directly and generates:
1. ``web/data/blog_data.json`` (schema_version "1.0", deterministic JSON)
2. Seven light/dark-safe editorial publication SVGs under ``docs/figures/``

Run with ``--check`` to verify exact byte-for-byte 0-diff parity against
already-committed files without writing to disk.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import chess

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
WEB_DATA_PATH = REPO_ROOT / "web" / "data" / "blog_data.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

# Editorial Warm Paper Palette (WCAG AA/AAA, safe on GitHub Light & Dark)
PAPER = "#FAF8F5"
SURFACE = "#FFFFFF"
SURFACE_WARM = "#F3EFE7"
HAIRLINE = "#DFD9CD"
GRID = "#ECE7DD"
INK = "#171717"
INK_SOFT = "#3A3835"
INK_MUTED = "#6B675E"
VERMILION = "#C83E2B"
VERMILION_SOFT = "#FBEAE7"
EMERALD = "#147A53"
EMERALD_SOFT = "#E6F4EE"
COBALT = "#1D5B9E"
COBALT_SOFT = "#E8F1FA"
AMBER = "#B86B14"
AMBER_SOFT = "#FDF3E3"
SLATE = "#7A756C"
BOARD_LIGHT = "#EFE6D5"
BOARD_DARK = "#8CA67B"

SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, monospace"

# Exact 5 verified board examples from test_fens_1000_strict.json
BOARD_EXAMPLE_SPECS: list[dict[str, str]] = [
    {
        "id": "pinned_c3",
        "title": "Pinned Knight on c3",
        "phase": "Opening",
        "fen": "r1bqk2r/ppp2ppp/2n5/3np3/1bB5/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
    },
    {
        "id": "kingside_castle",
        "title": "Kingside Castling vs Edge-Pawn Blunder",
        "phase": "Middlegame",
        "fen": "2rqk2r/1b1pb1p1/p1n2p2/3Bp2p/1Pp1P3/P3BN2/2Q2PPP/R3K2R w KQk - 2 17",
    },
    {
        "id": "c3_pawn_fork",
        "title": "Ignoring an Attacking Pawn on c3",
        "phase": "Opening",
        "fen": "r3kbnr/pp1q2pp/2n1bp2/2p1p3/4P3/2pP1NP1/PP1N1PBP/R1BQR1K1 w kq - 0 10",
    },
    {
        "id": "free_bishop_e6",
        "title": "Endgame Knight Capture on e6",
        "phase": "Endgame",
        "fen": "k4N2/5Pr1/pp1pb3/1np5/4p2B/2P2pP1/PP3R2/6K1 w - - 0 37",
    },
    {
        "id": "hanging_knight_g5",
        "title": "Hanging Knight on g5 Attacked by h6",
        "phase": "Middlegame",
        "fen": "r3k2r/ppqn1pb1/2p3pp/2p1p1N1/2P1P3/3P3P/PBP2PP1/R2Q1RK1 w kq - 0 14",
    },
]

FIGURE_FILENAMES = (
    "fig1_ablation_ladder.svg",
    "fig2_apawn_hack_board_and_diversity.svg",
    "fig3_pass_at_k_and_elo_mirage.svg",
    "fig4_coach_showdown_and_goodhart.svg",
    "fig5_board_probes_and_piece_trackers.svg",
    "fig6_lora_svd_and_slop_steering.svg",
    "fig7_logit_lens_and_circuit.svg",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_row(path: Path, index: int) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[index])


def _pct(val: float | None, digits: int = 2) -> float | None:
    if val is None:
        return None
    return round(val * 100.0, digits)


def _parse_final_eval_loss(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"['\"]eval_loss['\"]:\s*([0-9.]+)", text)
    if not matches:
        raise ValueError(f"Could not find eval_loss in {log_path}")
    return round(float(matches[-1]), 4)


def _parse_e0_eval_loss(repo_root: Path) -> float:
    exp_log = repo_root / "docs" / "EXPERIMENT_LOG.md"
    text = exp_log.read_text(encoding="utf-8")
    match = re.search(r"Final `eval_loss=([0-9.]+)`", text)
    if not match:
        raise ValueError(f"Could not parse E0 eval_loss from {exp_log}")
    return round(float(match.group(1)), 4)


def _uci_to_san(fen: str, uci: str) -> str | None:
    if not uci:
        return None
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return board.san(move)


def _extract_board_examples(results_dir: Path) -> list[dict[str, Any]]:
    details_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key, rel_path in [
        ("E2a", "phase2/e2a/e2a_test_fens_1000_strict.json"),
        ("E3a", "phase2/e3a/e3a_test_fens_1000_strict.json"),
        ("E3b", "phase2/e3b/e3b_test_fens_1000_strict.json"),
    ]:
        raw = _load_json(results_dir / rel_path)
        details_by_model[model_key] = {
            row["fen"]: row for row in raw["tiers"]["tier1_static"]["details"]
        }

    examples: list[dict[str, Any]] = []
    for spec in BOARD_EXAMPLE_SPECS:
        fen = spec["fen"]
        board = chess.Board(fen)
        e2a_row = details_by_model["E2a"][fen]
        best_move = e2a_row["best_move"]
        entry: dict[str, Any] = {
            "id": spec["id"],
            "title": spec["title"],
            "phase": spec["phase"],
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "best_move": best_move,
            "best_san": _uci_to_san(fen, best_move),
        }
        for model_key in ("E2a", "E3a", "E3b"):
            row = details_by_model[model_key][fen]
            move_uci = row["predicted_move"]
            cot = row.get("cot") or {}
            violations = [
                v.get("reason", str(v)) if isinstance(v, dict) else str(v)
                for v in cot.get("violations", [])
            ]
            entry[model_key] = {
                "move": move_uci,
                "san": _uci_to_san(fen, move_uci),
                "legal": bool(row["legal"]),
                "cp_loss": round(float(row["cp_loss"]), 1)
                if row.get("cp_loss") is not None
                else None,
                "wdl_loss_pct": round(float(row["wdl_loss"]), 2)
                if row.get("wdl_loss") is not None
                else None,
                "exact": bool(row.get("exact_match", False)),
                "cot_verified": int(cot.get("claims_verified", 0)),
                "cot_total": int(cot.get("claims_total", 0)),
                "cot_violations": violations,
            }
        examples.append(entry)
    return examples


def _extract_training_curve(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    out: list[dict[str, Any]] = []
    for row in raw["log_history"]:
        out.append(
            {
                "step": int(row["step"]),
                "reward": round(float(row["reward"]), 4),
                "reward_std": round(float(row["reward_std"]), 4),
                "kl": round(float(row["kl"]), 6),
                "loss": round(float(row["loss"]), 4),
                "completion_length": round(float(row["completion_length"]), 2),
                "clip_ratio": round(float(row["clip_ratio"]), 4),
            }
        )
    return out


def _extract_chessqa_categories(pce_raw: dict[str, Any]) -> dict[str, float]:
    per_cat = pce_raw["chessqa"]["per_category"]
    return {cat: round(float(info["accuracy"]) * 100.0, 2) for cat, info in sorted(per_cat.items())}


def _extract_sft_ladder(repo_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    base_ascii_s = _load_json(results_dir / "m3_base/base_think_off_strict.json")
    base_ascii_c = _load_json(results_dir / "m3_base/base_think_off_constrained.json")
    base_pl_s = _load_json(results_dir / "phase2/e1a/base_piece_list_200fen_strict.json")
    e0_s = _load_json(results_dir / "m3_sft/sft_think_on_strict.json")
    e0_c = _load_json(results_dir / "m3_sft/sft_think_on_constrained.json")
    e0_g_s = _load_json(results_dir / "m3_final/grpo_s2_think_on_strict.json")
    e0_g_c = _load_json(results_dir / "m3_final/grpo_s2_think_on_constrained.json")

    e1a_s = _load_json(results_dir / "phase2/e1a/e1a_200fen_strict.json")
    e1a_c = _load_json(results_dir / "phase2/e1a/e1a_200fen_constrained.json")
    e1a_pce = _load_json(results_dir / "phase2/e1a/e1a_puzzles_chessqa.json")

    e1b_s = _load_json(results_dir / "phase2/e1b/e1b_test_fens_1000_strict.json")
    e1b_c = _load_json(results_dir / "phase2/e1b/e1b_test_fens_1000_constrained.json")
    e1b_pce = _load_json(results_dir / "phase2/e1b/e1b_puzzles_chessqa_english.json")

    e2a_s = _load_json(results_dir / "phase2/e2a/e2a_test_fens_1000_strict.json")
    e2a_c = _load_json(results_dir / "phase2/e2a/e2a_test_fens_1000_constrained.json")
    e2a_pce = _load_json(results_dir / "phase2/e2a/e2a_puzzles_chessqa_english.json")

    e2b_s = _load_json(results_dir / "phase2/e2b/e2b_test_fens_1000_strict.json")
    e2b_c = _load_json(results_dir / "phase2/e2b/e2b_test_fens_1000_constrained.json")
    e2b_pce = _load_json(results_dir / "phase2/e2b/e2b_puzzles_chessqa_english.json")

    gemma26b = _load_json(results_dir / "phase2/e5/gemma26b_puzzles_chessqa.json")

    def _row(
        row_id: str,
        label: str,
        params: str,
        method: str,
        val_loss: float | None,
        strict_raw: dict[str, Any] | None,
        constr_raw: dict[str, Any] | None,
        pce_raw: dict[str, Any] | None,
        puzzles_lenient_pct: float | None = None,
    ) -> dict[str, Any]:
        t1_s = strict_raw["tiers"]["tier1_static"] if strict_raw else {}
        t2_s = strict_raw["tiers"].get("tier2_probes", {}) if strict_raw else {}
        t1_c = constr_raw["tiers"]["tier1_static"] if constr_raw else {}

        pos_acc = None
        if t2_s and "per_category" in t2_s and "piece_on_square" in t2_s["per_category"]:
            pos_acc = _pct(t2_s["per_category"]["piece_on_square"]["accuracy"], 2)

        puz_pct = None
        puz_legal_pct = None
        cqa_pct = None
        cqa_cats: dict[str, float] | None = None
        eng_pct = None
        eng_leak_pct = None
        if pce_raw:
            puz = pce_raw["puzzles"]
            puz_pct = _pct(puz["accuracy"], 2)
            puz_legal_pct = _pct(puz["legal_rate"], 2)
            cqa = pce_raw["chessqa"]
            cqa_pct = _pct(cqa["accuracy"], 2)
            cqa_cats = _extract_chessqa_categories(pce_raw)
            if "english_forgetting" in pce_raw:
                eng = pce_raw["english_forgetting"]
                eng_pct = _pct(eng["accuracy"], 2)
                eng_leak_pct = _pct(eng["chess_leakage_rate"], 2)

        return {
            "id": row_id,
            "label": label,
            "params": params,
            "method": method,
            "val_loss": val_loss,
            "strict_legal_pct": _pct(t1_s.get("legal_rate"), 2) if t1_s else None,
            "strict_acpl": round(float(t1_s["acpl"]), 2) if t1_s.get("acpl") is not None else None,
            "strict_mar_pct": _pct(t1_s.get("mar"), 2) if t1_s.get("mar") is not None else None,
            "constr_acpl": round(float(t1_c["acpl"]), 2) if t1_c.get("acpl") is not None else None,
            "piece_on_square_pct": pos_acc,
            "puzzles_900_pct": puz_pct,
            "puzzles_900_lenient_pct": puzzles_lenient_pct,
            "puzzles_900_legal_pct": puz_legal_pct,
            "chessqa_1000_pct": cqa_pct,
            "chessqa_categories": cqa_cats,
            "english_pct": eng_pct,
            "english_leak_pct": eng_leak_pct,
            "cot_v2_pct": _pct(t1_s.get("cot_claim_accuracy"), 2)
            if t1_s.get("cot_claim_accuracy") is not None
            else None,
            "cot_zero_viol_pct": _pct(t1_s.get("cot_zero_violation_rate"), 2)
            if t1_s.get("cot_zero_violation_rate") is not None
            else None,
        }

    return [
        _row(
            "Base_ASCII",
            "Base Qwen3-4B (ASCII)",
            "4B",
            "Zero-shot ASCII (Think OFF)",
            None,
            base_ascii_s,
            base_ascii_c,
            None,
        ),
        _row(
            "Base_PL",
            "Base Qwen3-4B (Piece-List)",
            "4B",
            "Zero-shot Piece-List",
            None,
            base_pl_s,
            None,
            None,
        ),
        _row(
            "E0",
            "E0 SFT (2.5K ASCII, r=16)",
            "4B",
            "LoRA r=16, lr=2e-5, ASCII",
            _parse_e0_eval_loss(repo_root),
            e0_s,
            e0_c,
            None,
        ),
        _row(
            "E0_GRPO",
            "E0 GRPO S2 (+400 steps)",
            "4B",
            "Staged GRPO S0->S2, ASCII",
            None,
            e0_g_s,
            e0_g_c,
            None,
        ),
        _row(
            "E1a",
            "E1a SFT (2.5K PL, r=16)",
            "4B",
            "All-Linear LoRA r=16, lr=2e-4, Piece-List",
            _parse_final_eval_loss(results_dir / "phase2/e1a/modal_sft_launch.log"),
            e1a_s,
            e1a_c,
            e1a_pce,
        ),
        _row(
            "E1b",
            "E1b SFT (20K Mix, r=16)",
            "4B",
            "All-Linear LoRA r=16, lr=2e-4, 20K Mix",
            _parse_final_eval_loss(results_dir / "phase2/e1b/sft_launch.log"),
            e1b_s,
            e1b_c,
            e1b_pce,
        ),
        _row(
            "E2a",
            "E2a SFT (20K Mix, r=64)",
            "4B",
            "All-Linear LoRA r=64, lr=2e-4, 20K Mix",
            _parse_final_eval_loss(results_dir / "phase2/e2a/sft_launch.log"),
            e2a_s,
            e2a_c,
            e2a_pce,
        ),
        _row(
            "E2b",
            "E2b Full-FT (20K Mix)",
            "4B",
            "100% Full Fine-Tune, lr=2e-5, 20K Mix",
            _parse_final_eval_loss(results_dir / "phase2/e2b/sft_launch.log"),
            e2b_s,
            e2b_c,
            e2b_pce,
        ),
        _row(
            "Gemma26B",
            "Gemma-4-26B (Zero-Shot)",
            "26B-A4B",
            "Zero-shot Frontier Teacher, Piece-List",
            None,
            None,
            None,
            gemma26b,
            puzzles_lenient_pct=_pct(
                gemma26b["puzzles"]["supplementary_lenient_san_diagnostic"]["lenient_solved_rate"],
                2,
            ),
        ),
    ]


def _extract_rlvr_section(results_dir: Path) -> dict[str, Any]:
    mda = _load_json(results_dir / "phase2/e3b/move_diversity_audit.json")
    constr_map = {
        "E2a": _load_json(results_dir / "phase2/e2a/e2a_test_fens_1000_constrained.json"),
        "E3a": _load_json(results_dir / "phase2/e3a/e3a_test_fens_1000_constrained.json"),
        "E3b": _load_json(results_dir / "phase2/e3b/e3b_test_fens_1000_constrained.json"),
    }
    pce_map = {
        "E2a": _load_json(results_dir / "phase2/e2a/e2a_puzzles_chessqa_english.json"),
        "E3a": _load_json(results_dir / "phase2/e3a/e3a_puzzles_chessqa_english.json"),
        "E3b": _load_json(results_dir / "phase2/e3b/e3b_puzzles_chessqa_english.json"),
    }

    models: dict[str, Any] = {}
    for key in ("E2a", "E3a", "E3b"):
        m = mda[key]
        tc = constr_map[key]["tiers"]["tier1_static"]
        pce = pce_map[key]
        models[key] = {
            "label": m["label"],
            "strict_legal_pct": round(float(m["legal_rate"]) * 100.0, 2),
            "exact_match_pct": round(float(m["exact_match_rate"]) * 100.0, 2),
            "mar_all_pct": round(float(m["mar_le50_over_all"]) * 100.0, 2),
            "mar_legal_pct": round(float(m["mar_le50_over_legal"]) * 100.0, 2),
            "strict_acpl": round(float(m["acpl"]), 2),
            "mean_wdl_loss_pct": round(float(m["mean_wdl_loss"]), 2),
            "constr_acpl": round(float(tc["acpl"]), 2),
            "constr_mar_pct": round(float(tc["mar"]) * 100.0, 2),
            "constr_exact_pct": round(float(tc["puzzle_accuracy"]) * 100.0, 2),
            "cot_v2_pct": round(float(m["cot_claim_accuracy"]) * 100.0, 2),
            "cot_zero_viol_pct": round(float(m["cot_zero_violation_rate"]) * 100.0, 2),
            "unique_moves": int(m["unique_moves"]),
            "top1_move": str(m["top1_move"]),
            "top1_freq_pct": round(float(m["top1_freq_pct"]), 2),
            "a_file_pawn_push_pct": round(float(m["a_file_pawn_push_pct"]), 2),
            "a_file_pawn_push_count": int(m["a_file_pawn_push_count"]),
            "a_file_vertical_move_pct": round(float(m["a_file_vertical_move_pct"]), 2),
            "top10_moves": [[mv, int(cnt)] for mv, cnt in m["top10_moves"]],
            "passes_anti_hack_gate": bool(m["passes_anti_hack_gate"]),
            "puzzles_900_pct": round(float(pce["puzzles"]["accuracy"]) * 100.0, 2),
            "puzzles_900_legal_pct": round(float(pce["puzzles"]["legal_rate"]) * 100.0, 2),
            "chessqa_1000_pct": round(float(pce["chessqa"]["accuracy"]) * 100.0, 2),
            "chessqa_categories": _extract_chessqa_categories(pce),
            "english_pct": round(float(pce["english_forgetting"]["accuracy"]) * 100.0, 2),
        }

    return {
        "models": models,
        "training_curves": {
            "E3a": _extract_training_curve(results_dir / "phase2/e3a/ckpt200_trainer_state.json"),
            "E3b": _extract_training_curve(results_dir / "phase2/e3b/ckpt200_trainer_state.json"),
        },
        "board_examples": _extract_board_examples(results_dir),
    }


def _extract_pass_at_k(results_dir: Path) -> dict[str, Any]:
    raw = _load_json(results_dir / "phase2/e3b/pass_at_k_e2a_vs_e3a_vs_e3b.json")
    models: dict[str, Any] = {}
    for key in ("E2a", "E3a", "E3b"):
        m = raw["models"][key]
        pak = {}
        for k_label in ("pass@1", "pass@2", "pass@4", "pass@8"):
            entry = m["pass_at_k"][k_label]
            item: dict[str, Any] = {
                "mean_pct": round(float(entry["mean"]) * 100.0, 2),
                "ci95_pct": [round(float(x) * 100.0, 2) for x in entry["bootstrap_ci_95"]],
            }
            if "solved_any" in entry:
                item["solved_any"] = int(entry["solved_any"])
            pak[k_label] = item
        models[key] = {
            "pass_at_k": pak,
            "strict_legal_pct": round(float(m["strict_legal_rate"]) * 100.0, 2),
            "mean_move_entropy_all_bits": round(float(m["mean_move_entropy_all_bits"]), 4),
            "mean_move_entropy_legal_bits": round(float(m["mean_move_entropy_legal_bits"]), 4),
            "cot_v2": {
                "claim_accuracy_pct": round(float(m["cot_v2"]["claim_accuracy"]) * 100.0, 2),
                "zero_violation_pct": round(
                    float(m["cot_v2"]["zero_violation_rate_all"]) * 100.0, 2
                ),
            },
        }
    return {
        "num_puzzles": int(raw["num_puzzles"]),
        "k": int(raw["k"]),
        "sampling": raw["sampling"],
        "models": models,
    }


def _extract_elo_mirage(results_dir: Path) -> dict[str, Any]:
    autopsy = _load_json(results_dir / "phase2/e3b/repetition_autopsy.json")
    bt_paths = {
        "E2a_std": results_dir / "phase2/e2a/e2a_bradley_terry_elo.json",
        "E2a_norep": results_dir / "phase2/e2a/e2a_norep_bradley_terry_elo.json",
        "E3b_std": results_dir / "phase2/e3b/e3b_bradley_terry_elo.json",
        "E3b_norep": results_dir / "phase2/e3b/e3b_norep_bradley_terry_elo.json",
    }

    suites: list[dict[str, Any]] = []
    for suite_id in ("E2a_std", "E2a_norep", "E3b_std", "E3b_norep"):
        au = autopsy[suite_id]
        bt = _load_json(bt_paths[suite_id])
        anchors: dict[str, Any] = {}
        for anchor_key, a_info in bt["anchors"].items():
            anchors[anchor_key] = {
                "anchor_elo": float(a_info["anchor_elo"]),
                "opponent_name": a_info["opponent_name"],
                "games": int(a_info["games"]),
                "wins": int(a_info["wins"]),
                "losses": int(a_info["losses"]),
                "draws": int(a_info["draws"]),
                "score_rate_pct": round(float(a_info["score_rate"]) * 100.0, 2),
            }
        suites.append(
            {
                "id": suite_id,
                "label": au["label"],
                "elo": round(float(au["bt_elo"]), 2),
                "ci95": [round(float(x), 2) for x in au["bt_ci_95"]],
                "wins": int(au["wins"]),
                "losses": int(au["losses"]),
                "draws": int(au["draws"]),
                "rep3_draws": int(au["threefold_repetition_endings"]),
                "rep3_pct_of_draws": round(float(au["pct_draws_from_threefold"]), 2),
                "stalemate_draws": int(au["stalemate_draws"]),
                "max_moves_draws": int(au["max_moves_draws"]),
                "model_rep2_moves": int(au["model_moves_causing_2fold"]),
                "total_model_moves": int(au["total_model_moves"]),
                "model_rep2_rate_pct": round(float(au["model_rep2_rate_pct"]), 3),
                "filtered_plies": int(au["repetition_filtered_plies"]),
                "maia1100_wins": int(bt["anchors"]["maia_1100"]["wins"]),
                "anchors": anchors,
            }
        )

    # E2b_std has a flat schema
    e2b_bt = _load_json(results_dir / "phase2/e2b/e2b_bradley_terry_elo.json")
    e2b_draws = int(e2b_bt["total_draws"])
    e2b_rep3 = int(e2b_bt["total_rep3_draws"])
    e2b_anchors: dict[str, Any] = {}
    for anchor_key, a_info in e2b_bt["per_opponent"].items():
        e2b_anchors[anchor_key] = {
            "anchor_elo": float(a_info["anchor_elo"]),
            "games": int(a_info["games"]),
            "score": float(a_info["score"]),
            "score_rate_pct": round(float(a_info["score_rate"]) * 100.0, 2),
        }
    suites.append(
        {
            "id": "E2b_std",
            "label": "E2b Full-FT Standard Constrained",
            "elo": round(float(e2b_bt["elo"]), 2),
            "ci95": [round(float(e2b_bt["ci_low"]), 2), round(float(e2b_bt["ci_high"]), 2)],
            "wins": None,
            "losses": None,
            "draws": e2b_draws,
            "rep3_draws": e2b_rep3,
            "rep3_pct_of_draws": round(100.0 * e2b_rep3 / e2b_draws, 2),
            "stalemate_draws": int(e2b_bt["total_stalemate_draws"]),
            "max_moves_draws": int(e2b_bt["total_max_moves_draws"]),
            "model_rep2_moves": None,
            "total_model_moves": None,
            "model_rep2_rate_pct": None,
            "filtered_plies": 0,
            "maia1100_wins": None,
            "anchors": e2b_anchors,
        }
    )

    return {"suites": suites}


def _extract_goodhart_06b(results_dir: Path) -> dict[str, Any]:
    raw = _load_json(results_dir / "verifier_v2_rescore.json")
    grounded: dict[str, Any] = {}
    for key in ("teacher_26b", "base_06b_grounded", "sft_06b_grounded"):
        g = raw["grounded_292"][key]
        slop = g["boilerplate_slop"]
        top_4g = slop["top_repeated_4grams"][0] if slop.get("top_repeated_4grams") else None
        grounded[key] = {
            "total_rows": int(g["total_rows"]),
            "v1_pass_count": int(g["v1_pass_count"]),
            "v1_pass_pct": round(float(g["v1_pass_pct"]), 2),
            "v2_pass_count": int(g["v2_pass_count"]),
            "v2_pass_pct": round(float(g["v2_pass_pct"]), 2),
            "v2_and_non_slop_count": int(g["v2_and_non_slop_count"]),
            "v2_and_non_slop_pct": round(float(g["v2_and_non_slop_pass_rate"]), 2),
            "slop_flagged_count": int(slop["flagged_count"]),
            "slop_flagged_pct": round(float(slop["flagged_rate"]) * 100.0, 2),
            "distinct_4gram_ratio": round(float(slop["distinct_4gram_ratio"]), 4),
            "top_4gram": {
                "phrase": top_4g["ngram"],
                "doc_count": int(top_4g["doc_count"]),
                "doc_freq_pct": round(float(top_4g["doc_freq"]) * 100.0, 2),
            }
            if top_4g
            else None,
            "violation_breakdown": g.get("violation_breakdown", {}),
        }

    four_cases: list[dict[str, Any]] = []
    for row_key in ("row_0", "row_5", "row_185", "row_240"):
        c = raw["four_case_audit"][row_key]
        four_cases.append(
            {
                "row_index": int(c["row_index"]),
                "task_type": c["task_type"],
                "fen": c["fen"],
                "best_move": c["best_move"],
                "best_san": _uci_to_san(c["fen"], c["best_move"]),
                "sft_06b": {
                    "v1_passed": bool(c["sft_06b"]["v1_passed"]),
                    "v2_passed": bool(c["sft_06b"]["v2_passed"]),
                    "violation_types": list(c["sft_06b"]["v2_violation_types"]),
                    "violation_reasons": [v["reason"] for v in c["sft_06b"]["v2_violations"]],
                    "response": c["sft_06b"]["response"],
                },
                "teacher_26b": {
                    "v1_passed": bool(c["teacher_26b"]["v1_passed"]),
                    "v2_passed": bool(c["teacher_26b"]["v2_passed"]),
                    "violation_types": list(c["teacher_26b"]["v2_violation_types"]),
                    "response": c["teacher_26b"]["response"],
                },
            }
        )

    return {"grounded_292": grounded, "four_case_audit": four_cases}


def _extract_e5_coach(results_dir: Path) -> dict[str, Any]:
    summary = _load_json(results_dir / "phase2/e5/e5_coach_comparison_summary.json")
    curves_raw = _load_json(results_dir / "phase2/e5/goodhart_arm_a_vs_arm_c_curves.json")
    gemma_raw = _load_json(results_dir / "phase2/e5/gemma26b_puzzles_chessqa.json")

    row_order = [
        ("row1_base_4b", "Base Qwen3-4B"),
        ("row2_arm_a_sft", "Arm A SFT"),
        ("row3_arm_a_grpo", "Arm A V2-GRPO"),
        ("row4_arm_b_sft", "Arm B SFT (E3b->SFT)"),
        ("row5_arm_b_grpo", "Arm B V2-GRPO (Transfer)"),
        ("row6_teacher_26b", "Gemma-4-26B Teacher"),
        ("row7_arm_c_v1_grpo", "Arm C Weak V1-GRPO"),
    ]
    rows: list[dict[str, Any]] = []
    for r_id, short_label in row_order:
        r = summary["rows"][r_id]
        rows.append(
            {
                "id": r_id,
                "short_label": short_label,
                "label": r["label"],
                "total_cases": int(r["total_cases"]),
                "v1_pass_pct": round(float(r["v1_pass_rate"]) * 100.0, 2),
                "v2_pass_pct": round(float(r["v2_pass_rate"]) * 100.0, 2),
                "v1_minus_v2_pp": round(float(r["v1_minus_v2_gap"]) * 100.0, 2),
                "v2_and_non_slop_pct": round(float(r["v2_and_non_slop_rate"]) * 100.0, 2),
                "spoiler_free_pct": round(float(r["socratic_l12_spoiler_free_rate"]) * 100.0, 2),
                "all_clean_pct": round(float(r["all_clean_rate"]) * 100.0, 2),
                "all_clean_ci95_pct": [round(float(x) * 100.0, 2) for x in r["all_clean_ci95"]],
                "fact_coverage_pct": round(float(r["mean_tactical_fact_coverage"]) * 100.0, 2),
                "slop_flagged_pct": round(float(r["slop_flagged_rate"]) * 100.0, 2),
                "distinct_4gram_ratio": round(float(r["distinct_4gram_ratio"]), 4),
            }
        )

    def _curve_points(raw_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pts: list[dict[str, Any]] = []
        for entry in raw_list:
            sl = entry["heldout_100_slice"]
            tr = entry.get("train")
            pts.append(
                {
                    "step": int(entry["step"]),
                    "train_reward": round(float(tr["reward"]), 4) if tr else None,
                    "train_kl": round(float(tr["kl"]), 6) if tr else None,
                    "v1_pass_pct": round(float(sl["v1_pass_rate"]) * 100.0, 2),
                    "v2_pass_pct": round(float(sl["v2_pass_rate"]) * 100.0, 2),
                    "v1_minus_v2_pp": round(float(sl["v1_minus_v2_gap"]) * 100.0, 2),
                    "all_clean_pct": round(float(sl["all_clean_rate"]) * 100.0, 2),
                    "fact_coverage_pct": round(float(sl["mean_tactical_fact_coverage"]) * 100.0, 2),
                    "distinct_4gram_ratio": round(float(sl["distinct_4gram_ratio"]), 4),
                    "violation_counts": sl["violation_counts"],
                }
            )
        return pts

    arm_a_pts = _curve_points(curves_raw["arm_a_coach_v2"]["curve"])
    arm_c_pts = _curve_points(curves_raw["arm_c_coach_v1"]["curve"])

    puz = gemma_raw["puzzles"]
    lenient = puz["supplementary_lenient_san_diagnostic"]
    gemma26b_anchor = {
        "model_id": gemma_raw["model_id"],
        "puzzles_strict_solved": int(puz["solved"]),
        "puzzles_total": int(puz["total_puzzles"]),
        "puzzles_strict_pct": round(float(puz["accuracy"]) * 100.0, 2),
        "puzzles_strict_legal_pct": round(float(puz["legal_rate"]) * 100.0, 2),
        "puzzles_lenient_solved": int(lenient["lenient_solved"]),
        "puzzles_lenient_pct": round(float(lenient["lenient_solved_rate"]) * 100.0, 2),
        "puzzles_lenient_legal_pct": round(float(lenient["lenient_legal_rate"]) * 100.0, 2),
        "mean_output_tokens": round(float(puz["mean_output_tokens"]), 2),
        "finish_reasons": puz["finish_reasons"],
        "chessqa_correct": int(gemma_raw["chessqa"]["correct"]),
        "chessqa_total": int(gemma_raw["chessqa"]["total"]),
        "chessqa_pct": round(float(gemma_raw["chessqa"]["accuracy"]) * 100.0, 2),
        "chessqa_categories": _extract_chessqa_categories(gemma_raw),
    }

    # Verbatim IDX 24 (postmortem) and IDX 2 (socratic_l2) comparisons
    e5_dir = results_dir / "phase2/e5"
    verbatim_specs = [
        (
            24,
            "Postmortem: Live V1 Goodhart False Pass vs Clean V2-GRPO",
            [
                ("arm_c_v1_grpo", "Arm C Weak V1-GRPO"),
                ("teacher_26b", "Gemma-4-26B Teacher"),
                ("arm_a_grpo", "Arm A V2-GRPO"),
                ("arm_b_grpo", "Arm B V2-GRPO (Transfer)"),
            ],
        ),
        (
            2,
            "Socratic L2: Zero-Spoiler Discipline vs Base Move Leak",
            [
                ("base_4b", "Base Qwen3-4B"),
                ("arm_a_grpo", "Arm A V2-GRPO"),
                ("arm_b_grpo", "Arm B V2-GRPO (Transfer)"),
                ("teacher_26b", "Gemma-4-26B Teacher"),
            ],
        ),
    ]
    verbatim_comparisons: list[dict[str, Any]] = []
    for idx, title, model_specs in verbatim_specs:
        ref_row = _load_jsonl_row(e5_dir / "coach_eval_arm_a_grpo.jsonl", idx)
        fen = ref_row["fen"]
        responses: list[dict[str, Any]] = []
        for file_key, display_label in model_specs:
            row = _load_jsonl_row(e5_dir / f"coach_eval_{file_key}.jsonl", idx)
            viols = [
                v.get("reason", str(v)) if isinstance(v, dict) else str(v)
                for v in row.get("violations", [])
            ]
            responses.append(
                {
                    "model_key": file_key,
                    "label": display_label,
                    "v1_passed": bool(row["v1_passed"]),
                    "v2_passed": bool(row["v2_passed"]),
                    "spoiler_free": bool(row["spoiler_free"]),
                    "all_clean": bool(row["all_clean"]),
                    "fact_coverage": round(float(row["tactical_fact_coverage"]), 4),
                    "violations": viols,
                    "response": row["response"],
                }
            )
        verbatim_comparisons.append(
            {
                "row_index": idx,
                "title": title,
                "task_type": ref_row["task_type"],
                "fen": fen,
                "best_move": ref_row["best_move"],
                "best_san": _uci_to_san(fen, ref_row["best_move"]),
                "student_move": ref_row["student_move"],
                "student_san": _uci_to_san(fen, ref_row["student_move"]),
                "responses": responses,
            }
        )

    return {
        "rows": rows,
        "deltas": summary["deltas"],
        "goodhart_curves": {
            "steps": [p["step"] for p in arm_a_pts],
            "arm_a": arm_a_pts,
            "arm_c": arm_c_pts,
        },
        "gemma26b_anchor": gemma26b_anchor,
        "verbatim_comparisons": verbatim_comparisons,
    }


def _extract_interp(results_dir: Path) -> dict[str, Any]:
    i1 = _load_json(results_dir / "interp/interp1_board_probes_and_steering.json")
    i2 = _load_json(results_dir / "interp/interp2_lora_svd_and_slop_steering.json")
    i3 = _load_json(results_dir / "interp/interp3_logit_lens_and_cot_sabotage.json")
    i4 = _load_json(results_dir / "interp/interp4_attention_circuits_and_ablation.json")

    layers = [int(x) for x in i1["layers_probed"]]
    heatmap: dict[str, Any] = {}
    for ckpt in ("Base_ASCII", "E0_ASCII", "E1a_PieceList", "E2a_r64", "E3b_SharpWDL"):
        info = i1["heatmap"][ckpt]
        layer_rows: dict[str, dict[str, float]] = {}
        for layer in layers:
            ld = info["layers"][str(layer)]
            layer_rows[str(layer)] = {
                "occupied_pct": round(float(ld["occupied_accuracy"]) * 100.0, 2),
                "overall_pct": round(float(ld["overall_accuracy"]) * 100.0, 2),
                "empty_pct": round(float(ld["empty_accuracy"]) * 100.0, 2),
            }
        heatmap[ckpt] = {
            "include_ascii": bool(info["include_ascii"]),
            "include_piece_list": bool(info["include_piece_list"]),
            "peak_layer": int(info["peak_layer"]),
            "peak_occupied_pct": round(float(info["peak_occupied_accuracy"]) * 100.0, 2),
            "layers": layer_rows,
        }

    surg = i1["causal_board_surgery"]
    causal_board_surgery = {
        "model": surg["model"],
        "layer": int(surg["layer"]),
        "steer_norm": round(float(surg["steer_norm"]), 4),
        "valid_cases": int(surg["valid_cases"]),
        "probe_move_flip_pct": round(float(surg["probe_move_flip_rate"]) * 100.0, 2),
        "random_control_move_flip_pct": round(
            float(surg["random_control_move_flip_rate"]) * 100.0, 2
        ),
        "causal_excess_flip_pp": round(float(surg["causal_excess_flip_pp"]), 2),
        "probe_erased_square_abandoned_count": int(surg["probe_erased_square_abandoned_count"]),
        "random_erased_square_abandoned_count": int(surg["random_erased_square_abandoned_count"]),
    }

    profiles: dict[str, Any] = {}
    for adapter_key in ("e3a_apawn_hack", "e3b_sharp_wdl", "arm_a_v2_grpo", "arm_c_v1_grpo"):
        p = i2["lora_svd_profiles"][adapter_key]
        profiles[adapter_key] = {
            "lora_r": int(p["lora_r"]),
            "num_layers": int(p["num_layers"]),
            "num_modules": int(p["num_modules"]),
            "total_frobenius": round(float(p["total_frobenius"]), 5),
            "attn_frobenius": round(float(p["attn_frobenius"]), 5),
            "mlp_frobenius": round(float(p["mlp_frobenius"]), 5),
            "mlp_energy_share_pct": round(float(p["mlp_energy_share"]) * 100.0, 2),
            "attn_energy_share_pct": round(float(p["attn_energy_share"]) * 100.0, 2),
            "peak_layer": int(p["peak_layer"]),
            "mean_effective_rank": round(float(p["mean_effective_rank"]), 3),
            "mean_top1_energy_pct": round(float(p["mean_top1_energy_fraction"]) * 100.0, 2),
            "per_layer": [
                {
                    "layer": int(lr["layer"]),
                    "attn_frobenius": round(float(lr["attn_frobenius"]), 5),
                    "mlp_frobenius": round(float(lr["mlp_frobenius"]), 5),
                    "total_frobenius": round(float(lr["total_frobenius"]), 5),
                    "mlp_share_pct": round(float(lr["mlp_share"]) * 100.0, 2),
                }
                for lr in p["per_layer"]
            ],
        }

    slop_raw = i2["slop_vector_steering"]
    slop_steering = {
        "peak_diff_layer": int(slop_raw["peak_diff_layer"]),
        "raw_diff_norm": round(float(slop_raw["raw_diff_norm"]), 4),
        "calibrated_unit_scale": round(float(slop_raw["calibrated_unit_scale"]), 4),
        "n_diff_prompts": int(slop_raw["n_diff_prompts"]),
        "n_sweep_prompts": int(slop_raw["n_sweep_prompts"]),
        "sweep": [
            {
                "alpha": float(s["alpha"]),
                "v1_pass_pct": round(float(s["v1_pass_rate"]) * 100.0, 2),
                "v2_pass_pct": round(float(s["v2_pass_rate"]) * 100.0, 2),
                "v1_minus_v2_pp": round(float(s["v1_minus_v2_gap"]) * 100.0, 2),
                "distinct_4gram_ratio": round(float(s["distinct_4gram_ratio"]), 4),
                "slop_flagged_pct": round(float(s["slop_flagged_rate"]) * 100.0, 2),
                "l12_spoiler_free_pct": round(float(s["l12_spoiler_free_rate"]) * 100.0, 2),
                "mean_words": round(float(s["mean_words"]), 2),
                "sample_output_0": s["sample_output_0"],
            }
            for s in slop_raw["sweep"]
        ],
    }

    ll = i3["logit_lens"]
    sab = i3["causal_cot_sabotage"]
    interp3_cot_sabotage = {
        "num_solved_analyzed": int(i3["num_solved_analyzed"]),
        "logit_lens": {
            "mean_prob_by_fraction": {
                k: round(float(v), 5) for k, v in ll["mean_prob_by_fraction"].items()
            },
            "post_think_mean_prob": round(float(ll["post_think_mean_prob"]), 5),
            "post_think_top1_match_pct": round(float(ll["post_think_top1_match"]) * 100.0, 2),
        },
        "sabotage": {
            "num_tested": int(sab["num_tested"]),
            "causal_move_flip_pct": round(float(sab["causal_move_flip_rate"]) * 100.0, 2),
            "adopted_distractor_pct": round(float(sab["adopted_distractor_rate"]) * 100.0, 2),
            "resisted_sabotage_pct": round(float(sab["resisted_sabotage_rate"]) * 100.0, 2),
            "examples": [
                {
                    "target_move": ex["target_move"],
                    "distractor_move": ex["distractor_move"],
                    "post_sabotage_move": ex["post_sabotage_move"],
                }
                for ex in sab["examples"]
            ],
        },
    }

    interp4_piece_tracker = {
        "num_attention_examples": int(i4["num_attention_examples"]),
        "num_ablation_puzzles": int(i4["num_ablation_puzzles"]),
        "total_heads": 36 * 32,
        "top5_heads": [
            {
                "layer": int(h["layer"]),
                "head": int(h["head"]),
                "label": f"L{int(h['layer'])}.H{int(h['head'])}",
                "mean_selectivity": round(float(h["mean_selectivity"]), 6),
            }
            for h in i4["top5_piece_tracker_heads"]
        ],
        "random5_heads": [
            {
                "layer": int(h["layer"]),
                "head": int(h["head"]),
                "label": f"L{int(h['layer'])}.H{int(h['head'])}",
            }
            for h in i4["layer_matched_random_heads"]
        ],
        "ablation": {
            "baseline_solve_pct": round(float(i4["baseline_solve_rate_on_subset"]) * 100.0, 2),
            "top5_retained_solve_pct": round(
                float(i4["top5_zero_ablation"]["retained_solve_rate"]) * 100.0, 2
            ),
            "top5_solve_drop_pp": round(float(i4["top5_zero_ablation"]["solve_drop_pp"]), 2),
            "top5_retained_legal_pct": round(
                float(i4["top5_zero_ablation"]["retained_legal_rate"]) * 100.0, 2
            ),
            "random5_retained_solve_pct": round(
                float(i4["random5_zero_ablation"]["retained_solve_rate"]) * 100.0, 2
            ),
            "random5_solve_drop_pp": round(float(i4["random5_zero_ablation"]["solve_drop_pp"]), 2),
            "random5_retained_legal_pct": round(
                float(i4["random5_zero_ablation"]["retained_legal_rate"]) * 100.0, 2
            ),
            "causal_excess_drop_pp": round(float(i4["causal_excess_drop_pp"]), 2),
            "drop_ratio": round(
                float(i4["top5_zero_ablation"]["solve_drop_pp"])
                / float(i4["random5_zero_ablation"]["solve_drop_pp"]),
                2,
            ),
        },
    }

    return {
        "interp1_probes": {
            "layers": layers,
            "n_train": int(i1["n_train"]),
            "n_test": int(i1["n_test"]),
            "heatmap": heatmap,
            "causal_board_surgery": causal_board_surgery,
        },
        "interp2_svd_slop": {
            "profiles": profiles,
            "comparisons": i2["adapter_comparisons"],
            "slop_steering": slop_steering,
        },
        "interp3_cot_sabotage": interp3_cot_sabotage,
        "interp4_piece_tracker": interp4_piece_tracker,
    }


def compile_blog_data(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Compile the complete canonical blog_data.json payload from raw results/."""
    results_dir = repo_root / "results"
    sft_ladder = _extract_sft_ladder(repo_root, results_dir)
    rlvr = _extract_rlvr_section(results_dir)
    pass_at_k = _extract_pass_at_k(results_dir)
    elo_mirage = _extract_elo_mirage(results_dir)
    goodhart_06b = _extract_goodhart_06b(results_dir)
    e5_coach = _extract_e5_coach(results_dir)
    interp = _extract_interp(results_dir)

    e5_rows_by_id = {r["id"]: r for r in e5_coach["rows"]}
    arm_a = e5_rows_by_id["row3_arm_a_grpo"]
    arm_b = e5_rows_by_id["row5_arm_b_grpo"]
    teacher = e5_rows_by_id["row6_teacher_26b"]
    e2a = rlvr["models"]["E2a"]
    e3a = rlvr["models"]["E3a"]
    e3b = rlvr["models"]["E3b"]
    suites_by_id = {s["id"]: s for s in elo_mirage["suites"]}
    e2a_std = suites_by_id["E2a_std"]
    e2a_norep = suites_by_id["E2a_norep"]
    e3b_norep = suites_by_id["E3b_norep"]
    i4_ab = interp["interp4_piece_tracker"]["ablation"]
    slop_sweep = interp["interp2_svd_slop"]["slop_steering"]["sweep"]

    hero_kpis = {
        "coach_all_clean_pct": arm_a["all_clean_pct"],
        "coach_teacher_all_clean_pct": teacher["all_clean_pct"],
        "coach_vs_teacher_pp": round(arm_a["all_clean_pct"] - teacher["all_clean_pct"], 2),
        "coach_slop_pct": arm_a["slop_flagged_pct"],
        "coach_spoiler_free_pct": arm_a["spoiler_free_pct"],
        "arm_b_fact_coverage_pct": arm_b["fact_coverage_pct"],
        "arm_b_fact_coverage_gain_pp": round(
            arm_b["fact_coverage_pct"] - arm_a["fact_coverage_pct"], 2
        ),
        "apawn_before_pct": e3a["a_file_pawn_push_pct"],
        "apawn_after_pct": e3b["a_file_pawn_push_pct"],
        "e2a_unique_moves": e2a["unique_moves"],
        "e3a_unique_moves": e3a["unique_moves"],
        "e3b_unique_moves": e3b["unique_moves"],
        "rlvr_legal_gain_pp": round(e3b["strict_legal_pct"] - e2a["strict_legal_pct"], 2),
        "rlvr_cot_gain_pp": round(e3b["cot_v2_pct"] - e2a["cot_v2_pct"], 2),
        "elo_mirage_pp": round(e2a_std["elo"] - e2a_norep["elo"], 2),
        "e2a_rep3_draw_pct": e2a_std["rep3_pct_of_draws"],
        "e3b_norep_rep3_draws": e3b_norep["rep3_draws"],
        "e3b_norep_total_draws": e3b_norep["draws"],
        "e3b_norep_rep3_draw_pct": e3b_norep["rep3_pct_of_draws"],
        "rlvr_clean_elo_lift": round(e3b_norep["elo"] - e2a_norep["elo"], 2),
        "piece_tracker_head_count": len(interp["interp4_piece_tracker"]["top5_heads"]),
        "total_attention_heads": interp["interp4_piece_tracker"]["total_heads"],
        "piece_tracker_head_pct": round(
            100.0
            * len(interp["interp4_piece_tracker"]["top5_heads"])
            / interp["interp4_piece_tracker"]["total_heads"],
            2,
        ),
        "piece_tracker_ablation_ratio": i4_ab["drop_ratio"],
        "top5_ablation_drop_pp": i4_ab["top5_solve_drop_pp"],
        "random5_ablation_drop_pp": i4_ab["random5_solve_drop_pp"],
        "sabotage_flip_rate_pct": interp["interp3_cot_sabotage"]["sabotage"][
            "causal_move_flip_pct"
        ],
        "slop_steer_min_pct": min(s["slop_flagged_pct"] for s in slop_sweep),
        "slop_steer_max_pct": max(s["slop_flagged_pct"] for s in slop_sweep),
    }

    return {
        "schema_version": "1.0",
        "hero_kpis": hero_kpis,
        "sft_ladder": sft_ladder,
        "rlvr_apawn_hack": rlvr,
        "pass_at_k": pass_at_k,
        "elo_mirage": elo_mirage,
        "goodhart_06b": goodhart_06b,
        "e5_coach": e5_coach,
        "interp": interp,
    }


# ---------------------------------------------------------------------------
# Editorial SVG Primitives
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _svg_open(width: int, height: int, title: str, defs: str = "") -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">',
        f"  <title>{_esc(title)}</title>",
    ]
    if defs:
        lines.append("  <defs>")
        lines.append(defs)
        lines.append("  </defs>")
    lines.append(
        f'  <rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="12" '
        f'fill="{PAPER}" stroke="{HAIRLINE}" stroke-width="1.5"/>'
    )
    return lines


def _svg_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    width: int,
    x: int = 36,
) -> list[str]:
    return [
        f'  <text x="{x}" y="34" font-family="{MONO}" font-size="11" font-weight="600" '
        f'letter-spacing="1.2" fill="{INK_MUTED}">{_esc(eyebrow)}</text>',
        f'  <text x="{x}" y="62" font-family="{SERIF}" font-size="22" font-weight="600" '
        f'fill="{INK}">{_esc(title)}</text>',
        f'  <text x="{x}" y="84" font-family="{SANS}" font-size="12.5" '
        f'fill="{INK_SOFT}">{_esc(subtitle)}</text>',
        f'  <line x1="{x}" y1="98" x2="{width - x}" y2="98" stroke="{HAIRLINE}" stroke-width="1"/>',
    ]


def _svg_footer(note: str, width: int, height: int, x: int = 36) -> list[str]:
    return [
        f'  <text x="{x}" y="{height - 16}" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK_MUTED}">{_esc(note)}</text>',
    ]


def _panel(x: float, y: float, w: float, h: float, fill: str = SURFACE) -> str:
    return (
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
        f'fill="{fill}" stroke="{HAIRLINE}" stroke-width="1"/>'
    )


def _pill(
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    bg: str,
    fg: str,
    font: str = MONO,
    font_size: float = 10.5,
) -> list[str]:
    return [
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}" '
        f'fill="{bg}"/>',
        f'  <text x="{x + w / 2:.1f}" y="{y + h / 2 + font_size * 0.34:.1f}" '
        f'text-anchor="middle" font-family="{font}" font-size="{font_size}" '
        f'font-weight="600" fill="{fg}">{_esc(text)}</text>',
    ]


# ---------------------------------------------------------------------------
# Pure-Vector SVG Chess Piece Symbols (Cburnett-inspired clean silhouettes)
# ---------------------------------------------------------------------------


def _chess_piece_defs() -> str:
    """Crisp vector chess piece symbols scaled to a 45x45 coordinate system."""
    w_fill = "#FFFFFF"
    w_stroke = "#1A1918"
    b_fill = "#22211F"
    b_stroke = "#141413"
    b_accent = "#FAF8F5"

    return f"""    <g id="wp">
      <path d="M22.5 9c-2.2 0-4 1.8-4 4 0 .9.3 1.7.8 2.4-1.9 1.1-3.3 3.2-3.3 5.6 0 2 1 3.8 2.4 5-3 1.1-7.4 5.5-7.4 13.5h23c0-8-4.4-12.4-7.4-13.5 1.5-1.2 2.4-3 2.4-5 0-2.4-1.3-4.5-3.3-5.6.5-.7.8-1.5.8-2.4 0-2.2-1.8-4-4-4z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </g>
    <g id="bp">
      <path d="M22.5 9c-2.2 0-4 1.8-4 4 0 .9.3 1.7.8 2.4-1.9 1.1-3.3 3.2-3.3 5.6 0 2 1 3.8 2.4 5-3 1.1-7.4 5.5-7.4 13.5h23c0-8-4.4-12.4-7.4-13.5 1.5-1.2 2.4-3 2.4-5 0-2.4-1.3-4.5-3.3-5.6.5-.7.8-1.5.8-2.4 0-2.2-1.8-4-4-4z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M13.5 36.5h18" stroke="{b_accent}" stroke-width="1.1" stroke-linecap="round"/>
    </g>
    <g id="wr">
      <path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M34 14l-3 3H14l-3-3" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M31 17v12.5H14V17" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6"/>
      <path d="M31 29.5l1.5 2.5h-20l1.5-2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M11 14h23" fill="none" stroke="{w_stroke}" stroke-width="1.5"/>
    </g>
    <g id="br">
      <path d="M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M14 29.5v-13h17v13H14z" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5"/>
      <path d="M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23"
            fill="none" stroke="{b_accent}" stroke-width="1.0" stroke-linecap="round"/>
    </g>
    <g id="wn">
      <path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.04-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.99-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.99 2.5-3c1 0 1 3 1 3"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="15" cy="15.5" r="1.2" fill="{w_stroke}"/>
      <path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0z" fill="{w_stroke}"/>
    </g>
    <g id="bn">
      <path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.04-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.99-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.99 2.5-3c1 0 1 3 1 3"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="15" cy="15.5" r="1.2" fill="{b_accent}"/>
      <path d="M24.5 10.4c.5 5 4.8 11 10.5 25.6" fill="none" stroke="{b_accent}" stroke-width="1.1" stroke-linecap="round"/>
    </g>
    <g id="wb">
      <path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <circle cx="22.5" cy="8" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6"/>
      <path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" fill="none" stroke="{w_stroke}" stroke-width="1.5" stroke-linecap="round"/>
    </g>
    <g id="bb">
      <path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <circle cx="22.5" cy="8" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5"/>
      <path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" fill="none" stroke="{b_accent}" stroke-width="1.3" stroke-linecap="round"/>
    </g>
    <g id="wq">
      <circle cx="6" cy="12" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.5"/>
      <circle cx="14" cy="9" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.5"/>
      <circle cx="22.5" cy="8" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.5"/>
      <circle cx="31" cy="9" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.5"/>
      <circle cx="39" cy="12" r="2.5" fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.5"/>
      <path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="none" stroke="{w_stroke}" stroke-width="1.4"/>
    </g>
    <g id="bq">
      <circle cx="6" cy="12" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.4"/>
      <circle cx="14" cy="9" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.4"/>
      <circle cx="22.5" cy="8" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.4"/>
      <circle cx="31" cy="9" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.4"/>
      <circle cx="39" cy="12" r="2.5" fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.4"/>
      <path d="M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M11 30c3.5-1 19.5-1 23 0M11.5 33.5c6-1 16-1 22 0M11.5 37c6 1 16 1 22 0"
            fill="none" stroke="{b_accent}" stroke-width="1.1" stroke-linecap="round"/>
    </g>
    <g id="wk">
      <path d="M22.5 11.63V6M20 8h5" fill="none" stroke="{w_stroke}" stroke-width="1.8" stroke-linecap="round"/>
      <path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"
            fill="{w_fill}" stroke="{w_stroke}" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M11.5 30c5.5-3 15.5-3 21 0M11.5 33.5c5.5-3 15.5-3 21 0M11.5 37c5.5-3 15.5-3 21 0"
            fill="none" stroke="{w_stroke}" stroke-width="1.4"/>
    </g>
    <g id="bk">
      <path d="M22.5 11.63V6M20 8h5" fill="none" stroke="{b_stroke}" stroke-width="1.8" stroke-linecap="round"/>
      <path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z"
            fill="{b_fill}" stroke="{b_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M11.5 30c5.5-3 15.5-3 21 0M11.5 33.5c5.5-3 15.5-3 21 0M11.5 37c5.5-3 15.5-3 21 0"
            fill="none" stroke="{b_accent}" stroke-width="1.15" stroke-linecap="round"/>
    </g>
    <marker id="arrow-vermilion" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1.2 L 8.8 5 L 0 8.8 z" fill="{VERMILION}"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1.2 L 8.8 5 L 0 8.8 z" fill="{EMERALD}"/>
    </marker>"""


# ---------------------------------------------------------------------------
# Figure 1: Multi-Panel Ablation Ladder (E0 -> E2a vs E2b vs 26B)
# ---------------------------------------------------------------------------


def render_fig1_ablation_ladder(data: dict[str, Any]) -> str:
    w, h = 1080, 560
    sft = {row["id"]: row for row in data["sft_ladder"]}

    lines = _svg_open(
        w,
        h,
        "Figure 1: ChessFM SFT Ablation Ladder — LoRA r=64 vs Full Fine-Tuning",
    )
    lines += _svg_header(
        "FIGURE 1 · REPRESENTATIONS & LORA WITHOUT REGRET",
        "Piece-List Coordinates + All-Linear LoRA r=64 Sweep Full Fine-Tuning",
        "Frozen 0%-leakage benchmarks comparing Qwen3-4B SFT ladder (E0 -> E2b) against zero-shot Gemma-4-26B.",
        w,
    )

    # Legend strip right under header
    legend_items = [
        ("ASCII / Early Stage", SLATE),
        ("Piece-List LoRA r=16", "#6A8DB5"),
        ("E2a: All-Linear LoRA r=64 (Winner)", COBALT),
        ("E2b: 100% Full Fine-Tuning", AMBER),
        ("Gemma-4-26B (Zero-Shot)", "#A39B8F"),
    ]
    lx = 36.0
    for lbl, col in legend_items:
        lines.append(f'  <rect x="{lx:.1f}" y="109" width="11" height="11" rx="2.5" fill="{col}"/>')
        lines.append(
            f'  <text x="{lx + 16:.1f}" y="118.5" font-family="{SANS}" font-size="11" '
            f'font-weight="600" fill="{INK_SOFT}">{_esc(lbl)}</text>'
        )
        lx += len(lbl) * 6.4 + 34

    def _draw_vbar_panel(
        px: float,
        py: float,
        pw: float,
        ph: float,
        panel_title: str,
        unit: str,
        bars: list[tuple[str, float, str, str]],
        y_max: float,
        y_ticks: list[float],
        extra_note: str | None = None,
        ghost_bar: tuple[int, float, str] | None = None,
    ) -> list[str]:
        out = [_panel(px, py, pw, ph)]
        out.append(
            f'  <text x="{px + 16:.1f}" y="{py + 24:.1f}" font-family="{SANS}" font-size="13" '
            f'font-weight="700" fill="{INK}">{_esc(panel_title)}</text>'
        )
        out.append(
            f'  <text x="{px + pw - 16:.1f}" y="{py + 24:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="10.5" fill="{INK_MUTED}">{_esc(unit)}</text>'
        )
        cx0, cy0 = px + 46, py + 44
        cw, ch = pw - 64, ph - 82

        for yt in y_ticks:
            yy = cy0 + ch - (yt / y_max) * ch
            out.append(
                f'  <line x1="{cx0:.1f}" y1="{yy:.1f}" x2="{cx0 + cw:.1f}" y2="{yy:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>'
            )
            yt_lbl = f"{int(yt)}" if yt == int(yt) else f"{yt:.1f}"
            out.append(
                f'  <text x="{cx0 - 8:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
                f'font-family="{MONO}" font-size="10" fill="{INK_MUTED}">{yt_lbl}</text>'
            )

        out.append(
            f'  <line x1="{cx0:.1f}" y1="{cy0 + ch:.1f}" x2="{cx0 + cw:.1f}" y2="{cy0 + ch:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="1.2"/>'
        )

        n = len(bars)
        slot = cw / n
        bw = min(46.0, slot * 0.62)
        for idx, (lbl, val, col, val_str) in enumerate(bars):
            bx = cx0 + idx * slot + (slot - bw) / 2
            bh = max(2.0, (val / y_max) * ch)
            by = cy0 + ch - bh

            if ghost_bar and ghost_bar[0] == idx:
                g_val, g_lbl = ghost_bar[1], ghost_bar[2]
                gh = (g_val / y_max) * ch
                gy = cy0 + ch - gh
                out.append(
                    f'  <rect x="{bx:.1f}" y="{gy:.1f}" width="{bw:.1f}" height="{gh:.1f}" '
                    f'rx="4" fill="{SURFACE_WARM}" stroke="{SLATE}" stroke-width="1.3" '
                    f'stroke-dasharray="4 3"/>'
                )
                out.append(
                    f'  <text x="{bx + bw / 2:.1f}" y="{gy - 6:.1f}" text-anchor="middle" '
                    f'font-family="{MONO}" font-size="9.5" font-weight="600" '
                    f'fill="{INK_SOFT}">{_esc(g_lbl)}</text>'
                )

            out.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'rx="4" fill="{col}"/>'
            )
            label_y = by - 6 if not (ghost_bar and ghost_bar[0] == idx) else by - 5
            if ghost_bar and ghost_bar[0] == idx:
                # Place strict 0.2% label subtly inside/above bottom so it never hits lenient label
                label_y = cy0 + ch - 8
            out.append(
                f'  <text x="{bx + bw / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10.5" font-weight="700" '
                f'fill="{col if col in (COBALT, EMERALD, VERMILION) else INK}">{_esc(val_str)}</text>'
            )
            out.append(
                f'  <text x="{bx + bw / 2:.1f}" y="{cy0 + ch + 16:.1f}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="10.5" font-weight="600" '
                f'fill="{INK_SOFT}">{_esc(lbl)}</text>'
            )

        if extra_note:
            out.append(
                f'  <text x="{px + 16:.1f}" y="{py + ph - 9:.1f}" font-family="{SANS}" '
                f'font-size="10" fill="{INK_MUTED}">{_esc(extra_note)}</text>'
            )
        return out

    pw, ph = 492.0, 192.0
    # Panel A: Strict Legal Rate (%)
    lines += _draw_vbar_panel(
        36,
        134,
        pw,
        ph,
        "A. Strict Legal Move Rate (Unconstrained)",
        "higher ↑",
        [
            (
                "Base",
                sft["Base_ASCII"]["strict_legal_pct"],
                SLATE,
                f"{sft['Base_ASCII']['strict_legal_pct']:.1f}%",
            ),
            (
                "E0 SFT",
                sft["E0"]["strict_legal_pct"],
                SLATE,
                f"{sft['E0']['strict_legal_pct']:.1f}%",
            ),
            (
                "E0 GRPO",
                sft["E0_GRPO"]["strict_legal_pct"],
                SLATE,
                f"{sft['E0_GRPO']['strict_legal_pct']:.1f}%",
            ),
            (
                "E1a",
                sft["E1a"]["strict_legal_pct"],
                "#6A8DB5",
                f"{sft['E1a']['strict_legal_pct']:.1f}%",
            ),
            (
                "E1b",
                sft["E1b"]["strict_legal_pct"],
                "#6A8DB5",
                f"{sft['E1b']['strict_legal_pct']:.1f}%",
            ),
            (
                "E2a r=64",
                sft["E2a"]["strict_legal_pct"],
                COBALT,
                f"{sft['E2a']['strict_legal_pct']:.1f}%",
            ),
            (
                "E2b FT",
                sft["E2b"]["strict_legal_pct"],
                AMBER,
                f"{sft['E2b']['strict_legal_pct']:.1f}%",
            ),
        ],
        y_max=80.0,
        y_ticks=[0, 20, 40, 60, 80],
        extra_note="Piece-List + 10x LR tripled strict legality at 2.5K rows (21.0% -> 63.5%).",
    )

    # Panel B: 900 Frozen Lichess Puzzles
    lines += _draw_vbar_panel(
        552,
        134,
        pw,
        ph,
        "B. 900 Frozen Lichess Puzzles (Solved)",
        "higher ↑",
        [
            (
                "E1a (2.5K)",
                sft["E1a"]["puzzles_900_pct"],
                "#6A8DB5",
                f"{sft['E1a']['puzzles_900_pct']:.1f}%",
            ),
            (
                "E1b (r=16)",
                sft["E1b"]["puzzles_900_pct"],
                "#6A8DB5",
                f"{sft['E1b']['puzzles_900_pct']:.1f}%",
            ),
            (
                "E2a (r=64)",
                sft["E2a"]["puzzles_900_pct"],
                COBALT,
                f"{sft['E2a']['puzzles_900_pct']:.1f}%",
            ),
            (
                "E2b (Full-FT)",
                sft["E2b"]["puzzles_900_pct"],
                AMBER,
                f"{sft['E2b']['puzzles_900_pct']:.1f}%",
            ),
            (
                "Gemma-26B",
                sft["Gemma26B"]["puzzles_900_pct"],
                "#A39B8F",
                f"{sft['Gemma26B']['puzzles_900_pct']:.2f}%",
            ),
        ],
        y_max=28.0,
        y_ticks=[0, 7, 14, 21, 28],
        extra_note="Dashed bar: Gemma-4-26B lenient SAN rescue (10.8%); 666/900 hit 512-tok cap.",
        ghost_bar=(
            4,
            sft["Gemma26B"]["puzzles_900_lenient_pct"],
            f"Lenient {sft['Gemma26B']['puzzles_900_lenient_pct']:.1f}%",
        ),
    )

    # Panel C: 1,000 Frozen ChessQA
    lines += _draw_vbar_panel(
        36,
        342,
        pw,
        ph,
        "C. 1,000 Frozen ChessQA (5 Categories)",
        "higher ↑",
        [
            (
                "E1a (2.5K)",
                sft["E1a"]["chessqa_1000_pct"],
                "#6A8DB5",
                f"{sft['E1a']['chessqa_1000_pct']:.1f}%",
            ),
            (
                "E1b (r=16)",
                sft["E1b"]["chessqa_1000_pct"],
                "#6A8DB5",
                f"{sft['E1b']['chessqa_1000_pct']:.1f}%",
            ),
            (
                "E2a (r=64)",
                sft["E2a"]["chessqa_1000_pct"],
                COBALT,
                f"{sft['E2a']['chessqa_1000_pct']:.1f}%",
            ),
            (
                "E2b (Full-FT)",
                sft["E2b"]["chessqa_1000_pct"],
                AMBER,
                f"{sft['E2b']['chessqa_1000_pct']:.1f}%",
            ),
            (
                "Gemma-26B",
                sft["Gemma26B"]["chessqa_1000_pct"],
                "#A39B8F",
                f"{sft['Gemma26B']['chessqa_1000_pct']:.1f}%",
            ),
        ],
        y_max=55.0,
        y_ticks=[0, 15, 30, 45],
        extra_note="E2a beats Full-FT by +14.0 pp and zero-shot 26B by +18.9 pp (English 90.0%, 0% leak).",
    )

    # Panel D: Strict ACPL (Lower is better)
    lines += _draw_vbar_panel(
        552,
        342,
        pw,
        ph,
        "D. Strict ACPL — Centipawn Loss (Legal Moves)",
        "lower ↓",
        [
            ("E0 SFT", sft["E0"]["strict_acpl"], SLATE, f"{sft['E0']['strict_acpl']:.1f}"),
            (
                "E1a (2.5K)",
                sft["E1a"]["strict_acpl"],
                "#6A8DB5",
                f"{sft['E1a']['strict_acpl']:.1f}",
            ),
            (
                "E1b (r=16)",
                sft["E1b"]["strict_acpl"],
                "#6A8DB5",
                f"{sft['E1b']['strict_acpl']:.1f}",
            ),
            ("E2a (r=64)", sft["E2a"]["strict_acpl"], EMERALD, f"{sft['E2a']['strict_acpl']:.1f}"),
            ("E2b (Full-FT)", sft["E2b"]["strict_acpl"], AMBER, f"{sft['E2b']['strict_acpl']:.1f}"),
        ],
        y_max=480.0,
        y_ticks=[0, 150, 300, 450],
        extra_note=f"Val Loss: E2a {sft['E2a']['val_loss']:.4f} < E1b {sft['E1b']['val_loss']:.4f} < E2b {sft['E2b']['val_loss']:.4f} < E1a {sft['E1a']['val_loss']:.4f}.",
    )

    lines += _svg_footer(
        "Source: ChessFM Phase I–II frozen 0%-leakage eval manifests (test_fens_1000, test_puzzles_900, test_chessqa_1000).",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 2: A-Pawn Hack Pure-Vector Chessboard + Move Diversity
# ---------------------------------------------------------------------------


def _draw_chessboard(
    fen: str,
    x0: float,
    y0: float,
    sq: float,
    highlights: dict[str, str],
    arrows: list[tuple[str, str, str, str]],
) -> list[str]:
    """Render a crisp 8x8 pure-vector chessboard with piece symbols and move arrows."""
    board = chess.Board(fen)
    out: list[str] = []
    size = sq * 8

    # Subtle outer border around board
    out.append(
        f'  <rect x="{x0 - 2:.1f}" y="{y0 - 2:.1f}" width="{size + 4:.1f}" '
        f'height="{size + 4:.1f}" rx="4" fill="{HAIRLINE}"/>'
    )

    piece_map = {
        "P": "wp",
        "N": "wn",
        "B": "wb",
        "R": "wr",
        "Q": "wq",
        "K": "wk",
        "p": "bp",
        "n": "bn",
        "b": "bb",
        "r": "br",
        "q": "bq",
        "k": "bk",
    }

    for rank_idx in range(8):
        rank_num = 8 - rank_idx
        for file_idx in range(8):
            file_char = "abcdefgh"[file_idx]
            sq_name = f"{file_char}{rank_num}"
            sx = x0 + file_idx * sq
            sy = y0 + rank_idx * sq
            is_light = (rank_idx + file_idx) % 2 == 0
            bg = BOARD_LIGHT if is_light else BOARD_DARK
            out.append(
                f'  <rect x="{sx:.1f}" y="{sy:.1f}" width="{sq:.1f}" height="{sq:.1f}" fill="{bg}"/>'
            )
            if sq_name in highlights:
                out.append(
                    f'  <rect x="{sx:.1f}" y="{sy:.1f}" width="{sq:.1f}" height="{sq:.1f}" '
                    f'fill="{highlights[sq_name]}" fill-opacity="0.34"/>'
                )

            # Rank labels on file 'a', file labels on rank '1'
            lbl_col = "#5B6D4E" if is_light else "#EFE6D5"
            if file_idx == 0:
                out.append(
                    f'  <text x="{sx + 3:.1f}" y="{sy + 11:.1f}" font-family="{MONO}" '
                    f'font-size="9.5" font-weight="700" fill="{lbl_col}">{rank_num}</text>'
                )
            if rank_idx == 7:
                out.append(
                    f'  <text x="{sx + sq - 3:.1f}" y="{sy + sq - 3:.1f}" text-anchor="end" '
                    f'font-family="{MONO}" font-size="9.5" font-weight="700" '
                    f'fill="{lbl_col}">{file_char}</text>'
                )

            chess_sq = chess.parse_square(sq_name)
            piece = board.piece_at(chess_sq)
            if piece is not None:
                sym = piece_map[piece.symbol()]
                scale = (sq * 0.88) / 45.0
                ox = sx + (sq - 45.0 * scale) / 2.0
                oy = sy + (sq - 45.0 * scale) / 2.0
                out.append(
                    f'  <use href="#{sym}" transform="translate({ox:.2f},{oy:.2f}) scale({scale:.4f})"/>'
                )

    def _sq_center(sq_name: str) -> tuple[float, float]:
        f_idx = "abcdefgh".index(sq_name[0])
        r_idx = 8 - int(sq_name[1])
        return x0 + (f_idx + 0.5) * sq, y0 + (r_idx + 0.5) * sq

    for src, dst, col, marker_id in arrows:
        x1, y1 = _sq_center(src)
        x2, y2 = _sq_center(dst)
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            x1s, y1s = x1 + ux * (sq * 0.14), y1 + uy * (sq * 0.14)
            x2s, y2s = x2 - ux * (sq * 0.18), y2 - uy * (sq * 0.18)
            out.append(
                f'  <line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" '
                f'stroke="{col}" stroke-width="4.4" stroke-linecap="round" '
                f'marker-end="url(#{marker_id})" opacity="0.92"/>'
            )

    return out


def render_fig2_apawn_hack_board_and_diversity(data: dict[str, Any]) -> str:
    w, h = 1080, 560
    rlvr = data["rlvr_apawn_hack"]
    e2a = rlvr["models"]["E2a"]
    e3a = rlvr["models"]["E3a"]
    e3b = rlvr["models"]["E3b"]
    ex0 = rlvr["board_examples"][0]

    lines = _svg_open(
        w,
        h,
        "Figure 2: The A-Pawn Hack Board Example and Move Diversity Recovery",
        defs=_chess_piece_defs(),
    )
    lines += _svg_header(
        "FIGURE 2 · CATCHING & FIXING THE A-PAWN HACK",
        "Naive WDL Reward Collapsed 60.3% of Moves into Edge Pawns; sharp_wdl Restored Chess",
        "Left: verified test_fens_1000 position (White to move). Right: aggregate move diversity across 1,000 unseen positions.",
        w,
    )

    # Left panel: Chessboard + legend
    lines.append(_panel(36, 114, 430, 412))
    lines.append(
        f'  <text x="54" y="138" font-family="{SANS}" font-size="13" font-weight="700" '
        f'fill="{INK}">{_esc(ex0["title"])} (White to Move)</text>'
    )
    lines += _draw_chessboard(
        ex0["fen"],
        x0=99,
        y0=150,
        sq=38.0,
        highlights={
            "a2": VERMILION,
            "a3": VERMILION,
            "c1": EMERALD,
            "d2": EMERALD,
        },
        arrows=[
            ("a2", "a3", VERMILION, "arrow-vermilion"),
            ("c1", "d2", EMERALD, "arrow-emerald"),
        ],
    )

    # Legend below board (spacious 2 rows)
    e3a_cp = ex0["E3a"]["cp_loss"]
    e3a_wdl = ex0["E3a"]["wdl_loss_pct"]
    lines.append(f'  <rect x="54" y="468" width="394" height="22" rx="5" fill="{VERMILION_SOFT}"/>')
    lines.append(f'  <circle cx="66" cy="479" r="4.5" fill="{VERMILION}"/>')
    lines.append(
        f'  <text x="77" y="483" font-family="{MONO}" font-size="10.5" font-weight="600" '
        f'fill="{VERMILION}">E3a Hack: a2a3 (a3) · +{e3a_cp:.0f} cp blunder ({e3a_wdl:.1f}% WDL)</text>'
    )
    lines.append(f'  <rect x="54" y="495" width="394" height="22" rx="5" fill="{EMERALD_SOFT}"/>')
    lines.append(f'  <circle cx="66" cy="506" r="4.5" fill="{EMERALD}"/>')
    lines.append(
        f'  <text x="77" y="510" font-family="{MONO}" font-size="10.5" font-weight="600" '
        f'fill="{EMERALD}">E3b Fix:  c1d2 (Bd2) · 0.0 cp exact best (unpins c3)</text>'
    )

    # Right top panel: A-File Share & Unique Moves
    lines.append(_panel(486, 114, 558, 198))
    lines.append(
        f'  <text x="504" y="138" font-family="{SANS}" font-size="13" font-weight="700" '
        f'fill="{INK}">A-File Pawn Push Share vs. Unique Moves (N=1,000 FENs)</text>'
    )

    rows_div = [
        ("E2a (SFT r=64)", e2a["a_file_pawn_push_pct"], e2a["unique_moves"], COBALT, "PASS"),
        ("E3a (Naive WDL)", e3a["a_file_pawn_push_pct"], e3a["unique_moves"], VERMILION, "HACK"),
        ("E3b (sharp_wdl)", e3b["a_file_pawn_push_pct"], e3b["unique_moves"], EMERALD, "PASS"),
    ]
    # Sub-headers
    lines.append(
        f'  <text x="640" y="158" font-family="{MONO}" font-size="10" font-weight="600" '
        f'fill="{INK_MUTED}">A-FILE PAWNS (%)</text>'
    )
    lines.append(
        f'  <text x="836" y="158" font-family="{MONO}" font-size="10" font-weight="600" '
        f'fill="{INK_MUTED}">UNIQUE MOVES</text>'
    )
    for idx, (lbl, apct, uniq, col, gate) in enumerate(rows_div):
        ry = 172 + idx * 44
        lines.append(
            f'  <text x="504" y="{ry + 16:.1f}" font-family="{SANS}" font-size="12" '
            f'font-weight="600" fill="{INK}">{_esc(lbl)}</text>'
        )
        # A-file bar (max 70%)
        lines.append(
            f'  <rect x="640" y="{ry + 3:.1f}" width="136" height="18" rx="4" fill="{SURFACE_WARM}"/>'
        )
        aw = max(4.0, (apct / 70.0) * 136.0)
        lines.append(
            f'  <rect x="640" y="{ry + 3:.1f}" width="{aw:.1f}" height="18" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="782" y="{ry + 16:.1f}" font-family="{MONO}" font-size="11" '
            f'font-weight="700" fill="{col}">{apct:.1f}%</text>'
        )

        # Unique moves bar (max 600)
        lines.append(
            f'  <rect x="836" y="{ry + 3:.1f}" width="114" height="18" rx="4" fill="{SURFACE_WARM}"/>'
        )
        uw = max(4.0, (uniq / 600.0) * 114.0)
        lines.append(
            f'  <rect x="836" y="{ry + 3:.1f}" width="{uw:.1f}" height="18" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="956" y="{ry + 16:.1f}" font-family="{MONO}" font-size="11" '
            f'font-weight="700" fill="{INK}">{uniq}</text>'
        )
        g_bg = VERMILION_SOFT if gate == "HACK" else EMERALD_SOFT
        g_fg = VERMILION if gate == "HACK" else EMERALD
        lines += _pill(992, ry + 3, 40, 18, gate, g_bg, g_fg, font_size=9.5)

    # Right bottom panel: Top-5 Moves comparison + Emergent CoT strip
    lines.append(_panel(486, 326, 558, 200))
    lines.append(
        f'  <text x="504" y="350" font-family="{SANS}" font-size="13" font-weight="700" '
        f'fill="{INK}">Top-5 Predicted Moves (of 1,000): Collapse vs. Recovery</text>'
    )

    # E3a Top 5 (left col) vs E3b Top 5 (right col)
    lines.append(
        f'  <text x="504" y="370" font-family="{MONO}" font-size="10.5" font-weight="700" '
        f'fill="{VERMILION}">E3a NAIVE WDL (ALL A-FILE)</text>'
    )
    lines.append(
        f'  <text x="776" y="370" font-family="{MONO}" font-size="10.5" font-weight="700" '
        f'fill="{EMERALD}">E3b SHARP_WDL (BALANCED)</text>'
    )

    for idx in range(5):
        ry = 382 + idx * 21
        mv_a, cnt_a = e3a["top10_moves"][idx]
        mv_b, cnt_b = e3b["top10_moves"][idx]

        # E3a bar (scale / 200)
        lines.append(
            f'  <text x="504" y="{ry + 11:.1f}" font-family="{MONO}" font-size="10.5" '
            f'font-weight="600" fill="{INK}">{mv_a}</text>'
        )
        wa = max(3.0, (cnt_a / 200.0) * 165.0)
        lines.append(
            f'  <rect x="548" y="{ry + 1:.1f}" width="{wa:.1f}" height="12" rx="3" fill="{VERMILION}"/>'
        )
        lines.append(
            f'  <text x="{554 + wa:.1f}" y="{ry + 11:.1f}" font-family="{MONO}" font-size="10" '
            f'font-weight="600" fill="{VERMILION}">{cnt_a}</text>'
        )

        # E3b bar (scale / 200 — same x scale!)
        lines.append(
            f'  <text x="776" y="{ry + 11:.1f}" font-family="{MONO}" font-size="10.5" '
            f'font-weight="600" fill="{INK}">{mv_b}</text>'
        )
        wb = max(3.0, (cnt_b / 200.0) * 165.0)
        lines.append(
            f'  <rect x="820" y="{ry + 1:.1f}" width="{wb:.1f}" height="12" rx="3" fill="{EMERALD}"/>'
        )
        lines.append(
            f'  <text x="{826 + wb:.1f}" y="{ry + 11:.1f}" font-family="{MONO}" font-size="10" '
            f'font-weight="600" fill="{EMERALD}">{cnt_b}</text>'
        )

    # Bottom emergent lift strip
    legal_gain = e3b["strict_legal_pct"] - e2a["strict_legal_pct"]
    cot_gain = e3b["cot_v2_pct"] - e2a["cot_v2_pct"]
    lines.append(f'  <rect x="504" y="492" width="522" height="24" rx="5" fill="{EMERALD_SOFT}"/>')
    lines.append(
        f'  <text x="765" y="508" text-anchor="middle" font-family="{SANS}" font-size="11" '
        f'font-weight="700" fill="{EMERALD}">E3b vs E2a: +{legal_gain:.1f} pp strict legal '
        f"({e2a['strict_legal_pct']:.1f}% -> {e3b['strict_legal_pct']:.1f}%) · "
        f"+{cot_gain:.2f} pp emergent CoT V2 (zero CoT reward)</text>"
    )

    lines += _svg_footer(
        "Source: results/phase2/e3b/move_diversity_audit.json & test_fens_1000_strict.json (same horizontal frequency scale on Top-5 bars).",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 3: pass@1..pass@8 Curves + Elo Mirage Dumbbell & 3-Fold Draws
# ---------------------------------------------------------------------------


def render_fig3_pass_at_k_and_elo_mirage(data: dict[str, Any]) -> str:
    w, h = 1080, 520
    pak = data["pass_at_k"]["models"]
    suites = {s["id"]: s for s in data["elo_mirage"]["suites"]}

    lines = _svg_open(
        w,
        h,
        "Figure 3: pass@1 to pass@8 Narrowing and the +231 Elo Repetition Mirage",
    )
    lines += _svg_header(
        "FIGURE 3 · TWO REALITY CHECKS (pass@k & REPETITION AUTOPSY)",
        "RLVR Narrows pass@8 While Lifting Legality — and Anti-Repetition Exposes +231 Elo Mirage",
        "Left: Stratified N=200 puzzles (k=8, T=0.7, 95% bootstrap CI). Right: 200-game 5-anchor Bradley-Terry Elo.",
        w,
    )

    # Left panel: pass@1..pass@8
    px, py, pw, ph = 36.0, 114.0, 496.0, 376.0
    lines.append(_panel(px, py, pw, ph))
    lines.append(
        f'  <text x="{px + 18:.1f}" y="{py + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">A. Stratified pass@k Curves (N=200 Puzzles, 95% CI)</text>'
    )

    cx0, cy0, cw, ch = px + 52, py + 52, pw - 122, 240.0
    ks = ["pass@1", "pass@2", "pass@4", "pass@8"]
    x_pos = [cx0 + i * (cw / 3.0) for i in range(4)]
    y_max = 55.0

    for yt in (0, 10, 20, 30, 40, 50):
        yy = cy0 + ch - (yt / y_max) * ch
        lines.append(
            f'  <line x1="{cx0:.1f}" y1="{yy:.1f}" x2="{cx0 + cw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{cx0 - 8:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="10" fill="{INK_MUTED}">{yt}%</text>'
        )

    for xi, k_lbl in zip(x_pos, ks, strict=True):
        lines.append(
            f'  <line x1="{xi:.1f}" y1="{cy0 + ch:.1f}" x2="{xi:.1f}" y2="{cy0 + ch + 5:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="1.2"/>'
        )
        lines.append(
            f'  <text x="{xi:.1f}" y="{cy0 + ch + 18:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10.5" font-weight="600" fill="{INK_SOFT}">{k_lbl}</text>'
        )

    series = [
        ("E2a", "E2a SFT", COBALT, "", -4),
        ("E3a", "E3a Hack", VERMILION, 'stroke-dasharray="5 3"', -6),
        ("E3b", "E3b Fix", EMERALD, "", 10),
    ]
    for m_key, m_lbl, col, dash, end_dy in series:
        m = pak[m_key]["pass_at_k"]
        means = [m[k]["mean_pct"] for k in ks]
        lo = [m[k]["ci95_pct"][0] for k in ks]
        hi = [m[k]["ci95_pct"][1] for k in ks]

        # CI ribbon
        top_pts = [
            f"{x:.1f},{cy0 + ch - (v / y_max) * ch:.1f}" for x, v in zip(x_pos, hi, strict=True)
        ]
        bot_pts = [
            f"{x:.1f},{cy0 + ch - (v / y_max) * ch:.1f}"
            for x, v in reversed(list(zip(x_pos, lo, strict=True)))
        ]
        lines.append(
            f'  <polygon points="{" ".join(top_pts + bot_pts)}" fill="{col}" fill-opacity="0.11"/>'
        )
        # Main polyline
        pts = " ".join(
            f"{x:.1f},{cy0 + ch - (v / y_max) * ch:.1f}" for x, v in zip(x_pos, means, strict=True)
        )
        lines.append(
            f'  <polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6" {dash}/>'
        )
        for xi, mv in zip(x_pos, means, strict=True):
            yy = cy0 + ch - (mv / y_max) * ch
            lines.append(
                f'  <circle cx="{xi:.1f}" cy="{yy:.1f}" r="4" fill="{SURFACE}" '
                f'stroke="{col}" stroke-width="2.2"/>'
            )
        # Endpoint label on right
        last_y = cy0 + ch - (means[-1] / y_max) * ch + end_dy
        lines.append(
            f'  <text x="{x_pos[-1] + 9:.1f}" y="{last_y:.1f}" font-family="{SANS}" '
            f'font-size="11" font-weight="700" fill="{col}">{_esc(m_lbl)} {means[-1]:.1f}%</text>'
        )

    # Bottom takeaway strip inside left panel
    e2a_leg = pak["E2a"]["strict_legal_pct"]
    e3b_leg = pak["E3b"]["strict_legal_pct"]
    e2a_cot = pak["E2a"]["cot_v2"]["claim_accuracy_pct"]
    e3b_cot = pak["E3b"]["cot_v2"]["claim_accuracy_pct"]
    lines.append(
        f'  <rect x="{px + 18:.1f}" y="{py + ph - 52:.1f}" width="{pw - 36:.1f}" height="38" '
        f'rx="6" fill="{SURFACE_WARM}"/>'
    )
    lines.append(
        f'  <text x="{px + 30:.1f}" y="{py + ph - 36:.1f}" font-family="{SANS}" font-size="11" '
        f'font-weight="600" fill="{INK}">Rollout Reliability (1,600 samples): '
        f"Strict Legal {e2a_leg:.1f}% -> {e3b_leg:.1f}% (+{e3b_leg - e2a_leg:.1f} pp)</text>"
    )
    lines.append(
        f'  <text x="{px + 30:.1f}" y="{py + ph - 21:.1f}" font-family="{SANS}" font-size="11" '
        f'fill="{INK_SOFT}">CoT V2 Faithfulness {e2a_cot:.1f}% -> {e3b_cot:.1f}% '
        f"(+{e3b_cot - e2a_cot:.1f} pp) · pass@8 narrowed -9.5 pp (Yue et al.)</text>"
    )

    # Right panel: Elo Mirage dumbbell + 3-fold draws
    rx, ry, rw, rh = 552.0, 114.0, 492.0, 376.0
    lines.append(_panel(rx, ry, rw, rh))
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">B. Bradley-Terry Elo: Standard vs. --avoid-repetition</text>'
    )

    # Dumbbell plot (Elo 200..660)
    ex0, ew = rx + 82, rw - 110
    elo_min, elo_max = 220.0, 650.0

    def _ex(v: float) -> float:
        return ex0 + ((v - elo_min) / (elo_max - elo_min)) * ew

    for et in (250, 350, 450, 550, 650):
        xx = _ex(et)
        lines.append(
            f'  <line x1="{xx:.1f}" y1="{ry + 44:.1f}" x2="{xx:.1f}" y2="{ry + 168:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{xx:.1f}" y="{ry + 182:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10" fill="{INK_MUTED}">{et}</text>'
        )

    dumbbell_rows = [
        ("E2a SFT", suites["E2a_std"], suites["E2a_norep"], COBALT, ry + 78),
        ("E3b Fix", suites["E3b_std"], suites["E3b_norep"], EMERALD, ry + 136),
    ]
    for lbl, std_s, nr_s, col, y_mid in dumbbell_rows:
        lines.append(
            f'  <text x="{rx + 18:.1f}" y="{y_mid + 4:.1f}" font-family="{SANS}" font-size="12" '
            f'font-weight="700" fill="{INK}">{_esc(lbl)}</text>'
        )
        x_std = _ex(std_s["elo"])
        x_nr = _ex(nr_s["elo"])
        # CI whiskers (subtle)
        lines.append(
            f'  <line x1="{_ex(std_s["ci95"][0]):.1f}" y1="{y_mid - 11:.1f}" '
            f'x2="{_ex(std_s["ci95"][1]):.1f}" y2="{y_mid - 11:.1f}" '
            f'stroke="{AMBER}" stroke-width="1.4" opacity="0.65"/>'
        )
        lines.append(
            f'  <line x1="{_ex(nr_s["ci95"][0]):.1f}" y1="{y_mid + 11:.1f}" '
            f'x2="{_ex(nr_s["ci95"][1]):.1f}" y2="{y_mid + 11:.1f}" '
            f'stroke="{col}" stroke-width="1.6" opacity="0.75"/>'
        )
        # Connector
        lines.append(
            f'  <line x1="{x_nr:.1f}" y1="{y_mid:.1f}" x2="{x_std:.1f}" y2="{y_mid:.1f}" '
            f'stroke="{INK_MUTED}" stroke-width="2" stroke-dasharray="3 3"/>'
        )
        # Standard dot (amber hollow)
        lines.append(
            f'  <circle cx="{x_std:.1f}" cy="{y_mid:.1f}" r="6.5" fill="{SURFACE}" '
            f'stroke="{AMBER}" stroke-width="2.5"/>'
        )
        lines.append(
            f'  <text x="{x_std:.1f}" y="{y_mid - 16:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10.5" font-weight="700" '
            f'fill="{AMBER}">{std_s["elo"]:.1f} (Std)</text>'
        )
        # No-rep dot (solid)
        lines.append(f'  <circle cx="{x_nr:.1f}" cy="{y_mid:.1f}" r="6.5" fill="{col}"/>')
        lines.append(
            f'  <text x="{x_nr:.1f}" y="{y_mid + 24:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10.5" font-weight="700" '
            f'fill="{col}">{nr_s["elo"]:.1f} (NoRep)</text>'
        )

    # Mirage & lift callout pills
    mirage_delta = suites["E2a_std"]["elo"] - suites["E2a_norep"]["elo"]
    clean_lift = suites["E3b_norep"]["elo"] - suites["E2a_norep"]["elo"]
    lines += _pill(
        rx + 18,
        ry + 194,
        214,
        24,
        f"E2a Mirage: -{mirage_delta:.1f} Elo",
        AMBER_SOFT,
        AMBER,
        font_size=10.5,
    )
    lines += _pill(
        rx + 242,
        ry + 194,
        232,
        24,
        f"Clean RLVR Lift: +{clean_lift:.1f} Elo",
        EMERALD_SOFT,
        EMERALD,
        font_size=10.5,
    )

    # Divider + 3-Fold Repetition Draw Share bars
    lines.append(
        f'  <line x1="{rx + 18:.1f}" y1="{ry + 230:.1f}" x2="{rx + rw - 18:.1f}" '
        f'y2="{ry + 230:.1f}" stroke="{HAIRLINE}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 250:.1f}" font-family="{SANS}" font-size="12" '
        f'font-weight="700" fill="{INK}">3-Fold Repetition Share of All Draws (Verified PGNs)</text>'
    )

    draw_rows = [
        ("E2a Standard", suites["E2a_std"], AMBER),
        ("E2a --avoid-rep", suites["E2a_norep"], COBALT),
        ("E3b Standard", suites["E3b_std"], AMBER),
        ("E3b --avoid-rep", suites["E3b_norep"], EMERALD),
    ]
    for idx, (lbl, st, col) in enumerate(draw_rows):
        dy = ry + 264 + idx * 26
        pct = st["rep3_pct_of_draws"]
        lines.append(
            f'  <text x="{rx + 18:.1f}" y="{dy + 12:.1f}" font-family="{SANS}" font-size="11" '
            f'font-weight="600" fill="{INK_SOFT}">{_esc(lbl)}</text>'
        )
        bx = rx + 132
        bw_max = 228.0
        lines.append(
            f'  <rect x="{bx:.1f}" y="{dy + 1:.1f}" width="{bw_max:.1f}" height="15" rx="4" fill="{SURFACE_WARM}"/>'
        )
        lines.append(
            f'  <rect x="{bx:.1f}" y="{dy + 1:.1f}" width="{(pct / 100.0) * bw_max:.1f}" height="15" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="{bx + bw_max + 8:.1f}" y="{dy + 12.5:.1f}" font-family="{MONO}" '
            f'font-size="10.5" font-weight="700" fill="{col}">'
            f"{st['rep3_draws']}/{st['draws']} ({pct:.1f}%)</text>"
        )

    lines += _svg_footer(
        "Source: results/phase2/e3b/pass_at_k_e2a_vs_e3a_vs_e3b.json & repetition_autopsy.json (E3b wins vs Maia-1100: 2 std, 1 norep; E2a: 0).",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 4: 0.6B Collapse + 7-Row Coach Showdown + Step 0-40 Goodhart Curves
# ---------------------------------------------------------------------------


def render_fig4_coach_showdown_and_goodhart(data: dict[str, Any]) -> str:
    w, h = 1080, 560
    g06 = data["goodhart_06b"]["grounded_292"]
    e5 = data["e5_coach"]
    curves = e5["goodhart_curves"]

    lines = _svg_open(
        w,
        h,
        "Figure 4: Goodhart's Law Collapse and 4B AI Chess Coach Showdown",
    )
    lines += _svg_header(
        "FIGURE 4 · GOODHART'S LAW TWICE & 4B AI CHESS COACH (E5)",
        "Strict Verifier V2 Exposes 0.6B Slop Collapse and Lifts a 4B Coach Above Its 26B Teacher",
        "Left: Qwen3-0.6B V1->V2 collapse (N=292). Center: 7-row test_coach_300 showdown. Right: Live Step 0-40 Goodhart gap.",
        w,
    )

    # Panel A: 0.6B Collapse (x=36..292, w=256)
    ax, ay, aw, ah = 36.0, 114.0, 256.0, 412.0
    lines.append(_panel(ax, ay, aw, ah))
    lines.append(
        f'  <text x="{ax + 16:.1f}" y="{ay + 24:.1f}" font-family="{SANS}" font-size="12.5" '
        f'font-weight="700" fill="{INK}">A. 0.6B Goodhart Collapse</text>'
    )
    lines.append(
        f'  <text x="{ax + 16:.1f}" y="{ay + 40:.1f}" font-family="{MONO}" font-size="10" '
        f'fill="{INK_MUTED}">N=292 Grounded Outputs</text>'
    )

    cy0, ch = ay + 66, 224.0
    for yt in (0, 25, 50, 75, 100):
        yy = cy0 + ch - (yt / 100.0) * ch
        lines.append(
            f'  <line x1="{ax + 36:.1f}" y1="{yy:.1f}" x2="{ax + aw - 16:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{ax + 31:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{yt}%</text>'
        )

    groups = [
        (
            "26B Teacher",
            ax + 54,
            [
                (g06["teacher_26b"]["v1_pass_pct"], AMBER, "V1"),
                (g06["teacher_26b"]["v2_and_non_slop_pct"], EMERALD, "V2+NS"),
            ],
        ),
        (
            "0.6B SFT",
            ax + 154,
            [
                (g06["sft_06b_grounded"]["v1_pass_pct"], AMBER, "V1"),
                (g06["sft_06b_grounded"]["v2_and_non_slop_pct"], VERMILION, "V2+NS"),
            ],
        ),
    ]
    for glbl, gx, gbars in groups:
        for bi, (val, col, _tag) in enumerate(gbars):
            bx = gx + bi * 36
            bh = (val / 100.0) * ch
            by = cy0 + ch - bh
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="30" height="{bh:.1f}" rx="4" fill="{col}"/>'
            )
            lines.append(
                f'  <text x="{bx + 15:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10" font-weight="700" fill="{col}">{val:.1f}%</text>'
            )
        lines.append(
            f'  <text x="{gx + 33:.1f}" y="{cy0 + ch + 18:.1f}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="11" font-weight="700" fill="{INK_SOFT}">{_esc(glbl)}</text>'
        )

    # Mini legend + callout box at bottom of Panel A
    lines.append(
        f'  <rect x="{ax + 16:.1f}" y="{ay + 322:.1f}" width="10" height="10" rx="2" fill="{AMBER}"/>'
    )
    lines.append(
        f'  <text x="{ax + 30:.1f}" y="{ay + 331:.1f}" font-family="{SANS}" font-size="10.5" fill="{INK_SOFT}">Weak V1</text>'
    )
    lines.append(
        f'  <rect x="{ax + 96:.1f}" y="{ay + 322:.1f}" width="10" height="10" rx="2" fill="{EMERALD}"/>'
    )
    lines.append(
        f'  <text x="{ax + 110:.1f}" y="{ay + 331:.1f}" font-family="{SANS}" font-size="10.5" fill="{INK_SOFT}">V2+Non-Slop</text>'
    )
    lines.append(
        f'  <rect x="{ax + 16:.1f}" y="{ay + 344:.1f}" width="{aw - 32:.1f}" height="52" rx="6" fill="{VERMILION_SOFT}"/>'
    )
    lines.append(
        f'  <text x="{ax + 26:.1f}" y="{ay + 362:.1f}" font-family="{SANS}" font-size="10.5" '
        f'font-weight="700" fill="{VERMILION}">79.8% Slop-Flagged (0.6B SFT)</text>'
    )
    lines.append(
        f'  <text x="{ax + 26:.1f}" y="{ay + 378:.1f}" font-family="{SANS}" font-size="10" '
        f'fill="{INK_SOFT}">Repeated "tension in the center"</text>'
    )
    lines.append(
        f'  <text x="{ax + 26:.1f}" y="{ay + 391:.1f}" font-family="{SANS}" font-size="10" '
        f'fill="{INK_SOFT}">on 41.1% of boards (120/292).</text>'
    )

    # Panel B: 7-Row test_coach_300 Showdown (x=308..768, w=460)
    bx, by, bw, bh_panel = 308.0, 114.0, 460.0, 412.0
    lines.append(_panel(bx, by, bw, bh_panel))
    lines.append(
        f'  <text x="{bx + 16:.1f}" y="{by + 24:.1f}" font-family="{SANS}" font-size="12.5" '
        f'font-weight="700" fill="{INK}">B. 7-Row test_coach_300 Showdown (N=300)</text>'
    )
    lines.append(
        f'  <text x="{bx + 16:.1f}" y="{by + 40:.1f}" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK_MUTED}">Bars: All-Clean % (95% CI) · Diamonds: Tactical Fact Coverage %</text>'
    )

    # Order rows cleanly by All-Clean descending (highlighting Arm A winner)
    rows_by_id = {r["id"]: r for r in e5["rows"]}
    ordered_ids = [
        ("row3_arm_a_grpo", "Arm A V2-GRPO", EMERALD),
        ("row5_arm_b_grpo", "Arm B V2-GRPO", COBALT),
        ("row7_arm_c_v1_grpo", "Arm C V1-GRPO", VERMILION),
        ("row6_teacher_26b", "Gemma-4-26B", AMBER),
        ("row2_arm_a_sft", "Arm A SFT", "#6A8DB5"),
        ("row4_arm_b_sft", "Arm B SFT", "#6A8DB5"),
        ("row1_base_4b", "Base Qwen3-4B", SLATE),
    ]

    bar_x0 = bx + 124
    bar_w = 248.0
    # X-axis scale 40%..100%
    xmin, xmax = 40.0, 100.0

    def _bx(v: float) -> float:
        return bar_x0 + ((max(xmin, min(xmax, v)) - xmin) / (xmax - xmin)) * bar_w

    for xt in (40, 60, 80, 100):
        xx = _bx(xt)
        lines.append(
            f'  <line x1="{xx:.1f}" y1="{by + 54:.1f}" x2="{xx:.1f}" y2="{by + 344:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{xx:.1f}" y="{by + 360:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{xt}%</text>'
        )

    for idx, (rid, lbl, col) in enumerate(ordered_ids):
        r = rows_by_id[rid]
        ry_row = by + 62 + idx * 40
        clean = r["all_clean_pct"]
        ci_lo, ci_hi = r["all_clean_ci95_pct"]
        cov = r["fact_coverage_pct"]

        lines.append(
            f'  <text x="{bx + 14:.1f}" y="{ry_row + 15:.1f}" font-family="{SANS}" font-size="11" '
            f'font-weight="{"700" if rid in ("row3_arm_a_grpo", "row5_arm_b_grpo") else "600"}" '
            f'fill="{INK}">{_esc(lbl)}</text>'
        )
        # Track
        lines.append(
            f'  <rect x="{bar_x0:.1f}" y="{ry_row + 2:.1f}" width="{bar_w:.1f}" height="20" rx="4" fill="{SURFACE_WARM}"/>'
        )
        # Filled bar
        fw = _bx(clean) - bar_x0
        lines.append(
            f'  <rect x="{bar_x0:.1f}" y="{ry_row + 2:.1f}" width="{fw:.1f}" height="20" rx="4" fill="{col}" fill-opacity="0.88"/>'
        )
        # 95% CI whisker
        lines.append(
            f'  <line x1="{_bx(ci_lo):.1f}" y1="{ry_row + 12:.1f}" x2="{_bx(ci_hi):.1f}" '
            f'y2="{ry_row + 12:.1f}" stroke="{INK}" stroke-width="1.6"/>'
        )
        # Fact coverage diamond
        cx_cov = _bx(cov)
        cy_cov = ry_row + 12
        lines.append(
            f'  <polygon points="{cx_cov:.1f},{cy_cov - 5:.1f} {cx_cov + 5:.1f},{cy_cov:.1f} '
            f'{cx_cov:.1f},{cy_cov + 5:.1f} {cx_cov - 5:.1f},{cy_cov:.1f}" '
            f'fill="{SURFACE}" stroke="{INK}" stroke-width="1.6"/>'
        )
        # Right value readout
        lines.append(
            f'  <text x="{bar_x0 + bar_w + 8:.1f}" y="{ry_row + 16:.1f}" font-family="{MONO}" '
            f'font-size="10.5" font-weight="700" fill="{col}">{clean:.1f}%</text>'
        )

    lines.append(
        f'  <rect x="{bx + 14:.1f}" y="{by + 370:.1f}" width="{bw - 28:.1f}" height="28" rx="6" fill="{EMERALD_SOFT}"/>'
    )
    lines.append(
        f'  <text x="{bx + bw / 2:.1f}" y="{by + 388:.1f}" text-anchor="middle" font-family="{SANS}" '
        f'font-size="10.5" font-weight="700" fill="{EMERALD}">'
        f"Arm A: 92.0% (+3.3 pp vs 26B, 0% slop) · Arm B: 81.7% Fact Cov (+4.5 pp)</text>"
    )

    # Panel C: Step 0-40 Goodhart Curves (x=784..1044, w=260)
    gx, gy, gw, gh = 784.0, 114.0, 260.0, 412.0
    lines.append(_panel(gx, gy, gw, gh))
    lines.append(
        f'  <text x="{gx + 16:.1f}" y="{gy + 24:.1f}" font-family="{SANS}" font-size="12.5" '
        f'font-weight="700" fill="{INK}">C. Live Goodhart (0->40)</text>'
    )
    lines.append(
        f'  <text x="{gx + 16:.1f}" y="{gy + 40:.1f}" font-family="{MONO}" font-size="10" '
        f'fill="{INK_MUTED}">Held-out N=100 Slice</text>'
    )

    px0, py0, pcw, pch = gx + 38, gy + 66, gw - 54, 214.0
    steps = curves["steps"]
    pmin, pmax = 78.0, 100.0

    def _px(step: int) -> float:
        return px0 + (step / 40.0) * pcw

    def _py(val: float) -> float:
        return py0 + pch - ((val - pmin) / (pmax - pmin)) * pch

    for yt in (80, 85, 90, 95, 100):
        yy = _py(yt)
        lines.append(
            f'  <line x1="{px0:.1f}" y1="{yy:.1f}" x2="{px0 + pcw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{px0 - 6:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{yt}%</text>'
        )

    for st in steps:
        xx = _px(st)
        lines.append(
            f'  <text x="{xx:.1f}" y="{py0 + pch + 16:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{st}</text>'
        )

    arm_a_c = curves["arm_a"]
    arm_c_c = curves["arm_c"]

    # Shade Arm C V1-V2 gap
    c_top = [f"{_px(p['step']):.1f},{_py(p['v1_pass_pct']):.1f}" for p in arm_c_c]
    c_bot = [f"{_px(p['step']):.1f},{_py(p['v2_pass_pct']):.1f}" for p in reversed(arm_c_c)]
    lines.append(
        f'  <polygon points="{" ".join(c_top + c_bot)}" fill="{VERMILION}" fill-opacity="0.12"/>'
    )

    curve_specs = [
        (arm_c_c, "v1_pass_pct", VERMILION, 'stroke-dasharray="4 3"', 2.2),
        (arm_c_c, "v2_pass_pct", VERMILION, "", 2.4),
        (arm_a_c, "v2_pass_pct", EMERALD, "", 2.6),
    ]
    for pts_data, field, col, dash, sw in curve_specs:
        poly = " ".join(f"{_px(p['step']):.1f},{_py(p[field]):.1f}" for p in pts_data)
        lines.append(
            f'  <polyline points="{poly}" fill="none" stroke="{col}" stroke-width="{sw}" {dash}/>'
        )
        for p in pts_data:
            lines.append(
                f'  <circle cx="{_px(p["step"]):.1f}" cy="{_py(p[field]):.1f}" r="3.2" '
                f'fill="{SURFACE}" stroke="{col}" stroke-width="1.8"/>'
            )

    # Legend inside Panel C
    leg_y = gy + 312
    for dy_i, (col, dash, lbl) in enumerate(
        [
            (VERMILION, 'stroke-dasharray="4 3"', "Arm C Weak V1 (97%)"),
            (VERMILION, "", "Arm C True V2 (10pp gap)"),
            (EMERALD, "", "Arm A True V2 (Winner)"),
        ]
    ):
        ly = leg_y + dy_i * 17
        lines.append(
            f'  <line x1="{gx + 16:.1f}" y1="{ly:.1f}" x2="{gx + 34:.1f}" y2="{ly:.1f}" '
            f'stroke="{col}" stroke-width="2.4" {dash}/>'
        )
        lines.append(
            f'  <text x="{gx + 40:.1f}" y="{ly + 3.5:.1f}" font-family="{SANS}" font-size="10.5" '
            f'fill="{INK_SOFT}">{_esc(lbl)}</text>'
        )

    lines.append(
        f'  <rect x="{gx + 16:.1f}" y="{gy + 368:.1f}" width="{gw - 32:.1f}" height="30" rx="6" fill="{SURFACE_WARM}"/>'
    )
    lines.append(
        f'  <text x="{gx + gw / 2:.1f}" y="{gy + 387:.1f}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="10" font-weight="600" fill="{INK}">Full N=300 Gap: 6.7 pp vs 3.7 pp</text>'
    )

    lines += _svg_footer(
        "Source: results/verifier_v2_rescore.json, results/phase2/e5/e5_coach_comparison_summary.json & goodhart_arm_a_vs_arm_c_curves.json.",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 5: Board Probes U-Shaped Rebound + Piece-Tracker Heads & Ablation
# ---------------------------------------------------------------------------


def render_fig5_board_probes_and_piece_trackers(data: dict[str, Any]) -> str:
    w, h = 1080, 520
    i1 = data["interp"]["interp1_probes"]
    i4 = data["interp"]["interp4_piece_tracker"]
    layers = i1["layers"]
    hm = i1["heatmap"]
    surg = i1["causal_board_surgery"]

    lines = _svg_open(
        w,
        h,
        "Figure 5: Linear Board-State Probes and Piece-Tracker Head Ablation",
    )
    lines += _svg_header(
        "FIGURE 5 · BOARD GEOMETRY & CAUSAL ATTENTION CIRCUITS (INTERP #1 & #4)",
        "Piece-List Prompts Unlock Late-Layer Board Rebound Driven by 5 Piece-Tracker Heads",
        "Left: 64-square occupied-square linear probes (800/200 split). Right: Top-5 Piece-Tracker heads & 3.0x causal ablation.",
        w,
    )

    # Left Panel: 5-checkpoint probe trajectories across L0..L35
    px, py, pw, ph = 36.0, 114.0, 520.0, 376.0
    lines.append(_panel(px, py, pw, ph))
    lines.append(
        f'  <text x="{px + 18:.1f}" y="{py + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">A. Occupied-Square Probe Accuracy (L0 -> L35)</text>'
    )

    cx0, cy0, cw, ch = px + 50, py + 50, pw - 152, 232.0
    ymin, ymax = 58.0, 75.0

    def _lx(layer: int) -> float:
        return cx0 + (layer / 35.0) * cw

    def _ly(val: float) -> float:
        return cy0 + ch - ((val - ymin) / (ymax - ymin)) * ch

    # Highlight L26..L35 rebound zone
    lines.append(
        f'  <rect x="{_lx(26):.1f}" y="{cy0:.1f}" width="{_lx(35) - _lx(26):.1f}" '
        f'height="{ch:.1f}" fill="{COBALT_SOFT}" fill-opacity="0.65"/>'
    )
    lines.append(
        f'  <text x="{(_lx(26) + _lx(35)) / 2:.1f}" y="{cy0 + 14:.1f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="9.5" font-weight="700" fill="{COBALT}">REBOUND</text>'
    )

    for yt in (60, 64, 68, 72):
        yy = _ly(yt)
        lines.append(
            f'  <line x1="{cx0:.1f}" y1="{yy:.1f}" x2="{cx0 + cw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{cx0 - 8:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="10" fill="{INK_MUTED}">{yt}%</text>'
        )

    for lr in layers:
        xx = _lx(lr)
        lines.append(
            f'  <line x1="{xx:.1f}" y1="{cy0 + ch:.1f}" x2="{xx:.1f}" y2="{cy0 + ch + 4:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="1.2"/>'
        )
        lines.append(
            f'  <text x="{xx:.1f}" y="{cy0 + ch + 17:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10" fill="{INK_SOFT}">L{lr}</text>'
        )

    probe_series = [
        ("E1a_PieceList", "E1a (PL)", COBALT, "", 2.4, -2),
        ("E3b_SharpWDL", "E3b (RLVR)", EMERALD, "", 2.6, 2),
        ("E2a_r64", "E2a (r=64)", "#3B8F84", "", 2.0, 14),
        ("E0_ASCII", "E0 (ASCII)", AMBER, 'stroke-dasharray="5 3"', 2.1, -3),
        ("Base_ASCII", "Base (ASCII)", SLATE, 'stroke-dasharray="3 3"', 2.0, 11),
    ]
    for ckpt, lbl, col, dash, sw, end_dy in probe_series:
        vals = [hm[ckpt]["layers"][str(lr)]["occupied_pct"] for lr in layers]
        poly = " ".join(f"{_lx(lr):.1f},{_ly(v):.1f}" for lr, v in zip(layers, vals, strict=True))
        lines.append(
            f'  <polyline points="{poly}" fill="none" stroke="{col}" stroke-width="{sw}" {dash}/>'
        )
        for lr, v in zip(layers, vals, strict=True):
            lines.append(
                f'  <circle cx="{_lx(lr):.1f}" cy="{_ly(v):.1f}" r="3.2" fill="{SURFACE}" '
                f'stroke="{col}" stroke-width="1.8"/>'
            )
        last_y = _ly(vals[-1]) + end_dy
        lines.append(
            f'  <text x="{_lx(35) + 8:.1f}" y="{last_y:.1f}" font-family="{SANS}" font-size="10.5" '
            f'font-weight="700" fill="{col}">{_esc(lbl)} {vals[-1]:.1f}%</text>'
        )

    # Bottom Layer-6 null control bar
    lines.append(
        f'  <rect x="{px + 18:.1f}" y="{py + ph - 52:.1f}" width="{pw - 36:.1f}" height="38" '
        f'rx="6" fill="{SURFACE_WARM}"/>'
    )
    lines.append(
        f'  <text x="{px + 30:.1f}" y="{py + ph - 36:.1f}" font-family="{SANS}" font-size="11" '
        f'font-weight="700" fill="{INK}">Honest Null Control — Layer-6 Causal Board Surgery (N=50):</text>'
    )
    lines.append(
        f'  <text x="{px + 30:.1f}" y="{py + ph - 21:.1f}" font-family="{MONO}" font-size="10.5" '
        f'fill="{INK_SOFT}">Probe Flip {surg["probe_move_flip_pct"]:.0f}% vs Random {surg["random_control_move_flip_pct"]:.0f}% '
        f"({surg['causal_excess_flip_pp']:.1f} pp) -> model reads KV cache late.</text>"
    )

    # Right Panel: Top-5 Piece-Tracker Heads + Causal Ablation
    rx, ry, rw, rh = 576.0, 114.0, 468.0, 376.0
    lines.append(_panel(rx, ry, rw, rh))
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">B. Piece-Tracker Heads (L26-L29)</text>'
    )
    lines += _pill(
        rx + rw - 148, ry + 11, 130, 22, "5 / 1,152 (0.43%)", COBALT_SOFT, COBALT, font_size=10.0
    )

    # Top half: Top-5 selectivity lollipop/bars
    top5 = i4["top5_heads"]
    max_sel = 0.021
    for idx, hd in enumerate(top5):
        hy = ry + 50 + idx * 27
        sel = hd["mean_selectivity"]
        lines.append(
            f'  <text x="{rx + 18:.1f}" y="{hy + 13:.1f}" font-family="{MONO}" font-size="11.5" '
            f'font-weight="700" fill="{INK}">{hd["label"]}</text>'
        )
        bx0 = rx + 90
        bw_max = 278.0
        lines.append(
            f'  <rect x="{bx0:.1f}" y="{hy + 2:.1f}" width="{bw_max:.1f}" height="15" rx="4" fill="{SURFACE_WARM}"/>'
        )
        fw = (sel / max_sel) * bw_max
        col = COBALT if idx == 0 else "#487BB3"
        lines.append(
            f'  <rect x="{bx0:.1f}" y="{hy + 2:.1f}" width="{fw:.1f}" height="15" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="{bx0 + bw_max + 8:.1f}" y="{hy + 13.5:.1f}" font-family="{MONO}" '
            f'font-size="10.5" font-weight="700" fill="{col}">+{sel:.4f}</text>'
        )

    # Divider
    lines.append(
        f'  <line x1="{rx + 18:.1f}" y1="{ry + 198:.1f}" x2="{rx + rw - 18:.1f}" '
        f'y2="{ry + 198:.1f}" stroke="{HAIRLINE}" stroke-width="1"/>'
    )
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 220:.1f}" font-family="{SANS}" font-size="13" '
        f'font-weight="700" fill="{INK}">C. Causal Zero-Ablation (N=100 Solved Puzzles)</text>'
    )
    ab = i4["ablation"]
    lines += _pill(
        rx + rw - 118,
        ry + 205,
        100,
        22,
        f"{ab['drop_ratio']:.1f}x Drop",
        VERMILION_SOFT,
        VERMILION,
        font_size=10.5,
    )

    ab_rows = [
        ("Baseline (Intact)", ab["baseline_solve_pct"], "0.0 pp", EMERALD),
        (
            "Ablate 5 Random L26-29",
            ab["random5_retained_solve_pct"],
            f"-{ab['random5_solve_drop_pp']:.1f} pp",
            AMBER,
        ),
        (
            "Ablate Top-5 Trackers",
            ab["top5_retained_solve_pct"],
            f"-{ab['top5_solve_drop_pp']:.1f} pp",
            VERMILION,
        ),
    ]
    for idx, (lbl, val, delta_lbl, col) in enumerate(ab_rows):
        ay_r = ry + 240 + idx * 36
        lines.append(
            f'  <text x="{rx + 18:.1f}" y="{ay_r + 14:.1f}" font-family="{SANS}" font-size="11.5" '
            f'font-weight="600" fill="{INK}">{_esc(lbl)}</text>'
        )
        abx0 = rx + 178
        abw = 188.0
        # Zoomed scale 70%..100% or honest 0%..100%? Honest 0%..100% is cleaner and zero distortion!
        lines.append(
            f'  <rect x="{abx0:.1f}" y="{ay_r + 1:.1f}" width="{abw:.1f}" height="18" rx="4" fill="{SURFACE_WARM}"/>'
        )
        lines.append(
            f'  <rect x="{abx0:.1f}" y="{ay_r + 1:.1f}" width="{(val / 100.0) * abw:.1f}" height="18" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="{abx0 + abw + 8:.1f}" y="{ay_r + 14:.1f}" font-family="{MONO}" '
            f'font-size="10.5" font-weight="700" fill="{col}">{val:.0f}% ({delta_lbl})</text>'
        )

    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + rh - 14:.1f}" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK_MUTED}">Legal rate stays 98-99% under ablation: these heads compute tactics, not syntax.</text>'
    )

    lines += _svg_footer(
        "Source: results/interp/interp1_board_probes_and_steering.json & interp4_attention_circuits_and_ablation.json.",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 6: 36-Layer LoRA QR-SVD Spectrum + Causal Slop Vector Steering
# ---------------------------------------------------------------------------


def render_fig6_lora_svd_and_slop_steering(data: dict[str, Any]) -> str:
    w, h = 1080, 520
    i2 = data["interp"]["interp2_svd_slop"]
    e3a_p = i2["profiles"]["e3a_apawn_hack"]
    e3b_p = i2["profiles"]["e3b_sharp_wdl"]
    slop = i2["slop_steering"]

    lines = _svg_open(
        w,
        h,
        "Figure 6: 36-Layer LoRA QR-SVD Weight Diffing and Goodhart Slop Vector Steering",
    )
    lines += _svg_header(
        "FIGURE 6 · WEIGHT DIFFING & CAUSAL STEERING (INTERP #2)",
        "Reward Hacking Spikes Output Layers L34-L35; Goodhart Slop Lives on One Linear Direction",
        "Left: Exact thin QR-SVD Frobenius norms across all 36 layers. Right: Layer-2 Slop Vector steering sweep (N=50).",
        w,
    )

    # Left panel: 36-layer LoRA QR-SVD
    px, py, pw, ph = 36.0, 114.0, 556.0, 376.0
    lines.append(_panel(px, py, pw, ph))
    lines.append(
        f'  <text x="{px + 18:.1f}" y="{py + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">A. 36-Layer LoRA Update Norm: E3a Hack vs. E3b Fix</text>'
    )
    lines += _pill(
        px + pw - 166,
        py + 11,
        148,
        22,
        f"MLP {e3b_p['mlp_energy_share_pct']:.1f}% · Attn {e3b_p['attn_energy_share_pct']:.1f}%",
        SURFACE_WARM,
        INK_SOFT,
        font_size=9.8,
    )

    cx0, cy0, cw, ch = px + 52, py + 58, pw - 74, 224.0
    ymin, ymax = 0.17, 0.26

    def _lx(layer: int) -> float:
        return cx0 + (layer / 35.0) * cw

    def _ly(val: float) -> float:
        return cy0 + ch - ((val - ymin) / (ymax - ymin)) * ch

    # Shaded L34-35 hack zone
    lines.append(
        f'  <rect x="{_lx(33.5):.1f}" y="{cy0:.1f}" width="{_lx(35) - _lx(33.5) + 4:.1f}" '
        f'height="{ch:.1f}" fill="{VERMILION_SOFT}" fill-opacity="0.8"/>'
    )
    # Shaded L26-29 tracker zone
    lines.append(
        f'  <rect x="{_lx(26):.1f}" y="{cy0:.1f}" width="{_lx(29) - _lx(26):.1f}" '
        f'height="{ch:.1f}" fill="{COBALT_SOFT}" fill-opacity="0.6"/>'
    )

    for yt in (0.18, 0.20, 0.22, 0.24, 0.26):
        yy = _ly(yt)
        lines.append(
            f'  <line x1="{cx0:.1f}" y1="{yy:.1f}" x2="{cx0 + cw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{cx0 - 7:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{yt:.2f}</text>'
        )

    for lr in (0, 5, 10, 15, 20, 25, 30, 35):
        xx = _lx(lr)
        lines.append(
            f'  <text x="{xx:.1f}" y="{cy0 + ch + 16:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_SOFT}">L{lr}</text>'
        )

    pts_a = " ".join(
        f"{_lx(r['layer']):.1f},{_ly(r['total_frobenius']):.1f}" for r in e3a_p["per_layer"]
    )
    pts_b = " ".join(
        f"{_lx(r['layer']):.1f},{_ly(r['total_frobenius']):.1f}" for r in e3b_p["per_layer"]
    )
    lines.append(
        f'  <polyline points="{pts_a}" fill="none" stroke="{VERMILION}" stroke-width="2.4"/>'
    )
    lines.append(
        f'  <polyline points="{pts_b}" fill="none" stroke="{EMERALD}" stroke-width="2.6"/>'
    )

    # Annotated callout dots for L17 (E3b peak), L29 (E3b > E3a), L35 (E3a peak)
    l17_v = e3b_p["per_layer"][17]["total_frobenius"]
    l29_v = e3b_p["per_layer"][29]["total_frobenius"]
    l35_v = e3a_p["per_layer"][35]["total_frobenius"]

    lines.append(f'  <circle cx="{_lx(17):.1f}" cy="{_ly(l17_v):.1f}" r="4.5" fill="{EMERALD}"/>')
    lines.append(
        f'  <text x="{_lx(17):.1f}" y="{_ly(l17_v) - 10:.1f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="10" font-weight="700" fill="{EMERALD}">L17 Peak (E3b)</text>'
    )
    lines.append(f'  <circle cx="{_lx(29):.1f}" cy="{_ly(l29_v):.1f}" r="4.5" fill="{COBALT}"/>')
    lines.append(
        f'  <text x="{_lx(27.5):.1f}" y="{cy0 + 13:.1f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="9.5" font-weight="700" fill="{COBALT}">L29 (+0.008)</text>'
    )
    lines.append(f'  <circle cx="{_lx(35):.1f}" cy="{_ly(l35_v):.1f}" r="4.5" fill="{VERMILION}"/>')
    lines.append(
        f'  <text x="{_lx(35) - 4:.1f}" y="{_ly(l35_v) - 9:.1f}" text-anchor="end" '
        f'font-family="{MONO}" font-size="10" font-weight="700" fill="{VERMILION}">L35 Hack Spike</text>'
    )

    # Bottom spectral telemetry comparison strip
    lines.append(
        f'  <rect x="{px + 18:.1f}" y="{py + ph - 58:.1f}" width="{pw - 36:.1f}" height="44" '
        f'rx="6" fill="{SURFACE_WARM}"/>'
    )
    lines.append(f'  <circle cx="{px + 32:.1f}" cy="{py + ph - 43:.1f}" r="4" fill="{VERMILION}"/>')
    lines.append(
        f'  <text x="{px + 42:.1f}" y="{py + ph - 39:.1f}" font-family="{MONO}" font-size="10.5" '
        f'font-weight="600" fill="{INK}">E3a Hack: ||dW||={e3a_p["total_frobenius"]:.3f} · '
        f"peak L{e3a_p['peak_layer']} · eff_rank {e3a_p['mean_effective_rank']:.2f} "
        f"(top-1 {e3a_p['mean_top1_energy_pct']:.1f}%)</text>"
    )
    lines.append(f'  <circle cx="{px + 32:.1f}" cy="{py + ph - 25:.1f}" r="4" fill="{EMERALD}"/>')
    lines.append(
        f'  <text x="{px + 42:.1f}" y="{py + ph - 21:.1f}" font-family="{MONO}" font-size="10.5" '
        f'font-weight="600" fill="{INK}">E3b Fix:  ||dW||={e3b_p["total_frobenius"]:.3f} · '
        f"peak L{e3b_p['peak_layer']} · eff_rank {e3b_p['mean_effective_rank']:.2f} "
        f"(top-1 {e3b_p['mean_top1_energy_pct']:.1f}%)</text>"
    )

    # Right panel: Slop Vector Steering Sweep (alpha = -2..+2)
    rx, ry, rw, rh = 612.0, 114.0, 432.0, 376.0
    lines.append(_panel(rx, ry, rw, rh))
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">B. Causal "Slop Vector" Steering (Layer 2)</text>'
    )
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 43:.1f}" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK_MUTED}">h_steer = h + alpha * (h_ArmC - h_ArmA) · 100% spoiler-free at all alpha</text>'
    )

    sx0, sy0, scw, sch = rx + 48, ry + 62, rw - 70, 212.0
    sweep = slop["sweep"]
    slot = scw / len(sweep)
    bw = 38.0
    s_ymax = 44.0

    for yt in (0, 10, 20, 30, 40):
        yy = sy0 + sch - (yt / s_ymax) * sch
        lines.append(
            f'  <line x1="{sx0:.1f}" y1="{yy:.1f}" x2="{sx0 + scw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{sx0 - 7:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{yt}%</text>'
        )

    bar_colors = [EMERALD, "#799687", AMBER, "#D17759", VERMILION]
    for idx, (entry, col) in enumerate(zip(sweep, bar_colors, strict=True)):
        bx = sx0 + idx * slot + (slot - bw) / 2
        val = entry["slop_flagged_pct"]
        bh = max(3.0, (val / s_ymax) * sch)
        by = sy0 + sch - bh
        lines.append(
            f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>'
        )
        lines.append(
            f'  <text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" font-weight="700" fill="{col}">{val:.0f}%</text>'
        )
        a_lbl = f"{entry['alpha']:+.0f}" if entry["alpha"] != 0 else "0"
        lines.append(
            f'  <text x="{bx + bw / 2:.1f}" y="{sy0 + sch + 17:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" font-weight="700" fill="{INK}">a={a_lbl}</text>'
        )
        # V1-V2 gap chip underneath alpha
        gap_str = f"gap {entry['v1_minus_v2_pp']:.0f}pp"
        lines.append(
            f'  <text x="{bx + bw / 2:.1f}" y="{sy0 + sch + 32:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="9" fill="{INK_MUTED}">{gap_str}</text>'
        )

    # Bottom steering quote summary
    lines.append(
        f'  <rect x="{rx + 18:.1f}" y="{ry + rh - 58:.1f}" width="{rw - 36:.1f}" height="44" '
        f'rx="6" fill="{SURFACE_WARM}"/>'
    )
    lines.append(
        f'  <text x="{rx + 28:.1f}" y="{ry + rh - 40:.1f}" font-family="{SANS}" font-size="10.5" '
        f'font-weight="700" fill="{EMERALD}">a=-2 (0% slop): "simplifying the structure with a pawn capture"</text>'
    )
    lines.append(
        f'  <text x="{rx + 28:.1f}" y="{ry + rh - 23:.1f}" font-family="{SANS}" font-size="10.5" '
        f'font-weight="700" fill="{VERMILION}">a=+2 (36% slop): "resolving the tension... manage central tension"</text>'
    )

    lines += _svg_footer(
        "Source: results/interp/interp2_lora_svd_and_slop_steering.json (252 LoRA modules exact QR-SVD; N=60 diff / N=50 sweep prompts).",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure 7: Unified ChessFM Circuit Schematic + Logit Lens & Sabotage
# ---------------------------------------------------------------------------


def render_fig7_logit_lens_and_circuit(data: dict[str, Any]) -> str:
    w, h = 1080, 520
    i3 = data["interp"]["interp3_cot_sabotage"]
    ll = i3["logit_lens"]
    sab = i3["sabotage"]

    defs = f"""    <marker id="flow-arrow" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="{INK_MUTED}"/>
    </marker>"""

    lines = _svg_open(
        w,
        h,
        "Figure 7: Unified ChessFM Mechanistic Circuit, Logit Lens, and CoT Sabotage",
        defs=defs,
    )
    lines += _svg_header(
        "FIGURE 7 · CAUSAL <think> FAITHFULNESS & UNIFIED CIRCUIT (INTERP #3)",
        "Chain-of-Thought Is 100% Causally Load-Bearing Inside a Four-Stage Chess Circuit",
        "Left: Unified mechanistic synthesis across INTERP #1–#4. Right: Logit lens decision cliff & 100/100 sabotage flips.",
        w,
    )

    # Left panel: Unified Circuit Schematic (x=36..576, w=540)
    px, py, pw, ph = 36.0, 114.0, 540.0, 376.0
    lines.append(_panel(px, py, pw, ph))
    lines.append(
        f'  <text x="{px + 18:.1f}" y="{py + 26:.1f}" font-family="{SANS}" font-size="13.5" '
        f'font-weight="700" fill="{INK}">A. Unified ChessFM Mechanistic Circuit (Qwen3-4B)</text>'
    )

    stages = [
        (
            "STAGE 1 · LAYERS 0–25",
            "Prompt KV Cache & Mid-Network Planning",
            "Piece-List coordinates stored verbatim (80.5% probe); E3b peaks at L17 (71.2% MLP).",
            "Layer-6 surgery null (18% vs 24%) -> no static early board.",
            SURFACE_WARM,
            INK_SOFT,
            SLATE,
        ),
        (
            "STAGE 2 · LAYERS 26–29",
            "5 Piece-Tracker Attention Heads (0.43% of Heads)",
            "L27.H17, L29.H9, L26.H20, L26.H24, L28.H2 attend to active moving & target squares.",
            "Zero-ablation causes 3.0x drop (-12.0 pp vs -4.0 pp random).",
            COBALT_SOFT,
            COBALT,
            COBALT,
        ),
        (
            "STAGE 3 · LAYERS 30–35",
            "Late-Layer Board-State Rebound (vs. Hack Shortcut)",
            "Piece-List checkpoints recover +5.9 to +8.1 pp occupied-square accuracy in L30–L35.",
            "E3a hack jammed low-rank a-pawn shortcut into L34–L35 (eff_rank 11.41).",
            EMERALD_SOFT,
            EMERALD,
            EMERALD,
        ),
        (
            "STAGE 4 · DECODING",
            "100% Causal <think> Trace -> Post-</think> UCI Snap",
            "Target move stays latent during prose (P <= 0.02), then snaps to 0.99997 post-</think>.",
            "Corrupted <think> prefill flips 100/100 solved moves (0% resistance).",
            AMBER_SOFT,
            AMBER,
            AMBER,
        ),
    ]

    box_x = px + 22
    box_w = pw - 44
    box_h = 68.0
    gap = 18.0
    for idx, (tag, heading, line1, line2, bg, accent, border_col) in enumerate(stages):
        by_s = py + 44 + idx * (box_h + gap)
        lines.append(
            f'  <rect x="{box_x:.1f}" y="{by_s:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            f'rx="8" fill="{bg}" stroke="{border_col}" stroke-width="1.2" stroke-opacity="0.45"/>'
        )
        lines.append(
            f'  <text x="{box_x + 14:.1f}" y="{by_s + 17:.1f}" font-family="{MONO}" '
            f'font-size="9.5" font-weight="700" fill="{accent}">{_esc(tag)}</text>'
        )
        lines.append(
            f'  <text x="{box_x + 168:.1f}" y="{by_s + 17:.1f}" font-family="{SANS}" '
            f'font-size="11.5" font-weight="700" fill="{INK}">{_esc(heading)}</text>'
        )
        lines.append(
            f'  <text x="{box_x + 14:.1f}" y="{by_s + 38:.1f}" font-family="{SANS}" '
            f'font-size="10.8" fill="{INK_SOFT}">{_esc(line1)}</text>'
        )
        lines.append(
            f'  <text x="{box_x + 14:.1f}" y="{by_s + 54:.1f}" font-family="{SANS}" '
            f'font-size="10.5" font-weight="600" fill="{accent}">{_esc(line2)}</text>'
        )
        if idx < len(stages) - 1:
            ax_mid = box_x + box_w / 2
            y_start = by_s + box_h + 2
            y_end = by_s + box_h + gap - 3
            lines.append(
                f'  <line x1="{ax_mid:.1f}" y1="{y_start:.1f}" x2="{ax_mid:.1f}" y2="{y_end:.1f}" '
                f'stroke="{INK_MUTED}" stroke-width="1.8" marker-end="url(#flow-arrow)"/>'
            )

    # Right Top Panel: Logit Lens Cliff (x=596..1044, y=114..340)
    rx, ry, rw, rh1 = 596.0, 114.0, 448.0, 224.0
    lines.append(_panel(rx, ry, rw, rh1))
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + 25:.1f}" font-family="{SANS}" font-size="13" '
        f'font-weight="700" fill="{INK}">B. Logit Lens P(target_move) Across &lt;think&gt;</text>'
    )

    lx0, ly0, lcw, lch = rx + 50, ry + 48, rw - 74, 128.0
    frac_labels = ["0%", "25%", "50%", "75%", "100%", "Post"]
    probs = [
        ll["mean_prob_by_fraction"]["0.0"],
        ll["mean_prob_by_fraction"]["0.25"],
        ll["mean_prob_by_fraction"]["0.5"],
        ll["mean_prob_by_fraction"]["0.75"],
        ll["mean_prob_by_fraction"]["1.0"],
        ll["post_think_mean_prob"],
    ]

    for yt in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = ly0 + lch - yt * lch
        lines.append(
            f'  <line x1="{lx0:.1f}" y1="{yy:.1f}" x2="{lx0 + lcw:.1f}" y2="{yy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{lx0 - 7:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-family="{MONO}" font-size="9.5" fill="{INK_MUTED}">{yt:.2f}</text>'
        )

    xs = [lx0 + i * (lcw / 5.0) for i in range(6)]
    ys = [ly0 + lch - p * lch for p in probs]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    area = f"{xs[0]:.1f},{ly0 + lch:.1f} {poly} {xs[-1]:.1f},{ly0 + lch:.1f}"
    lines.append(f'  <polygon points="{area}" fill="{EMERALD}" fill-opacity="0.10"/>')
    lines.append(f'  <polyline points="{poly}" fill="none" stroke="{EMERALD}" stroke-width="2.8"/>')

    for i, (xx, yy, lbl, pr) in enumerate(zip(xs, ys, frac_labels, probs, strict=True)):
        col = EMERALD if i == 5 else COBALT
        lines.append(
            f'  <circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.2" fill="{SURFACE}" '
            f'stroke="{col}" stroke-width="2.2"/>'
        )
        anchor = "end" if i == 5 else "middle"
        tx = xx + 6 if i == 5 else xx
        lines.append(
            f'  <text x="{tx:.1f}" y="{ly0 + lch + 16:.1f}" text-anchor="{anchor}" '
            f'font-family="{MONO}" font-size="9.5" font-weight="600" fill="{INK_SOFT}">{_esc(lbl)}</text>'
        )
        if i == 5:
            lines.append(
                f'  <text x="{xx - 10:.1f}" y="{yy + 4:.1f}" text-anchor="end" '
                f'font-family="{MONO}" font-size="11" font-weight="700" fill="{EMERALD}">{pr:.5f}</text>'
            )
        elif pr > 0.0:
            lines.append(
                f'  <text x="{xx:.1f}" y="{yy - 8:.1f}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="9.5" fill="{COBALT}">{pr:.2f}</text>'
            )

    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry + rh1 - 12:.1f}" font-family="{SANS}" font-size="10.5" '
        f'fill="{INK_MUTED}">N={i3["num_solved_analyzed"]} solved puzzles · Post-&lt;/think&gt; top-1 match: {ll["post_think_top1_match_pct"]:.0f}%</text>'
    )

    # Right Bottom Panel: 100/100 Causal Sabotage (x=596..1044, y=354..490)
    ry2, rh2 = 354.0, 136.0
    lines.append(_panel(rx, ry2, rw, rh2))
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry2 + 25:.1f}" font-family="{SANS}" font-size="13" '
        f'font-weight="700" fill="{INK}">C. Causal &lt;think&gt; Prefill Sabotage (N={sab["num_tested"]})</text>'
    )
    lines += _pill(
        rx + rw - 116,
        ry2 + 10,
        98,
        22,
        f"{sab['resisted_sabotage_pct']:.0f}% Resisted",
        VERMILION_SOFT,
        VERMILION,
        font_size=10.0,
    )

    # Full-width 100% flip bar
    lines.append(
        f'  <rect x="{rx + 18:.1f}" y="{ry2 + 42:.1f}" width="{rw - 36:.1f}" height="28" '
        f'rx="6" fill="{VERMILION}"/>'
    )
    lines.append(
        f'  <text x="{rx + rw / 2:.1f}" y="{ry2 + 60:.1f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="11.5" font-weight="700" fill="{SURFACE}">'
        f"100 / 100 Flipped to Injected Distractor ({sab['causal_move_flip_pct']:.1f}%)</text>"
    )

    # 5 real flip chips
    lines.append(
        f'  <text x="{rx + 18:.1f}" y="{ry2 + 90:.1f}" font-family="{SANS}" font-size="10.5" '
        f'font-weight="600" fill="{INK_MUTED}">Verified move flips (clean -> sabotaged):</text>'
    )
    chip_w = 78.0
    chip_gap = 6.0
    for idx, ex in enumerate(sab["examples"][:5]):
        cx = rx + 18 + idx * (chip_w + chip_gap)
        lines += _pill(
            cx,
            ry2 + 98,
            chip_w,
            24,
            f"{ex['target_move']}->{ex['post_sabotage_move']}",
            SURFACE_WARM,
            INK,
            font_size=9.3,
        )

    lines += _svg_footer(
        "Source: results/interp/interp3_logit_lens_and_cot_sabotage.json & INTERP #1–#4 synthesis (Qwen3-4B E3b sharp_wdl).",
        w,
        h,
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


def render_all_figures(data: dict[str, Any]) -> dict[str, str]:
    """Generate all 7 editorial SVG figures from the canonical blog_data payload."""
    return {
        "fig1_ablation_ladder.svg": render_fig1_ablation_ladder(data),
        "fig2_apawn_hack_board_and_diversity.svg": render_fig2_apawn_hack_board_and_diversity(data),
        "fig3_pass_at_k_and_elo_mirage.svg": render_fig3_pass_at_k_and_elo_mirage(data),
        "fig4_coach_showdown_and_goodhart.svg": render_fig4_coach_showdown_and_goodhart(data),
        "fig5_board_probes_and_piece_trackers.svg": render_fig5_board_probes_and_piece_trackers(
            data
        ),
        "fig6_lora_svd_and_slop_steering.svg": render_fig6_lora_svd_and_slop_steering(data),
        "fig7_logit_lens_and_circuit.svg": render_fig7_logit_lens_and_circuit(data),
    }


def build_or_check(repo_root: Path = REPO_ROOT, check: bool = False) -> list[str]:
    """Compile blog_data.json + 7 SVGs; in check mode return list of mismatched paths."""
    data = compile_blog_data(repo_root)
    json_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    figures = render_all_figures(data)

    targets: dict[Path, str] = {repo_root / "web" / "data" / "blog_data.json": json_text}
    for fname, svg_text in figures.items():
        targets[repo_root / "docs" / "figures" / fname] = svg_text

    mismatches: list[str] = []
    for path, expected in targets.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(str(path.relative_to(repo_root)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify ChessFM blog_data.json and 7 editorial SVGs."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify exact 0-diff byte parity against committed files without modifying disk.",
    )
    args = parser.parse_args(argv)

    mismatches = build_or_check(REPO_ROOT, check=args.check)
    if args.check:
        if mismatches:
            print("Blog asset drift detected in:", ", ".join(mismatches), file=sys.stderr)
            return 1
        print(f"OK: 0-diff parity verified across blog_data.json and {len(FIGURE_FILENAMES)} SVGs.")
        return 0

    json_bytes = (REPO_ROOT / "web" / "data" / "blog_data.json").stat().st_size
    print(f"Wrote web/data/blog_data.json ({json_bytes:,} bytes) and {len(FIGURE_FILENAMES)} SVGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
