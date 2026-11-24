"""Tests for deterministic blog data compilation and publication SVG figures."""

from __future__ import annotations

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import chess
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOG_DATA_PATH = REPO_ROOT / "web" / "data" / "blog_data.json"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

EXPECTED_FIGURES = (
    "fig1_ablation_ladder.svg",
    "fig2_apawn_hack_board_and_diversity.svg",
    "fig3_pass_at_k_and_elo_mirage.svg",
    "fig4_coach_showdown_and_goodhart.svg",
    "fig5_board_probes_and_piece_trackers.svg",
    "fig6_lora_svd_and_slop_steering.svg",
    "fig7_logit_lens_and_circuit.svg",
)

FORBIDDEN_PATTERNS = (
    re.compile(r"\bresume\b", re.IGNORECASE),
    re.compile(r"\binterview\b", re.IGNORECASE),
    re.compile(r"\bwhiteboard\b", re.IGNORECASE),
    re.compile(r"\bModal Starter\b", re.IGNORECASE),
    re.compile(r"\bfree credit", re.IGNORECASE),
    re.compile(r"\bbudget cap\b", re.IGNORECASE),
    re.compile(r"\bhiring\b", re.IGNORECASE),
    re.compile(r"\bportfolio\b", re.IGNORECASE),
)


def _load_builder_module():
    script_path = REPO_ROOT / "scripts" / "build_blog_assets.py"
    spec = importlib.util.spec_from_file_location("build_blog_assets", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_no_forbidden_wording(text: str, label: str) -> None:
    for pat in FORBIDDEN_PATTERNS:
        assert pat.search(text) is None, f"Forbidden wording {pat.pattern!r} found in {label}"


def test_blog_data_json_schema_and_verified_invariants() -> None:
    assert BLOG_DATA_PATH.exists(), "web/data/blog_data.json must be committed"
    raw_text = BLOG_DATA_PATH.read_text(encoding="utf-8")
    _assert_no_forbidden_wording(raw_text, "web/data/blog_data.json")
    data = json.loads(raw_text)

    assert data["schema_version"] == "1.0"
    required_top = {
        "schema_version",
        "hero_kpis",
        "sft_ladder",
        "rlvr_apawn_hack",
        "pass_at_k",
        "elo_mirage",
        "goodhart_06b",
        "e5_coach",
        "interp",
    }
    assert required_top.issubset(data.keys())

    # Hero KPIs
    kpi = data["hero_kpis"]
    assert kpi["coach_all_clean_pct"] == 92.0
    assert kpi["coach_vs_teacher_pp"] == 3.33
    assert kpi["coach_slop_pct"] == 0.0
    assert kpi["apawn_before_pct"] == 60.3
    assert kpi["apawn_after_pct"] == 1.5
    assert kpi["e2a_unique_moves"] == 528
    assert kpi["e3a_unique_moves"] == 206
    assert kpi["e3b_unique_moves"] == 443
    assert kpi["rlvr_legal_gain_pp"] == 10.3
    assert kpi["rlvr_cot_gain_pp"] == 10.02
    assert kpi["elo_mirage_pp"] == 230.92
    assert kpi["e2a_rep3_draw_pct"] == 89.36
    assert kpi["rlvr_clean_elo_lift"] == 27.31
    assert kpi["piece_tracker_head_pct"] == 0.43
    assert kpi["piece_tracker_ablation_ratio"] == 3.0
    assert kpi["top5_ablation_drop_pp"] == 12.0
    assert kpi["random5_ablation_drop_pp"] == 4.0
    assert kpi["sabotage_flip_rate_pct"] == 100.0

    # SFT ladder stages
    sft_ids = [row["id"] for row in data["sft_ladder"]]
    assert sft_ids == [
        "Base_ASCII",
        "Base_PL",
        "E0",
        "E0_GRPO",
        "E1a",
        "E1b",
        "E2a",
        "E2b",
        "Gemma26B",
    ]
    sft_by_id = {row["id"]: row for row in data["sft_ladder"]}
    assert sft_by_id["E2a"]["strict_legal_pct"] == 66.2
    assert sft_by_id["E2a"]["val_loss"] == 0.266
    assert sft_by_id["E2b"]["val_loss"] == 0.2842
    assert sft_by_id["Gemma26B"]["puzzles_900_lenient_pct"] == 10.78

    # RLVR & 5 verified FEN board examples
    rlvr = data["rlvr_apawn_hack"]
    assert rlvr["models"]["E3a"]["a_file_pawn_push_pct"] == 60.3
    assert rlvr["models"]["E3b"]["a_file_pawn_push_pct"] == 1.5
    assert rlvr["models"]["E2a"]["unique_moves"] == 528
    assert rlvr["models"]["E3a"]["unique_moves"] == 206
    assert rlvr["models"]["E3b"]["unique_moves"] == 443
    assert len(rlvr["training_curves"]["E3a"]) == 20
    assert len(rlvr["training_curves"]["E3b"]) == 20

    examples = rlvr["board_examples"]
    assert len(examples) == 5
    for ex in examples:
        board = chess.Board(ex["fen"])
        best_mv = chess.Move.from_uci(ex["best_move"])
        assert best_mv in board.legal_moves
        e3b_mv = chess.Move.from_uci(ex["E3b"]["move"])
        assert e3b_mv in board.legal_moves
        assert ex["E3b"]["cp_loss"] == 0.0

    # Elo Mirage verified 22/56 = 39.29% on E3b_norep
    suites = {s["id"]: s for s in data["elo_mirage"]["suites"]}
    assert suites["E3b_norep"]["rep3_draws"] == 22
    assert suites["E3b_norep"]["draws"] == 56
    assert suites["E3b_norep"]["rep3_pct_of_draws"] == 39.29
    assert suites["E2a_std"]["rep3_draws"] == 84
    assert suites["E2a_std"]["draws"] == 94

    # Goodhart 0.6B verified 100.0% teacher V1 pass
    g06 = data["goodhart_06b"]
    assert g06["grounded_292"]["teacher_26b"]["v1_pass_pct"] == 100.0
    assert g06["grounded_292"]["teacher_26b"]["v1_pass_count"] == 292
    assert g06["grounded_292"]["sft_06b_grounded"]["v2_and_non_slop_pct"] == 12.67
    assert len(g06["four_case_audit"]) == 4

    # E5 Coach showdown
    e5_rows = {r["id"]: r for r in data["e5_coach"]["rows"]}
    assert len(e5_rows) == 7
    assert e5_rows["row3_arm_a_grpo"]["all_clean_pct"] == 92.0
    assert e5_rows["row5_arm_b_grpo"]["all_clean_pct"] == 91.0
    assert e5_rows["row5_arm_b_grpo"]["fact_coverage_pct"] == 81.67
    assert e5_rows["row6_teacher_26b"]["all_clean_pct"] == 88.67
    assert data["e5_coach"]["goodhart_curves"]["steps"] == [0, 10, 20, 30, 40]
    assert len(data["e5_coach"]["verbatim_comparisons"]) == 2

    # Interp #1-#4 invariants
    interp = data["interp"]
    assert interp["interp1_probes"]["layers"] == [0, 6, 12, 18, 24, 30, 35]
    assert len(interp["interp1_probes"]["heatmap"]) == 5
    assert interp["interp1_probes"]["causal_board_surgery"]["probe_move_flip_pct"] == 18.0
    assert interp["interp1_probes"]["causal_board_surgery"]["random_control_move_flip_pct"] == 24.0

    slop_sweep = interp["interp2_svd_slop"]["slop_steering"]["sweep"]
    assert [s["alpha"] for s in slop_sweep] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert slop_sweep[0]["slop_flagged_pct"] == 0.0
    assert slop_sweep[-1]["slop_flagged_pct"] == 36.0
    for adapter in ("e3a_apawn_hack", "e3b_sharp_wdl", "arm_a_v2_grpo", "arm_c_v1_grpo"):
        assert len(interp["interp2_svd_slop"]["profiles"][adapter]["per_layer"]) == 36

    assert interp["interp3_cot_sabotage"]["sabotage"]["num_tested"] == 100
    assert interp["interp3_cot_sabotage"]["sabotage"]["causal_move_flip_pct"] == 100.0
    assert interp["interp3_cot_sabotage"]["logit_lens"]["post_think_mean_prob"] == 0.99997

    ab = interp["interp4_piece_tracker"]["ablation"]
    assert ab["top5_solve_drop_pp"] == 12.0
    assert ab["random5_solve_drop_pp"] == 4.0
    assert ab["drop_ratio"] == 3.0


@pytest.mark.parametrize("filename", EXPECTED_FIGURES)
def test_publication_svgs_are_valid_safe_and_themed(filename: str) -> None:
    svg_path = FIGURES_DIR / filename
    assert svg_path.exists(), f"Missing publication SVG: {svg_path}"
    content = svg_path.read_text(encoding="utf-8")
    assert len(content) > 2000

    # Strict XML parse
    root = ET.fromstring(content)
    assert root.tag.endswith("svg")
    assert root.attrib.get("viewBox"), f"{filename} missing viewBox"
    assert root.attrib.get("width") and root.attrib.get("height")

    # Explicit warm-paper backdrop for GitHub Light & Dark readability
    assert "#FAF8F5" in content, f"{filename} missing explicit #FAF8F5 opaque backdrop"

    # Security & GitHub SVG sandbox compliance
    lowered = content.lower()
    assert "<script" not in lowered
    assert "@import" not in lowered
    assert "http://" not in lowered.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in lowered

    _assert_no_forbidden_wording(content, filename)


@pytest.mark.skipif(
    not (REPO_ROOT / "results" / "interp").exists(),
    reason="Raw gitignored results/ absent in CI",
)
def test_blog_assets_zero_diff_parity_against_raw_results() -> None:
    builder = _load_builder_module()
    mismatches = builder.build_or_check(REPO_ROOT, check=True)
    assert mismatches == [], f"Out-of-date blog assets vs raw results/: {mismatches}"
