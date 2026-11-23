"""Unit tests for Bradley-Terry Elo MLE, bootstrap CIs, and English forgetting benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chessfm.eval.bradley_terry import (
    DEFAULT_ANCHOR_RATINGS,
    BradleyTerryEstimate,
    OpponentMatchResult,
    estimate_bradley_terry_elo,
    expected_score,
    results_from_game_records,
    solve_bradley_terry_elo,
)
from chessfm.eval.forgetting import (
    CATEGORIES,
    evaluate_forgetting_benchmark,
    export_forgetting_jsonl,
    get_forgetting_benchmark,
    score_forgetting_response,
)

# ---------------------------------------------------------------------------
# Bradley-Terry Elo Solver & Stratified Bootstrap CI
# ---------------------------------------------------------------------------


def test_default_anchor_ratings_calibrated() -> None:
    assert DEFAULT_ANCHOR_RATINGS == {
        "random": 200.0,
        "sf_skill_0": 800.0,
        "sf_skill_1": 950.0,
        "maia_1100": 1100.0,
        "sf_uci_1320": 1320.0,
    }


def test_expected_score_symmetry_and_extremes() -> None:
    assert expected_score(1200.0, 1200.0) == pytest.approx(0.5)
    assert expected_score(1600.0, 1200.0) == pytest.approx(10.0 / 11.0, rel=1e-6)
    assert expected_score(800.0, 1200.0) == pytest.approx(1.0 / 11.0, rel=1e-6)
    assert expected_score(5000.0, 200.0) == pytest.approx(1.0, abs=1e-9)
    assert expected_score(200.0, 5000.0) == pytest.approx(0.0, abs=1e-9)


def test_opponent_match_result_validation_and_wdl() -> None:
    res = OpponentMatchResult.from_wdl("maia_1100", 1100.0, wins=10, draws=4, losses=6)
    assert res.games == 20
    assert res.score == pytest.approx(12.0)
    assert res.score_rate == pytest.approx(0.6)

    with pytest.raises(ValueError, match="non-empty"):
        OpponentMatchResult("", 1000.0, (1.0,))
    with pytest.raises(ValueError, match="finite"):
        OpponentMatchResult("opp", float("nan"), (1.0,))
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        OpponentMatchResult("opp", 1000.0, (1.5,))
    with pytest.raises(ValueError, match="non-negative"):
        OpponentMatchResult.from_wdl("opp", 1000.0, -1, 0, 1)


def test_solve_bradley_terry_exact_single_and_multi_opponent() -> None:
    # 50% vs 1100 -> exact 1100.0
    res_even = [OpponentMatchResult.from_wdl("maia_1100", 1100.0, 10, 0, 10)]
    elo, reg, bound = solve_bradley_terry_elo(res_even)
    assert elo == pytest.approx(1100.0, abs=1e-6)
    assert reg is False
    assert bound == "exact"

    # Multi-opponent: true rating 1000.0 -> synthetic expected scores match exact MLE.
    true_elo = 1000.0
    multi = []
    for name, anchor in DEFAULT_ANCHOR_RATINGS.items():
        p = expected_score(true_elo, anchor)
        multi.append(OpponentMatchResult(name, anchor, tuple([p] * 40)))
    est_elo, reg_m, bound_m = solve_bradley_terry_elo(multi)
    assert est_elo == pytest.approx(true_elo, abs=1e-5)
    assert reg_m is False
    assert bound_m == "exact"


def test_solve_bradley_terry_zero_and_hundred_percent_regularization() -> None:
    # 0% win rate -> Jeffreys regularized upper bound, clamped safely
    all_loss = [
        OpponentMatchResult.from_wdl("sf_skill_0", 800.0, 0, 0, 20),
        OpponentMatchResult.from_wdl("maia_1100", 1100.0, 0, 0, 20),
    ]
    elo_low, reg_low, bound_low = solve_bradley_terry_elo(all_loss)
    assert reg_low is True
    assert bound_low == "upper"
    assert 800.0 - 800.0 <= elo_low < 800.0

    # 100% win rate -> Jeffreys regularized lower bound, clamped safely
    all_win = [
        OpponentMatchResult.from_wdl("random", 200.0, 20, 0, 0),
        OpponentMatchResult.from_wdl("sf_skill_0", 800.0, 20, 0, 0),
    ]
    elo_hi, reg_hi, bound_hi = solve_bradley_terry_elo(all_win)
    assert reg_hi is True
    assert bound_hi == "lower"
    assert 800.0 < elo_hi <= 800.0 + 800.0


def test_solve_bradley_terry_validation_errors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        solve_bradley_terry_elo([])
    with pytest.raises(ValueError, match="at least one played game"):
        solve_bradley_terry_elo([OpponentMatchResult("random", 200.0, ())])
    with pytest.raises(ValueError, match="prior_strength"):
        solve_bradley_terry_elo(
            [OpponentMatchResult.from_wdl("random", 200.0, 5, 0, 5)], prior_strength=0.0
        )


def test_estimate_bradley_terry_elo_bootstrap_and_serialization() -> None:
    results = [
        OpponentMatchResult.from_wdl("random", 200.0, 38, 1, 1),
        OpponentMatchResult.from_wdl("sf_skill_0", 800.0, 24, 6, 10),
        OpponentMatchResult.from_wdl("sf_skill_1", 950.0, 16, 8, 16),
        OpponentMatchResult.from_wdl("maia_1100", 1100.0, 10, 6, 24),
        OpponentMatchResult.from_wdl("sf_uci_1320", 1320.0, 3, 5, 32),
    ]
    est1 = estimate_bradley_terry_elo(results, n_bootstrap=200, ci_level=0.95, seed=123)
    est2 = estimate_bradley_terry_elo(results, n_bootstrap=200, ci_level=0.95, seed=123)
    assert isinstance(est1, BradleyTerryEstimate)
    assert est1 == est2  # deterministic seed
    assert est1.ci_low < est1.elo < est1.ci_high
    assert est1.std_error > 0.0
    assert est1.total_games == 200
    assert est1.regularized is False
    assert est1.bound_type == "exact"
    assert set(est1.per_opponent.keys()) == set(DEFAULT_ANCHOR_RATINGS.keys())

    payload = est1.to_dict()
    assert payload["total_games"] == 200
    assert payload["ci_low"] < payload["elo"] < payload["ci_high"]
    assert math.isclose(payload["score_rate"], est1.score_rate, abs_tol=1e-5)

    # n_bootstrap <= 1 path
    est_no_boot = estimate_bradley_terry_elo(results, n_bootstrap=0)
    assert est_no_boot.ci_low == est_no_boot.elo == est_no_boot.ci_high
    assert est_no_boot.std_error == 0.0

    with pytest.raises(ValueError):
        estimate_bradley_terry_elo([], n_bootstrap=10)
    with pytest.raises(ValueError):
        estimate_bradley_terry_elo(results, n_bootstrap=-1)
    with pytest.raises(ValueError):
        estimate_bradley_terry_elo(results, ci_level=1.0)


def test_solve_bradley_terry_unscaled_foc_and_single_anchor_precision() -> None:
    # Single-anchor 99/100 and 1/100 closed-form precision (< 1e-7 Elo error).
    for anchor in (200.0, 1000.0, 2500.0):
        for wins, losses in ((99, 1), (1, 99)):
            p = wins / (wins + losses)
            exact_mle = anchor + 400.0 * math.log10(p / (1.0 - p))
            res = [OpponentMatchResult.from_wdl("anchor", anchor, wins, 0, losses)]
            est, reg, bound = solve_bradley_terry_elo(res)
            assert reg is False
            assert bound == "exact"
            assert abs(est - exact_mle) < 1e-7
            residual = abs(wins - (wins + losses) * expected_score(est, anchor))
            assert residual < 1e-9

    # Multi-opponent wide-spread unscaled FOC residual < 1e-9.
    wide_results = [
        OpponentMatchResult.from_wdl("opp1", 100.0, 19, 1, 0),
        OpponentMatchResult.from_wdl("opp2", 800.0, 15, 3, 2),
        OpponentMatchResult.from_wdl("opp3", 1500.0, 6, 4, 10),
        OpponentMatchResult.from_wdl("opp4", 2700.0, 0, 1, 19),
    ]
    est_wide, reg_wide, _ = solve_bradley_terry_elo(wide_results)
    assert reg_wide is False
    unscaled_residual = abs(
        sum(r.score - r.games * expected_score(est_wide, r.anchor_elo) for r in wide_results)
    )
    assert unscaled_residual < 1e-9


def test_estimate_bradley_terry_rejects_duplicate_opponent_names() -> None:
    dup_results = [
        OpponentMatchResult.from_wdl("maia_1100", 1100.0, 10, 0, 10),
        OpponentMatchResult.from_wdl("maia_1100", 1100.0, 12, 0, 8),
    ]
    with pytest.raises(ValueError, match="Duplicate opponent name"):
        estimate_bradley_terry_elo(dup_results, n_bootstrap=10)


def test_results_from_game_records() -> None:
    records = [
        {"opponent": "random", "result": "win"},
        {"opponent": "random", "score": 0.5},
        {"opponent": "maia_1100", "result": "loss"},
        {"opponent": "maia_1100", "result": "1/2-1/2"},
        {"opponent": "custom_bot", "score": 1.0, "anchor_elo": 1250.0},
    ]
    grouped = results_from_game_records(records)
    by_name = {r.opponent: r for r in grouped}
    assert by_name["random"].anchor_elo == 200.0
    assert by_name["random"].outcomes == (1.0, 0.5)
    assert by_name["maia_1100"].anchor_elo == 1100.0
    assert by_name["maia_1100"].outcomes == (0.0, 0.5)
    assert by_name["custom_bot"].anchor_elo == 1250.0

    with pytest.raises(KeyError, match="No anchor_elo"):
        results_from_game_records([{"opponent": "unknown_opp", "score": 1.0}])
    with pytest.raises(ValueError, match="unrecognized result"):
        results_from_game_records([{"opponent": "random", "result": "bogus"}])
    with pytest.raises(ValueError, match="must include 'score' or 'result'"):
        results_from_game_records([{"opponent": "random"}])
    with pytest.raises(ValueError, match="Conflicting anchor_elo"):
        results_from_game_records(
            [
                {"opponent": "bot", "score": 1.0, "anchor_elo": 1000.0},
                {"opponent": "bot", "score": 0.0, "anchor_elo": 1100.0},
            ]
        )


# ---------------------------------------------------------------------------
# General-English Forgetting Sanity Benchmark (100 items)
# ---------------------------------------------------------------------------


def test_forgetting_benchmark_structure_and_deep_copy() -> None:
    items = get_forgetting_benchmark()
    assert len(items) == 100

    ids = [it["id"] for it in items]
    assert len(set(ids)) == 100

    cat_counts: dict[str, int] = {}
    for it in items:
        cat_counts[it["category"]] = cat_counts.get(it["category"], 0) + 1
        assert {"id", "category", "prompt", "answer", "answer_type", "choices"} <= set(it.keys())
        if it["answer_type"] == "mcq":
            assert it["answer"] in ("A", "B", "C", "D")
            assert set(it["choices"].keys()) == {"A", "B", "C", "D"}
        else:
            assert it["answer_type"] == "exact"
            assert isinstance(it["answer"], str) and len(it["answer"]) > 0

    for cat in CATEGORIES:
        assert cat_counts[cat] == 25

    # Deep copy isolation
    items[0]["answer"] = "MUTATED"
    assert get_forgetting_benchmark()[0]["answer"] != "MUTATED"


def test_export_forgetting_jsonl(tmp_path: Path) -> None:
    out_file = tmp_path / "nested" / "test_english_100.jsonl"
    n = export_forgetting_jsonl(out_file)
    assert n == 100
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 100
    assert lines[0]["id"] == "rc_01"


def test_score_forgetting_response_and_chess_leakage() -> None:
    items = get_forgetting_benchmark()
    mcq_item = next(it for it in items if it["answer_type"] == "mcq")
    exact_item = next(it for it in items if it["answer_type"] == "exact")

    # Clean MCQ with <think> stripped
    scored_ok = score_forgetting_response(
        mcq_item, f"<think>Let me read carefully.</think>\nAnswer: {mcq_item['answer']}"
    )
    assert scored_ok["correct"] is True
    assert scored_ok["chess_leakage"] is False

    # Chess leakage: <think> + UCI move spam on non-chess prompt
    scored_leak = score_forgetting_response(
        mcq_item,
        "<think>Position assessment: White to move. Best move e2e4.</think>\ne2e4",
    )
    assert scored_leak["correct"] is False
    assert scored_leak["chess_leakage"] is True

    # Pure UCI move without think tags
    scored_uci_only = score_forgetting_response(exact_item, "g1f3")
    assert scored_uci_only["correct"] is False
    assert scored_uci_only["chess_leakage"] is True

    # Exact answer wrapped in quotes
    scored_exact = score_forgetting_response(exact_item, f'"{exact_item["answer"]}"')
    assert scored_exact["correct"] is True
    assert scored_exact["chess_leakage"] is False


def test_evaluate_forgetting_benchmark_perfect_and_leaking() -> None:
    summary_perfect = evaluate_forgetting_benchmark(lambda it: it["answer"])
    assert summary_perfect["total"] == 100
    assert summary_perfect["correct"] == 100
    assert summary_perfect["accuracy"] == pytest.approx(1.0)
    assert summary_perfect["chess_leakage_rate"] == pytest.approx(0.0)
    assert set(summary_perfect["per_category"].keys()) == set(CATEGORIES)
    for cat in CATEGORIES:
        assert summary_perfect["per_category"][cat]["accuracy"] == pytest.approx(1.0)

    summary_bad = evaluate_forgetting_benchmark(
        lambda _it: "<think>Best move is e2e4</think>\ne2e4"
    )
    assert summary_bad["correct"] == 0
    assert summary_bad["accuracy"] == pytest.approx(0.0)
    assert summary_bad["chess_leakage_rate"] == pytest.approx(1.0)


def test_forgetting_mcq_answer_keys_are_balanced() -> None:
    items = [it for it in get_forgetting_benchmark() if it["answer_type"] == "mcq"]
    assert len(items) == 75
    counts = {k: sum(1 for it in items if it["answer"] == k) for k in ("A", "B", "C", "D")}
    assert min(counts.values()) >= 17
    assert max(counts.values()) <= 20
    assert sum(counts.values()) == 75


def test_forgetting_parser_orphan_think_markdown_and_article_false_positives() -> None:
    mcq_item = {
        "id": "rc_test",
        "category": "reading_comprehension",
        "prompt": "Q",
        "answer": "B",
        "answer_type": "mcq",
        "choices": {"A": "x", "B": "y", "C": "z", "D": "w"},
    }
    exact_item = {
        "id": "if_test",
        "category": "instruction_following",
        "prompt": "Reply BLUE",
        "answer": "BLUE",
        "answer_type": "exact",
        "choices": {},
    }

    # Orphan </think> tag (Qwen3 enable_thinking=True prompt prefix).
    s_mcq_orphan = score_forgetting_response(mcq_item, "Let me think... it's B.\n</think>\nB")
    assert s_mcq_orphan["predicted"] == "B"
    assert s_mcq_orphan["correct"] is True

    s_exact_orphan = score_forgetting_response(exact_item, "Let me think.\n</think>\nBLUE")
    assert s_exact_orphan["predicted"] == "BLUE"
    assert s_exact_orphan["correct"] is True

    # Markdown bold and code wrappers.
    for raw in ("**B**", "`B`", "Answer: **B**", "Answer: `B`", "The answer is **B**."):
        scored = score_forgetting_response(mcq_item, raw)
        assert scored["predicted"] == "B", f"Failed on {raw!r}"
        assert scored["correct"] is True, f"Failed on {raw!r}"

    # English article "a" / "A" must NOT produce false positive "A".
    mcq_a = dict(mcq_item, answer="A")
    for prose in ("The answer is a mystery", "A dog barked loudly"):
        scored = score_forgetting_response(mcq_a, prose)
        assert scored["predicted"] == "", f"False positive on {prose!r}: {scored}"
        assert scored["correct"] is False
