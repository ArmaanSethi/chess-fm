#!/usr/bin/env python3
"""Stratified pass@k, move-entropy, and CoT V2 faithfulness evaluation on chess puzzles.

Evaluates a model with ``k`` stochastic rollouts per puzzle (e.g. ``k=8``,
``temperature=0.7, top_p=0.95``) on a deterministic stratified slice of
``test_puzzles_900.jsonl`` to measure whether RLVR expands the reasoning
frontier (``pass@8``) or sharpens sampling efficiency (``pass@1`` vs
``pass@8``), alongside Shannon move entropy (in bits) and Strict Verifier V2
chain-of-thought faithfulness across all ``N * k`` sampled rollouts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

from chessfm.coach.verifier import verify_think_trace
from chessfm.config import DEFAULT_MAX_NEW_TOKENS, build_chat_prompt
from chessfm.eval.suite import (
    _detect_hardware,
    _has_valid_think_format,
    _resolve_difficulty_band,
    _resolve_puzzle_target,
)
from chessfm.rl.evaluate import wilson_ci
from chessfm.rl.move_utils import parse_move_from_output
from chessfm.training.manifest import _get_git_sha


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON lines from ``path``."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


INVALID_MOVE_BUCKET = "<invalid>"
_THINK_CONTENT_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _stratum_key(puzzle: dict[str, Any]) -> str:
    """Return a deterministic stratification bucket key for a puzzle row."""
    split_group = str(puzzle.get("split_group") or "").strip()
    if split_group == "theme":
        theme = str(puzzle.get("primary_theme") or "unknown").strip()
        return f"theme:{theme}"
    if split_group == "difficulty":
        band = _resolve_difficulty_band(puzzle) or "unknown"
        return f"difficulty:{band}"
    theme = str(puzzle.get("primary_theme") or "").strip()
    if theme:
        return f"theme:{theme}"
    band = _resolve_difficulty_band(puzzle)
    if band:
        return f"difficulty:{band}"
    return "default"


def stratified_puzzle_sample(
    puzzles: Sequence[dict[str, Any]],
    n: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample ``n`` puzzles stratified across theme & difficulty buckets.

    Uses proportional largest-remainder allocation so both ``theme`` and
    ``difficulty`` splits retain their exact population proportions. Each
    returned dict is a shallow copy annotated with ``original_index``,
    ``puzzle_id``, and ``stratum`` for full auditability.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    total = len(puzzles)
    if n == 0 or total == 0:
        return []

    indexed: list[dict[str, Any]] = []
    for orig_idx, row in enumerate(puzzles):
        copy = dict(row)
        copy["original_index"] = orig_idx
        copy["puzzle_id"] = str(row.get("puzzle_id") or f"idx_{orig_idx:04d}")
        copy["stratum"] = _stratum_key(row)
        indexed.append(copy)

    if n >= total:
        return indexed

    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in indexed:
        buckets.setdefault(item["stratum"], []).append(item)

    rng = random.Random(seed)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])

    # Largest-remainder proportional allocation across sorted strata.
    exact_quotas = {key: (len(items) * n) / total for key, items in buckets.items()}
    allocations = {
        key: min(len(buckets[key]), int(math.floor(q))) for key, q in exact_quotas.items()
    }
    remaining = n - sum(allocations.values())
    if remaining > 0:
        remainders = sorted(
            buckets.keys(),
            key=lambda k: (-(exact_quotas[k] - allocations[k]), k),
        )
        for key in remainders:
            if remaining <= 0:
                break
            if allocations[key] < len(buckets[key]):
                allocations[key] += 1
                remaining -= 1

    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        selected.extend(buckets[key][: allocations[key]])

    # Preserve original dataset ordering for clean, reproducible batching.
    selected.sort(key=lambda item: item["original_index"])
    return selected


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the exact unbiased ``pass@k`` estimator (Chen et al., 2021).

    ``pass@k = 1 - C(n - c, k) / C(n, k)``, returning ``1.0`` whenever
    ``n - c < k`` and ``0.0`` when ``c <= 0``.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n ({n}), got {k}")
    if c < 0 or c > n:
        raise ValueError(f"c must satisfy 0 <= c <= n ({n}), got {c}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI for the mean of ``values``."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        v = float(values[0])
        return (v, v)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    rng = random.Random(seed)
    vals = [float(v) for v in values]
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()
    alpha = (1.0 - ci_level) / 2.0
    lo_idx = min(n_bootstrap - 1, max(0, int(math.floor(alpha * n_bootstrap))))
    hi_idx = min(n_bootstrap - 1, max(0, int(math.ceil((1.0 - alpha) * n_bootstrap)) - 1))
    return (boot_means[lo_idx], boot_means[hi_idx])


def shannon_entropy_bits(labels: Sequence[str]) -> float:
    """Compute discrete Shannon entropy ``H = -sum(p * log2(p))`` in bits."""
    total = len(labels)
    if total <= 1:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log2(p)
    return max(0.0, entropy)


def _extract_think_word_count(text: str) -> int:
    """Return the word count inside ``<think>...</think>`` (0 if absent)."""
    match = _THINK_CONTENT_RE.search(text or "")
    if not match:
        return 0
    return len(match.group(1).split())


def evaluate_pass_at_k_rollouts(
    puzzles: Sequence[dict[str, Any]],
    rollouts_per_puzzle: Sequence[Sequence[str]],
    *,
    k_values: Sequence[int] = (1, 2, 4, 8),
    verify_cot: bool = True,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score pre-generated stochastic rollouts for pass@k, entropy, and CoT V2 faithfulness.

    Returns ``(summary_metrics, per_puzzle_records)``.
    """
    if len(puzzles) != len(rollouts_per_puzzle):
        raise ValueError(
            f"puzzles length ({len(puzzles)}) must match rollouts_per_puzzle length "
            f"({len(rollouts_per_puzzle)})."
        )

    n_puzzles = len(puzzles)
    if n_puzzles == 0:
        return (
            {
                "num_puzzles": 0,
                "k": 0,
                "total_rollouts": 0,
                "pass_at_k": {},
                "entropy_unit": "bits",
                "mean_move_entropy_all_bits": 0.0,
                "mean_move_entropy_legal_bits": 0.0,
                "puzzles_with_legal_rollout": 0,
                "mean_unique_predictions": 0.0,
                "mean_unique_legal_moves": 0.0,
                "strict_legal_rollouts": 0,
                "strict_legal_rate": 0.0,
                "format_adherence": 0.0,
                "mean_think_words": 0.0,
                "cot_v2": {
                    "traces_evaluated": 0,
                    "traces_with_claims": 0,
                    "claims_total": 0,
                    "claims_verified": 0,
                    "claim_accuracy": None,
                    "zero_violation_rate_all": None,
                    "zero_violation_rate_with_claims": None,
                    "mean_claims_per_trace": None,
                },
                "by_split_group": {},
                "by_difficulty": {},
                "puzzle_ids": [],
                "original_indices": [],
            },
            [],
        )

    k = len(rollouts_per_puzzle[0])
    if k < 1:
        raise ValueError("Each puzzle must have at least 1 rollout.")
    for idx, rlist in enumerate(rollouts_per_puzzle):
        if len(rlist) != k:
            raise ValueError(
                f"Inconsistent rollout count at puzzle index {idx}: expected {k}, got {len(rlist)}."
            )

    valid_ms = sorted({int(m) for m in k_values if 1 <= int(m) <= k})
    if 1 not in valid_ms:
        valid_ms.insert(0, 1)
    if k not in valid_ms:
        valid_ms.append(k)

    per_m_scores: dict[int, list[float]] = {m: [] for m in valid_ms}
    all_entropies: list[float] = []
    legal_entropies: list[float] = []
    unique_preds_list: list[int] = []
    unique_legal_list: list[int] = []

    total_rollouts = 0
    strict_legal_rollouts = 0
    formatted_rollouts = 0
    total_think_words = 0

    cot_traces_evaluated = 0
    cot_traces_with_claims = 0
    cot_zero_viol_all = 0
    cot_zero_viol_with_claims = 0
    cot_claims_total = 0
    cot_claims_verified = 0

    split_buckets: dict[str, dict[str, list[float]]] = {}
    diff_buckets: dict[str, dict[str, list[float]]] = {}
    records: list[dict[str, Any]] = []

    for idx, (puzzle, completions) in enumerate(zip(puzzles, rollouts_per_puzzle, strict=True)):
        fen = puzzle["fen"]
        target = _resolve_puzzle_target(puzzle)
        board = chess.Board(fen)
        legal_set = {m.uci() for m in board.legal_moves}

        rollout_details: list[dict[str, Any]] = []
        all_buckets: list[str] = []
        legal_preds: list[str] = []
        correct_count = 0

        for rollout_idx, raw_output in enumerate(completions):
            total_rollouts += 1
            formatted = _has_valid_think_format(raw_output)
            if formatted:
                formatted_rollouts += 1
            think_words = _extract_think_word_count(raw_output)
            total_think_words += think_words

            parsed_move = parse_move_from_output(raw_output)
            is_legal = bool(parsed_move is not None and parsed_move in legal_set)
            if is_legal:
                strict_legal_rollouts += 1
                assert parsed_move is not None
                all_buckets.append(parsed_move)
                legal_preds.append(parsed_move)
            else:
                all_buckets.append(INVALID_MOVE_BUCKET)

            is_solved = bool(is_legal and target is not None and parsed_move == target)
            if is_solved:
                correct_count += 1

            cot_dict = None
            if verify_cot and raw_output and "<think>" in raw_output.lower():
                cot_res = verify_think_trace(fen, raw_output)
                cot_traces_evaluated += 1
                cot_claims_total += cot_res.claims_total
                cot_claims_verified += cot_res.claims_verified
                if cot_res.zero_violation:
                    cot_zero_viol_all += 1
                if cot_res.claims_total > 0:
                    cot_traces_with_claims += 1
                    if cot_res.zero_violation:
                        cot_zero_viol_with_claims += 1
                cot_dict = cot_res.to_dict()

            rollout_details.append(
                {
                    "rollout_index": rollout_idx,
                    "predicted_move": parsed_move,
                    "legal": is_legal,
                    "formatted": formatted,
                    "solved": is_solved,
                    "think_words": think_words,
                    "completion": raw_output,
                    "cot": cot_dict,
                }
            )

        puzzle_pass: dict[str, float] = {}
        for m in valid_ms:
            score_m = unbiased_pass_at_k(k, correct_count, m)
            per_m_scores[m].append(score_m)
            puzzle_pass[f"pass@{m}"] = round(score_m, 6)

        ent_all = shannon_entropy_bits(all_buckets)
        all_entropies.append(ent_all)
        ent_legal = shannon_entropy_bits(legal_preds) if legal_preds else None
        if ent_legal is not None:
            legal_entropies.append(ent_legal)

        unique_preds_list.append(len(set(all_buckets)))
        unique_legal_list.append(len(set(legal_preds)))

        split_group = str(puzzle.get("split_group") or "").strip()
        if split_group:
            sb = split_buckets.setdefault(split_group, {"p1": [], "pk": [], "ent": []})
            sb["p1"].append(per_m_scores[1][-1])
            sb["pk"].append(per_m_scores[k][-1])
            sb["ent"].append(ent_all)

        diff_band = _resolve_difficulty_band(puzzle)
        if diff_band:
            db = diff_buckets.setdefault(diff_band, {"p1": [], "pk": [], "ent": []})
            db["p1"].append(per_m_scores[1][-1])
            db["pk"].append(per_m_scores[k][-1])
            db["ent"].append(ent_all)

        records.append(
            {
                "puzzle_id": str(puzzle.get("puzzle_id") or f"idx_{idx:04d}"),
                "original_index": int(puzzle.get("original_index", idx)),
                "stratum": str(puzzle.get("stratum") or _stratum_key(puzzle)),
                "fen": fen,
                "target_move": target,
                "rating": puzzle.get("rating"),
                "split_group": puzzle.get("split_group"),
                "primary_theme": puzzle.get("primary_theme"),
                "difficulty_band": diff_band,
                "k": k,
                "correct_count": correct_count,
                "solved_any": correct_count >= 1,
                "pass_at_k": puzzle_pass,
                "move_entropy_all_bits": round(ent_all, 6),
                "move_entropy_legal_bits": (round(ent_legal, 6) if ent_legal is not None else None),
                "unique_predictions": len(set(all_buckets)),
                "unique_legal_moves": len(set(legal_preds)),
                "rollouts": rollout_details,
            }
        )

    pass_summary: dict[str, dict[str, Any]] = {}
    for m_idx, m in enumerate(valid_ms):
        vals = per_m_scores[m]
        mean_val = sum(vals) / n_puzzles
        ci_lo, ci_hi = bootstrap_mean_ci(
            vals,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed + m_idx,
        )
        entry: dict[str, Any] = {
            "mean": round(mean_val, 4),
            "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        if m == k:
            solved_any_count = sum(1 for r in records if r["solved_any"])
            w_lo, w_hi = wilson_ci(solved_any_count, n_puzzles - solved_any_count, 0)
            entry["solved_any"] = solved_any_count
            entry["wilson_ci_95"] = [round(w_lo, 4), round(w_hi, 4)]
        pass_summary[f"pass@{m}"] = entry

    ent_all_ci = bootstrap_mean_ci(
        all_entropies, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed + 100
    )
    ent_legal_ci = (
        bootstrap_mean_ci(
            legal_entropies, n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed + 101
        )
        if legal_entropies
        else (0.0, 0.0)
    )

    def _summarize_group_buckets(
        buckets: dict[str, dict[str, list[float]]],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key in sorted(buckets):
            b = buckets[key]
            cnt = len(b["p1"])
            out[key] = {
                "total": cnt,
                "pass@1": round(sum(b["p1"]) / cnt, 4) if cnt else 0.0,
                f"pass@{k}": round(sum(b["pk"]) / cnt, 4) if cnt else 0.0,
                "mean_move_entropy_all_bits": round(sum(b["ent"]) / cnt, 4) if cnt else 0.0,
            }
        return out

    summary = {
        "num_puzzles": n_puzzles,
        "k": k,
        "total_rollouts": total_rollouts,
        "pass_at_k": pass_summary,
        "entropy_unit": "bits",
        "mean_move_entropy_all_bits": round(sum(all_entropies) / n_puzzles, 4),
        "mean_move_entropy_all_ci_95": [round(ent_all_ci[0], 4), round(ent_all_ci[1], 4)],
        "mean_move_entropy_legal_bits": (
            round(sum(legal_entropies) / len(legal_entropies), 4) if legal_entropies else 0.0
        ),
        "mean_move_entropy_legal_ci_95": [round(ent_legal_ci[0], 4), round(ent_legal_ci[1], 4)],
        "puzzles_with_legal_rollout": len(legal_entropies),
        "mean_unique_predictions": round(sum(unique_preds_list) / n_puzzles, 3),
        "mean_unique_legal_moves": round(sum(unique_legal_list) / n_puzzles, 3),
        "strict_legal_rollouts": strict_legal_rollouts,
        "strict_legal_rate": round(strict_legal_rollouts / max(1, total_rollouts), 4),
        "format_adherence": round(formatted_rollouts / max(1, total_rollouts), 4),
        "mean_think_words": round(total_think_words / max(1, total_rollouts), 2),
        "cot_v2": {
            "traces_evaluated": cot_traces_evaluated,
            "traces_with_claims": cot_traces_with_claims,
            "claims_total": cot_claims_total,
            "claims_verified": cot_claims_verified,
            "claim_accuracy": (
                round(cot_claims_verified / cot_claims_total, 4) if cot_claims_total > 0 else None
            ),
            "zero_violation_rate_all": (
                round(cot_zero_viol_all / cot_traces_evaluated, 4)
                if cot_traces_evaluated > 0
                else None
            ),
            "zero_violation_rate_with_claims": (
                round(cot_zero_viol_with_claims / cot_traces_with_claims, 4)
                if cot_traces_with_claims > 0
                else None
            ),
            "mean_claims_per_trace": (
                round(cot_claims_total / cot_traces_evaluated, 2)
                if cot_traces_evaluated > 0
                else None
            ),
        },
        "by_split_group": _summarize_group_buckets(split_buckets),
        "by_difficulty": _summarize_group_buckets(diff_buckets),
        "puzzle_ids": [r["puzzle_id"] for r in records],
        "original_indices": [r["original_index"] for r in records],
    }
    return summary, records


def sample_completions_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
    seed: int = 42,
) -> list[str]:
    """Generate stochastic completions in left-padded batches without mutating tokenizer state."""
    if not prompts:
        return []
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0 for stochastic pass@k sampling, got {temperature}"
        )
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    has_padding_side = hasattr(tokenizer, "padding_side")
    orig_padding_side = getattr(tokenizer, "padding_side", None)
    has_pad_token_id = hasattr(tokenizer, "pad_token_id")
    orig_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    has_pad_token = hasattr(tokenizer, "pad_token")
    orig_pad_token = getattr(tokenizer, "pad_token", None)

    try:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            no_grad_ctx = torch.no_grad()
        except ImportError:
            no_grad_ctx = contextlib.nullcontext()

        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                tokenizer.pad_token_id = eos_id
            eos_tok = getattr(tokenizer, "eos_token", None)
            if getattr(tokenizer, "pad_token", None) is None and eos_tok is not None:
                tokenizer.pad_token = eos_tok

        all_outputs: list[str] = []
        prompt_list = list(prompts)
        total_batches = math.ceil(len(prompt_list) / batch_size)
        for batch_idx in range(total_batches):
            chunk = prompt_list[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True)
            device = getattr(model, "device", None)
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            input_ids = (
                inputs["input_ids"]
                if isinstance(inputs, dict) or hasattr(inputs, "__getitem__")
                else inputs.input_ids
            )
            padded_len = int(input_ids.shape[1])

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

            with no_grad_ctx:
                outputs = model.generate(**inputs, **gen_kwargs)

            for row in range(len(chunk)):
                gen_tokens = outputs[row][padded_len:]
                all_outputs.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(
                    f"[pass@k] Completed batch {batch_idx + 1}/{total_batches} "
                    f"({len(all_outputs)}/{len(prompt_list)} rollouts)",
                    flush=True,
                )
        return all_outputs
    finally:
        if has_padding_side:
            tokenizer.padding_side = orig_padding_side
        elif hasattr(tokenizer, "padding_side"):
            delattr(tokenizer, "padding_side")

        if has_pad_token_id:
            tokenizer.pad_token_id = orig_pad_token_id
        elif hasattr(tokenizer, "pad_token_id"):
            delattr(tokenizer, "pad_token_id")

        if has_pad_token:
            tokenizer.pad_token = orig_pad_token
        elif hasattr(tokenizer, "pad_token"):
            delattr(tokenizer, "pad_token")


def compare_pass_at_k_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    baseline_label: str = "E2a",
    candidate_label: str = "E3a",
) -> dict[str, Any]:
    """Build a structured side-by-side comparison report (e.g. E2a vs E3a)."""
    b_ids = baseline_report.get("puzzle_ids", [])
    c_ids = candidate_report.get("puzzle_ids", [])
    if b_ids != c_ids:
        raise ValueError("Baseline and candidate reports evaluated different puzzle slices.")

    b_pass = baseline_report.get("pass_at_k", {})
    c_pass = candidate_report.get("pass_at_k", {})
    pass_deltas: dict[str, dict[str, Any]] = {}
    for key in sorted(set(b_pass) & set(c_pass), key=lambda s: int(s.split("@")[1])):
        b_mean = float(b_pass[key]["mean"])
        c_mean = float(c_pass[key]["mean"])
        pass_deltas[key] = {
            baseline_label: b_pass[key],
            candidate_label: c_pass[key],
            "delta_pp": round((c_mean - b_mean) * 100.0, 2),
        }

    b_cot = baseline_report.get("cot_v2", {})
    c_cot = candidate_report.get("cot_v2", {})

    return {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _get_git_sha(),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_model_id": baseline_report.get("model_id"),
        "candidate_model_id": candidate_report.get("model_id"),
        "num_puzzles": baseline_report.get("num_puzzles"),
        "k": baseline_report.get("k"),
        "sampling": candidate_report.get("sampling"),
        "puzzle_ids_match": True,
        "pass_at_k_comparison": pass_deltas,
        "entropy_comparison": {
            "unit": "bits",
            "all_rollouts": {
                baseline_label: baseline_report.get("mean_move_entropy_all_bits"),
                candidate_label: candidate_report.get("mean_move_entropy_all_bits"),
                "delta_bits": round(
                    float(candidate_report.get("mean_move_entropy_all_bits", 0.0))
                    - float(baseline_report.get("mean_move_entropy_all_bits", 0.0)),
                    4,
                ),
            },
            "legal_only": {
                baseline_label: baseline_report.get("mean_move_entropy_legal_bits"),
                candidate_label: candidate_report.get("mean_move_entropy_legal_bits"),
                "delta_bits": round(
                    float(candidate_report.get("mean_move_entropy_legal_bits", 0.0))
                    - float(baseline_report.get("mean_move_entropy_legal_bits", 0.0)),
                    4,
                ),
            },
        },
        "strict_legality_comparison": {
            baseline_label: baseline_report.get("strict_legal_rate"),
            candidate_label: candidate_report.get("strict_legal_rate"),
            "delta_pp": round(
                (
                    float(candidate_report.get("strict_legal_rate", 0.0))
                    - float(baseline_report.get("strict_legal_rate", 0.0))
                )
                * 100.0,
                2,
            ),
        },
        "cot_v2_comparison": {
            baseline_label: b_cot,
            candidate_label: c_cot,
        },
        "baseline": baseline_report,
        "candidate": candidate_report,
    }


def run_pass_at_k_evaluation(
    model,
    tokenizer,
    *,
    model_id: str,
    puzzles_path: Path,
    num_puzzles: int = 200,
    k: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    include_ascii: bool = False,
    include_piece_list: bool = True,
    include_legal_moves: bool = False,
    enable_thinking: bool = True,
    seed: int = 42,
    n_bootstrap: int = 5000,
    sample_fn: Callable[..., list[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run full stratified pass@k evaluation and return ``(report, raw_records)``."""
    start_t = time.monotonic()
    all_puzzles = _load_jsonl(puzzles_path)
    sampled_puzzles = stratified_puzzle_sample(all_puzzles, n=num_puzzles, seed=seed)

    base_prompts = [
        build_chat_prompt(
            p["fen"],
            tokenizer=tokenizer,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=enable_thinking,
        )
        for p in sampled_puzzles
    ]
    expanded_prompts: list[str] = []
    for prompt in base_prompts:
        expanded_prompts.extend([prompt] * k)

    generator = sample_fn if sample_fn is not None else sample_completions_batched
    flat_completions = generator(
        model,
        tokenizer,
        expanded_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        seed=seed,
    )

    rollouts_per_puzzle = [
        flat_completions[i * k : (i + 1) * k] for i in range(len(sampled_puzzles))
    ]
    metrics, records = evaluate_pass_at_k_rollouts(
        sampled_puzzles,
        rollouts_per_puzzle,
        k_values=(1, 2, 4, 8),
        verify_cot=True,
        n_bootstrap=n_bootstrap,
        ci_level=0.95,
        seed=seed,
    )
    wall_s = round(time.monotonic() - start_t, 2)

    report = {
        "schema_version": "1.0",
        "git_sha": _get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "puzzles_source": str(puzzles_path),
        "hardware": _detect_hardware(),
        "wall_clock_seconds": wall_s,
        "sampling": {
            "do_sample": True,
            "k": k,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "seed": seed,
        },
        "prompt_config": {
            "enable_thinking": bool(enable_thinking),
            "include_ascii": bool(include_ascii),
            "include_piece_list": bool(include_piece_list),
            "include_legal_moves": bool(include_legal_moves),
        },
        **metrics,
    }
    return report, records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for pass@k puzzle evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate pass@1 vs pass@k, move entropy, and CoT V2 faithfulness on chess puzzles"
    )
    parser.add_argument("--model", type=str, required=True, help="Model HF ID or checkpoint path")
    parser.add_argument(
        "--puzzles",
        type=Path,
        required=True,
        help="Path to puzzle JSONL (e.g. test_puzzles_900.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for summary JSON report",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional output path for per-puzzle raw rollouts JSONL",
    )
    parser.add_argument("--num-puzzles", type=int, default=200, help="Stratified puzzle count")
    parser.add_argument("--k", type=int, default=8, help="Stochastic rollouts per puzzle")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top_p")
    parser.add_argument("--batch-size", type=int, default=16, help="Batched generation size")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens per rollout",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--include-ascii", action="store_true")
    parser.add_argument("--include-piece-list", action="store_true")
    parser.add_argument("--include-legal-moves", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from chessfm.load_model import load_model

    model, tokenizer = load_model(args.model, for_inference=True)
    report, records = run_pass_at_k_evaluation(
        model,
        tokenizer,
        model_id=args.model,
        puzzles_path=args.puzzles,
        num_puzzles=args.num_puzzles,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        include_ascii=args.include_ascii,
        include_piece_list=args.include_piece_list,
        include_legal_moves=args.include_legal_moves,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pass@k summary report to {args.output}", flush=True)

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_output.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved pass@k raw rollouts to {args.raw_output}", flush=True)


if __name__ == "__main__":
    main()
