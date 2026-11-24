#!/usr/bin/env python3
"""Generate a deterministic, anonymized 25-position Blind Human Audit Packet for Phase 4 (E5).

Selects 5 stratified positions per task type (25 total, fixed ``seed=2026``)
from ``test_coach_300.jsonl``, randomly permutes model labels independently for
each position (``Model A``, ``Model B``, ``Model C``, ...), and writes:

- ``blind_audit_25.md``: Human-gradeable packet with FEN, ASCII board,
  verified ``tactical_facts``, anonymized model responses, and grading rubric
  (zero real model-name leakage).
- ``blind_audit_25_key.json``: Unblinded label mapping and automated V1 /
  Strict V2 / spoiler / tactical-fact-coverage scores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from chessfm.coach.coach_rewards import clean_coach_completion, verify_claims_v1
from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.tactical_facts import (
    compute_tactical_fact_coverage,
    extract_tactical_facts,
)
from chessfm.coach.verifier import verify_claims
from chessfm.config import render_ascii_board
from chessfm.data_generation.distill_open_teacher import VALID_TASK_TYPES, infer_task_type
from chessfm.fens import read_jsonl

_ANON_LABELS: tuple[str, ...] = (
    "Model X",
    "Model Y",
    "Model Z",
    "Model W",
    "Model V",
    "Model U",
)


def _extract_response(row: dict[str, Any]) -> str:
    if isinstance(row.get("response"), str):
        return clean_coach_completion(row["response"])
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return clean_coach_completion(m["content"])
    return clean_coach_completion(str(row.get("completion") or ""))


def _select_stratified_positions(
    test_rows: list[dict[str, Any]],
    *,
    per_task: int = 5,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Select ``per_task`` positions per task type, mixing severities deterministically."""
    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TASK_TYPES}
    for row in test_rows:
        tt = infer_task_type(row)
        if tt in by_task:
            by_task[tt].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for tt in VALID_TASK_TYPES:
        pool = list(by_task[tt])
        if len(pool) < per_task:
            raise ValueError(
                f"Task {tt!r} has only {len(pool)} rows, fewer than requested per_task={per_task}"
            )
        # Prioritize a deterministic mix of blunder / mistake / inaccuracy when available.
        blunders = [r for r in pool if str(r.get("severity", "")).lower() == "blunder"]
        mistakes = [r for r in pool if str(r.get("severity", "")).lower() == "mistake"]
        others = [r for r in pool if r not in blunders and r not in mistakes]
        rng.shuffle(blunders)
        rng.shuffle(mistakes)
        rng.shuffle(others)

        chosen: list[dict[str, Any]] = []
        for bucket, quota in ((blunders, 2), (mistakes, 1)):
            chosen.extend(bucket[: min(quota, per_task - len(chosen))])
        remainder = [r for r in (blunders[2:] + mistakes[1:] + others) if r not in chosen]
        rng.shuffle(remainder)
        chosen.extend(remainder[: per_task - len(chosen)])
        selected.extend(chosen[:per_task])
    return selected


def build_blind_audit(
    model_outputs: dict[str, list[dict[str, Any]]],
    test_rows: list[dict[str, Any]],
    *,
    per_task: int = 5,
    seed: int = 2026,
) -> tuple[str, dict[str, Any]]:
    """Build anonymized Markdown audit packet and unblinded JSON key.

    Returns ``(markdown_text, key_dict)``.
    """
    if not model_outputs:
        raise ValueError("model_outputs must contain at least one model")
    model_names = sorted(model_outputs.keys())
    if len(model_names) > len(_ANON_LABELS):
        raise ValueError(f"At most {len(_ANON_LABELS)} models supported, got {len(model_names)}")

    # Index each model's outputs by (fen, task_type).
    indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for name in model_names:
        m_map: dict[tuple[str, str], dict[str, Any]] = {}
        for r in model_outputs[name]:
            fen = r.get("fen")
            if isinstance(fen, str) and fen.strip():
                m_map[(fen, infer_task_type(r))] = r
        indexed[name] = m_map

    # Restrict to test rows present across all models.
    common_rows = [
        r
        for r in test_rows
        if isinstance(r.get("fen"), str)
        and all((r["fen"], infer_task_type(r)) in indexed[m] for m in model_names)
    ]
    chosen_rows = _select_stratified_positions(common_rows, per_task=per_task, seed=seed)

    rng = random.Random(seed)
    anon_labels = list(_ANON_LABELS[: len(model_names)])

    md_lines: list[str] = [
        "# Blind Human Coaching Audit Packet (N=25)",
        "",
        "Each position below shows the verified board state, deterministic `python-chess` "
        "tactical facts, and anonymized responses (`"
        + " / ".join(anon_labels)
        + "`) whose label assignment is freshly randomized per position.",
        "",
        "## Grading Rubric (Score Each Model Independently)",
        "- **Factual Accuracy (1–5):** Every piece, square, attack, check, and continuation matches the board.",
        "- **Pedagogical Clarity (1–5):** Concrete, rating-appropriate explanation of *why* the move works or fails.",
        "- **Specificity / Anti-Slop (1–5):** Position-specific prose rather than generic filler.",
        "- **Spoiler Compliance (Pass/Fail on L1/L2):** Never reveals the best move or both origin+dest squares on `socratic_l1`/`socratic_l2`.",
        "",
        "---",
        "",
    ]

    key_entries: list[dict[str, Any]] = []
    for idx, row in enumerate(chosen_rows, start=1):
        fen = row["fen"]
        task_type = infer_task_type(row)
        best_move = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
        student_move = str(row.get("student_move") or "").strip()
        pv = row.get("pv") or []
        cp_loss = row.get("cp_loss")

        tf = extract_tactical_facts(
            fen,
            student_move=student_move or None,
            best_move=best_move,
            pv=pv if isinstance(pv, list) else None,
            cp_loss=cp_loss,
        )
        ascii_board = render_ascii_board(fen)

        perm = list(model_names)
        rng.shuffle(perm)
        label_to_model = dict(zip(anon_labels, perm, strict=True))

        md_lines.extend(
            [
                f"## Position #{idx} — `{task_type}` (Student Elo: {row.get('student_elo', 1200)}, Severity: `{row.get('severity', 'n/a')}`)",
                "",
                f"- **FEN:** `{fen}`",
                f"- **Side to Move:** `{tf['side_to_move']}`",
                f"- **Best Move:** `{tf['best_move_san']}` (`{tf['best_move_uci']}`)"
                + (
                    f" | **Student Move:** `{tf['student_move_san']}` (`{tf['student_move_uci']}`, `cp_loss={cp_loss}`)"
                    if tf["student_move_san"]
                    else ""
                ),
                f"- **Verified Tactical Motif(s):** `{', '.join(tf['tactical_motifs'])}` | **Gives Check:** `{tf['gives_check']}` | **Center Tension:** `{tf['has_center_tension']}`",
                f"- **Pieces Attacked by Best Move:** `{tf['pieces_attacked_by_best_move'] or 'none'}`",
                f"- **Verified PV:** `{' '.join(tf['best_pv_san'])}`",
                "",
                "```text",
                ascii_board,
                "```",
                "",
            ]
        )

        auto_scores: dict[str, dict[str, Any]] = {}
        for label, real_model in label_to_model.items():
            out_row = indexed[real_model][(fen, task_type)]
            resp = _extract_response(out_row)
            v1 = verify_claims_v1(fen, resp)
            v2 = verify_claims(fen, resp, engine=None, check_boilerplate=True)
            spoiler = False
            if task_type in ("socratic_l1", "socratic_l2") and best_move:
                spoiler = bool(contains_move_spoiler(resp, best_move, fen))
            cov = compute_tactical_fact_coverage(resp, tf, task_type=task_type)

            auto_scores[label] = {
                "model": real_model,
                "v1_passed": bool(v1["passed"]),
                "v2_passed": bool(v2["passed"]),
                "spoiler_free": not spoiler,
                "tactical_fact_coverage": cov,
                "violations": [v["type"] for v in v2["violations"]],
            }

            md_lines.extend(
                [
                    f"### {label}",
                    f"> {resp}",
                    "",
                ]
            )

        md_lines.extend(["---", ""])
        key_entries.append(
            {
                "position_index": idx,
                "fen": fen,
                "task_type": task_type,
                "best_move_uci": best_move,
                "best_move_san": tf["best_move_san"],
                "student_move_uci": student_move or None,
                "severity": row.get("severity"),
                "label_to_model": label_to_model,
                "automated_scores": auto_scores,
            }
        )

    markdown = "\n".join(md_lines)
    key_dict = {
        "seed": seed,
        "per_task": per_task,
        "total_positions": len(key_entries),
        "models": model_names,
        "anon_labels": anon_labels,
        "positions": key_entries,
    }
    return markdown, key_dict


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected 'model_name=path.jsonl', got {spec!r}")
    name, path_str = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty model name in spec {spec!r}")
    return name, Path(path_str.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build anonymized 25-position blind coach audit packet."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Model output JSONLs as label=path.jsonl (e.g. arm_a=results/.../arm_a.jsonl)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data_generation/full/phase2/test_coach_300.jsonl"),
        help="Frozen coaching benchmark JSONL",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/phase2/e5/blind_audit_25.md"),
        help="Output path for anonymized Markdown audit packet",
    )
    parser.add_argument(
        "--output-key",
        type=Path,
        default=Path("results/phase2/e5/blind_audit_25_key.json"),
        help="Output path for unblinded JSON key",
    )
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    model_outputs: dict[str, list[dict[str, Any]]] = {}
    for spec in args.inputs:
        name, path = parse_model_spec(spec)
        model_outputs[name] = read_jsonl(path)

    test_rows = read_jsonl(args.benchmark)
    md_text, key_dict = build_blind_audit(
        model_outputs,
        test_rows,
        per_task=args.per_task,
        seed=args.seed,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_key.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")
    args.output_key.write_text(json.dumps(key_dict, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote blind audit packet ({key_dict['total_positions']} positions, "
        f"{len(key_dict['models'])} models) -> {args.output_md} & {args.output_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
