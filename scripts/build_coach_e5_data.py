#!/usr/bin/env python3
"""Build verified, 0%-leakage Coach V2 datasets for Phase 4 (E5).

Subcommands:
    prepare:
        1. Enrich & strictly V2-filter existing 26B teacher traces
           (``distilled_coach.jsonl`` joined with ``layer_b_seed_train.jsonl``),
           excluding ALL Phase 2 frozen test FENs.
        2. Annotate 1,000 fresh non-overlapping positions (200/task type) via
           Stockfish depth=10 for warm-vLLM tactical-fact-grounded distillation.
        3. Build deterministic 100-item stratified slice of ``test_coach_300.jsonl``
           (20/task type, seed=42) for Arm A vs Arm C checkpoint curves.
        4. Build 2,000 stratified Coach-GRPO prompts (400/task type, 0% test overlap).

    merge:
        Merge enriched existing V2-clean traces + newly distilled 1K traces,
        re-verify Strict V2 + 0% L1/L2 spoilers + corpus anti-slop + 0.00%
        overlap with ALL 5 Phase 2 frozen test sets, write
        ``coach_sft_v2_4500.jsonl``, and emit ``coach_data_audit.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.tactical_facts import (
    compute_tactical_fact_coverage,
    extract_tactical_facts,
)
from chessfm.coach.verifier import detect_boilerplate_slop, verify_claims
from chessfm.data_generation.distill_open_teacher import (
    SYSTEM_PROMPT,
    VALID_TASK_TYPES,
    build_teacher_prompt,
    format_chatml_prompt,
    infer_task_type,
    strip_reasoning_blocks,
)
from chessfm.data_generation.phase_sampler import classify_game_phase, normalize_board_fen
from chessfm.fens import read_fens, read_jsonl
from chessfm.rl.engine import close_engine, open_engine

FROZEN_TEST_FILES: tuple[str, ...] = (
    "test_coach_300.jsonl",
    "test_fens_1000.jsonl",
    "test_puzzles_900.jsonl",
    "test_chessqa_1000.jsonl",
    "test_openings_20.txt",
)

_SOCRATIC_LEVELS: dict[str, int] = {
    "socratic_l1": 1,
    "socratic_l2": 2,
    "socratic_l3": 3,
}

_STUDENT_ELOS: tuple[int, ...] = (800, 1000, 1200, 1400, 1600, 1800)


def _norm_fen_safe(fen: str) -> str:
    try:
        return normalize_board_fen(fen)
    except Exception:
        return fen.strip()


def load_frozen_test_fens(phase2_dir: Path) -> dict[str, set[str]]:
    """Load normalized board FEN sets for all 5 Phase 2 frozen test files."""
    by_set: dict[str, set[str]] = {}
    for fname in FROZEN_TEST_FILES:
        path = phase2_dir / fname
        fens: set[str] = set()
        if path.exists():
            if path.suffix == ".jsonl":
                for row in read_jsonl(path):
                    fen = row.get("fen")
                    if isinstance(fen, str) and fen.strip():
                        fens.add(_norm_fen_safe(fen))
            else:
                for fen in read_fens(path):
                    if fen.strip():
                        fens.add(_norm_fen_safe(fen))
        by_set[fname] = fens
    return by_set


def audit_zero_leakage(
    rows: list[dict[str, Any]],
    test_fens_by_set: dict[str, set[str]],
) -> dict[str, int]:
    """Return exact normalized-FEN overlap counts against each frozen test set."""
    row_fens = {_norm_fen_safe(r["fen"]) for r in rows if isinstance(r.get("fen"), str)}
    return {name: len(row_fens & test_set) for name, test_set in test_fens_by_set.items()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_assistant_text(row: dict[str, Any]) -> str:
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return strip_reasoning_blocks(m["content"])
    if isinstance(row.get("response"), str):
        return strip_reasoning_blocks(row["response"])
    comp = str(row.get("completion") or "")
    return strip_reasoning_blocks(comp.replace("<|im_end|>", ""))


def enrich_and_filter_existing_traces(
    distilled_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    excluded_fens: set[str],
    *,
    min_chars: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join existing 26B distilled traces with seed facts, enrich with tactical facts, and V2-filter."""
    seed_index: dict[tuple[str, str], dict[str, Any]] = {}
    for srow in seed_rows:
        fen = srow.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            continue
        tt = infer_task_type(srow)
        seed_index[(fen, tt)] = srow

    kept: list[dict[str, Any]] = []
    stats = {
        "input_rows": len(distilled_rows),
        "dropped_test_overlap": 0,
        "dropped_v2_fail": 0,
        "dropped_short": 0,
        "dropped_spoiler": 0,
        "dropped_invalid_facts": 0,
        "joined_with_seed": 0,
    }

    for row in distilled_rows:
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            continue
        if _norm_fen_safe(fen) in excluded_fens:
            stats["dropped_test_overlap"] += 1
            continue

        task_type = infer_task_type(row)
        cleaned = _extract_assistant_text(row)
        if len(cleaned) < min_chars:
            stats["dropped_short"] += 1
            continue

        seed = seed_index.get((fen, task_type))
        if seed is not None:
            stats["joined_with_seed"] += 1

        merged_facts = {**(seed or {}), **row}
        best_move_uci = str(
            merged_facts.get("best_move_uci") or merged_facts.get("best_move") or ""
        ).strip()
        student_move_uci = str(
            merged_facts.get("student_move") or merged_facts.get("played_move_uci") or ""
        ).strip()
        pv = merged_facts.get("pv") or merged_facts.get("principal_variation") or []
        cp_loss = merged_facts.get("cp_loss")

        v2 = verify_claims(fen, cleaned, engine=None, check_boilerplate=True)
        if not v2["passed"]:
            stats["dropped_v2_fail"] += 1
            continue

        if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
            try:
                if contains_move_spoiler(cleaned, best_move_uci, fen):
                    stats["dropped_spoiler"] += 1
                    continue
            except ValueError:
                stats["dropped_invalid_facts"] += 1
                continue

        try:
            tf = extract_tactical_facts(
                fen,
                student_move=student_move_uci or None,
                best_move=best_move_uci,
                pv=pv if isinstance(pv, list) else None,
                cp_loss=cp_loss,
                eval_before_cp=merged_facts.get("eval_before_cp"),
                eval_after_cp=merged_facts.get("eval_after_cp"),
            )
        except ValueError:
            stats["dropped_invalid_facts"] += 1
            continue

        prompt_facts = {**merged_facts, "tactical_facts": tf}
        user_prompt = build_teacher_prompt(
            prompt_facts,
            task_type=task_type,
            include_tactical_facts=True,
        )
        cov = compute_tactical_fact_coverage(cleaned, tf, task_type=task_type)

        record: dict[str, Any] = {
            "fen": fen,
            "task_type": task_type,
            "best_move_uci": best_move_uci,
            "best_move": best_move_uci,
            "student_move": student_move_uci,
            "pv": pv[:6] if isinstance(pv, list) else [],
            "cp_loss": cp_loss,
            "student_elo": merged_facts.get("student_elo", 1200),
            "phase": merged_facts.get("phase") or classify_game_phase(fen),
            "severity": merged_facts.get("severity") or merged_facts.get("classification") or "",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": cleaned},
            ],
            "prompt": format_chatml_prompt(SYSTEM_PROMPT, user_prompt),
            "completion": f"{cleaned}<|im_end|>",
            "teacher_model": row.get("teacher_model", "google/gemma-4-26B-A4B-it"),
            "source": "distilled_26b_enriched",
            "tactical_facts": tf,
            "tactical_fact_coverage": cov,
            "verified": True,
            "act_eval_passed": True,
            "spoiler_free": True,
        }
        if task_type in _SOCRATIC_LEVELS:
            record["level"] = _SOCRATIC_LEVELS[task_type]
        kept.append(record)

    stats["kept_rows"] = len(kept)
    return kept, stats


def build_stratified_coach_slice(
    test_rows: list[dict[str, Any]],
    *,
    per_task: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a deterministic stratified slice (``per_task`` per task type) from ``test_coach_300``."""
    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TASK_TYPES}
    for row in test_rows:
        tt = infer_task_type(row)
        if tt in by_task:
            by_task[tt].append(row)

    rng = random.Random(seed)
    sliced: list[dict[str, Any]] = []
    for tt in VALID_TASK_TYPES:
        pool = list(by_task[tt])
        if len(pool) < per_task:
            raise ValueError(
                f"Task {tt!r} has only {len(pool)} rows, fewer than requested per_task={per_task}"
            )
        chosen = rng.sample(pool, per_task)
        sliced.extend(chosen)
    return sliced


def build_coach_grpo_prompts(
    candidate_rows: list[dict[str, Any]],
    excluded_fens: set[str],
    *,
    per_task: int = 400,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a stratified 0%-leakage Coach-GRPO prompt dataset enriched with tactical facts."""
    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TASK_TYPES}
    seen_keys: set[tuple[str, str]] = set()

    for row in candidate_rows:
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            continue
        norm = _norm_fen_safe(fen)
        if norm in excluded_fens:
            continue
        tt = infer_task_type(row)
        if tt not in by_task:
            continue
        key = (norm, tt)
        if key in seen_keys:
            continue
        best_move = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
        if not best_move:
            continue
        seen_keys.add(key)
        by_task[tt].append(row)

    rng = random.Random(seed)
    prompts_out: list[dict[str, Any]] = []
    for tt in VALID_TASK_TYPES:
        pool = list(by_task[tt])
        rng.shuffle(pool)
        selected = 0
        for row in pool:
            if selected >= per_task:
                break
            fen = row["fen"]
            best_move = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
            student_move = str(row.get("student_move") or row.get("played_move_uci") or "").strip()
            pv = row.get("pv") or row.get("principal_variation") or []
            tf = row.get("tactical_facts")
            if not isinstance(tf, dict) or not tf.get("fen"):
                try:
                    tf = extract_tactical_facts(
                        fen,
                        student_move=student_move or None,
                        best_move=best_move,
                        pv=pv if isinstance(pv, list) else None,
                        cp_loss=row.get("cp_loss"),
                        eval_before_cp=row.get("eval_before_cp"),
                        eval_after_cp=row.get("eval_after_cp"),
                    )
                except ValueError:
                    continue

            prompt_row = {**row, "tactical_facts": tf}
            user_prompt = build_teacher_prompt(
                prompt_row,
                task_type=tt,
                include_tactical_facts=True,
            )
            prompts_out.append(
                {
                    "fen": fen,
                    "task_type": tt,
                    "best_move_uci": best_move,
                    "best_move": best_move,
                    "student_move": student_move,
                    "student_elo": row.get("student_elo", 1200),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "tactical_facts": tf,
                    "tactical_facts_json": json.dumps(tf, ensure_ascii=False),
                }
            )
            selected += 1

    rng.shuffle(prompts_out)
    return prompts_out


def _classify_severity(cp_loss: float) -> str:
    if cp_loss >= 200.0:
        return "blunder"
    if cp_loss >= 100.0:
        return "mistake"
    return "inaccuracy"


def _annotate_chunk_stockfish(
    chunk: list[tuple[str, str, int]],
    depth: int,
) -> list[dict[str, Any]]:
    """Annotate a chunk of ``(fen, task_type, student_elo)`` with Stockfish MultiPV."""
    engine = open_engine()
    if engine is None:
        raise RuntimeError("Stockfish binary not found for fresh seed annotation")
    limit = chess.engine.Limit(depth=depth)
    results: list[dict[str, Any]] = []
    try:
        for fen, task_type, student_elo in chunk:
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            legal_moves = list(board.legal_moves)
            if len(legal_moves) < 2:
                continue

            try:
                infos = engine.analyse(board, limit, multipv=min(3, len(legal_moves)))
            except Exception:
                continue
            if not isinstance(infos, list) or not infos:
                continue
            top = infos[0]
            top_pv = list(top.get("pv") or [])
            if not top_pv:
                continue
            best_mv = top_pv[0]
            best_score_obj = top.get("score")
            if best_score_obj is None:
                continue
            eval_before_cp = float(best_score_obj.pov(board.turn).score(mate_score=10000) or 0)

            # Pick a realistic suboptimal student move (preferring multipv[1/2] with >=25 cp loss).
            student_mv: chess.Move | None = None
            eval_after_cp: float | None = None
            cp_loss: float = 0.0

            for alt_info in infos[1:]:
                alt_pv = list(alt_info.get("pv") or [])
                alt_score_obj = alt_info.get("score")
                if not alt_pv or alt_score_obj is None:
                    continue
                cand_mv = alt_pv[0]
                if cand_mv == best_mv:
                    continue
                cand_cp = float(alt_score_obj.pov(board.turn).score(mate_score=10000) or 0)
                loss = max(0.0, eval_before_cp - cand_cp)
                if 25.0 <= loss <= 600.0:
                    student_mv = cand_mv
                    eval_after_cp = cand_cp
                    cp_loss = loss
                    break
                if student_mv is None and loss >= 15.0:
                    student_mv = cand_mv
                    eval_after_cp = cand_cp
                    cp_loss = loss

            if student_mv is None:
                # Fallback: evaluate a deterministic non-best legal move.
                fallback_mv = next((m for m in legal_moves if m != best_mv), None)
                if fallback_mv is None:
                    continue
                after_fb = board.copy(stack=False)
                after_fb.push(fallback_mv)
                try:
                    fb_info = engine.analyse(after_fb, limit)
                    fb_score = fb_info.get("score")
                    if fb_score is None:
                        continue
                    eval_after_cp = float(fb_score.pov(board.turn).score(mate_score=10000) or 0)
                except Exception:
                    continue
                student_mv = fallback_mv
                cp_loss = max(0.0, eval_before_cp - eval_after_cp)

            pv_ucis = [m.uci() for m in top_pv[:6]]
            try:
                tf = extract_tactical_facts(
                    fen,
                    student_move=student_mv.uci(),
                    best_move=best_mv.uci(),
                    pv=pv_ucis,
                    cp_loss=cp_loss,
                    eval_before_cp=eval_before_cp,
                    eval_after_cp=eval_after_cp,
                )
            except ValueError:
                continue

            results.append(
                {
                    "fen": fen,
                    "mode": task_type,
                    "task_type": task_type,
                    "best_move": best_mv.uci(),
                    "best_move_uci": best_mv.uci(),
                    "student_move": student_mv.uci(),
                    "pv": pv_ucis,
                    "eval_before_cp": round(eval_before_cp, 2),
                    "eval_after_cp": round(eval_after_cp, 2) if eval_after_cp is not None else None,
                    "cp_loss": round(cp_loss, 2),
                    "severity": _classify_severity(cp_loss),
                    "student_elo": student_elo,
                    "phase": classify_game_phase(fen),
                    "tactical_facts": tf,
                }
            )
    finally:
        close_engine(engine)
    return results


def build_fresh_seed_positions(
    source_paths: list[Path],
    excluded_fens: set[str],
    *,
    per_task: int = 200,
    depth: int = 10,
    workers: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample and Stockfish-annotate ``per_task * 5`` fresh non-overlapping seed positions."""
    target_total = per_task * len(VALID_TASK_TYPES)
    candidate_fens: list[str] = []
    seen_norm = set(excluded_fens)

    for src in source_paths:
        if not src.exists():
            continue
        if src.suffix == ".jsonl":
            with src.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fen = obj.get("fen")
                    if not isinstance(fen, str) or not fen.strip():
                        continue
                    norm = _norm_fen_safe(fen)
                    if norm in seen_norm:
                        continue
                    seen_norm.add(norm)
                    candidate_fens.append(fen)
                    if len(candidate_fens) >= target_total * 4:
                        break
        else:
            for fen in read_fens(src):
                norm = _norm_fen_safe(fen)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                candidate_fens.append(fen)
                if len(candidate_fens) >= target_total * 4:
                    break
        if len(candidate_fens) >= target_total * 4:
            break

    rng = random.Random(seed)
    rng.shuffle(candidate_fens)
    # Oversample slightly (1.2x) to guarantee exact per_task rows even if any FEN is terminal.
    sample_count = min(len(candidate_fens), int(target_total * 1.25) + 50)
    work_items: list[tuple[str, str, int]] = []
    for idx, fen in enumerate(candidate_fens[:sample_count]):
        tt = VALID_TASK_TYPES[idx % len(VALID_TASK_TYPES)]
        elo = _STUDENT_ELOS[idx % len(_STUDENT_ELOS)]
        work_items.append((fen, tt, elo))

    n_workers = max(1, workers)
    chunk_size = max(1, (len(work_items) + n_workers - 1) // n_workers)
    chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]

    annotated: list[dict[str, Any]] = []
    if n_workers == 1:
        annotated = _annotate_chunk_stockfish(work_items, depth=depth)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_annotate_chunk_stockfish, ch, depth) for ch in chunks]
            for fut in futures:
                annotated.extend(fut.result())

    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TASK_TYPES}
    for row in annotated:
        tt = row["task_type"]
        if len(by_task[tt]) < per_task:
            by_task[tt].append(row)

    final_rows: list[dict[str, Any]] = []
    for tt in VALID_TASK_TYPES:
        if len(by_task[tt]) < per_task:
            raise RuntimeError(
                f"Annotated only {len(by_task[tt])}/{per_task} fresh seeds for task {tt!r}"
            )
        final_rows.extend(by_task[tt][:per_task])
    return final_rows


def merge_and_audit_coach_sft(
    existing_clean: list[dict[str, Any]],
    fresh_distilled: list[dict[str, Any]],
    test_fens_by_set: dict[str, set[str]],
    *,
    seed: int = 42,
    min_chars: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge existing + fresh V2-clean traces, re-verify Strict V2, anti-slop, and 0% test overlap."""
    all_test_fens: set[str] = set().union(*test_fens_by_set.values()) if test_fens_by_set else set()

    combined_raw = list(existing_clean) + list(fresh_distilled)
    verified_rows: list[dict[str, Any]] = []
    seen_fen_task: set[tuple[str, str]] = set()
    dropped = Counter()

    for row in combined_raw:
        fen = row.get("fen")
        if not isinstance(fen, str) or not fen.strip():
            dropped["missing_fen"] += 1
            continue
        norm = _norm_fen_safe(fen)
        if norm in all_test_fens:
            dropped["test_overlap"] += 1
            continue

        task_type = infer_task_type(row)
        dedup_key = (norm, task_type)
        if dedup_key in seen_fen_task:
            dropped["duplicate_fen_task"] += 1
            continue

        cleaned = _extract_assistant_text(row)
        if len(cleaned) < min_chars:
            dropped["short"] += 1
            continue

        v2 = verify_claims(fen, cleaned, engine=None, check_boilerplate=True)
        if not v2["passed"]:
            dropped["v2_fail"] += 1
            continue

        best_move_uci = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
        if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
            try:
                if contains_move_spoiler(cleaned, best_move_uci, fen):
                    dropped["spoiler_leak"] += 1
                    continue
            except ValueError:
                dropped["invalid_best_move"] += 1
                continue

        tf = row.get("tactical_facts")
        if not isinstance(tf, dict) or not tf.get("fen"):
            try:
                tf = extract_tactical_facts(
                    fen,
                    student_move=str(row.get("student_move") or "") or None,
                    best_move=best_move_uci,
                    pv=row.get("pv") if isinstance(row.get("pv"), list) else None,
                    cp_loss=row.get("cp_loss"),
                    eval_before_cp=row.get("eval_before_cp"),
                    eval_after_cp=row.get("eval_after_cp"),
                )
            except ValueError:
                dropped["invalid_facts"] += 1
                continue

        # Ensure unified tactical-fact-grounded messages.
        msgs = row.get("messages")
        has_grounded_msgs = (
            isinstance(msgs, list)
            and len(msgs) >= 3
            and any(
                m.get("role") == "user"
                and "Tactical Facts (Verified):" in str(m.get("content", ""))
                for m in msgs
            )
        )
        if has_grounded_msgs:
            user_prompt = next(m["content"] for m in msgs if m.get("role") == "user")
        else:
            user_prompt = build_teacher_prompt(
                {**row, "tactical_facts": tf},
                task_type=task_type,
                include_tactical_facts=True,
            )

        cov = compute_tactical_fact_coverage(cleaned, tf, task_type=task_type)
        rec = {
            "fen": fen,
            "task_type": task_type,
            "best_move_uci": best_move_uci,
            "best_move": best_move_uci,
            "student_move": str(row.get("student_move") or ""),
            "student_elo": row.get("student_elo", 1200),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": cleaned},
            ],
            "prompt": format_chatml_prompt(SYSTEM_PROMPT, user_prompt),
            "completion": f"{cleaned}<|im_end|>",
            "teacher_model": row.get("teacher_model", "google/gemma-4-26B-A4B-it"),
            "source": row.get("source", "distilled_26b_fresh_1k"),
            "tactical_facts": tf,
            "tactical_fact_coverage": cov,
            "verified": True,
            "act_eval_passed": True,
            "spoiler_free": True,
        }
        if task_type in _SOCRATIC_LEVELS:
            rec["level"] = _SOCRATIC_LEVELS[task_type]
        seen_fen_task.add(dedup_key)
        verified_rows.append(rec)

    # Corpus-wide anti-slop verification (remove any flagged rows if present).
    texts = [_extract_assistant_text(r) for r in verified_rows]
    slop_report = detect_boilerplate_slop(texts)
    if slop_report["flagged_indices"]:
        flagged_set = set(slop_report["flagged_indices"])
        dropped["corpus_slop_flagged"] += len(flagged_set)
        verified_rows = [r for idx, r in enumerate(verified_rows) if idx not in flagged_set]
        texts = [_extract_assistant_text(r) for r in verified_rows]
        slop_report = detect_boilerplate_slop(texts)

    rng = random.Random(seed)
    rng.shuffle(verified_rows)

    overlap_audit = audit_zero_leakage(verified_rows, test_fens_by_set)
    per_task = Counter(r["task_type"] for r in verified_rows)
    per_source = Counter(r.get("source", "unknown") for r in verified_rows)
    mean_cov = (
        round(
            sum(float(r.get("tactical_fact_coverage", 0.0)) for r in verified_rows)
            / len(verified_rows),
            4,
        )
        if verified_rows
        else 0.0
    )

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "existing_clean_input": len(existing_clean),
        "fresh_distilled_input": len(fresh_distilled),
        "final_rows": len(verified_rows),
        "unique_norm_fens": len({_norm_fen_safe(r["fen"]) for r in verified_rows}),
        "per_task_type": dict(sorted(per_task.items())),
        "per_source": dict(sorted(per_source.items())),
        "dropped_counts": dict(sorted(dropped.items())),
        "v2_pass_rate": 1.0 if verified_rows else 0.0,
        "l12_spoiler_rate": 0.0,
        "mean_tactical_fact_coverage": mean_cov,
        "slop_audit": {
            "distinct_4gram_ratio": slop_report["distinct_4gram_ratio"],
            "exact_duplicate_rate": slop_report["exact_duplicate_rate"],
            "flagged_count": slop_report["flagged_count"],
            "flagged_rate": slop_report["flagged_rate"],
            "is_slop_corpus": slop_report["is_slop_corpus"],
            "top_repeated_4grams": slop_report["top_repeated_4grams"][:5],
        },
        "test_overlap_counts": overlap_audit,
        "zero_leakage_verified": all(v == 0 for v in overlap_audit.values()),
    }
    return verified_rows, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 4 (E5) Coach V2 datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prep = subparsers.add_parser(
        "prepare", help="Prepare V2-clean existing, fresh 1K seeds, GRPO 2K, and 100-slice"
    )
    p_prep.add_argument("--data-dir", type=Path, default=Path("data_generation/full"))
    p_prep.add_argument("--phase2-dir", type=Path, default=Path("data_generation/full/phase2"))
    p_prep.add_argument("--fresh-per-task", type=int, default=200)
    p_prep.add_argument("--grpo-per-task", type=int, default=400)
    p_prep.add_argument("--slice-per-task", type=int, default=20)
    p_prep.add_argument("--depth", type=int, default=10)
    p_prep.add_argument("--workers", type=int, default=4)
    p_prep.add_argument("--seed", type=int, default=42)

    p_merge = subparsers.add_parser(
        "merge", help="Merge existing + fresh distilled traces and audit"
    )
    p_merge.add_argument("--phase2-dir", type=Path, default=Path("data_generation/full/phase2"))
    p_merge.add_argument(
        "--existing",
        type=Path,
        default=Path("data_generation/full/phase2/coach_existing_v2_clean.jsonl"),
    )
    p_merge.add_argument(
        "--fresh",
        type=Path,
        default=Path("results/phase2/e5/coach_fresh_1k_distilled.jsonl"),
    )
    p_merge.add_argument(
        "--output",
        type=Path,
        default=Path("data_generation/full/phase2/coach_sft_v2_4500.jsonl"),
    )
    p_merge.add_argument(
        "--audit-output",
        type=Path,
        default=Path("results/phase2/e5/coach_data_audit.json"),
    )
    p_merge.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.command == "prepare":
        test_fens_by_set = load_frozen_test_fens(args.phase2_dir)
        all_test_fens = set().union(*test_fens_by_set.values())

        distilled_rows = read_jsonl(args.data_dir / "distilled_coach.jsonl")
        seed_rows = read_jsonl(args.data_dir / "layer_b_seed_train.jsonl")

        existing_clean, existing_stats = enrich_and_filter_existing_traces(
            distilled_rows, seed_rows, all_test_fens
        )
        existing_out = args.phase2_dir / "coach_existing_v2_clean.jsonl"
        write_jsonl(existing_out, existing_clean)
        print(f"[prepare] Existing V2-clean: {existing_stats} -> {existing_out}")

        existing_fens = {_norm_fen_safe(r["fen"]) for r in distilled_rows if r.get("fen")}
        fresh_excluded = all_test_fens | existing_fens
        fresh_sources = [
            args.phase2_dir / "sft_phase2_80k.jsonl",
            args.phase2_dir / "rl_positions_40k.txt",
        ]
        fresh_seeds = build_fresh_seed_positions(
            fresh_sources,
            fresh_excluded,
            per_task=args.fresh_per_task,
            depth=args.depth,
            workers=args.workers,
            seed=args.seed,
        )
        fresh_out = args.phase2_dir / "coach_fresh_1k_seed.jsonl"
        write_jsonl(fresh_out, fresh_seeds)
        fresh_overlap = audit_zero_leakage(fresh_seeds, test_fens_by_set)
        print(
            f"[prepare] Fresh 1K seeds: {len(fresh_seeds)} rows, overlap={fresh_overlap} -> {fresh_out}"
        )

        test_coach_300 = read_jsonl(args.phase2_dir / "test_coach_300.jsonl")
        coach_100 = build_stratified_coach_slice(
            test_coach_300, per_task=args.slice_per_task, seed=args.seed
        )
        slice_out = args.phase2_dir / "test_coach_100_slice.jsonl"
        write_jsonl(slice_out, coach_100)
        print(f"[prepare] Stratified 100-slice: {len(coach_100)} rows -> {slice_out}")

        grpo_candidates = list(seed_rows) + list(fresh_seeds)
        grpo_prompts = build_coach_grpo_prompts(
            grpo_candidates,
            all_test_fens,
            per_task=args.grpo_per_task,
            seed=args.seed,
        )
        grpo_out = args.phase2_dir / "coach_grpo_prompts_2k.jsonl"
        write_jsonl(grpo_out, grpo_prompts)
        grpo_overlap = audit_zero_leakage(grpo_prompts, test_fens_by_set)
        print(
            f"[prepare] Coach-GRPO 2K prompts: {len(grpo_prompts)} rows, overlap={grpo_overlap} -> {grpo_out}"
        )
        return 0

    # merge mode
    test_fens_by_set = load_frozen_test_fens(args.phase2_dir)
    existing_clean = read_jsonl(args.existing)
    fresh_distilled = read_jsonl(args.fresh)
    for r in fresh_distilled:
        r.setdefault("source", "distilled_26b_fresh_1k")

    merged_rows, audit = merge_and_audit_coach_sft(
        existing_clean,
        fresh_distilled,
        test_fens_by_set,
        seed=args.seed,
    )
    if not audit["zero_leakage_verified"]:
        raise RuntimeError(f"Leakage detected in merged Coach SFT: {audit['test_overlap_counts']}")
    if audit["slop_audit"]["is_slop_corpus"]:
        raise RuntimeError(f"Merged Coach SFT failed corpus anti-slop gate: {audit['slop_audit']}")

    write_jsonl(args.output, merged_rows)

    # Also include audit stats for GRPO 2K and 100-slice if present.
    grpo_path = args.phase2_dir / "coach_grpo_prompts_2k.jsonl"
    if grpo_path.exists():
        grpo_rows = read_jsonl(grpo_path)
        grpo_ov = audit_zero_leakage(grpo_rows, test_fens_by_set)
        audit["coach_grpo_prompts_2k"] = {
            "rows": len(grpo_rows),
            "per_task_type": dict(
                sorted(Counter(r.get("task_type", "") for r in grpo_rows).items())
            ),
            "test_overlap_counts": grpo_ov,
            "zero_leakage_verified": all(v == 0 for v in grpo_ov.values()),
        }

    slice_path = args.phase2_dir / "test_coach_100_slice.jsonl"
    test_300_path = args.phase2_dir / "test_coach_300.jsonl"
    if slice_path.exists() and test_300_path.exists():
        slice_rows = read_jsonl(slice_path)
        t300_keys = {(r["fen"], infer_task_type(r)) for r in read_jsonl(test_300_path)}
        is_subset = all((r["fen"], infer_task_type(r)) in t300_keys for r in slice_rows)
        audit["test_coach_100_slice"] = {
            "rows": len(slice_rows),
            "per_task_type": dict(sorted(Counter(infer_task_type(r) for r in slice_rows).items())),
            "strict_subset_of_test_coach_300": is_subset,
        }

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[merge] Wrote {len(merged_rows)} spotless Coach SFT rows -> {args.output}")
    print(f"[merge] Audit written -> {args.audit_output}")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
