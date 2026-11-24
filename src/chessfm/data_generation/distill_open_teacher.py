#!/usr/bin/env python3
"""Rejection-sampled open-weight coaching distillation pipeline.

Queries a self-hosted open-weight model (for example ``google/gemma-4-26B-A4B-it``
or ``Qwen/Qwen3.6-35B-A3B`` served via vLLM or Ollama's OpenAI-compatible
endpoint) to convert Stockfish + Maia structured position facts into natural
coaching prose, then filters every candidate through deterministic ACT-Eval
claim verification and Socratic spoiler checks.

Design principles:
- Zero proprietary API dependencies: talks to any OpenAI-compatible open-weight
  server (``vLLM``, ``llama.cpp``, ``Ollama``) or an injected Python callable.
- Strict grounding: every accepted example passes ``verify_claims()`` with zero
  violations, and Level 1 / Level 2 Socratic hints pass
  ``contains_move_spoiler() == False``.
- Reproducible auditing: writes an explicit acceptance/rejection audit summary
  next to the output JSONL.

Usage:
    python -m chessfm.data_generation.distill_open_teacher \\
        --input data_generation/coaching_sample.jsonl \\
        --output data_generation/full/distilled_coach.jsonl \\
        --base-url http://127.0.0.1:8001/v1 \\
        --model google/gemma-4-26B-A4B-it
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import requests

from chessfm.coach.coach_rewards import verify_claims_v1
from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.tactical_facts import (
    compute_tactical_fact_coverage,
    extract_tactical_facts,
    format_tactical_facts_for_prompt,
)
from chessfm.coach.verifier import verify_claims
from chessfm.config import DEFAULT_TEACHER_MODEL
from chessfm.fens import read_jsonl

VALID_TASK_TYPES = ("postmortem", "socratic_l1", "socratic_l2", "socratic_l3", "why_not")

_SOCRATIC_LEVELS: dict[str, int] = {
    "socratic_l1": 1,
    "socratic_l2": 2,
    "socratic_l3": 3,
}

SYSTEM_PROMPT = (
    "You are an accurate, concise chess coach. You are given verified engine facts "
    "(FEN, best move, principal variation, evaluation delta, and human Elo context). "
    "Write clear, pedagogical coaching grounded strictly in those facts. Never invent "
    "illegal moves, piece placements, or variations that contradict the position."
)

# Strips reasoning blocks from both Qwen (<think>...</think>) and Gemma 4
# (<|channel>thought...<channel|>) before auditing user-facing coaching text.
_THINK_BLOCK_RE = re.compile(
    r"(?:<think>.*?</think>|<\|channel>thought.*?<channel\|>)",
    re.DOTALL | re.IGNORECASE,
)

# Retained so unit tests that monkeypatch ``requests.post`` continue to work
# while production calls use a thread-local ``requests.Session`` for keep-alive.
_ORIG_REQUESTS_POST = requests.post


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string (YYYY-MM-DDTHH:MM:SSZ)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head_sha() -> str:
    """Return the 40-char git HEAD SHA, or ``'unknown'`` outside a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if len(out) == 40:
            return out
    except Exception:
        pass
    return "unknown"


def _is_valid_fen(fen: Any) -> bool:
    """Return True if ``fen`` is a non-empty string representing a valid chess FEN."""
    if not isinstance(fen, str) or not fen.strip():
        return False
    try:
        chess.Board(fen)
    except ValueError:
        return False
    return True


def _is_valid_legal_uci(fen: str, move_uci: Any) -> bool:
    """Return True if ``move_uci`` is a valid legal UCI move on ``fen``."""
    if not isinstance(move_uci, str) or not move_uci.strip():
        return False
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci.strip())
        return move in board.legal_moves
    except (ValueError, TypeError, AttributeError, chess.InvalidMoveError):
        return False


def _uci_to_san(fen: str, move_str: str) -> str:
    """Convert a legal UCI move string to SAN on ``fen``; return ``''`` if invalid."""
    if not isinstance(fen, str) or not fen or not isinstance(move_str, str) or not move_str:
        return ""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_str.strip())
        if move in board.legal_moves:
            return board.san(move)
    except (ValueError, TypeError, AttributeError, chess.InvalidMoveError):
        return ""
    return ""


def _is_non_coaching_row(row: dict[str, Any]) -> bool:
    """Return True for raw P1 move-prediction rows mixed into coaching corpora."""
    has_coaching_key = bool(
        row.get("task_type") or row.get("mode") or row.get("hint_level") is not None
    )
    return row.get("source") == "engine_best_line" and not has_coaching_key


def format_chatml_prompt(system_prompt: str, user_prompt: str) -> str:
    """Format system + user turns as a ChatML prompt ending with the assistant header."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def strip_reasoning_blocks(text: str) -> str:
    """Remove model reasoning blocks and return trimmed user-facing coaching text."""
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def infer_task_type(row: dict[str, Any]) -> str:
    """Infer a normalized coaching task type from an input dataset row."""
    raw = str(row.get("task_type") or row.get("mode") or "").strip().lower()
    if raw in VALID_TASK_TYPES:
        return raw
    level = row.get("hint_level")
    if level in (1, "1"):
        return "socratic_l1"
    if level in (2, "2"):
        return "socratic_l2"
    if level in (3, "3"):
        return "socratic_l3"
    return "postmortem"


def _resolve_tactical_facts(facts: dict[str, Any]) -> dict[str, Any] | None:
    """Return precomputed or freshly extracted tactical facts for ``facts``."""
    existing = facts.get("tactical_facts")
    if isinstance(existing, dict) and existing.get("fen"):
        return existing

    fen = facts.get("fen")
    best_move = str(facts.get("best_move_uci") or facts.get("best_move") or "").strip()
    if not isinstance(fen, str) or not fen.strip() or not best_move:
        return None
    student_move = str(facts.get("played_move_uci") or facts.get("student_move") or "").strip()
    pv = facts.get("principal_variation") or facts.get("pv")
    pv_list = [str(m) for m in pv] if isinstance(pv, list) else None
    try:
        return extract_tactical_facts(
            fen,
            student_move=student_move or None,
            best_move=best_move,
            pv=pv_list,
            cp_loss=facts.get("cp_loss"),
            eval_before_cp=facts.get("eval_before_cp"),
            eval_after_cp=facts.get("eval_after_cp"),
        )
    except ValueError:
        return None


def build_teacher_prompt(
    facts: dict[str, Any],
    task_type: str = "postmortem",
    *,
    include_tactical_facts: bool = False,
) -> str:
    """Construct a structured, engine-grounded prompt for an open-weight teacher."""
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {task_type!r}. Choose from: {', '.join(VALID_TASK_TYPES)}"
        )

    fen = facts["fen"]
    best_move_uci = str(facts.get("best_move_uci") or facts.get("best_move") or "")
    best_move_san = str(facts.get("best_move_san") or "")
    if not best_move_san and best_move_uci:
        best_move_san = _uci_to_san(fen, best_move_uci)

    played_move_san = str(facts.get("played_move_san") or "")
    played_move_raw = str(facts.get("played_move_uci") or facts.get("student_move") or "")
    if played_move_san:
        played_move = played_move_san
    elif played_move_raw:
        played_move = _uci_to_san(fen, played_move_raw) or played_move_raw
    else:
        played_move = ""

    classification = str(facts.get("classification") or facts.get("severity") or "")
    cp_loss = facts.get("cp_loss")
    pv = facts.get("principal_variation") or facts.get("pv") or []
    student_elo = facts.get("student_elo") or 1200

    maia_prob = facts.get("maia_human_prob")
    if maia_prob is None:
        maia_prob = facts.get("maia_prob")

    payload: dict[str, Any] = {
        "fen": fen,
        "student_elo": student_elo,
        "best_move_uci": best_move_uci,
        "best_move_san": best_move_san,
        "played_move": played_move,
        "classification": classification,
        "cp_loss": cp_loss,
        "principal_variation": pv[:6] if isinstance(pv, list) else pv,
        "maia_human_prob": maia_prob,
        "phase": facts.get("phase"),
        "best_motifs": facts.get("best_motifs"),
        "student_motifs": facts.get("student_motifs"),
    }
    compact_facts = json.dumps({k: v for k, v in payload.items() if v not in ("", None, [])})

    if task_type == "socratic_l1":
        instruction = (
            "Write a Level 1 Socratic hint (1-2 sentences) describing only the high-level "
            "strategic or tactical theme. Do NOT name the best move, the moving piece's "
            "origin square, or the destination square."
        )
    elif task_type == "socratic_l2":
        instruction = (
            "Write a Level 2 Socratic hint (1-2 sentences) pointing attention to the active "
            "area of the board or key motif without revealing the exact best move or both "
            "origin and destination squares."
        )
    elif task_type == "socratic_l3":
        instruction = (
            "Write a Level 3 concrete hint (2-3 sentences) explaining the tactical mechanism "
            "and key continuation clearly for the student's rating."
        )
    elif task_type == "why_not":
        instruction = (
            "Write a concise 2-4 sentence 'Why Not?' comparison tailored to the student's Elo. "
            "Contrast the student's played move against the engine's best move using only the "
            "grounded engine facts, explaining concretely why the played move falls short and "
            "why the best move is stronger."
        )
    else:
        instruction = (
            "Write a concise 2-4 sentence coaching explanation tailored to the student's Elo. "
            "Explain why the best move works (and why the played move falls short if provided), "
            "staying 100% consistent with the board and engine facts."
        )

    if not include_tactical_facts:
        return f"Engine Facts (JSON):\n{compact_facts}\n\nTask ({task_type}):\n{instruction}"

    tf = _resolve_tactical_facts(facts)
    tf_block = (
        f"\n\n{format_tactical_facts_for_prompt(tf, task_type=task_type)}" if tf is not None else ""
    )
    grounding_note = (
        " Ground every claim strictly in the Tactical Facts above: never claim a check if "
        "gives_check is false, never claim center tension if has_center_tension is false, "
        "always respect side_to_move, and use varied position-specific prose rather than "
        "generic boilerplate."
    )
    if task_type in ("socratic_l1", "socratic_l2"):
        grounding_note += (
            " Strictly avoid spoiling the best move or naming both origin and destination squares."
        )
    return f"Engine Facts (JSON):\n{compact_facts}{tf_block}\n\nTask ({task_type}):\n{instruction}{grounding_note}"


class OpenAICompatibleTeacher:
    """Minimal client for self-hosted OpenAI-compatible endpoints (vLLM, Ollama, llama.cpp)."""

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_TEACHER_MODEL,
        api_key: str = "EMPTY",
        temperature: float = 0.4,
        max_tokens: int = 384,
        timeout: int = 120,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        post_fn = (
            requests.post if requests.post is not _ORIG_REQUESTS_POST else self._get_session().post
        )
        resp = post_fn(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0].get("message") or {}
        content = message.get("content") or ""
        return str(content)


@dataclass
class VerificationDecision:
    """Result of deterministic ACT-Eval + spoiler filtering on one candidate."""

    accepted: bool
    cleaned_text: str
    reason: str = "accepted"
    violations: list[dict[str, Any]] = field(default_factory=list)


def verify_candidate(
    fen: str,
    raw_text: str,
    *,
    task_type: str = "postmortem",
    best_move_uci: str = "",
    min_chars: int = 20,
) -> VerificationDecision:
    """Deterministically verify one teacher completion before accepting it."""
    cleaned = strip_reasoning_blocks(raw_text)
    if not _is_valid_fen(fen):
        return VerificationDecision(accepted=False, cleaned_text=cleaned, reason="invalid_fen")

    if len(cleaned) < min_chars:
        return VerificationDecision(accepted=False, cleaned_text=cleaned, reason="empty_or_short")

    try:
        report = verify_claims(fen, cleaned)
    except (ValueError, TypeError, AttributeError):
        return VerificationDecision(accepted=False, cleaned_text=cleaned, reason="invalid_fen")
    if not report["passed"]:
        return VerificationDecision(
            accepted=False,
            cleaned_text=cleaned,
            reason="claim_violation",
            violations=list(report.get("violations", [])),
        )

    if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
        try:
            has_spoiler = contains_move_spoiler(cleaned, best_move_uci, fen)
        except (ValueError, TypeError, AttributeError, chess.InvalidMoveError):
            return VerificationDecision(
                accepted=False,
                cleaned_text=cleaned,
                reason="invalid_best_move",
            )
        if has_spoiler:
            return VerificationDecision(
                accepted=False,
                cleaned_text=cleaned,
                reason="spoiler_leak",
            )

    return VerificationDecision(accepted=True, cleaned_text=cleaned, reason="accepted")


def _distill_single_row(
    row: dict[str, Any],
    teacher_fn: Callable[[str, str], str],
    *,
    teacher_model: str,
    max_attempts: int,
    min_chars: int,
    include_tactical_facts: bool = False,
    single_shot: bool = False,
) -> tuple[dict[str, Any] | None, int, dict[str, int]]:
    """Run rejection sampling (or single-shot eval) for a single row."""
    fen = row["fen"]
    task_type = infer_task_type(row)
    best_move_uci = str(row.get("best_move_uci") or row.get("best_move") or "")
    tactical_facts = _resolve_tactical_facts(row) if include_tactical_facts else None
    prompt_row = (
        {**row, "tactical_facts": tactical_facts}
        if include_tactical_facts and tactical_facts is not None
        else row
    )
    user_prompt = build_teacher_prompt(
        prompt_row,
        task_type=task_type,
        include_tactical_facts=include_tactical_facts,
    )
    local_rejections: dict[str, int] = {}
    attempts = 0
    effective_attempts = 1 if single_shot else max_attempts

    for attempt in range(1, effective_attempts + 1):
        attempts += 1
        try:
            raw_text = teacher_fn(SYSTEM_PROMPT, user_prompt)
        except Exception:
            local_rejections["backend_error"] = local_rejections.get("backend_error", 0) + 1
            if single_shot:
                raw_text = ""
            else:
                continue

        if single_shot:
            cleaned = strip_reasoning_blocks(raw_text)
            v2_report = verify_claims(fen, cleaned, engine=None, check_boilerplate=True)
            v1_report = verify_claims_v1(fen, cleaned)
            act_eval_passed = bool(v2_report["passed"]) and len(cleaned) >= min_chars
            v1_passed = bool(v1_report["passed"]) and len(cleaned) >= min_chars
            spoiler = False
            if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
                try:
                    spoiler = bool(contains_move_spoiler(cleaned, best_move_uci, fen))
                except ValueError:
                    spoiler = False
            tf_cov = (
                compute_tactical_fact_coverage(cleaned, tactical_facts, task_type=task_type)
                if tactical_facts is not None
                else 0.0
            )
            record: dict[str, Any] = {
                "fen": fen,
                "task_type": task_type,
                "best_move_uci": best_move_uci,
                "best_move": best_move_uci,
                "student_move": str(row.get("student_move") or row.get("played_move_uci") or ""),
                "student_elo": row.get("student_elo", 1200),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": cleaned},
                ],
                "prompt": format_chatml_prompt(SYSTEM_PROMPT, user_prompt),
                "completion": f"{cleaned}<|im_end|>",
                "response": cleaned,
                "teacher_model": teacher_model,
                "attempts_used": 1,
                "verified": bool(act_eval_passed and not spoiler),
                "act_eval_passed": act_eval_passed,
                "v1_passed": v1_passed,
                "spoiler_free": not spoiler,
                "violations": list(v2_report.get("violations", [])),
                "tactical_fact_coverage": tf_cov,
            }
            if tactical_facts is not None:
                record["tactical_facts"] = tactical_facts
            if task_type in _SOCRATIC_LEVELS:
                record["level"] = _SOCRATIC_LEVELS[task_type]
            if not act_eval_passed:
                reason = "empty_or_short" if len(cleaned) < min_chars else "claim_violation"
                local_rejections[reason] = local_rejections.get(reason, 0) + 1
            elif spoiler:
                local_rejections["spoiler_leak"] = local_rejections.get("spoiler_leak", 0) + 1
            return record, attempts, local_rejections

        decision = verify_candidate(
            fen=fen,
            raw_text=raw_text,
            task_type=task_type,
            best_move_uci=best_move_uci,
            min_chars=min_chars,
        )
        if decision.accepted:
            record = {
                "fen": fen,
                "task_type": task_type,
                "best_move_uci": best_move_uci,
                "best_move": best_move_uci,
                "student_elo": row.get("student_elo", 1200),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": decision.cleaned_text},
                ],
                "prompt": format_chatml_prompt(SYSTEM_PROMPT, user_prompt),
                "completion": f"{decision.cleaned_text}<|im_end|>",
                "teacher_model": teacher_model,
                "attempts_used": attempt,
                "verified": True,
                "act_eval_passed": True,
                "spoiler_free": True,
            }
            if include_tactical_facts and tactical_facts is not None:
                record["tactical_facts"] = tactical_facts
                record["tactical_fact_coverage"] = compute_tactical_fact_coverage(
                    decision.cleaned_text, tactical_facts, task_type=task_type
                )
            if task_type in _SOCRATIC_LEVELS:
                record["level"] = _SOCRATIC_LEVELS[task_type]
            return record, attempts, local_rejections

        local_rejections[decision.reason] = local_rejections.get(decision.reason, 0) + 1
        if decision.reason in ("invalid_fen", "invalid_best_move"):
            break

    return None, attempts, local_rejections


def distill_rows(
    rows: list[dict[str, Any]],
    teacher_fn: Callable[[str, str], str],
    *,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    max_attempts: int = 3,
    min_chars: int = 20,
    workers: int = 1,
    coaching_only: bool = True,
    include_tactical_facts: bool = False,
    single_shot: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run rejection-sampled distillation (or single-shot eval) across ``rows``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    start_time = time.monotonic()
    effective_workers = max(1, workers)
    skipped_non_coaching = 0
    eligible_rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {
        "empty_or_short": 0,
        "claim_violation": 0,
        "spoiler_leak": 0,
        "backend_error": 0,
        "invalid_fen": 0,
        "invalid_best_move": 0,
    }
    for row in rows:
        if "fen" not in row or row.get("fen") is None or row.get("fen") == "":
            continue
        if coaching_only and _is_non_coaching_row(row):
            skipped_non_coaching += 1
            continue
        fen = row["fen"]
        if not _is_valid_fen(fen):
            rejection_counts["invalid_fen"] += 1
            continue
        task_type = infer_task_type(row)
        best_move_uci = str(row.get("best_move_uci") or row.get("best_move") or "")
        if (
            task_type in ("socratic_l1", "socratic_l2")
            and best_move_uci
            and not _is_valid_legal_uci(fen, best_move_uci)
        ):
            rejection_counts["invalid_best_move"] += 1
            continue
        eligible_rows.append(row)
    per_task_type: dict[str, int] = {}
    total_attempts = 0
    accepted_rows: list[dict[str, Any]] = []

    def _process(r: dict[str, Any]) -> tuple[dict[str, Any] | None, int, dict[str, int]]:
        return _distill_single_row(
            r,
            teacher_fn,
            teacher_model=teacher_model,
            max_attempts=max_attempts,
            min_chars=min_chars,
            include_tactical_facts=include_tactical_facts,
            single_shot=single_shot,
        )

    if effective_workers == 1:
        results = [_process(r) for r in eligible_rows]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(_process, eligible_rows))

    for record, attempts, local_rejections in results:
        total_attempts += attempts
        for reason, count in local_rejections.items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count
        if record is not None:
            accepted_rows.append(record)
            task_key = record["task_type"]
            per_task_type[task_key] = per_task_type.get(task_key, 0) + 1

    processed = len(eligible_rows)
    verified_count = sum(1 for r in accepted_rows if r.get("verified", True))
    audit = {
        "teacher_model": teacher_model,
        "timestamp_utc": _utc_timestamp(),
        "git_sha": _git_head_sha(),
        "wall_clock_seconds": round(time.monotonic() - start_time, 3),
        "workers": effective_workers,
        "include_tactical_facts": include_tactical_facts,
        "single_shot": single_shot,
        "input_rows": processed,
        "skipped_non_coaching_rows": skipped_non_coaching,
        "accepted_rows": len(accepted_rows),
        "verified_rows": verified_count,
        "acceptance_rate": round(verified_count / max(1, processed), 4),
        "total_attempts": total_attempts,
        "rejection_counts": rejection_counts,
        "per_task_type": per_task_type,
    }
    return accepted_rows, audit


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for open-weight rejection-sampled coaching distillation."""
    parser = argparse.ArgumentParser(
        description="Generate ACT-Eval-verified coaching distillation rows from an open-weight teacher."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_generation/coaching_sample.jsonl"),
        help="Input JSONL containing FEN + engine facts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_generation/full/distilled_coach.jsonl"),
        help="Output JSONL for verified teacher completions",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="OpenAI-compatible server URL (vLLM / Ollama / llama.cpp)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEACHER_MODEL,
        help=f"Open-weight teacher model ID (default: {DEFAULT_TEACHER_MODEL})",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Max rejection-sampling tries")
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Concurrent worker threads for teacher API requests (default: 24)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on input rows")
    parser.add_argument(
        "--include-non-coaching",
        action="store_true",
        help="Include raw engine_best_line rows instead of filtering to coaching rows only",
    )
    parser.add_argument(
        "--include-tactical-facts",
        action="store_true",
        help="Inject deterministic python-chess Tactical Facts into teacher prompts and output rows",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help=(
            "Single-shot evaluation mode: max_attempts=1, default temperature=0.2, writes EVERY "
            "row without dropping failures while recording honest V1/V2/spoiler/coverage metrics"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (defaults to 0.2 under --single-shot, else 0.4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input rows and print sample teacher prompt without calling the server",
    )
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    coaching_only = not args.include_non_coaching

    if args.dry_run:
        valid = [
            r
            for r in rows
            if _is_valid_fen(r.get("fen")) and (not coaching_only or not _is_non_coaching_row(r))
        ]
        print(
            f"[dry-run] Loaded {len(valid)}/{len(rows)} coaching rows with FENs from {args.input}"
        )
        if valid:
            task = infer_task_type(valid[0])
            print(f"[dry-run] Sample prompt ({task}, model={args.model}):\n")
            print(
                build_teacher_prompt(
                    valid[0],
                    task_type=task,
                    include_tactical_facts=args.include_tactical_facts,
                )
            )
        return 0

    temperature = (
        args.temperature if args.temperature is not None else (0.2 if args.single_shot else 0.4)
    )
    effective_attempts = 1 if args.single_shot else args.max_attempts
    teacher = OpenAICompatibleTeacher(
        base_url=args.base_url,
        model=args.model,
        temperature=temperature,
    )
    accepted, audit = distill_rows(
        rows,
        teacher,
        teacher_model=args.model,
        max_attempts=effective_attempts,
        workers=args.workers,
        coaching_only=coaching_only,
        include_tactical_facts=args.include_tactical_facts,
        single_shot=args.single_shot,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Distillation complete: {audit['accepted_rows']}/{audit['input_rows']} written "
        f"(verified {audit['verified_rows']}/{audit['input_rows']} = {audit['acceptance_rate']:.1%}) "
        f"-> {args.output} (audit: {audit_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
