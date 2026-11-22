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
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.verifier import verify_claims
from chessfm.config import DEFAULT_TEACHER_MODEL
from chessfm.fens import read_jsonl

VALID_TASK_TYPES = ("postmortem", "socratic_l1", "socratic_l2", "socratic_l3")

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


def build_teacher_prompt(facts: dict[str, Any], task_type: str = "postmortem") -> str:
    """Construct a structured, engine-grounded prompt for an open-weight teacher."""
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {task_type!r}. Choose from: {', '.join(VALID_TASK_TYPES)}"
        )

    fen = facts["fen"]
    best_move_uci = facts.get("best_move_uci") or facts.get("best_move") or ""
    best_move_san = facts.get("best_move_san") or ""
    played_move = facts.get("played_move_san") or facts.get("played_move_uci") or ""
    classification = facts.get("classification") or ""
    cp_loss = facts.get("cp_loss")
    pv = facts.get("principal_variation") or facts.get("pv") or []
    student_elo = facts.get("student_elo") or 1200
    maia_prob = facts.get("maia_human_prob")

    payload = {
        "fen": fen,
        "student_elo": student_elo,
        "best_move_uci": best_move_uci,
        "best_move_san": best_move_san,
        "played_move": played_move,
        "classification": classification,
        "cp_loss": cp_loss,
        "principal_variation": pv[:6] if isinstance(pv, list) else pv,
        "maia_human_prob": maia_prob,
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
    else:
        instruction = (
            "Write a concise 2-4 sentence coaching explanation tailored to the student's Elo. "
            "Explain why the best move works (and why the played move falls short if provided), "
            "staying 100% consistent with the board and engine facts."
        )

    return f"Engine Facts (JSON):\n{compact_facts}\n\nTask ({task_type}):\n{instruction}"


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

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            self.endpoint,
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])


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
    if len(cleaned) < min_chars:
        return VerificationDecision(accepted=False, cleaned_text=cleaned, reason="empty_or_short")

    report = verify_claims(fen, cleaned)
    if not report["passed"]:
        return VerificationDecision(
            accepted=False,
            cleaned_text=cleaned,
            reason="claim_violation",
            violations=list(report.get("violations", [])),
        )

    if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
        if contains_move_spoiler(cleaned, best_move_uci, fen):
            return VerificationDecision(
                accepted=False,
                cleaned_text=cleaned,
                reason="spoiler_leak",
            )

    return VerificationDecision(accepted=True, cleaned_text=cleaned, reason="accepted")


def distill_rows(
    rows: list[dict[str, Any]],
    teacher_fn: Callable[[str, str], str],
    *,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    max_attempts: int = 3,
    min_chars: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run rejection-sampled distillation across ``rows`` and return ``(accepted, audit)``."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    accepted_rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {
        "empty_or_short": 0,
        "claim_violation": 0,
        "spoiler_leak": 0,
        "backend_error": 0,
    }
    total_attempts = 0

    for row in rows:
        fen = row.get("fen")
        if not fen:
            continue

        task_type = infer_task_type(row)
        best_move_uci = str(row.get("best_move_uci") or row.get("best_move") or "")
        user_prompt = build_teacher_prompt(row, task_type=task_type)

        for attempt in range(1, max_attempts + 1):
            total_attempts += 1
            try:
                raw_text = teacher_fn(SYSTEM_PROMPT, user_prompt)
            except Exception:
                rejection_counts["backend_error"] += 1
                continue

            decision = verify_candidate(
                fen=fen,
                raw_text=raw_text,
                task_type=task_type,
                best_move_uci=best_move_uci,
                min_chars=min_chars,
            )
            if decision.accepted:
                accepted_rows.append(
                    {
                        "fen": fen,
                        "task_type": task_type,
                        "best_move_uci": best_move_uci,
                        "student_elo": row.get("student_elo", 1200),
                        "prompt": user_prompt,
                        "completion": decision.cleaned_text,
                        "teacher_model": teacher_model,
                        "attempts_used": attempt,
                        "verified": True,
                    }
                )
                break
            rejection_counts[decision.reason] = rejection_counts.get(decision.reason, 0) + 1

    processed = sum(1 for r in rows if r.get("fen"))
    audit = {
        "teacher_model": teacher_model,
        "input_rows": processed,
        "accepted_rows": len(accepted_rows),
        "acceptance_rate": round(len(accepted_rows) / max(1, processed), 4),
        "total_attempts": total_attempts,
        "rejection_counts": rejection_counts,
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
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on input rows")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input rows and print sample teacher prompt without calling the server",
    )
    args = parser.parse_args(argv)

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        valid = [r for r in rows if r.get("fen")]
        print(f"[dry-run] Loaded {len(valid)}/{len(rows)} rows with FENs from {args.input}")
        if valid:
            task = infer_task_type(valid[0])
            print(f"[dry-run] Sample prompt ({task}, model={args.model}):\n")
            print(build_teacher_prompt(valid[0], task_type=task))
        return 0

    teacher = OpenAICompatibleTeacher(base_url=args.base_url, model=args.model)
    accepted, audit = distill_rows(
        rows,
        teacher,
        teacher_model=args.model,
        max_attempts=args.max_attempts,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Distillation complete: {audit['accepted_rows']}/{audit['input_rows']} accepted "
        f"({audit['acceptance_rate']:.1%}) -> {args.output} (audit: {audit_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
