#!/usr/bin/env python3
"""Stockfish-FREE Coach-GRPO reward functions (Strict V2 + Anti-Slop and Weak V1).

Provides two reward modes for GRPO training of AI Chess Coach models:

- ``coach_v2`` (production / Arms A & B): Strict Verifier V2 factual pass
  (``+1.0`` / ``-1.0``) + task-aware spoiler-safe tactical fact coverage
  (``+0.5 * cov``) + 0%-spoiler adherence (``+0.2`` / ``-1.0`` on L1/L2)
  + cross-FEN rolling anti-slop 4-gram penalty (``-0.8``) + degenerate short
  output guard (``-1.5``).
- ``coach_v1`` (controlled Goodhart's Law ablation / Arm C): Weak Verifier V1
  only (basic non-hyphenated ``piece_on_square``, raw move legality, and
  ``side_to_move``; zero V2 geometry/polarity/boilerplate checks, zero anti-slop
  penalty, zero tactical fact coverage bonus) + basic L1/L2 spoiler check.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, deque
from typing import Any

import chess

from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.tactical_facts import (
    compute_tactical_fact_coverage,
    extract_tactical_facts,
)
from chessfm.coach.verifier import (
    _CONTEXTUAL_PAWN_MOVE_RE,
    _FOLLOWUP_PAWN_MOVE_RE,
    _PIECE_ON_SQUARE_RE,
    _SAN_MOVE_RE,
    _SIDE_TO_MOVE_RE,
    _SPECIAL_TOKEN_RE,
    _UCI_MOVE_RE,
    _VARIATION_RESET_RE,
    _is_negated,
    _overlaps,
    _parse_move_on_board,
    _same_sentence,
    _verify_piece_on_square,
    detect_boilerplate_slop,
    verify_claims,
)
from chessfm.data_generation.phase_sampler import normalize_board_fen

logger = logging.getLogger("chessfm.coach_rewards")

VALID_COACH_REWARD_MODES: tuple[str, ...] = ("coach_v2", "coach_v1")

_REASONING_BLOCK_RE = re.compile(
    r"(?:<think>.*?</think>|<\|channel>thought.*?<channel\|>)",
    re.DOTALL | re.IGNORECASE,
)

# Verified M4 Goodhart boilerplate 4-grams observed during V1->V2 rescoring.
KNOWN_SLOP_4GRAMS: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        ("tension", "in", "the", "center"),
        ("at", "the", "tension", "in"),
        ("closely", "at", "the", "tension"),
        ("look", "closely", "at", "the"),
        ("create", "a", "powerful", "attacking"),
        ("a", "powerful", "attacking", "sequence"),
        ("prepare", "for", "a", "decisive"),
        ("for", "a", "decisive", "central"),
        ("a", "decisive", "central", "breakthrough"),
        ("allows", "them", "to", "create"),
    }
)

_FEN_IN_JSON_RE = re.compile(r'"fen"\s*:\s*"([^"]+)"')
_FEN_IN_TEXT_RE = re.compile(r"Position\s*\(FEN\)\s*:\s*([^\n]+)")


def clean_coach_completion(text: str) -> str:
    """Strip reasoning tags and chat special tokens from a coach completion."""
    stripped = _REASONING_BLOCK_RE.sub("", text or "")
    stripped = _SPECIAL_TOKEN_RE.sub("", stripped)
    return stripped.strip()


def _extract_4grams(text: str) -> frozenset[tuple[str, str, str, str]]:
    """Extract normalized lowercase 4-grams matching ``detect_boilerplate_slop``."""
    cleaned = _SPECIAL_TOKEN_RE.sub("", text or "").strip()
    tokens = re.findall(r"[a-z0-9]+", cleaned.lower())
    if len(tokens) < 4:
        return frozenset()
    return frozenset(
        (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]) for i in range(len(tokens) - 3)
    )


def _extract_fen_fallback(prompt: str) -> str | None:
    """Fallback FEN extraction from a coach prompt when ``fen`` kwarg is omitted."""
    if not prompt:
        return None
    m_json = _FEN_IN_JSON_RE.search(prompt)
    if m_json:
        return m_json.group(1).strip()
    m_txt = _FEN_IN_TEXT_RE.search(prompt)
    if m_txt:
        return m_txt.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Weak Verifier V1 (Reproduces Original Pre-V2 Semantics for Arm C Showdown)
# ---------------------------------------------------------------------------


def verify_claims_v1(fen: str, text: str) -> dict[str, Any]:
    """Verify coaching prose using ONLY original weak Verifier V1 semantics.

    Checks ONLY:
    - Non-hyphenated ``piece_on_square`` claims (``"knight on f6"``; explicitly
      skips hyphenated ``"e8-pawn"``, determiner square-first ``"the f7 pawn"``,
      and rank-1/8 pawn impossibility checks).
    - Basic UCI / SAN / contextual pawn move legality (explicitly skips
      ``color_polarity_violation`` and ``invalid_check_annotation`` checks).
    - Explicit ``side_to_move`` assertions (``"White is to move"``).

    Intentionally skips ALL V2-only checks: ``impossible_pawn_rank``,
    ``fake_check``, ``fake_checkmate``, ``invalid_attack_claim``,
    ``color_polarity_violation``, ``invalid_check_annotation``, and
    ``hollow_boilerplate``.
    """
    root_board = chess.Board(fen)
    var_board = root_board.copy()
    cleaned_text = _SPECIAL_TOKEN_RE.sub("", text or "")

    claims: list[dict[str, Any]] = []
    occupied_move_spans: list[tuple[int, int]] = []
    piece_spans: list[tuple[int, int]] = []

    # 1. Basic non-hyphenated piece-on-square ONLY (V1).
    for m in _PIECE_ON_SQUARE_RE.finditer(cleaned_text):
        span = (m.start(), m.end())
        if _overlaps(span, piece_spans):
            continue
        piece_spans.append(span)
        color = m.group("color").lower() if m.group("color") else None
        claims.append(
            {
                "type": "piece_on_square",
                "raw": m.group(0),
                "color": color,
                "piece": m.group("piece").lower(),
                "square": m.group("square").lower(),
                "span": span,
            }
        )

    # 2. UCI moves.
    for m in _UCI_MOVE_RE.finditer(cleaned_text):
        span = (m.start(1), m.end(1))
        if _overlaps(span, occupied_move_spans):
            continue
        occupied_move_spans.append(span)
        claims.append(
            {
                "type": "move",
                "notation": "uci",
                "move": m.group(1).lower(),
                "raw": m.group(1),
                "span": span,
            }
        )

    # 3. SAN moves.
    for m in _SAN_MOVE_RE.finditer(cleaned_text):
        span = (m.start(1), m.end(1))
        if _overlaps(span, occupied_move_spans):
            continue
        occupied_move_spans.append(span)
        claims.append(
            {
                "type": "move",
                "notation": "san",
                "move": m.group(1),
                "raw": m.group(1),
                "span": span,
            }
        )

    # 4. Contextual bare pawn moves.
    for m in _CONTEXTUAL_PAWN_MOVE_RE.finditer(cleaned_text):
        span = (m.start("move"), m.end("move"))
        if _overlaps(span, occupied_move_spans) or _overlaps(span, piece_spans):
            continue
        occupied_move_spans.append(span)
        claims.append(
            {
                "type": "move",
                "notation": "san",
                "move": m.group("move").lower(),
                "raw": m.group("move"),
                "span": span,
            }
        )

    added = True
    while added:
        added = False
        for end_pos in sorted(end for _, end in occupied_move_spans):
            fm = _FOLLOWUP_PAWN_MOVE_RE.match(cleaned_text[end_pos:])
            if fm:
                span = (end_pos + fm.start("move"), end_pos + fm.end("move"))
                if not _overlaps(span, occupied_move_spans) and not _overlaps(span, piece_spans):
                    occupied_move_spans.append(span)
                    claims.append(
                        {
                            "type": "move",
                            "notation": "san",
                            "move": fm.group("move").lower(),
                            "raw": fm.group("move"),
                            "span": span,
                        }
                    )
                    added = True

    # 5. Explicit side-to-move assertions.
    for m in _SIDE_TO_MOVE_RE.finditer(cleaned_text):
        if _is_negated(cleaned_text, m.start()):
            continue
        claims.append(
            {
                "type": "side_to_move",
                "color": m.group("color").lower(),
                "raw": m.group(0),
                "span": (m.start(), m.end()),
            }
        )

    claims.sort(key=lambda c: c["span"][0])

    violations: list[dict[str, Any]] = []
    verified_claims = 0
    last_move_span: tuple[int, int] | None = None
    last_move_after: chess.Board | None = None

    for claim in claims:
        ctype = claim["type"]

        if ctype == "piece_on_square":
            err = _verify_piece_on_square(root_board, claim)
            if err is not None and (
                last_move_span is not None
                and last_move_after is not None
                and _same_sentence(cleaned_text, last_move_span[1], claim["span"][0])
            ):
                if _verify_piece_on_square(last_move_after, claim) is None:
                    err = None
            if err is not None:
                violations.append({"type": "piece_on_square", "claim": claim, "reason": err})
            else:
                verified_claims += 1

        elif ctype == "move":
            move_str = claim["move"]
            notation = claim["notation"]
            if last_move_span is not None:
                between = cleaned_text[last_move_span[1] : claim["span"][0]]
                if _VARIATION_RESET_RE.search(between):
                    var_board = root_board.copy()

            move, err = _parse_move_on_board(var_board, move_str, notation)
            if move is None and var_board.fen() != root_board.fen():
                move, err = _parse_move_on_board(root_board, move_str, notation)
                if move is not None:
                    var_board = root_board.copy()

            if move is None:
                violations.append(
                    {
                        "type": "illegal_move",
                        "claim": claim,
                        "reason": err or f"Illegal move '{move_str}'",
                    }
                )
                last_move_span = None
                last_move_after = None
                continue

            var_board.push(move)
            last_move_span = claim["span"]
            last_move_after = var_board.copy()
            verified_claims += 1

        elif ctype == "side_to_move":
            ref_board = (
                last_move_after
                if (
                    last_move_span is not None
                    and last_move_after is not None
                    and _same_sentence(cleaned_text, last_move_span[1], claim["span"][0])
                )
                else root_board
            )
            actual_turn = "white" if ref_board.turn == chess.WHITE else "black"
            if claim["color"] != actual_turn:
                violations.append(
                    {
                        "type": "side_to_move",
                        "claim": claim,
                        "reason": (
                            f"Claimed {claim['color'].title()} to move, "
                            f"but it is {actual_turn.title()} to move"
                        ),
                    }
                )
            else:
                verified_claims += 1

    total_claims = len(claims)
    factuality_score = verified_claims / total_claims if total_claims > 0 else 1.0
    return {
        "passed": len(violations) == 0,
        "factuality_score": factuality_score,
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Cross-FEN Rolling Anti-Slop Tracker (GRPO-Safe Across G=8 Same-FEN Rollouts)
# ---------------------------------------------------------------------------


class CrossFenSlopTracker:
    """Rolling cross-FEN 4-gram repetition tracker for GRPO anti-slop penalties.

    Computes 4-gram document frequency strictly across **distinct normalized
    FENs** other than the current position, so ``G=8`` rollouts for the same
    FEN in a GRPO batch never falsely penalize accurate position-specific
    phrases (e.g. ``"the bishop on b5"``).
    """

    def __init__(
        self,
        maxlen: int = 256,
        max_doc_freq_ratio: float = 0.20,
        min_doc_count: int = 3,
        min_distinct_fens: int = 8,
        blocklist: frozenset[tuple[str, str, str, str]] = KNOWN_SLOP_4GRAMS,
    ) -> None:
        self.maxlen = maxlen
        self.max_doc_freq_ratio = max_doc_freq_ratio
        self.min_doc_count = min_doc_count
        self.min_distinct_fens = min_distinct_fens
        self.blocklist = blocklist
        self.window: deque[tuple[str, frozenset[tuple[str, str, str, str]]]] = deque(maxlen=maxlen)

    def _normalize_fen(self, fen: str) -> str:
        try:
            return normalize_board_fen(fen)
        except Exception:
            return fen.strip()

    def is_slop(self, fen: str, text: str) -> bool:
        """Return True if ``text`` hits the slop blocklist or repeats cross-FEN 4-grams."""
        grams = _extract_4grams(text)
        if not grams:
            return False
        if grams & self.blocklist:
            return True

        norm_fen = self._normalize_fen(fen)
        # Union 4-grams per distinct other FEN in the rolling window.
        other_fen_grams: dict[str, set[tuple[str, str, str, str]]] = {}
        for entry_fen, entry_grams in self.window:
            if entry_fen == norm_fen:
                continue
            bucket = other_fen_grams.setdefault(entry_fen, set())
            bucket.update(entry_grams)

        n_other_fens = len(other_fen_grams)
        if n_other_fens < self.min_distinct_fens:
            return False

        doc_freq: Counter[tuple[str, str, str, str]] = Counter()
        for f_grams in other_fen_grams.values():
            for ng in grams & f_grams:
                doc_freq[ng] += 1

        for count in doc_freq.values():
            if count >= self.min_doc_count and (count / n_other_fens) >= self.max_doc_freq_ratio:
                return True
        return False

    def check_and_update(self, fen: str, text: str) -> bool:
        """Check whether ``text`` is cross-FEN slop and record it in the rolling window."""
        flagged = self.is_slop(fen, text)
        norm_fen = self._normalize_fen(fen)
        self.window.append((norm_fen, _extract_4grams(text)))
        return flagged


# ---------------------------------------------------------------------------
# Reward Scoring & TRL GRPO Reward Factory
# ---------------------------------------------------------------------------


def score_coach_completion(
    fen: str,
    completion: str,
    *,
    task_type: str = "postmortem",
    best_move_uci: str = "",
    tactical_facts: dict[str, Any] | None = None,
    mode: str = "coach_v2",
    slop_tracker: CrossFenSlopTracker | None = None,
    min_chars: int = 20,
) -> dict[str, Any]:
    """Score one coaching completion under ``coach_v2`` or ``coach_v1`` (Stockfish-FREE)."""
    if mode not in VALID_COACH_REWARD_MODES:
        raise ValueError(f"Unsupported coach reward mode {mode!r}. Use: {VALID_COACH_REWARD_MODES}")

    cleaned = clean_coach_completion(completion)
    if len(cleaned) < min_chars:
        return {
            "total": -1.5,
            "mode": mode,
            "degenerate": True,
            "verifier_passed": False,
            "spoiler_free": True,
            "coverage": 0.0,
            "slop_hit": False,
        }

    # Spoiler check applies to L1/L2 in both V1 and V2 modes.
    spoiled = False
    if task_type in ("socratic_l1", "socratic_l2") and best_move_uci:
        try:
            spoiled = bool(contains_move_spoiler(cleaned, best_move_uci, fen))
        except ValueError:
            spoiled = False
    r_spoiler = -1.0 if spoiled else 0.2

    if mode == "coach_v1":
        v1 = verify_claims_v1(fen, cleaned)
        r_v1 = 1.0 if v1["passed"] else -1.0
        total = round(r_v1 + r_spoiler, 4)
        return {
            "total": total,
            "mode": mode,
            "degenerate": False,
            "verifier_passed": bool(v1["passed"]),
            "spoiler_free": not spoiled,
            "coverage": 0.0,
            "slop_hit": False,
            "violations": v1["violations"],
        }

    # Strict Verifier V2 + tactical fact coverage + cross-FEN anti-slop.
    v2 = verify_claims(fen, cleaned, engine=None, check_boilerplate=True)
    r_v2 = 1.0 if v2["passed"] else -1.0

    cov = (
        compute_tactical_fact_coverage(cleaned, tactical_facts, task_type=task_type)
        if tactical_facts
        else 0.0
    )
    r_cov = 0.5 * cov

    if slop_tracker is not None:
        slop_hit = slop_tracker.check_and_update(fen, cleaned)
    else:
        slop_hit = bool(_extract_4grams(cleaned) & KNOWN_SLOP_4GRAMS)
    if any(v.get("type") == "hollow_boilerplate" for v in v2["violations"]):
        slop_hit = True
    r_slop = -0.8 if slop_hit else 0.0

    total = round(r_v2 + r_cov + r_spoiler + r_slop, 4)
    return {
        "total": total,
        "mode": mode,
        "degenerate": False,
        "verifier_passed": bool(v2["passed"]),
        "spoiler_free": not spoiled,
        "coverage": round(cov, 4),
        "slop_hit": slop_hit,
        "violations": v2["violations"],
    }


def make_coach_reward_func(
    mode: str = "coach_v2",
    *,
    slop_tracker: CrossFenSlopTracker | None = None,
    min_chars: int = 20,
):
    """Build a TRL-compatible Stockfish-FREE reward function for Coach-GRPO.

    Expects TRL ``GRPOTrainer`` to call
    ``reward_func(prompts=..., completions=..., **reward_kwargs)`` where
    ``reward_kwargs`` carries parallel lists for ``fen``, ``task_type``,
    ``best_move_uci``, and ``tactical_facts_json`` (automatically forwarded by
    TRL 0.16 from dataset columns).
    """
    if mode not in VALID_COACH_REWARD_MODES:
        raise ValueError(f"Unsupported coach reward mode {mode!r}. Use: {VALID_COACH_REWARD_MODES}")

    active_tracker = (
        slop_tracker
        if slop_tracker is not None
        else (CrossFenSlopTracker() if mode == "coach_v2" else None)
    )

    def _get_col(kwargs: dict[str, Any], key: str, idx: int, default: Any = None) -> Any:
        val = kwargs.get(key)
        if isinstance(val, (list, tuple)) and idx < len(val):
            return val[idx]
        return default

    def coach_reward_fn(
        prompts: list[str],
        completions: list[str],
        **reward_kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for idx, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            fen = _get_col(reward_kwargs, "fen", idx) or _extract_fen_fallback(prompt)
            if not fen:
                logger.error("Missing FEN in coach GRPO sample %d; returning neutral 0.0", idx)
                rewards.append(0.0)
                continue

            task_type = str(_get_col(reward_kwargs, "task_type", idx, "postmortem") or "postmortem")
            best_move_uci = str(_get_col(reward_kwargs, "best_move_uci", idx, "") or "")

            tf_raw = _get_col(reward_kwargs, "tactical_facts_json", idx)
            if tf_raw is None:
                tf_raw = _get_col(reward_kwargs, "tactical_facts", idx)
            tactical_facts: dict[str, Any] | None = None
            if isinstance(tf_raw, dict):
                tactical_facts = tf_raw
            elif isinstance(tf_raw, str) and tf_raw.strip():
                try:
                    parsed = json.loads(tf_raw)
                    if isinstance(parsed, dict):
                        tactical_facts = parsed
                except json.JSONDecodeError:
                    tactical_facts = None

            try:
                res = score_coach_completion(
                    fen=fen,
                    completion=completion,
                    task_type=task_type,
                    best_move_uci=best_move_uci,
                    tactical_facts=tactical_facts,
                    mode=mode,
                    slop_tracker=active_tracker,
                    min_chars=min_chars,
                )
                rewards.append(float(res["total"]))
            except ValueError as exc:
                logger.error("Coach reward error on sample %d: %s; returning neutral 0.0", idx, exc)
                rewards.append(0.0)
        return rewards

    coach_reward_fn.mode = mode  # type: ignore[attr-defined]
    coach_reward_fn.slop_tracker = active_tracker  # type: ignore[attr-defined]
    return coach_reward_fn


def summarize_coach_eval(
    rows: list[dict[str, Any]],
    *,
    min_chars: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score coaching evaluation outputs under V1, Strict V2, Anti-Slop, Spoiler, and Coverage.

    Used identically by Modal ``run_coach_benchmark_l4`` and local Row 6
    Teacher scoring so all 6 rows of the E5 comparison table share one
    authoritative metric definition.
    """
    cleaned_texts: list[str] = []
    for row in rows:
        raw = row.get("response")
        if not isinstance(raw, str):
            msgs = row.get("messages")
            if isinstance(msgs, list):
                raw = next(
                    (m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"),
                    "",
                )
            else:
                raw = str(row.get("completion") or "")
        cleaned_texts.append(clean_coach_completion(str(raw)))

    slop_report = detect_boilerplate_slop(cleaned_texts)
    flagged_set = set(slop_report["flagged_indices"])

    enriched: list[dict[str, Any]] = []
    v1_passed_total = 0
    v2_passed_total = 0
    v2_non_slop_total = 0
    all_clean_total = 0
    socratic_l12_total = 0
    socratic_l12_clean = 0
    cov_sum = 0.0
    violation_counts: Counter[str] = Counter()
    per_task: dict[str, dict[str, Any]] = {}

    for idx, (row, cleaned) in enumerate(zip(rows, cleaned_texts, strict=True)):
        fen = str(row["fen"])
        task_type = str(row.get("task_type") or row.get("mode") or "postmortem").strip().lower()
        best_move = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
        student_move = str(row.get("student_move") or row.get("played_move_uci") or "").strip()

        raw_pv = row.get("pv") or row.get("principal_variation")
        pv_list = list(raw_pv) if isinstance(raw_pv, list) else []

        tf = row.get("tactical_facts")
        if not isinstance(tf, dict) or not tf.get("fen"):
            tf = None
            if best_move:
                try:
                    tf = extract_tactical_facts(
                        fen,
                        student_move=student_move or None,
                        best_move=best_move,
                        pv=pv_list or None,
                        cp_loss=row.get("cp_loss"),
                        eval_before_cp=row.get("eval_before_cp"),
                        eval_after_cp=row.get("eval_after_cp"),
                    )
                except ValueError:
                    tf = None

        non_empty = len(cleaned) >= min_chars
        v1_rep = verify_claims_v1(fen, cleaned)
        v2_rep = verify_claims(fen, cleaned, engine=None, check_boilerplate=True)

        v1_ok = bool(v1_rep["passed"]) and non_empty
        v2_ok = bool(v2_rep["passed"]) and non_empty
        violations = list(v2_rep.get("violations", []))
        for v in violations:
            violation_counts[str(v.get("type", "unknown"))] += 1
        if not non_empty:
            violation_counts["empty_or_short"] += 1

        has_hollow = any(v.get("type") == "hollow_boilerplate" for v in violations)
        non_slop = (idx not in flagged_set) and (not has_hollow)
        v2_non_slop = v2_ok and non_slop

        spoiler = False
        if task_type in ("socratic_l1", "socratic_l2") and best_move:
            socratic_l12_total += 1
            try:
                spoiler = bool(contains_move_spoiler(cleaned, best_move, fen))
            except ValueError:
                spoiler = False
            if not spoiler:
                socratic_l12_clean += 1
        spoiler_free = not spoiler
        all_clean = v2_non_slop and spoiler_free

        cov = compute_tactical_fact_coverage(cleaned, tf, task_type=task_type) if tf else 0.0
        cov_sum += cov

        if v1_ok:
            v1_passed_total += 1
        if v2_ok:
            v2_passed_total += 1
        if v2_non_slop:
            v2_non_slop_total += 1
        if all_clean:
            all_clean_total += 1

        bucket = per_task.setdefault(
            task_type,
            {
                "total": 0,
                "v1_passed": 0,
                "v2_passed": 0,
                "act_passed": 0,
                "v2_and_non_slop": 0,
                "all_clean": 0,
                "spoiler_free": 0,
                "_cov_sum": 0.0,
            },
        )
        bucket["total"] += 1
        if v1_ok:
            bucket["v1_passed"] += 1
        if v2_ok:
            bucket["v2_passed"] += 1
            bucket["act_passed"] += 1
        if v2_non_slop:
            bucket["v2_and_non_slop"] += 1
        if all_clean:
            bucket["all_clean"] += 1
        if spoiler_free:
            bucket["spoiler_free"] += 1
        bucket["_cov_sum"] += cov

        enriched.append(
            {
                "fen": fen,
                "task_type": task_type,
                "best_move": best_move,
                "student_move": student_move,
                "pv": pv_list,
                "response": cleaned,
                "v1_passed": v1_ok,
                "v2_passed": v2_ok,
                "act_eval_passed": v2_ok,
                "non_slop": non_slop,
                "v2_and_non_slop": v2_non_slop,
                "spoiler_free": spoiler_free,
                "all_clean": all_clean,
                "tactical_fact_coverage": cov,
                "violations": violations,
            }
        )

    n = len(enriched)
    denom = max(1, n)
    per_task_out: dict[str, dict[str, Any]] = {}
    for tt, b in sorted(per_task.items()):
        t_n = max(1, b["total"])
        per_task_out[tt] = {
            "total": b["total"],
            "v1_passed": b["v1_passed"],
            "v1_pass_rate": round(b["v1_passed"] / t_n, 4),
            "v2_passed": b["v2_passed"],
            "v2_pass_rate": round(b["v2_passed"] / t_n, 4),
            "act_passed": b["act_passed"],
            "v2_and_non_slop": b["v2_and_non_slop"],
            "v2_and_non_slop_rate": round(b["v2_and_non_slop"] / t_n, 4),
            "all_clean": b["all_clean"],
            "all_clean_rate": round(b["all_clean"] / t_n, 4),
            "spoiler_free": b["spoiler_free"],
            "spoiler_free_rate": round(b["spoiler_free"] / t_n, 4),
            "mean_tactical_fact_coverage": round(b["_cov_sum"] / t_n, 4),
        }

    summary = {
        "total_cases": n,
        "v1_passed": v1_passed_total,
        "v1_pass_rate": round(v1_passed_total / denom, 4) if n else 0.0,
        "v2_passed": v2_passed_total,
        "v2_pass_rate": round(v2_passed_total / denom, 4) if n else 0.0,
        "act_eval_passed": v2_passed_total,
        "act_eval_pass_rate": round(v2_passed_total / denom, 4) if n else 0.0,
        "v2_and_non_slop": v2_non_slop_total,
        "v2_and_non_slop_rate": round(v2_non_slop_total / denom, 4) if n else 0.0,
        "socratic_l12_total": socratic_l12_total,
        "socratic_l12_spoiler_free": socratic_l12_clean,
        "socratic_l12_spoiler_free_rate": (
            round(socratic_l12_clean / socratic_l12_total, 4) if socratic_l12_total else 1.0
        ),
        "all_clean": all_clean_total,
        "all_clean_rate": round(all_clean_total / denom, 4) if n else 0.0,
        "mean_tactical_fact_coverage": round(cov_sum / denom, 4) if n else 0.0,
        "distinct_4gram_ratio": slop_report["distinct_4gram_ratio"],
        "slop_audit": {
            "distinct_4gram_ratio": slop_report["distinct_4gram_ratio"],
            "exact_duplicate_rate": slop_report["exact_duplicate_rate"],
            "flagged_count": slop_report["flagged_count"],
            "flagged_rate": slop_report["flagged_rate"],
            "is_slop_corpus": slop_report["is_slop_corpus"],
            "top_repeated_4grams": slop_report["top_repeated_4grams"][:5],
        },
        "violation_breakdown": dict(sorted(violation_counts.items())),
        "per_task_type": per_task_out,
    }
    return enriched, summary
