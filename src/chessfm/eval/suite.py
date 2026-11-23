#!/usr/bin/env python3
"""4-Tier Evaluation Suite for ChessFM.

Implements a structured, reproducible evaluation harness across four tiers:

- **Tier 1 (Static Engine Metrics)**: Legal-move rate, ``<think>...</think>``
  format adherence, Average Centipawn Loss (ACPL), Move Accuracy Rate (MAR),
  mean WDL win-probability loss, and exact puzzle accuracy.
- **Tier 2 (Board-State Probing / ChessQA)**: Deterministic board-state
  comprehension probes (side to move, piece on square, move legality, legal
  move count) with overall and per-category accuracy.
- **Tier 3 (Coaching Faithfulness / ACT-Eval)**: Deterministic claim
  verification via :func:`chessfm.coach.verifier.verify_claims` and Socratic
  spoiler detection via :func:`chessfm.coach.socratic.contains_move_spoiler`.
- **Tier 4 (Paired Color-Balanced Match Summary)**: Overall and White/Black
  split W-D-L, 95% Wilson confidence interval, estimated Elo, and
  reproducibility metadata.

Usage::

    python -m chessfm.eval.suite --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessfm.coach.classifier import win_prob_from_cp
from chessfm.coach.socratic import contains_move_spoiler, generate_socratic_hint
from chessfm.coach.verifier import verify_claims
from chessfm.config import (
    CHATML_ASSISTANT_OPEN,
    CHATML_USER_CLOSE,
    CHATML_USER_OPEN,
    DEFAULT_STOCKFISH_DEPTH,
    build_chat_prompt,
    render_ascii_board,
)
from chessfm.fens import read_jsonl
from chessfm.rl.engine import close_engine, open_engine
from chessfm.rl.evaluate import estimated_elo, generate_completion, git_revision, wilson_ci
from chessfm.rl.move_utils import parse_move_from_output
from chessfm.training.manifest import _get_git_sha

_MATE_SCORE_CP = 10000
_MAR_CP_THRESHOLD = 50.0

_PIECE_TYPE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_ILLEGAL_CANDIDATE_UCIS: tuple[str, ...] = (
    "a1h8",
    "a8h1",
    "h1a8",
    "h8a1",
    "e1e8",
    "e8e1",
    "b1g8",
    "g1b8",
    "a1b7",
    "h8g2",
)


def _has_valid_think_format(output: str) -> bool:
    """Return True if ``output`` contains a well-ordered ``<think>...</think>`` block."""
    if not output:
        return False
    open_idx = output.find("<think>")
    close_idx = output.find("</think>")
    return open_idx != -1 and close_idx != -1 and open_idx < close_idx


def _score_to_cp(score: chess.engine.PovScore | None, turn: chess.Color) -> int:
    """Convert a python-chess PovScore to centipawns from ``turn``'s perspective."""
    if score is None:
        return 0
    cp = score.pov(turn).score(mate_score=_MATE_SCORE_CP)
    return int(cp) if cp is not None else 0


# ---------------------------------------------------------------------------
# Tier 1: Static Engine Metrics
# ---------------------------------------------------------------------------


def evaluate_static_positions(
    positions: list[dict[str, Any]],
    predict_fn: Callable[[str], str],
    engine: Any | None = None,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
) -> dict[str, Any]:
    """Evaluate raw model predictions on a static set of FEN positions.

    Args:
        positions: List of dicts containing ``fen`` and optional ``best_move``
            (UCI string).
        predict_fn: Callable accepting a FEN string and returning raw model
            output (expected to contain ``<think>...</think>`` and a UCI move).
        engine: Optional active ``chess.engine.SimpleEngine`` (or compatible
            mock) used to compute ACPL, WDL loss, and engine-backed MAR.
        depth: Stockfish search depth in plies when ``engine`` is provided.

    Returns:
        Summary dictionary with ``total_positions``, ``legal_rate``,
        ``format_adherence``, ``acpl``, ``mar``, ``mean_wdl_loss``,
        ``puzzle_accuracy``, and per-position ``details``.
    """
    total = len(positions)
    if total == 0:
        return {
            "total_positions": 0,
            "legal_moves": 0,
            "legal_rate": 0.0,
            "format_adherence": 0.0,
            "acpl": None,
            "mar": None,
            "mean_wdl_loss": None,
            "puzzle_accuracy": None,
            "details": [],
        }

    limit = chess.engine.Limit(depth=int(depth))
    legal_count = 0
    formatted_count = 0
    cp_losses: list[float] = []
    wdl_losses: list[float] = []
    accurate_legal_count = 0
    mar_evaluable_legal_count = 0
    puzzle_total = 0
    puzzle_exact = 0
    details: list[dict[str, Any]] = []

    for pos in positions:
        fen = pos["fen"]
        best_move_raw = pos.get("best_move")
        best_move = best_move_raw.strip().lower() if isinstance(best_move_raw, str) else None
        board = chess.Board(fen)

        raw_output = predict_fn(fen)
        formatted = _has_valid_think_format(raw_output)
        if formatted:
            formatted_count += 1

        parsed_move = parse_move_from_output(raw_output)
        move_obj: chess.Move | None = None
        is_legal = False
        if parsed_move is not None:
            try:
                candidate = chess.Move.from_uci(parsed_move)
                if candidate in board.legal_moves:
                    move_obj = candidate
                    is_legal = True
            except ValueError:
                is_legal = False

        if is_legal:
            legal_count += 1

        exact_match: bool | None = None
        if best_move is not None:
            puzzle_total += 1
            exact_match = is_legal and parsed_move == best_move
            if exact_match:
                puzzle_exact += 1

        cp_loss: float | None = None
        wdl_loss: float | None = None
        if is_legal and move_obj is not None and engine is not None:
            root_info = engine.analyse(board, limit)
            best_eval_cp = _score_to_cp(root_info.get("score"), board.turn)
            root_pv = list(root_info.get("pv", []))
            engine_best_uci = root_pv[0].uci() if root_pv else None

            if engine_best_uci is not None and parsed_move == engine_best_uci:
                move_eval_cp = best_eval_cp
            else:
                after = board.copy()
                after.push(move_obj)
                if after.is_checkmate():
                    move_eval_cp = _MATE_SCORE_CP
                elif after.is_game_over():
                    move_eval_cp = 0
                else:
                    after_info = engine.analyse(after, limit)
                    move_eval_cp = _score_to_cp(after_info.get("score"), board.turn)

            cp_loss = max(0.0, float(best_eval_cp) - float(move_eval_cp))
            wdl_loss = max(
                0.0,
                win_prob_from_cp(best_eval_cp) - win_prob_from_cp(move_eval_cp),
            )
            cp_losses.append(cp_loss)
            wdl_losses.append(wdl_loss)

        is_accurate: bool | None = None
        if is_legal and (cp_loss is not None or best_move is not None):
            mar_evaluable_legal_count += 1
            is_accurate = bool(exact_match) or (
                cp_loss is not None and cp_loss <= _MAR_CP_THRESHOLD
            )
            if is_accurate:
                accurate_legal_count += 1

        details.append(
            {
                "fen": fen,
                "predicted_move": parsed_move,
                "best_move": best_move,
                "legal": is_legal,
                "formatted": formatted,
                "cp_loss": None if cp_loss is None else round(cp_loss, 2),
                "wdl_loss": None if wdl_loss is None else round(wdl_loss, 4),
                "accurate": is_accurate,
                "exact_match": exact_match,
            }
        )

    acpl = round(sum(cp_losses) / len(cp_losses), 2) if cp_losses else None
    mean_wdl_loss = round(sum(wdl_losses) / len(wdl_losses), 4) if wdl_losses else None
    if engine is not None or puzzle_total > 0:
        mar: float | None = (
            round(accurate_legal_count / mar_evaluable_legal_count, 4)
            if mar_evaluable_legal_count > 0
            else 0.0
        )
    else:
        mar = None
    puzzle_accuracy = round(puzzle_exact / puzzle_total, 4) if puzzle_total > 0 else None

    return {
        "total_positions": total,
        "legal_moves": legal_count,
        "legal_rate": round(legal_count / total, 4),
        "format_adherence": round(formatted_count / total, 4),
        "acpl": acpl,
        "mar": mar,
        "mean_wdl_loss": mean_wdl_loss,
        "puzzle_accuracy": puzzle_accuracy,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Tier 2: Board-State Probing / ChessQA
# ---------------------------------------------------------------------------


def _find_deterministic_illegal_move(board: chess.Board) -> str:
    """Return a deterministic UCI move string that is illegal in ``board``."""
    legal_set = {m.uci() for m in board.legal_moves}
    for uci in _ILLEGAL_CANDIDATE_UCIS:
        if uci not in legal_set:
            return uci
    # Fallback: scan all square pairs deterministically.
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            uci = chess.Move(from_sq, to_sq).uci()
            if uci not in legal_set:
                return uci
    raise RuntimeError("Could not construct an illegal move")  # pragma: no cover


def generate_board_probes(fen: str) -> list[dict[str, Any]]:
    """Generate deterministic ground-truth board-state probes for ``fen``.

    Produces probes across four categories:
    - ``side_to_move``: which color moves next (``'white'`` or ``'black'``).
    - ``piece_on_square``: color and piece type on a deterministically chosen
      occupied square.
    - ``move_legality``: one legal move probe (when legal moves exist) and one
      illegal move probe (``'yes'`` / ``'no'``).
    - ``legal_move_count``: exact integer number of legal moves.
    """
    board = chess.Board(fen)
    digest = hashlib.sha256(fen.encode("utf-8")).digest()
    probes: list[dict[str, Any]] = []

    # 1. Side to move.
    side = "white" if board.turn == chess.WHITE else "black"
    probes.append(
        {
            "fen": fen,
            "category": "side_to_move",
            "question": "Which side is to move in this position? Answer 'white' or 'black'.",
            "expected": side,
        }
    )

    # 2. Piece on a deterministically selected occupied square.
    occupied = sorted(board.piece_map().keys())
    if occupied:
        sq = occupied[digest[0] % len(occupied)]
        piece = board.piece_at(sq)
        assert piece is not None
        sq_name = chess.square_name(sq)
        color_name = "white" if piece.color == chess.WHITE else "black"
        piece_name = _PIECE_TYPE_NAMES[piece.piece_type]
        probes.append(
            {
                "fen": fen,
                "category": "piece_on_square",
                "square": sq_name,
                "question": (
                    f"What piece is on square {sq_name}? "
                    "Answer as '<color> <piece>' (e.g. 'white knight')."
                ),
                "expected": f"{color_name} {piece_name}",
            }
        )

    # 3. Move legality (one legal if available + one guaranteed illegal).
    legal_ucis = sorted(m.uci() for m in board.legal_moves)
    if legal_ucis:
        chosen_legal = legal_ucis[digest[1] % len(legal_ucis)]
        probes.append(
            {
                "fen": fen,
                "category": "move_legality",
                "move": chosen_legal,
                "question": (
                    f"Is the UCI move {chosen_legal} legal in this position? Answer 'yes' or 'no'."
                ),
                "expected": "yes",
            }
        )

    illegal_uci = _find_deterministic_illegal_move(board)
    probes.append(
        {
            "fen": fen,
            "category": "move_legality",
            "move": illegal_uci,
            "question": (
                f"Is the UCI move {illegal_uci} legal in this position? Answer 'yes' or 'no'."
            ),
            "expected": "no",
        }
    )

    # 4. Number of legal moves.
    probes.append(
        {
            "fen": fen,
            "category": "legal_move_count",
            "question": "How many legal moves are available in this position? Answer with an integer.",
            "expected": str(len(legal_ucis)),
        }
    )

    return probes


def _normalize_probe_answer(category: str, answer: Any) -> str:
    """Normalize a model's probe answer for fair deterministic comparison."""
    text = re.sub(r"\s+", " ", str(answer).strip().lower()).rstrip(".,!?")
    if category == "side_to_move":
        if text in ("w", "white"):
            return "white"
        if text in ("b", "black"):
            return "black"
        return text
    if category == "move_legality":
        if text in ("yes", "true", "legal", "y", "1"):
            return "yes"
        if text in ("no", "false", "illegal", "n", "0"):
            return "no"
        return text
    if category == "legal_move_count":
        match = re.search(r"-?\d+", text)
        return match.group(0) if match else text
    return text


def evaluate_board_probes(
    fens: list[str],
    answer_fn: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Run deterministic ChessQA board-state probes across ``fens``.

    Args:
        fens: List of FEN strings to probe.
        answer_fn: Callable accepting a probe dict (from
            :func:`generate_board_probes`) and returning the model's answer.

    Returns:
        Dictionary with ``total_probes``, ``correct``, ``accuracy``,
        ``per_category``, and per-probe ``results``.
    """
    per_cat_counts: dict[str, dict[str, int]] = {}
    results: list[dict[str, Any]] = []
    total_probes = 0
    total_correct = 0

    for fen in fens:
        for probe in generate_board_probes(fen):
            cat = probe["category"]
            raw_answer = answer_fn(probe)
            normalized = _normalize_probe_answer(cat, raw_answer)
            is_correct = normalized == probe["expected"]

            total_probes += 1
            if is_correct:
                total_correct += 1

            bucket = per_cat_counts.setdefault(cat, {"total": 0, "correct": 0})
            bucket["total"] += 1
            if is_correct:
                bucket["correct"] += 1

            results.append(
                {
                    "fen": fen,
                    "category": cat,
                    "question": probe["question"],
                    "expected": probe["expected"],
                    "predicted": normalized,
                    "correct": is_correct,
                }
            )

    per_category = {
        cat: {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": round(counts["correct"] / counts["total"], 4)
            if counts["total"] > 0
            else 0.0,
        }
        for cat, counts in per_cat_counts.items()
    }

    return {
        "total_probes": total_probes,
        "correct": total_correct,
        "accuracy": round(total_correct / total_probes, 4) if total_probes > 0 else 0.0,
        "per_category": per_category,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tier 3: Coaching Faithfulness / ACT-Eval
# ---------------------------------------------------------------------------


def _extract_coach_text(output: Any) -> str:
    """Normalize ``coach_fn`` return values (plain string or hint dict) to text."""
    if isinstance(output, dict):
        for key in ("hint", "text", "explanation", "response", "completion"):
            if key in output and isinstance(output[key], str):
                return output[key]
        return str(output)
    return str(output)


def evaluate_coaching_faithfulness(
    cases: list[dict[str, Any]],
    coach_fn: Callable[[dict[str, Any]], Any],
    engine: Any | None = None,
) -> dict[str, Any]:
    """Evaluate coaching prose factuality and Socratic spoiler adherence.

    Runs :func:`chessfm.coach.verifier.verify_claims` on every output from
    ``coach_fn(case)``. For Socratic Level 1 and Level 2 cases that supply a
    ``best_move``, also checks :func:`chessfm.coach.socratic.contains_move_spoiler`.

    Args:
        cases: List of coaching evaluation cases (each containing ``fen`` and
            optionally ``level`` and ``best_move``).
        coach_fn: Callable accepting a case dict and returning coaching prose
            (or a dict containing a ``hint``/``text`` field).
        engine: Optional Stockfish engine passed to ``verify_claims`` for
            blunder checking.

    Returns:
        Dictionary with ``total_cases``, ``mean_factuality_score``,
        ``pass_rate``, ``spoiler_rate``, and per-case ``cases``.
    """
    total_cases = len(cases)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "mean_factuality_score": 0.0,
            "pass_rate": 0.0,
            "spoiler_rate": 0.0,
            "cases": [],
        }

    factuality_scores: list[float] = []
    passed_count = 0
    spoiler_checked = 0
    spoiler_count = 0
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        fen = case["fen"]
        level = case.get("level")
        best_move = case.get("best_move")

        response_text = _extract_coach_text(coach_fn(case))
        verification = verify_claims(fen, response_text, engine=engine)
        factuality_scores.append(float(verification["factuality_score"]))

        spoiler: bool | None = None
        if level in (1, 2) and isinstance(best_move, str) and best_move.strip():
            spoiler_checked += 1
            spoiler = contains_move_spoiler(response_text, best_move.strip(), fen)
            if spoiler:
                spoiler_count += 1

        case_passed = bool(verification["passed"]) and (spoiler is not True)
        if case_passed:
            passed_count += 1

        case_reports.append(
            {
                "fen": fen,
                "level": level,
                "best_move": best_move,
                "response": response_text,
                "passed": case_passed,
                "factuality_score": round(float(verification["factuality_score"]), 4),
                "total_claims": verification["total_claims"],
                "verified_claims": verification["verified_claims"],
                "violations": verification["violations"],
                "spoiler": spoiler,
            }
        )

    return {
        "total_cases": total_cases,
        "mean_factuality_score": round(sum(factuality_scores) / total_cases, 4),
        "pass_rate": round(passed_count / total_cases, 4),
        "spoiler_rate": round(spoiler_count / spoiler_checked, 4) if spoiler_checked > 0 else 0.0,
        "cases": case_reports,
    }


# ---------------------------------------------------------------------------
# Tier 4: Paired Color-Balanced Match Summary
# ---------------------------------------------------------------------------


def _resolve_game_outcome(record: dict[str, Any], color: str) -> str:
    """Resolve a game record to ``'win'``, ``'loss'``, or ``'draw'``."""
    outcome = record.get("outcome")
    if isinstance(outcome, str):
        norm = outcome.strip().lower()
        if norm in ("win", "w"):
            return "win"
        if norm in ("loss", "l"):
            return "loss"
        if norm in ("draw", "d"):
            return "draw"

    result = record.get("result")
    if isinstance(result, str):
        res = result.strip()
        if res == "1-0":
            return "win" if color == "white" else "loss"
        if res == "0-1":
            return "win" if color == "black" else "loss"
        if res in ("1/2-1/2", "draw"):
            return "draw"

    if record.get("wins"):
        return "win"
    if record.get("losses"):
        return "loss"
    if record.get("draws"):
        return "draw"

    raise ValueError(f"Could not determine game outcome from record: {record!r}")


def _split_stats(wins: int, losses: int, draws: int) -> dict[str, Any]:
    """Build a compact W-D-L and score dictionary."""
    games = wins + losses + draws
    score = (wins + 0.5 * draws) / games if games > 0 else 0.0
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4),
    }


def summarize_paired_match(
    game_records: list[dict[str, Any]],
    opponent_name: str,
    opponent_elo: int,
    model_id: str = "unknown",
    decoding_mode: str = "strict",
) -> dict[str, Any]:
    """Summarize a paired, color-balanced match with Wilson CI and Elo estimate.

    Each entry in ``game_records`` should specify the model's color via
    ``color`` (or ``model_color``: ``'white'`` / ``'black'``) and its outcome
    via ``outcome`` (``'win'`` / ``'loss'`` / ``'draw'``), PGN ``result``
    (``'1-0'`` / ``'0-1'`` / ``'1/2-1/2'``), or ``wins``/``losses``/``draws``
    counters.
    """
    counts: dict[str, dict[str, int]] = {
        "white": {"win": 0, "loss": 0, "draw": 0},
        "black": {"win": 0, "loss": 0, "draw": 0},
    }

    for rec in game_records:
        raw_color = rec.get("color", rec.get("model_color"))
        if not isinstance(raw_color, str) or raw_color.strip().lower() not in ("white", "black"):
            raise ValueError(f"Invalid or missing color in game record: {rec!r}")
        color = raw_color.strip().lower()
        outcome = _resolve_game_outcome(rec, color)
        counts[color][outcome] += 1

    w_wins = counts["white"]["win"]
    w_losses = counts["white"]["loss"]
    w_draws = counts["white"]["draw"]
    b_wins = counts["black"]["win"]
    b_losses = counts["black"]["loss"]
    b_draws = counts["black"]["draw"]

    total_wins = w_wins + b_wins
    total_losses = w_losses + b_losses
    total_draws = w_draws + b_draws
    total_games = total_wins + total_losses + total_draws

    ci_low, ci_high = wilson_ci(total_wins, total_losses, total_draws)
    elo = (
        round(estimated_elo(total_wins, total_losses, total_draws, opponent_elo), 1)
        if total_games > 0
        else None
    )

    white_split = _split_stats(w_wins, w_losses, w_draws)
    black_split = _split_stats(b_wins, b_losses, b_draws)

    return {
        "opponent_name": opponent_name,
        "opponent_elo": int(opponent_elo),
        "overall": _split_stats(total_wins, total_losses, total_draws),
        "white": white_split,
        "black": black_split,
        "color_balanced": total_games > 0 and white_split["games"] == black_split["games"],
        "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "estimated_elo": elo,
        "metadata": {
            "git_sha": git_revision(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "decoding_mode": decoding_mode,
        },
    }


# ---------------------------------------------------------------------------
# Offline Demo & CLI
# ---------------------------------------------------------------------------


def run_demo_suite() -> dict[str, Any]:
    """Run a fast, deterministic, offline demo across all 4 evaluation tiers."""
    demo_positions = [
        {"fen": chess.STARTING_FEN, "best_move": "e2e4"},
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "best_move": "h5f7",
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "best_move": "g1f3",
        },
    ]
    demo_answers = {p["fen"]: p["best_move"] for p in demo_positions}

    def _demo_predict(fen: str) -> str:
        move = demo_answers[fen]
        return f"<think>Develop actively and apply pressure with {move}.</think> {move}"

    tier1 = evaluate_static_positions(demo_positions, _demo_predict, engine=None)

    demo_fens = [p["fen"] for p in demo_positions]
    tier2 = evaluate_board_probes(demo_fens, lambda probe: probe["expected"])

    demo_coach_cases = [
        {
            "fen": demo_positions[1]["fen"],
            "best_move": "h5f7",
            "level": 1,
        },
        {
            "fen": demo_positions[1]["fen"],
            "best_move": "h5f7",
            "level": 2,
        },
        {
            "fen": demo_positions[1]["fen"],
            "best_move": "h5f7",
            "level": 3,
        },
    ]

    def _demo_coach(case: dict[str, Any]) -> dict[str, Any]:
        return generate_socratic_hint(
            fen=case["fen"],
            best_move_uci=case["best_move"],
            level=case["level"],
        )

    tier3 = evaluate_coaching_faithfulness(demo_coach_cases, _demo_coach, engine=None)

    demo_games = [
        {"color": "white", "result": "1-0"},
        {"color": "black", "result": "0-1"},
        {"color": "white", "result": "1/2-1/2"},
        {"color": "black", "result": "1-0"},
    ]
    tier4 = summarize_paired_match(
        demo_games,
        opponent_name="Stockfish L1 (demo)",
        opponent_elo=900,
        model_id="chessfm-demo",
        decoding_mode="strict",
    )

    return {
        "tier1_static_engine": tier1,
        "tier2_board_probes": tier2,
        "tier3_coaching_faithfulness": tier3,
        "tier4_paired_match": tier4,
    }


def _detect_hardware() -> str:
    """Return CUDA device name if available, otherwise ``'cpu'``."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def build_eval_manifest(
    *,
    model_id: str,
    decoding_mode: str = "strict",
    enable_thinking: bool = True,
    include_ascii: bool = False,
    hardware: str | None = None,
    wall_clock_seconds: float = 0.0,
    tier1_static: dict[str, Any] | None = None,
    tier2_probes: dict[str, Any] | None = None,
    tier3_act_eval: dict[str, Any] | None = None,
    tier4_match: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the standardized ``eval_manifest.json`` (schema v1.0) envelope."""
    return {
        "schema_version": "1.0",
        "git_sha": _get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "decoding_mode": decoding_mode,
        "enable_thinking": bool(enable_thinking),
        "include_ascii": bool(include_ascii),
        "hardware": hardware if hardware is not None else _detect_hardware(),
        "wall_clock_seconds": round(float(wall_clock_seconds), 2),
        "tiers": {
            "tier1_static": tier1_static or {},
            "tier2_probes": tier2_probes or {},
            "tier3_act_eval": tier3_act_eval or {},
            "tier4_match": tier4_match or {},
        },
    }


def _build_probe_prompt(
    probe: dict[str, Any],
    tokenizer: Any = None,
    include_ascii: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Format a Tier 2 ChessQA board probe as a chat prompt."""
    fen = probe["fen"]
    board_block = f"Board:\n{render_ascii_board(fen)}\n\n" if include_ascii else ""
    raw = f"Position (FEN): {fen}\n{board_block}Question: {probe['question']}\nAnswer concisely."
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": raw}]
        try:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except (ValueError, AttributeError, NotImplementedError):
            pass
    return f"{CHATML_USER_OPEN}{raw}{CHATML_USER_CLOSE}{CHATML_ASSISTANT_OPEN}"


def _extract_probe_answer(raw_completion: str) -> str:
    """Strip any ``<think>...</think>`` block and return the concise probe answer."""
    text = raw_completion or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def evaluate_model_suite(
    model: Any,
    tokenizer: Any,
    *,
    model_id: str,
    positions: list[dict[str, Any]],
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    engine: Any | None = None,
    run_probes: bool = True,
    coaching_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Tier 1 (+ optional Tier 2 and Tier 3) evaluation on a loaded model."""
    start = time.monotonic()

    def _predict(fen: str) -> str:
        prompt = build_chat_prompt(
            fen,
            tokenizer=tokenizer,
            include_ascii=include_ascii,
            enable_thinking=enable_thinking,
        )
        return generate_completion(
            model,
            tokenizer,
            prompt,
            decoding_mode=decoding_mode,
            fen=fen,
            require_think_close=enable_thinking,
        )

    tier1 = evaluate_static_positions(positions, _predict, engine=engine, depth=depth)

    tier2: dict[str, Any] = {}
    if run_probes and positions:
        fens = [p["fen"] for p in positions]

        def _answer_probe(probe: dict[str, Any]) -> str:
            probe_prompt = _build_probe_prompt(
                probe,
                tokenizer=tokenizer,
                include_ascii=include_ascii,
                enable_thinking=False,
            )
            raw = generate_completion(
                model,
                tokenizer,
                probe_prompt,
                max_new_tokens=64,
                decoding_mode="strict",
            )
            return _extract_probe_answer(raw)

        tier2 = evaluate_board_probes(fens, _answer_probe)

    tier3: dict[str, Any] = {}
    if coaching_cases:
        tier3 = evaluate_coaching_faithfulness(
            coaching_cases,
            lambda case: _extract_coach_text(case),
            engine=engine,
        )

    elapsed = time.monotonic() - start
    return build_eval_manifest(
        model_id=model_id,
        decoding_mode=decoding_mode,
        enable_thinking=enable_thinking,
        include_ascii=include_ascii,
        wall_clock_seconds=elapsed,
        tier1_static=tier1,
        tier2_probes=tier2,
        tier3_act_eval=tier3,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the 4-Tier ChessFM Evaluation Suite."""
    parser = argparse.ArgumentParser(description="ChessFM 4-Tier Evaluation Suite")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a fast deterministic offline demo across all 4 tiers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model ID or local checkpoint path for real Tier 1/2 evaluation",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data_generation/eval_benchmark.jsonl"),
        help="Path to JSONL benchmark positions (with 'fen' and optional 'best_move')",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="Optional cap on benchmark positions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_STOCKFISH_DEPTH,
        help="Stockfish search depth for Tier 1 ACPL/MAR/WDL analysis",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include ASCII board rendering in prompts (must match training prompt format)",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable <think> reasoning prompt mode (--no-enable-thinking for Think-OFF ablation)",
    )
    parser.add_argument(
        "--decoding-mode",
        type=str,
        default="strict",
        choices=["strict", "constrained"],
        help="Decoding mode for Tier 1: 'strict' or 'constrained'",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip Tier 2 ChessQA board-state probes when running --model",
    )
    parser.add_argument(
        "--coaching-jsonl",
        type=Path,
        default=None,
        help="JSONL of coaching responses to audit with Tier 3 ACT-Eval faithfulness/spoiler checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary / eval_manifest.json report",
    )
    args = parser.parse_args(argv)

    if not args.demo and not args.model and args.coaching_jsonl is None:
        parser.error(
            "Pass --demo for the offline demo, --model for real Tier 1/2 evaluation, "
            "or --coaching-jsonl for Tier 3 ACT-Eval auditing."
        )

    if args.demo:
        report: dict[str, Any] = run_demo_suite()
    else:
        coaching_cases = read_jsonl(args.coaching_jsonl) if args.coaching_jsonl else None
        engine = open_engine()
        try:
            if args.model:
                from chessfm.load_model import load_model

                positions = read_jsonl(args.benchmark)
                if args.max_positions > 0:
                    positions = positions[: args.max_positions]
                model, tokenizer = load_model(args.model, for_inference=True)
                report = evaluate_model_suite(
                    model,
                    tokenizer,
                    model_id=args.model,
                    positions=positions,
                    decoding_mode=args.decoding_mode,
                    include_ascii=args.include_ascii,
                    enable_thinking=args.enable_thinking,
                    depth=args.depth,
                    engine=engine,
                    run_probes=not args.skip_probes,
                    coaching_cases=coaching_cases,
                )
            else:
                assert coaching_cases is not None
                start = time.monotonic()
                tier3 = evaluate_coaching_faithfulness(
                    coaching_cases,
                    lambda case: _extract_coach_text(
                        case.get("completion", case.get("response", case))
                    ),
                    engine=engine,
                )
                report = build_eval_manifest(
                    model_id=str(args.coaching_jsonl),
                    decoding_mode=args.decoding_mode,
                    enable_thinking=args.enable_thinking,
                    include_ascii=args.include_ascii,
                    wall_clock_seconds=time.monotonic() - start,
                    tier3_act_eval=tier3,
                )
        finally:
            close_engine(engine)

    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
