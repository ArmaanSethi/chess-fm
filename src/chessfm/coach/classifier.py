"""Move quality classification and Partial Strategy Tree construction for coaching.

Implements Lichess-aligned win-probability conversion and move labeling, plus
Stockfish MultiPV Partial Strategy Tree extraction for grounded LLM coaching
verbalization (Cui et al., 2026).
"""

from __future__ import annotations

import math
from typing import Any

import chess
import chess.engine

from chessfm.config import DEFAULT_STOCKFISH_DEPTH

# Lichess logistic scaling constant mapping centipawns to win probability.
_LICHESS_CP_COEFF = 0.00368208
_MATE_SCORE_CP = 10000


def win_prob_from_cp(cp: float) -> float:
    """Convert a centipawn evaluation to win probability in ``[0.0, 100.0]``.

    Uses the Lichess logistic formula:
    ``50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)``.
    """
    exponent = max(-500.0, min(500.0, -_LICHESS_CP_COEFF * float(cp)))
    prob = 50.0 + 50.0 * (2.0 / (1.0 + math.exp(exponent)) - 1.0)
    return max(0.0, min(100.0, prob))


def classify_move(
    eval_before_cp: float,
    eval_after_cp: float,
    is_best_move: bool = False,
) -> dict[str, Any]:
    """Classify a move using centipawn loss and Lichess win-probability drop.

    Both ``eval_before_cp`` and ``eval_after_cp`` must be expressed from the
    moving player's perspective.
    """
    cp_loss = max(0.0, float(eval_before_cp) - float(eval_after_cp))
    win_prob_before = win_prob_from_cp(eval_before_cp)
    win_prob_after = win_prob_from_cp(eval_after_cp)
    win_prob_drop = max(0.0, win_prob_before - win_prob_after)

    if is_best_move:
        label = "Best"
    elif win_prob_before >= 80.0 and 35.0 <= win_prob_after <= 60.0 and win_prob_drop >= 20.0:
        label = "Missed Win"
    elif win_prob_drop >= 20.0 or cp_loss >= 250.0:
        label = "Blunder"
    elif win_prob_drop >= 10.0 or cp_loss >= 100.0:
        label = "Mistake"
    elif win_prob_drop >= 5.0 or cp_loss >= 50.0:
        label = "Inaccuracy"
    elif win_prob_drop >= 2.0 or cp_loss >= 20.0:
        label = "Good"
    else:
        label = "Excellent"

    return {
        "label": label,
        "cp_loss": round(cp_loss, 2),
        "win_prob_before": round(win_prob_before, 2),
        "win_prob_after": round(win_prob_after, 2),
        "win_prob_drop": round(win_prob_drop, 2),
        "is_best_move": bool(is_best_move),
    }


def _score_to_cp(score: chess.engine.PovScore | None, turn: chess.Color) -> int:
    """Convert a python-chess PovScore to centipawns from ``turn``'s perspective."""
    if score is None:
        return 0
    cp = score.pov(turn).score(mate_score=_MATE_SCORE_CP)
    return int(cp) if cp is not None else 0


def _format_pv(
    board: chess.Board,
    pv: list[chess.Move],
    pv_plies: int,
) -> tuple[list[str], list[str]]:
    """Format a principal variation into parallel UCI and SAN lists."""
    sim = board.copy()
    pv_uci: list[str] = []
    pv_san: list[str] = []
    for move in pv[: max(0, pv_plies)]:
        if move not in sim.legal_moves:
            break
        pv_san.append(sim.san(move))
        pv_uci.append(move.uci())
        sim.push(move)
    return pv_uci, pv_san


def build_partial_strategy_tree(
    fen: str,
    engine: Any,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    multipv: int = 3,
    pv_plies: int = 4,
    student_move: str | None = None,
) -> dict[str, Any]:
    """Build a compact, JSON-serializable Partial Strategy Tree via Stockfish MultiPV.

    Args:
        fen: Board position in FEN notation.
        engine: Active ``chess.engine.SimpleEngine`` (or compatible mock).
        depth: Search depth in plies.
        multipv: Number of top candidate lines to extract.
        pv_plies: Maximum number of plies per principal variation.
        student_move: Optional UCI move played by the student to evaluate as a
            dedicated comparison branch.

    Returns:
        Dictionary with keys ``fen``, ``side_to_move``, ``best_move``,
        ``top_lines``, and ``student_branch``.
    """
    board = chess.Board(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"

    legal_count = board.legal_moves.count()
    if legal_count == 0:
        return {
            "fen": fen,
            "side_to_move": side_to_move,
            "best_move": None,
            "top_lines": [],
            "student_branch": None,
        }

    actual_multipv = max(1, min(int(multipv), legal_count))
    limit = chess.engine.Limit(depth=int(depth))
    raw_infos = engine.analyse(board, limit, multipv=actual_multipv)
    infos = raw_infos if isinstance(raw_infos, list) else [raw_infos]

    top_lines: list[dict[str, Any]] = []
    for info in infos:
        pv = list(info.get("pv", []))
        if not pv:
            continue
        first_move = pv[0]
        if first_move not in board.legal_moves:
            continue
        pv_uci, pv_san = _format_pv(board, pv, pv_plies)
        top_lines.append(
            {
                "move": first_move.uci(),
                "san": board.san(first_move),
                "eval_cp": _score_to_cp(info.get("score"), board.turn),
                "pv_uci": pv_uci,
                "pv_san": pv_san,
            }
        )

    best_move = top_lines[0]["move"] if top_lines else None
    best_eval_cp = top_lines[0]["eval_cp"] if top_lines else 0

    student_branch: dict[str, Any] | None = None
    if student_move is not None:
        try:
            move_obj = chess.Move.from_uci(student_move)
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move: {student_move!r}") from exc
        if move_obj not in board.legal_moves:
            raise ValueError(f"Illegal move {student_move!r} for FEN {fen!r}")

        matched = next((line for line in top_lines if line["move"] == student_move), None)
        if matched is not None:
            student_branch = {
                **matched,
                "classification": classify_move(
                    eval_before_cp=best_eval_cp,
                    eval_after_cp=matched["eval_cp"],
                    is_best_move=(student_move == best_move),
                ),
            }
        else:
            san = board.san(move_obj)
            board_after = board.copy()
            board_after.push(move_obj)
            if board_after.is_checkmate():
                student_eval_cp = _MATE_SCORE_CP
                pv_uci, pv_san = [student_move], [san]
            elif board_after.is_game_over():
                student_eval_cp = 0
                pv_uci, pv_san = [student_move], [san]
            else:
                reply_info = engine.analyse(board_after, limit)
                student_eval_cp = _score_to_cp(reply_info.get("score"), board.turn)
                full_pv = [move_obj, *list(reply_info.get("pv", []))]
                pv_uci, pv_san = _format_pv(board, full_pv, pv_plies)

            student_branch = {
                "move": student_move,
                "san": san,
                "eval_cp": student_eval_cp,
                "pv_uci": pv_uci,
                "pv_san": pv_san,
                "classification": classify_move(
                    eval_before_cp=best_eval_cp,
                    eval_after_cp=student_eval_cp,
                    is_best_move=(student_move == best_move),
                ),
            }

    return {
        "fen": fen,
        "side_to_move": side_to_move,
        "best_move": best_move,
        "top_lines": top_lines,
        "student_branch": student_branch,
    }
