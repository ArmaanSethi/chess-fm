#!/usr/bin/env python3
"""Full-game PGN review and cross-game weakness profiling for ChessFM Coach.

Provides:
- ``review_pgn_game``: parses a PGN, analyzes every ply with Stockfish (or a
  lightweight material/tactical heuristic fallback when ``engine is None``),
  classifies moves, computes accuracy %, identifies the top turning points, and
  tags suboptimal moves with structured weakness categories.
- ``build_weakness_profile``: aggregates recurring weakness categories across
  one or more game reviews, ranks the top 3 study priorities, and extracts
  deduplicated practice drill positions.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

from chessfm.rl.engine import _cp_from_white, close_engine, open_engine

# Standard centipawn piece values for heuristic evaluation and material counts.
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Center squares rewarded lightly by the heuristic fallback.
_CENTER_SQUARES = frozenset({chess.D4, chess.E4, chess.D5, chess.E5})
_EXTENDED_CENTER = frozenset(
    {
        chess.C3,
        chess.D3,
        chess.E3,
        chess.F3,
        chess.C4,
        chess.F4,
        chess.C5,
        chess.F5,
        chess.C6,
        chess.D6,
        chess.E6,
        chess.F6,
    }
)

# Canonical weakness categories and coaching metadata.
WEAKNESS_CATEGORIES: tuple[str, ...] = (
    "opening_inaccuracy",
    "hanging_piece",
    "missed_tactic",
    "missed_mate",
    "endgame_conversion",
    "positional_drift",
)

WEAKNESS_METADATA: dict[str, dict[str, str]] = {
    "opening_inaccuracy": {
        "description": "Early-game inaccuracies in development, center control, or king safety.",
        "recommendation": "Review opening principles: develop minor pieces, control the center, and castle early.",
    },
    "hanging_piece": {
        "description": "Leaving pieces undefended or stepping onto squares where they can be won.",
        "recommendation": "Run a blunder check before every move: verify all captures, checks, and undefended pieces.",
    },
    "missed_tactic": {
        "description": "Overlooking forcing tactical shots such as forks, pins, skewers, or winning captures.",
        "recommendation": "Practice daily tactical puzzles focused on checks, captures, and threats (CCT).",
    },
    "missed_mate": {
        "description": "Missing a forced checkmate sequence when the opponent's king is exposed.",
        "recommendation": "Drill mate-in-1 through mate-in-3 patterns and always scan forcing checks first.",
    },
    "endgame_conversion": {
        "description": "Losing evaluation in simplified positions with few pieces remaining.",
        "recommendation": "Study king activity, passed pawns, and fundamental rook and pawn endgames.",
    },
    "positional_drift": {
        "description": "Gradual strategic concessions in the middlegame without an immediate tactical blow.",
        "recommendation": "Improve your worst-placed piece and formulate a concrete plan around pawn structure.",
    },
}

_SUBOPTIMAL_CLASSES = frozenset({"Inaccuracy", "Mistake", "Blunder"})
_ENGINE_ERRORS = (chess.engine.EngineError, OSError)


def cp_to_win_prob(cp: float) -> float:
    """Convert a centipawn score (mover's perspective) to win probability in [0, 100].

    Uses the standard Lichess logistic scaling so WDL swings are comparable
    across Stockfish and heuristic evaluations.
    """
    clamped = max(-10000.0, min(10000.0, float(cp)))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * clamped)) - 1.0)


def move_accuracy(wdl_swing: float) -> float:
    """Compute per-move accuracy percentage in [0, 100] from a WDL swing."""
    swing = max(0.0, float(wdl_swing))
    if swing == 0.0:
        return 100.0
    acc = 103.1668 * math.exp(-0.04354 * swing) - 3.1669
    return max(0.0, min(100.0, acc))


def _non_pawn_material(board: chess.Board) -> int:
    """Return total non-king, non-pawn material on ``board`` in centipawns."""
    total = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count = len(board.pieces(piece_type, chess.WHITE)) + len(
            board.pieces(piece_type, chess.BLACK)
        )
        total += count * PIECE_VALUES[piece_type]
    return total


def _is_endgame(board: chess.Board) -> bool:
    """Return True if ``board`` is in an endgame phase."""
    npm = _non_pawn_material(board)
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    return npm <= 2600 or (queens == 0 and npm <= 3400)


def _static_eval(board: chess.Board, for_color: chess.Color) -> float:
    """Fast static material + center/mobility heuristic in centipawns."""
    if board.is_checkmate():
        return -10000.0 if board.turn == for_color else 10000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = 0.0
    for sq, piece in board.piece_map().items():
        val = float(PIECE_VALUES[piece.piece_type])
        if sq in _CENTER_SQUARES:
            val += 15.0 if piece.piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP) else 5.0
        elif sq in _EXTENDED_CENTER and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            val += 8.0
        if piece.color == for_color:
            score += val
        else:
            score -= val

    # Penalize pieces of the side that just moved if they are left en prise
    # (attacked by the side to move and either undefended or attacked by a
    # strictly lower-value piece).
    opponent = board.turn
    mover = not opponent
    worst_loss = 0.0
    for sq, piece in board.piece_map().items():
        if piece.color != mover or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(opponent, sq)
        if not attackers:
            continue
        defenders = board.attackers(mover, sq)
        piece_val = float(PIECE_VALUES[piece.piece_type])
        if not defenders:
            worst_loss = max(worst_loss, piece_val * 0.9)
        else:
            min_attacker = min(
                PIECE_VALUES[board.piece_at(att).piece_type]
                for att in attackers
                if board.piece_at(att) is not None
            )
            if min_attacker < piece_val:
                worst_loss = max(worst_loss, (piece_val - min_attacker) * 0.85)

    if worst_loss > 0.0:
        if mover == for_color:
            score -= worst_loss
        else:
            score += worst_loss

    return score


def _heuristic_analyze(board: chess.Board) -> tuple[str | None, float, int | None]:
    """Lightweight 1-ply tactical/material analyzer used when Stockfish is absent.

    Returns ``(best_move_uci, best_cp_for_turn, mate_in_plies_or_none)``.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        if board.is_checkmate():
            return None, -10000.0, 0
        return None, 0.0, None

    turn = board.turn
    best_move_uci: str | None = None
    best_score = -float("inf")
    mate_found: int | None = None

    for move in legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move.uci(), 10000.0, 1

        # Check if opponent has an immediate mate-in-1 after this move.
        opp_has_mate = False
        if board.is_check() or len(list(board.legal_moves)) <= 35:
            for reply in board.legal_moves:
                board.push(reply)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    opp_has_mate = True
                    break

        if opp_has_mate:
            val = -9900.0
        else:
            val = _static_eval(board, turn)
        board.pop()

        if val > best_score:
            best_score = val
            best_move_uci = move.uci()

    return best_move_uci, best_score, mate_found


def _analyze_position(
    board: chess.Board,
    engine: Any | None,
    depth: int,
    for_color: chess.Color,
) -> tuple[str | None, float, int | None]:
    """Analyze ``board`` with Stockfish (or heuristic fallback) from ``for_color``'s view.

    Returns ``(best_move_uci, cp_score, mate_score_for_color)``.
    """
    if engine is not None:
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            pv = info.get("pv")
            best_uci = pv[0].uci() if pv else None
            cp = _cp_from_white(info, for_color)
            score = info["score"].white()
            mate: int | None = None
            if score.is_mate():
                raw_mate = score.mate()
                if raw_mate is not None:
                    mate = raw_mate if for_color == chess.WHITE else -raw_mate
            return best_uci, cp, mate
        except _ENGINE_ERRORS:
            pass

    best_uci, turn_cp, turn_mate = _heuristic_analyze(board)
    cp = turn_cp if board.turn == for_color else -turn_cp
    mate = turn_mate if (turn_mate is None or board.turn == for_color) else -turn_mate
    return best_uci, cp, mate


def _classify_move(
    played_uci: str,
    best_uci: str | None,
    cp_loss: float,
    wdl_swing: float,
    missed_mate: bool,
    blundered_mate: bool,
) -> str:
    """Classify a move into Best / Good / Inaccuracy / Mistake / Blunder."""
    if played_uci == best_uci:
        return "Best"
    if missed_mate or blundered_mate or cp_loss > 200.0 or wdl_swing > 20.0:
        return "Blunder"
    if cp_loss > 100.0 or wdl_swing > 10.0:
        return "Mistake"
    if cp_loss > 50.0 or wdl_swing > 5.0:
        return "Inaccuracy"
    if cp_loss > 15.0 or wdl_swing > 2.0:
        return "Good"
    return "Best"


def _has_hanging_piece(board_after: chess.Board, mover: chess.Color) -> bool:
    """Return True if ``mover`` has a non-pawn piece left en prise in ``board_after``."""
    opponent = not mover
    for sq, piece in board_after.piece_map().items():
        if piece.color != mover or piece.piece_type in (chess.KING, chess.PAWN):
            continue
        attackers = board_after.attackers(opponent, sq)
        if not attackers:
            continue
        defenders = board_after.attackers(mover, sq)
        if not defenders:
            return True
        piece_val = PIECE_VALUES[piece.piece_type]
        min_attacker = min(
            PIECE_VALUES[board_after.piece_at(att).piece_type]
            for att in attackers
            if board_after.piece_at(att) is not None
        )
        if min_attacker < piece_val:
            return True
    return False


def _tag_weakness(
    board_before: chess.Board,
    board_after: chess.Board,
    mover: chess.Color,
    best_uci: str | None,
    cp_before: float,
    cp_after: float,
    cp_loss: float,
    mate_before: int | None,
    ply: int,
) -> str:
    """Assign a weakness category to a suboptimal move."""
    if (mate_before is not None and mate_before > 0) or (cp_before >= 9000.0 and cp_after < 9000.0):
        return "missed_mate"

    if cp_loss >= 150.0 and _has_hanging_piece(board_after, mover):
        return "hanging_piece"

    if best_uci is not None and cp_loss >= 120.0:
        try:
            best_move = chess.Move.from_uci(best_uci)
            if best_move in board_before.legal_moves:
                is_tactical = (
                    board_before.is_capture(best_move)
                    or best_move.promotion is not None
                    or board_before.gives_check(best_move)
                )
                if is_tactical:
                    return "missed_tactic"
        except ValueError:
            pass

    if ply <= 16:
        return "opening_inaccuracy"

    if _is_endgame(board_before):
        return "endgame_conversion"

    return "positional_drift"


def _normalize_student_color(student_color: str) -> tuple[str, set[chess.Color]]:
    """Validate and normalize ``student_color`` into label + set of chess colors."""
    norm = str(student_color).strip().lower()
    if norm in ("white", "w"):
        return "white", {chess.WHITE}
    if norm in ("black", "b"):
        return "black", {chess.BLACK}
    if norm == "both":
        return "both", {chess.WHITE, chess.BLACK}
    raise ValueError(
        f"Invalid student_color {student_color!r}; expected 'white', 'black', or 'both'."
    )


def review_pgn_game(
    pgn_text: str,
    engine: Any | None = None,
    student_color: str = "white",
    student_elo: int = 1200,
    depth: int = 8,
    max_critical: int = 5,
) -> dict[str, Any]:
    """Review a chess game from PGN text and identify critical turning points.

    Args:
        pgn_text: Full PGN string (headers optional, movetext required for move
            analysis).
        engine: Optional open ``chess.engine.SimpleEngine`` (Stockfish). When
            ``None``, uses a fast deterministic material/tactical heuristic so
            unit tests and CPU-only flows run without Stockfish.
        student_color: ``"white"``, ``"black"``, or ``"both"``.
        student_elo: Approximate student Elo rating for metadata/profiling.
        depth: Stockfish search depth when ``engine`` is provided.
        max_critical: Maximum number of critical turning points to return.

    Returns:
        Structured game review dictionary containing ``accuracy``, move
        ``counts``, ``weakness_counts``, ``critical_moments``, and per-ply
        ``moves``.
    """
    if not isinstance(pgn_text, str) or not pgn_text.strip():
        raise ValueError("pgn_text must be a non-empty PGN string.")

    norm_color, target_colors = _normalize_student_color(student_color)
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse a valid PGN game from pgn_text.")

    board = game.board()
    mainline_moves = list(game.mainline_moves())

    counts: dict[str, int] = {
        "Best": 0,
        "Good": 0,
        "Inaccuracy": 0,
        "Mistake": 0,
        "Blunder": 0,
    }
    weakness_counts: dict[str, int] = {cat: 0 for cat in WEAKNESS_CATEGORIES}
    all_moves: list[dict[str, Any]] = []
    student_accuracies: list[float] = []

    for ply_idx, move in enumerate(mainline_moves, start=1):
        mover = board.turn
        move_number = board.fullmove_number
        fen_before = board.fen()
        played_uci = move.uci()
        played_san = board.san(move)

        best_uci, cp_before, mate_before = _analyze_position(board, engine, depth, mover)
        best_san: str | None = None
        if best_uci is not None:
            try:
                bm = chess.Move.from_uci(best_uci)
                if bm in board.legal_moves:
                    best_san = board.san(bm)
            except ValueError:
                best_san = None

        board_before_copy = board.copy(stack=False)
        board.push(move)

        if board.is_checkmate():
            cp_after = 10000.0
            mate_after: int | None = 0
        elif board.is_stalemate() or board.is_insufficient_material():
            cp_after = 0.0
            mate_after = None
        else:
            _, cp_after, mate_after = _analyze_position(board, engine, depth, mover)

        cp_loss = 0.0 if played_uci == best_uci else max(0.0, cp_before - cp_after)
        wdl_before = cp_to_win_prob(cp_before)
        wdl_after = cp_to_win_prob(cp_after)
        wdl_swing = 0.0 if played_uci == best_uci else max(0.0, wdl_before - wdl_after)

        missed_mate = bool(
            ((mate_before is not None and mate_before > 0) or cp_before >= 9000.0)
            and cp_after < 9000.0
            and played_uci != best_uci
        )
        blundered_mate = bool(
            (mate_after is not None and mate_after < 0)
            or (cp_before > -9000.0 and cp_after <= -9000.0)
        )

        classification = _classify_move(
            played_uci=played_uci,
            best_uci=best_uci,
            cp_loss=cp_loss,
            wdl_swing=wdl_swing,
            missed_mate=missed_mate,
            blundered_mate=blundered_mate,
        )

        category: str | None = None
        if classification in _SUBOPTIMAL_CLASSES:
            category = _tag_weakness(
                board_before=board_before_copy,
                board_after=board,
                mover=mover,
                best_uci=best_uci,
                cp_before=cp_before,
                cp_after=cp_after,
                cp_loss=cp_loss,
                mate_before=mate_before,
                ply=ply_idx,
            )

        is_student_move = mover in target_colors
        move_acc = move_accuracy(wdl_swing)

        record: dict[str, Any] = {
            "ply": ply_idx,
            "move_number": move_number,
            "color": "white" if mover == chess.WHITE else "black",
            "is_student": is_student_move,
            "fen": fen_before,
            "played_move": played_uci,
            "played_san": played_san,
            "best_move": best_uci,
            "best_san": best_san,
            "cp_before": round(cp_before, 1),
            "cp_after": round(cp_after, 1),
            "cp_loss": round(cp_loss, 1),
            "wdl_before": round(wdl_before, 2),
            "wdl_after": round(wdl_after, 2),
            "wdl_swing": round(wdl_swing, 2),
            "accuracy": round(move_acc, 1),
            "classification": classification,
            "category": category,
        }
        all_moves.append(record)

        if is_student_move:
            counts[classification] += 1
            student_accuracies.append(move_acc)
            if category is not None:
                weakness_counts[category] += 1

    avg_accuracy = (
        round(sum(student_accuracies) / len(student_accuracies), 1) if student_accuracies else 100.0
    )

    # Select top turning points for the student sorted by WDL swing then cp_loss.
    student_candidates = [
        m
        for m in all_moves
        if m["is_student"] and (m["wdl_swing"] > 0.0 or m["classification"] in _SUBOPTIMAL_CLASSES)
    ]
    student_candidates.sort(key=lambda m: (m["wdl_swing"], m["cp_loss"]), reverse=True)
    critical_moments = student_candidates[: max(0, max_critical)]

    return {
        "headers": dict(game.headers),
        "student_color": norm_color,
        "student_elo": student_elo,
        "engine_used": "stockfish" if engine is not None else "heuristic",
        "total_plies": len(all_moves),
        "student_moves": len(student_accuracies),
        "accuracy": avg_accuracy,
        "counts": counts,
        "weakness_counts": {k: v for k, v in weakness_counts.items() if v > 0},
        "critical_moments": critical_moments,
        "moves": all_moves,
    }


def build_weakness_profile(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate recurring weaknesses across one or more game reviews.

    Ranks the top 3 study priorities and extracts deduplicated practice drill
    positions formatted as ``{"fen", "best_move", "category", "student_mistake"}``.
    """
    if not reviews:
        return {
            "games_analyzed": 0,
            "average_accuracy": 0.0,
            "total_mistakes": 0,
            "category_breakdown": {},
            "top_priorities": [],
            "drills": [],
        }

    cat_stats: dict[str, dict[str, float | int]] = {}
    drill_candidates: list[tuple[float, float, dict[str, Any]]] = []
    accuracies: list[float] = []

    for review in reviews:
        if "accuracy" in review and review["accuracy"] is not None:
            accuracies.append(float(review["accuracy"]))

        # Prefer full move list; fall back to critical_moments if only those are present.
        source_moves = review.get("moves")
        if source_moves is None:
            source_moves = review.get("critical_moments", [])

        for m in source_moves:
            if not m.get("is_student", True):
                continue
            category = m.get("category")
            if not category:
                continue
            cp_loss = float(m.get("cp_loss", 0.0))
            wdl_swing = float(m.get("wdl_swing", 0.0))

            entry = cat_stats.setdefault(
                category,
                {"count": 0, "total_cp_loss": 0.0, "total_wdl_swing": 0.0},
            )
            entry["count"] = int(entry["count"]) + 1
            entry["total_cp_loss"] = float(entry["total_cp_loss"]) + cp_loss
            entry["total_wdl_swing"] = float(entry["total_wdl_swing"]) + wdl_swing

            fen = m.get("fen")
            best_move = m.get("best_move")
            student_mistake = m.get("played_move") or m.get("student_mistake")
            if fen and best_move and student_mistake:
                drill_candidates.append(
                    (
                        wdl_swing,
                        cp_loss,
                        {
                            "fen": fen,
                            "best_move": best_move,
                            "category": category,
                            "student_mistake": student_mistake,
                        },
                    )
                )

    total_mistakes = sum(int(s["count"]) for s in cat_stats.values())
    category_breakdown: dict[str, dict[str, Any]] = {}
    ranked: list[tuple[float, dict[str, Any]]] = []

    for cat, stats in cat_stats.items():
        count = int(stats["count"])
        total_cp = float(stats["total_cp_loss"])
        avg_cp = round(total_cp / count, 1) if count else 0.0
        avg_wdl = round(float(stats["total_wdl_swing"]) / count, 2) if count else 0.0
        # Priority score balances frequency and severity.
        severity_score = round(count * (1.0 + avg_wdl / 25.0) + total_cp / 500.0, 3)
        meta = WEAKNESS_METADATA.get(
            cat,
            {
                "description": f"Recurring errors in category '{cat}'.",
                "recommendation": "Review annotated games and drill these positions.",
            },
        )
        summary = {
            "category": cat,
            "count": count,
            "total_cp_loss": round(total_cp, 1),
            "avg_cp_loss": avg_cp,
            "avg_wdl_swing": avg_wdl,
            "priority_score": severity_score,
            "description": meta["description"],
            "recommendation": meta["recommendation"],
        }
        category_breakdown[cat] = summary
        ranked.append((severity_score, summary))

    ranked.sort(key=lambda item: (item[0], item[1]["count"]), reverse=True)
    top_priorities = [item[1] for item in ranked[:3]]
    top_cats = {p["category"] for p in top_priorities}

    # Deduplicate drills by (fen, best_move), prioritizing top study categories & largest swings.
    drill_candidates.sort(
        key=lambda item: (item[2]["category"] in top_cats, item[0], item[1]),
        reverse=True,
    )
    seen: set[tuple[str, str]] = set()
    drills: list[dict[str, Any]] = []
    for _, _, drill in drill_candidates:
        key = (drill["fen"], drill["best_move"])
        if key in seen:
            continue
        seen.add(key)
        drills.append(drill)

    avg_acc = round(sum(accuracies) / len(accuracies), 1) if accuracies else 0.0

    return {
        "games_analyzed": len(reviews),
        "average_accuracy": avg_acc,
        "total_mistakes": total_mistakes,
        "category_breakdown": category_breakdown,
        "top_priorities": top_priorities,
        "drills": drills,
    }


def format_review_report(review: dict[str, Any]) -> str:
    """Format a ``review_pgn_game`` result into a crisp terminal report."""
    headers = review.get("headers", {})
    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    event = headers.get("Event", "Game Review")
    result = headers.get("Result", "*")

    lines = [
        f"ChessFM Game Review — {event}",
        f"Players: {white} vs {black} ({result})",
        (
            f"Student: {review['student_color']} (Elo {review['student_elo']}) | "
            f"Engine: {review['engine_used']} | "
            f"Accuracy: {review['accuracy']:.1f}% "
            f"({review['student_moves']}/{review['total_plies']} plies)"
        ),
    ]

    counts = review.get("counts", {})
    lines.append(
        "Move Quality: "
        + " | ".join(
            f"{label}: {counts.get(label, 0)}"
            for label in ("Best", "Good", "Inaccuracy", "Mistake", "Blunder")
        )
    )

    weakness_counts = review.get("weakness_counts", {})
    if weakness_counts:
        summary = ", ".join(f"{cat} ({cnt})" for cat, cnt in weakness_counts.items())
        lines.append(f"Weaknesses: {summary}")

    critical = review.get("critical_moments", [])
    if critical:
        lines.append("Critical Moments:")
        for idx, moment in enumerate(critical, start=1):
            move_prefix = (
                f"{moment['move_number']}."
                if moment["color"] == "white"
                else f"{moment['move_number']}..."
            )
            best_label = moment.get("best_san") or moment.get("best_move") or "-"
            cat_suffix = f" [{moment['category']}]" if moment.get("category") else ""
            lines.append(
                f"  {idx}. {move_prefix} {moment['played_san']} ({moment['classification']}) "
                f"— best: {best_label} | -{moment['cp_loss']:.0f} cp, "
                f"WDL swing {moment['wdl_swing']:.1f}%{cat_suffix}"
            )
    else:
        lines.append("Critical Moments: none")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m chessfm.coach.game_review``."""
    parser = argparse.ArgumentParser(
        prog="python -m chessfm.coach.game_review",
        description="Review a chess game from a PGN file using Stockfish (with heuristic fallback).",
    )
    parser.add_argument(
        "--pgn",
        type=Path,
        required=True,
        help="Path to the PGN file to review.",
    )
    parser.add_argument(
        "--color",
        default="white",
        choices=("white", "black", "both"),
        help="Student side to evaluate (default: white).",
    )
    parser.add_argument(
        "--elo",
        type=int,
        default=1200,
        help="Approximate student Elo rating (default: 1200).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Stockfish analysis depth when available (default: 8).",
    )
    parser.add_argument(
        "--max-critical",
        type=int,
        default=5,
        help="Maximum number of critical turning points to report (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output the structured game review as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for reviewing a PGN file from the terminal."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.pgn.is_file():
        print(f"Error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1

    pgn_text = args.pgn.read_text(encoding="utf-8")
    engine = open_engine(threads=1, hash_mb=16)
    try:
        review = review_pgn_game(
            pgn_text,
            engine=engine,
            student_color=args.color,
            student_elo=args.elo,
            depth=args.depth,
            max_critical=args.max_critical,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_engine(engine)

    if args.as_json:
        print(json.dumps(review, indent=2))
    else:
        print(format_review_report(review))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
