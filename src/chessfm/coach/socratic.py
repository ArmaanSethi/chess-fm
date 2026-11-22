#!/usr/bin/env python3
"""3-Level Socratic Hint Engine with a strict 0%-spoiler guarantee.

Generates progressive pedagogical hints for a chess position:
- Level 1 (Nudge): High-level strategic theme or piece type only (no origin
  square, no destination square, no move).
- Level 2 (Guided): Tactical motif, threat, or target area/square (no exact
  move and never both origin and destination squares).
- Level 3 (Solution): Full verified move (SAN + UCI), short verified
  continuation (when ``pv_ucis`` is provided), and a clear explanation.
"""

from __future__ import annotations

import re
from typing import Any

import chess

_PIECE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

_CENTER_SQUARES = frozenset({chess.D4, chess.D5, chess.E4, chess.E5})
_EXTENDED_CENTER = frozenset(
    {
        chess.C3,
        chess.D3,
        chess.E3,
        chess.F3,
        chess.C4,
        chess.D4,
        chess.E4,
        chess.F4,
        chess.C5,
        chess.D5,
        chess.E5,
        chess.F5,
        chess.C6,
        chess.D6,
        chess.E6,
        chess.F6,
    }
)


def _token_pattern(token: str) -> re.Pattern[str]:
    """Build a boundary-aware regex matching ``token`` in prose."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", re.IGNORECASE)


def contains_move_spoiler(hint_text: str, best_move_uci: str, fen: str) -> bool:
    """Return True if ``hint_text`` leaks ``best_move_uci`` in any form.

    Detects:
    - Exact UCI move (including promotion suffix and base 4-char coordinate
      form, as well as hyphen/arrow variants like ``e2-e4`` or ``e2->e4``).
    - Exact SAN form (e.g. ``Nf3``, ``Bxf7+``, ``O-O``, ``e8=Q``), with or
      without trailing ``+``/``#`` and ``0-0``/``0-0-0`` castling aliases.
    - Co-occurrence of both the origin and destination squares anywhere in
      ``hint_text``.
    """
    if not hint_text:
        return False

    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"Move '{best_move_uci}' is not legal in FEN '{fen}'")

    uci_lower = best_move_uci.lower()
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)

    # 1. Exact UCI (and 4-char base for promotions, plus hyphen/arrow forms).
    uci_variants = {
        uci_lower,
        f"{from_sq}{to_sq}",
        f"{from_sq}-{to_sq}",
        f"{from_sq}->{to_sq}",
        f"{from_sq}\u2192{to_sq}",
    }
    for variant in uci_variants:
        if _token_pattern(variant).search(hint_text):
            return True

    # 2. SAN form (full and stripped of +/#, plus 0-0 / 0-0-0 aliases).
    san = board.san(move)
    san_base = san.rstrip("+#")
    san_variants = {san, san_base}
    if san_base == "O-O":
        san_variants.update({"0-0", "0-0+", "0-0#"})
    elif san_base == "O-O-O":
        san_variants.update({"0-0-0", "0-0-0+", "0-0-0#"})

    for variant in san_variants:
        if _token_pattern(variant).search(hint_text):
            return True

    # 3. Both origin and destination squares mentioned anywhere in the hint.
    has_from = bool(_token_pattern(from_sq).search(hint_text))
    has_to = bool(_token_pattern(to_sq).search(hint_text))
    if has_from and has_to:
        return True

    return False


def _contains_any_move_square(hint_text: str, move: chess.Move) -> bool:
    """Return True if either the origin or destination square appears in text."""
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    return bool(
        _token_pattern(from_sq).search(hint_text) or _token_pattern(to_sq).search(hint_text)
    )


def _board_zone(square: chess.Square) -> str:
    """Describe the board region of ``square`` without naming the square."""
    if square in _CENTER_SQUARES:
        return "the center"
    file_idx = chess.square_file(square)
    if file_idx >= 5:
        return "the kingside"
    if file_idx <= 2:
        return "the queenside"
    return "the central files"


def _analyze_move(board: chess.Board, move: chess.Move) -> dict[str, Any]:
    """Extract deterministic tactical and strategic features of ``move``."""
    piece = board.piece_at(move.from_square)
    assert piece is not None
    piece_name = _PIECE_NAMES[piece.piece_type]
    mover_color = board.turn
    enemy_color = not mover_color

    is_castling = board.is_castling(move)
    is_kingside_castle = board.is_kingside_castling(move)
    is_promotion = move.promotion is not None
    promo_name = _PIECE_NAMES[move.promotion] if move.promotion else None
    is_en_passant = board.is_en_passant(move)
    is_capture = board.is_capture(move)

    captured_piece = (
        chess.Piece(chess.PAWN, enemy_color) if is_en_passant else board.piece_at(move.to_square)
    )
    captured_name = _PIECE_NAMES[captured_piece.piece_type] if captured_piece else None

    gives_check = board.gives_check(move)

    after = board.copy()
    after.push(move)
    is_checkmate = after.is_checkmate()
    is_discovered_check = gives_check and (move.to_square not in after.checkers())

    # Detect fork: moved piece attacks >= 2 valuable enemy pieces (or undefended pieces).
    attacked_squares = after.attacks(move.to_square)
    high_value_targets: list[str] = []
    for sq in attacked_squares:
        target = after.piece_at(sq)
        if target is None or target.color != enemy_color:
            continue
        is_higher = _PIECE_VALUES[target.piece_type] > _PIECE_VALUES[piece.piece_type]
        is_king = target.piece_type == chess.KING
        is_undefended = not after.is_attacked_by(enemy_color, sq)
        if is_king or is_higher or (is_undefended and target.piece_type != chess.PAWN):
            high_value_targets.append(_PIECE_NAMES[target.piece_type])
    is_fork = len(high_value_targets) >= 2

    # Detect development of a minor piece off the back rank.
    back_rank = 0 if mover_color == chess.WHITE else 7
    is_development = (
        piece.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(move.from_square) == back_rank
        and chess.square_rank(move.to_square) != back_rank
    )

    # Detect rook moving to the 7th/2nd rank.
    seventh_rank = 6 if mover_color == chess.WHITE else 1
    is_seventh_rank_rook = (
        piece.piece_type == chess.ROOK and chess.square_rank(move.to_square) == seventh_rank
    )

    # Collect detected themes in priority order.
    detected_themes: list[str] = []
    if is_checkmate:
        enemy_king_sq = after.king(enemy_color)
        enemy_back = 7 if mover_color == chess.WHITE else 0
        if enemy_king_sq is not None and chess.square_rank(enemy_king_sq) == enemy_back:
            detected_themes.append("back-rank checkmate")
        else:
            detected_themes.append("checkmate")
    if is_promotion:
        detected_themes.append("pawn promotion")
    if is_castling:
        detected_themes.append("king safety")
    if is_discovered_check:
        detected_themes.append("discovered check")
    elif gives_check and not is_checkmate:
        detected_themes.append("forcing check")
    if is_fork:
        detected_themes.append("fork")
    if is_capture:
        detected_themes.append("winning material")
    if is_seventh_rank_rook:
        detected_themes.append("7th rank invasion")
    if is_development:
        detected_themes.append("piece development")
    if move.to_square in _CENTER_SQUARES:
        detected_themes.append("center control")
    if not detected_themes:
        detected_themes.append("piece activity")

    return {
        "piece_type": piece.piece_type,
        "piece_name": piece_name,
        "is_castling": is_castling,
        "is_kingside_castle": is_kingside_castle,
        "is_promotion": is_promotion,
        "promo_name": promo_name,
        "is_en_passant": is_en_passant,
        "is_capture": is_capture,
        "captured_name": captured_name,
        "gives_check": gives_check,
        "is_checkmate": is_checkmate,
        "is_discovered_check": is_discovered_check,
        "is_fork": is_fork,
        "is_development": is_development,
        "is_seventh_rank_rook": is_seventh_rank_rook,
        "to_file": chess.FILE_NAMES[chess.square_file(move.to_square)],
        "to_square_name": chess.square_name(move.to_square),
        "zone": _board_zone(move.to_square),
        "detected_themes": detected_themes,
    }


def _build_level_1_hint(info: dict[str, Any], clean_themes: list[str]) -> str:
    """Build a Level 1 nudge: strategic theme or piece type only, zero squares."""
    piece_name = info["piece_name"]
    theme_note = f" Focus on the theme of {clean_themes[0]}." if clean_themes else ""

    if info["is_checkmate"]:
        return (
            f"Look for a decisive forcing move with your {piece_name} — "
            f"there is an immediate mating idea available."
        )
    if info["is_castling"]:
        return f"Think about king safety and how to bring your rooks into the game.{theme_note}"
    if info["is_promotion"]:
        return (
            "Pay close attention to your most advanced pawn — "
            "it can dramatically change the material balance."
        )
    if info["is_fork"]:
        return (
            f"Look for a tactical idea with your {piece_name} that puts "
            "pressure on more than one target at the same time."
        )
    if info["is_discovered_check"]:
        return (
            f"Consider moving your {piece_name} to unleash a hidden attack "
            "against the opponent's king."
        )
    if info["gives_check"]:
        return f"Look for a forcing check with your {piece_name} to seize the initiative."
    if info["is_capture"]:
        return (
            f"Check whether your {piece_name} can capture an enemy piece "
            f"to win material or remove a key defender.{theme_note}"
        )
    if info["is_development"]:
        return f"Focus on developing your {piece_name} to a more active role.{theme_note}"
    return f"Take a closer look at how your {piece_name} can improve your position.{theme_note}"


def _build_level_2_hint(info: dict[str, Any], clean_themes: list[str]) -> str:
    """Build a Level 2 guided hint: motif, threat, or target area/square."""
    piece_name = info["piece_name"]
    zone = info["zone"]
    to_file = info["to_file"]
    to_sq = info["to_square_name"]
    is_pawn = info["piece_type"] == chess.PAWN
    theme_prefix = f"Theme: {clean_themes[0]}. " if clean_themes else ""

    if info["is_castling"]:
        side = "kingside" if info["is_kingside_castle"] else "queenside"
        return f"{theme_prefix}Tuck your king to safety on the {side} while connecting your rooks."

    if info["is_promotion"]:
        return (
            f"{theme_prefix}Advance your passed pawn along the {to_file}-file "
            "to the promotion rank to create a decisive material advantage."
        )

    if is_pawn:
        # Never mention the bare destination square for pawn moves because
        # pawn-push SAN (e.g. "e4") equals the destination square name.
        if info["is_capture"] and info["captured_name"]:
            return (
                f"{theme_prefix}Use a pawn capture toward the {to_file}-file "
                f"in {zone} to remove the opponent's {info['captured_name']}."
            )
        if info["gives_check"]:
            return (
                f"{theme_prefix}Push a pawn along the {to_file}-file in {zone} "
                "to deliver a forcing check."
            )
        return (
            f"{theme_prefix}Advance a pawn along the {to_file}-file to gain "
            f"space and control in {zone}."
        )

    # Non-pawn, non-castling moves: safe to reference the target square (never
    # the origin square, and never the SAN string).
    if info["is_checkmate"]:
        return (
            f"{theme_prefix}Target the {to_sq} square in {zone} with your "
            f"{piece_name} to deliver checkmate."
        )
    if info["is_fork"]:
        return (
            f"{theme_prefix}Look at the {to_sq} square in {zone} — your "
            f"{piece_name} can land there to fork multiple enemy pieces."
        )
    if info["is_discovered_check"]:
        return (
            f"{theme_prefix}Moving your {piece_name} toward {zone} uncovers "
            "a discovered check against the enemy king."
        )
    if info["gives_check"]:
        return (
            f"{theme_prefix}Aim your {piece_name} at the {to_sq} square in "
            f"{zone} to put the opponent's king in check."
        )
    if info["is_capture"] and info["captured_name"]:
        return (
            f"{theme_prefix}Target the opponent's {info['captured_name']} on "
            f"{to_sq} in {zone} with your {piece_name}."
        )
    if info["is_seventh_rank_rook"]:
        return (
            f"{theme_prefix}Infiltrate the 7th rank on the {to_file}-file "
            "with your rook to pressure the opponent's camp."
        )
    return (
        f"{theme_prefix}Aim your {piece_name} toward the {to_sq} square to "
        f"increase pressure in {zone}."
    )


def _verify_continuation(
    board: chess.Board,
    best_move: chess.Move,
    pv_ucis: list[str] | None,
    max_plies: int = 6,
) -> list[str]:
    """Validate ``pv_ucis`` against ``board`` and return a SAN continuation."""
    if not pv_ucis:
        return []

    sim = board.copy()
    san_line: list[str] = []

    # If pv_ucis already starts with best_move, walk from root; otherwise push
    # best_move first and treat pv_ucis as the follow-up replies.
    ucis = list(pv_ucis)
    if ucis[0].lower() != best_move.uci().lower():
        san_line.append(sim.san(best_move))
        sim.push(best_move)

    for uci in ucis:
        if len(san_line) >= max_plies:
            break
        try:
            mv = chess.Move.from_uci(uci.lower())
        except ValueError:
            break
        if mv not in sim.legal_moves:
            break
        san_line.append(sim.san(mv))
        sim.push(mv)

    return san_line


def _build_level_3_hint(
    board: chess.Board,
    move: chess.Move,
    info: dict[str, Any],
    continuation_san: list[str],
) -> str:
    """Build a Level 3 full solution with verified move, line, and explanation."""
    san = board.san(move)
    uci = move.uci()

    reasons: list[str] = []
    if info["is_checkmate"]:
        reasons.append("delivers immediate checkmate")
    elif info["is_discovered_check"]:
        reasons.append("unleashes a discovered check")
    elif info["gives_check"]:
        reasons.append("gives a forcing check")

    if info["is_promotion"] and info["promo_name"]:
        reasons.append(f"promotes the pawn to a {info['promo_name']}")
    if info["is_castling"]:
        side = "kingside" if info["is_kingside_castle"] else "queenside"
        reasons.append(f"castles {side} to safeguard the king and activate the rook")
    if info["is_fork"]:
        reasons.append("forks multiple enemy targets")
    if info["is_capture"] and info["captured_name"]:
        if info["is_en_passant"]:
            reasons.append("captures en passant")
        else:
            reasons.append(f"captures the {info['captured_name']} on {info['to_square_name']}")
    if not reasons:
        if info["is_development"]:
            reasons.append(
                f"develops the {info['piece_name']} to an active square in {info['zone']}"
            )
        else:
            reasons.append(
                f"improves the {info['piece_name']} and strengthens control over {info['zone']}"
            )

    explanation = f"Play {san} ({uci}), which {' and '.join(reasons)}."
    if len(continuation_san) > 1:
        line_str = " ".join(continuation_san)
        explanation += f" Verified continuation: {line_str}."
    return explanation


def generate_socratic_hint(
    fen: str,
    best_move_uci: str,
    level: int = 1,
    pv_ucis: list[str] | None = None,
    themes: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a progressive Socratic hint for ``best_move_uci`` in ``fen``.

    Args:
        fen: Position FEN string.
        best_move_uci: Best move in UCI format (must be legal in ``fen``).
        level: Hint level (``1`` = nudge, ``2`` = guided, ``3`` = full solution).
        pv_ucis: Optional principal variation in UCI format for Level 3.
        themes: Optional tactical/strategic theme tags (sanitized on Levels 1-2).

    Returns:
        Dictionary containing ``level``, ``hint``, ``spoiler_free``, ``themes``,
        ``move_san``, ``move_uci``, and ``continuation_san``.
    """
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2, or 3 (got {level})")

    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move_uci.lower())
    if move not in board.legal_moves:
        raise ValueError(f"Move '{best_move_uci}' is not legal in FEN '{fen}'")

    info = _analyze_move(board, move)
    raw_themes = list(themes) if themes else info["detected_themes"]

    # Filter out any caller-supplied theme strings that would leak the move or
    # (for Level 1) either square of the move.
    clean_themes = [
        t
        for t in raw_themes
        if not contains_move_spoiler(t, move.uci(), fen)
        and (level != 1 or not _contains_any_move_square(t, move))
    ]

    if level == 1:
        hint = _build_level_1_hint(info, clean_themes)
        if contains_move_spoiler(hint, move.uci(), fen) or _contains_any_move_square(hint, move):
            hint = f"Look closely at what your {info['piece_name']} can accomplish."
        assert not contains_move_spoiler(hint, move.uci(), fen)
        assert not _contains_any_move_square(hint, move)
        return {
            "level": 1,
            "hint": hint,
            "spoiler_free": True,
            "themes": clean_themes or info["detected_themes"],
            "move_san": None,
            "move_uci": None,
            "continuation_san": [],
        }

    if level == 2:
        hint = _build_level_2_hint(info, clean_themes)
        if contains_move_spoiler(hint, move.uci(), fen):
            hint = (
                f"Look for a tactical opportunity with your {info['piece_name']} in {info['zone']}."
            )
        assert not contains_move_spoiler(hint, move.uci(), fen)
        return {
            "level": 2,
            "hint": hint,
            "spoiler_free": True,
            "themes": clean_themes or info["detected_themes"],
            "move_san": None,
            "move_uci": None,
            "continuation_san": [],
        }

    # Level 3: full verified move + verified continuation + explanation.
    continuation_san = _verify_continuation(board, move, pv_ucis)
    hint = _build_level_3_hint(board, move, info, continuation_san)
    return {
        "level": 3,
        "hint": hint,
        "spoiler_free": False,
        "themes": raw_themes,
        "move_san": board.san(move),
        "move_uci": move.uci(),
        "continuation_san": continuation_san,
    }
