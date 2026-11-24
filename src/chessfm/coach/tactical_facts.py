#!/usr/bin/env python3
"""Deterministic ``python-chess`` tactical fact extraction for grounded AI Chess Coaching.

Computes exact, hallucination-free tactical and positional facts for a chess
position and candidate moves so small and large coaching models verbalize
verified board truth rather than guessing geometry from raw FEN strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import chess

from chessfm.coach.socratic import _board_zone, contains_move_spoiler
from chessfm.coach.verifier import PIECE_TYPE_TO_NAME, _has_center_tension
from chessfm.rl.rewards import cp_to_expected_score

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

_SLIDER_RAYS: dict[chess.PieceType, tuple[tuple[int, int], ...]] = {
    chess.BISHOP: ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    chess.ROOK: ((1, 0), (-1, 0), (0, 1), (0, -1)),
    chess.QUEEN: ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)),
}

_UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
_SQUARE_TOKEN_RE = re.compile(r"\b([a-h][1-8])\b", re.IGNORECASE)

_MOTIF_SYNONYMS: dict[str, tuple[str, ...]] = {
    "back_rank_mate": ("back rank", "back-rank", "checkmate", "mate"),
    "checkmate": ("checkmate", "mate", "mating"),
    "stalemate": ("stalemate", "draw"),
    "discovered_check": ("discovered check", "discovered", "unleash", "uncover", "check"),
    "discovered_attack": ("discovered attack", "discovered", "unleash", "uncover", "opens"),
    "fork": ("fork", "double attack", "two targets", "two pieces", "simultaneously"),
    "pin": ("pin", "pinned", "pinning"),
    "skewer": ("skewer", "skewering", "x-ray", "xray", "behind"),
    "trapped_piece": ("trap", "trapped", "trapping", "no escape", "no safe square"),
    "back_rank_threat": ("back rank", "back-rank", "last rank", "8th rank", "1st rank"),
    "underpromotion": ("underpromot", "promot", "knight", "rook", "bishop"),
    "promotion": ("promot", "queen", "passed pawn", "queening"),
    "hanging_piece_capture": ("hanging", "undefended", "unprotected", "free", "capture", "win"),
    "en_passant": ("en passant", "capture", "pawn"),
    "castling": ("castl", "king safety", "tuck", "connect"),
    "king_safety": ("king safety", "safe", "castl", "shelter"),
    "check": ("check", "forcing"),
    "capture": ("capture", "captures", "capturing", "take", "takes", "taking", "win", "wins"),
    "development": ("develop", "active", "activate", "minor piece"),
    "center_control": ("center", "central", "space"),
    "positional_improvement": ("improve", "active", "activity", "control", "pressure", "coord"),
}

_ZONE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "the center": ("center", "central"),
    "the kingside": ("kingside", "king side", "king-side", "g-file", "h-file", "f-file"),
    "the queenside": ("queenside", "queen side", "queen-side", "a-file", "b-file", "c-file"),
    "the central files": ("center", "central", "d-file", "e-file"),
}


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _token_regex(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class PieceSquareRef:
    """Deterministic reference to a piece on a board square."""

    color: str
    piece: str
    square: str

    def label(self) -> str:
        """Return Verifier-V2-exact phrasing, e.g. ``'black bishop on b5'``."""
        return f"{self.color} {self.piece} on {self.square}"


@dataclass(frozen=True)
class HangingPieceFact:
    """A non-king piece that is undefended or attacked by a lower-value piece."""

    color: str
    piece: str
    square: str
    attackers: tuple[str, ...]
    defenders: tuple[str, ...]
    reason: str  # "undefended" | "attacked_by_lower_value"

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece": f"{self.color} {self.piece} on {self.square}",
            "color": self.color,
            "piece_type": self.piece,
            "square": self.square,
            "attackers": list(self.attackers),
            "defenders": list(self.defenders),
            "reason": self.reason,
        }


@dataclass
class TacticalFacts:
    """Complete deterministic tactical facts extracted for a coaching prompt."""

    fen: str
    side_to_move: str
    piece_moved: str
    piece_moved_type: str
    from_square: str
    to_square: str
    best_move_uci: str
    best_move_san: str
    student_move_uci: str | None
    student_move_san: str | None
    captured_piece: str | None
    promotion_piece: str | None
    is_castling: str | None
    is_en_passant: bool
    gives_check: bool
    is_checkmate: bool
    is_stalemate: bool
    has_center_tension: bool
    board_zone: str
    hanging_pieces_with_attackers: list[dict[str, Any]]
    student_hanging_after_move: list[dict[str, Any]]
    pieces_attacked_by_best_move: list[str]
    pieces_defended: list[str]
    tactical_motifs: list[str]
    tactical_motif: str
    cp_loss: float | None
    wdl_loss: float | None
    wdl_loss_estimated: bool
    best_pv_san: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of all tactical facts."""
        return {
            "fen": self.fen,
            "side_to_move": self.side_to_move,
            "piece_moved": self.piece_moved,
            "piece_moved_type": self.piece_moved_type,
            "from_square": self.from_square,
            "to_square": self.to_square,
            "best_move_uci": self.best_move_uci,
            "best_move_san": self.best_move_san,
            "student_move_uci": self.student_move_uci,
            "student_move_san": self.student_move_san,
            "captured_piece": self.captured_piece,
            "promotion_piece": self.promotion_piece,
            "is_castling": self.is_castling,
            "is_en_passant": self.is_en_passant,
            "gives_check": self.gives_check,
            "is_checkmate": self.is_checkmate,
            "is_stalemate": self.is_stalemate,
            "has_center_tension": self.has_center_tension,
            "board_zone": self.board_zone,
            "hanging_pieces_with_attackers": self.hanging_pieces_with_attackers,
            "student_hanging_after_move": self.student_hanging_after_move,
            "pieces_attacked_by_best_move": self.pieces_attacked_by_best_move,
            "pieces_defended": self.pieces_defended,
            "tactical_motifs": self.tactical_motifs,
            "tactical_motif": self.tactical_motif,
            "cp_loss": self.cp_loss,
            "wdl_loss": self.wdl_loss,
            "wdl_loss_estimated": self.wdl_loss_estimated,
            "best_pv_san": self.best_pv_san,
        }


def _parse_move_uci_or_san(board: chess.Board, move_str: str) -> chess.Move:
    """Parse a legal move given in either UCI or SAN notation."""
    cleaned = move_str.strip()
    if not cleaned:
        raise ValueError("Empty move string")

    if _UCI_PATTERN.match(cleaned):
        try:
            mv = chess.Move.from_uci(cleaned.lower())
            if mv in board.legal_moves:
                return mv
        except ValueError:
            pass

    try:
        mv = board.parse_san(cleaned)
    except (
        chess.IllegalMoveError,
        chess.InvalidMoveError,
        chess.AmbiguousMoveError,
        ValueError,
    ) as exc:
        raise ValueError(f"Illegal or invalid move {move_str!r} for FEN {board.fen()!r}") from exc

    if mv not in board.legal_moves:
        raise ValueError(f"Move {move_str!r} is not legal in FEN {board.fen()!r}")
    return mv


def _piece_ref(board: chess.Board, sq: chess.Square) -> PieceSquareRef | None:
    piece = board.piece_at(sq)
    if piece is None:
        return None
    return PieceSquareRef(
        color=_color_name(piece.color),
        piece=PIECE_TYPE_TO_NAME[piece.piece_type],
        square=chess.square_name(sq),
    )


def _find_hanging_pieces(
    board: chess.Board,
    *,
    color_filter: chess.Color | None = None,
) -> list[HangingPieceFact]:
    """Find all hanging non-king pieces with exact attackers and defenders."""
    hanging: list[HangingPieceFact] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        if color_filter is not None and piece.color != color_filter:
            continue

        attackers_sqs = sorted(board.attackers(not piece.color, sq))
        if not attackers_sqs:
            continue
        defenders_sqs = sorted(board.attackers(piece.color, sq))

        attacker_vals = [
            _PIECE_VALUES[board.piece_at(a).piece_type]  # type: ignore[union-attr]
            for a in attackers_sqs
            if board.piece_at(a) is not None
        ]
        if not attacker_vals:
            continue

        piece_val = _PIECE_VALUES[piece.piece_type]
        if not defenders_sqs:
            reason = "undefended"
        elif min(attacker_vals) < piece_val:
            reason = "attacked_by_lower_value"
        else:
            continue

        attackers = tuple(
            ref.label() for a in attackers_sqs if (ref := _piece_ref(board, a)) is not None
        )
        defenders = tuple(
            ref.label() for d in defenders_sqs if (ref := _piece_ref(board, d)) is not None
        )
        hanging.append(
            HangingPieceFact(
                color=_color_name(piece.color),
                piece=PIECE_TYPE_TO_NAME[piece.piece_type],
                square=chess.square_name(sq),
                attackers=attackers,
                defenders=defenders,
                reason=reason,
            )
        )
    return hanging


def _castling_rook_dest(board: chess.Board, move: chess.Move) -> chess.Square | None:
    """Return the destination square of the castled rook if ``move`` is castling."""
    if not board.is_castling(move):
        return None
    rank = chess.square_rank(move.from_square)
    file_idx = 5 if board.is_kingside_castling(move) else 3
    return chess.square(file_idx, rank)


def _detect_discovered_attacks(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    mover: chess.Color,
) -> list[chess.Square]:
    """Find enemy pieces newly attacked by a stationary friendly piece after ``move``."""
    if board_before.is_castling(move):
        return []
    enemy = not mover
    discovered: list[chess.Square] = []
    seen: set[chess.Square] = set()

    for sq in chess.SQUARES:
        if sq == move.to_square:
            continue
        p_after = board_after.piece_at(sq)
        if p_after is None or p_after.color != mover:
            continue
        if p_after.piece_type not in _SLIDER_RAYS:
            continue
        if board_before.piece_at(sq) != p_after:
            continue
        new_attacks = board_after.attacks(sq) & ~board_before.attacks(sq)
        for target_sq in sorted(new_attacks):
            target_piece = board_after.piece_at(target_sq)
            if target_piece is not None and target_piece.color == enemy and target_sq not in seen:
                seen.add(target_sq)
                discovered.append(target_sq)
    return discovered


def _is_fork_after_move(
    board_after: chess.Board,
    attacker_sq: chess.Square,
    mover: chess.Color,
) -> bool:
    """Return True if the piece on ``attacker_sq`` forks >=2 valuable/undefended enemy pieces."""
    piece = board_after.piece_at(attacker_sq)
    if piece is None:
        return False
    enemy = not mover
    mover_val = _PIECE_VALUES[piece.piece_type]
    targets = 0
    for sq in board_after.attacks(attacker_sq):
        target = board_after.piece_at(sq)
        if target is None or target.color != enemy:
            continue
        if target.piece_type == chess.PAWN:
            continue
        target_val = _PIECE_VALUES[target.piece_type]
        is_undefended = not board_after.is_attacked_by(enemy, sq)
        if piece.piece_type in (chess.PAWN, chess.KNIGHT):
            targets += 1
        elif target.piece_type == chess.KING or target_val > mover_val or is_undefended:
            targets += 1
    return targets >= 2


def _detect_pin_and_skewer(
    board_after: chess.Board,
    attacker_squares: list[chess.Square],
    mover: chess.Color,
) -> tuple[bool, bool]:
    """Detect whether any slider in ``attacker_squares`` creates a pin or a skewer."""
    enemy = not mover
    has_pin = False
    has_skewer = False

    for attacker_sq in attacker_squares:
        piece = board_after.piece_at(attacker_sq)
        if piece is None or piece.piece_type not in _SLIDER_RAYS:
            continue

        # 1. Absolute pin against enemy king via python-chess pin mask.
        for sq in board_after.attacks(attacker_sq):
            target = board_after.piece_at(sq)
            if target is not None and target.color == enemy and target.piece_type != chess.KING:
                if board_after.is_pinned(enemy, sq) and attacker_sq in board_after.pin(enemy, sq):
                    has_pin = True

        # 2. Ray scan for relative pin and skewer.
        start_f = chess.square_file(attacker_sq)
        start_r = chess.square_rank(attacker_sq)
        for df, dr in _SLIDER_RAYS[piece.piece_type]:
            f, r = start_f + df, start_r + dr
            first_enemy: tuple[chess.Square, chess.Piece] | None = None
            while 0 <= f < 8 and 0 <= r < 8:
                sq = chess.square(f, r)
                occ = board_after.piece_at(sq)
                if occ is not None:
                    if occ.color == mover:
                        break
                    if first_enemy is None:
                        first_enemy = (sq, occ)
                    else:
                        _, p1 = first_enemy
                        v1 = _PIECE_VALUES[p1.piece_type]
                        v2 = _PIECE_VALUES[occ.piece_type]
                        if p1.piece_type != chess.KING and v2 > v1:
                            has_pin = True
                        if p1.piece_type in (chess.KING, chess.QUEEN, chess.ROOK) and v1 > v2:
                            if occ.piece_type != chess.PAWN or not board_after.is_attacked_by(
                                enemy, sq
                            ):
                                has_skewer = True
                        break
                f += df
                r += dr
    return has_pin, has_skewer


def _is_back_rank_box_blocked(board: chess.Board, enemy: chess.Color) -> tuple[bool, int]:
    """Check if ``enemy`` king sits on its back rank with blocked front escape squares.

    Returns ``(on_back_rank, open_front_squares_count)``.
    """
    king_sq = board.king(enemy)
    if king_sq is None:
        return False, 3
    back_rank = 7 if enemy == chess.BLACK else 0
    if chess.square_rank(king_sq) != back_rank:
        return False, 3
    front_rank = 6 if enemy == chess.BLACK else 1
    kf = chess.square_file(king_sq)
    open_front = 0
    for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
        sq = chess.square(f, front_rank)
        occ = board.piece_at(sq)
        if occ is None or occ.color != enemy:
            open_front += 1
    return True, open_front


def _detect_back_rank_threat(
    board_after: chess.Board,
    attacker_squares: list[chess.Square],
    mover: chess.Color,
) -> bool:
    """Return True if mover's rook/queen targets an enemy king boxed on the back rank."""
    enemy = not mover
    on_back, open_front = _is_back_rank_box_blocked(board_after, enemy)
    if not on_back or open_front > 1:
        return False
    enemy_back_rank = 7 if enemy == chess.BLACK else 0
    for sq in attacker_squares:
        piece = board_after.piece_at(sq)
        if piece is None or piece.piece_type not in (chess.ROOK, chess.QUEEN):
            continue
        if chess.square_rank(sq) == enemy_back_rank:
            return True
        if any(chess.square_rank(a) == enemy_back_rank for a in board_after.attacks(sq)):
            return True
    return False


def _detect_trapped_piece(board_after: chess.Board, mover: chess.Color) -> bool:
    """Return True if an attacked enemy minor/major piece has no safe legal escape square."""
    enemy = not mover
    if board_after.is_game_over():
        return False
    for sq in chess.SQUARES:
        piece = board_after.piece_at(sq)
        if piece is None or piece.color != enemy:
            continue
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if not board_after.is_attacked_by(mover, sq):
            continue

        piece_val = _PIECE_VALUES[piece.piece_type]
        legal_escapes = [m for m in board_after.legal_moves if m.from_square == sq]
        has_safe_escape = False
        for esc in legal_escapes:
            cap_piece = board_after.piece_at(esc.to_square)
            if cap_piece is not None and _PIECE_VALUES[cap_piece.piece_type] >= piece_val:
                has_safe_escape = True
                break
            sim = board_after.copy(stack=False)
            sim.push(esc)
            attackers = list(sim.attackers(mover, esc.to_square))
            if not attackers:
                has_safe_escape = True
                break
            min_att = min(
                _PIECE_VALUES[sim.piece_at(a).piece_type]  # type: ignore[union-attr]
                for a in attackers
                if sim.piece_at(a) is not None
            )
            is_defended = sim.is_attacked_by(enemy, esc.to_square)
            if min_att >= piece_val and is_defended:
                has_safe_escape = True
                break
        if not has_safe_escape:
            return True
    return False


def _build_verified_pv_san(
    board: chess.Board,
    best_move: chess.Move,
    pv: list[str] | None,
    max_plies: int = 6,
) -> list[str]:
    """Validate PV moves (UCI or SAN) and return up to ``max_plies`` SAN strings."""
    sim = board.copy(stack=False)
    best_san = sim.san(best_move)
    sim.push(best_move)
    san_line = [best_san]

    if sim.is_checkmate() or sim.is_stalemate() or not pv:
        return san_line

    raw_moves = list(pv)
    # Skip leading element if it repeats best_move (UCI or SAN).
    if raw_moves:
        first = str(raw_moves[0]).strip()
        if first.lower() == best_move.uci().lower() or first.rstrip("+#") == best_san.rstrip("+#"):
            raw_moves = raw_moves[1:]

    for raw in raw_moves:
        if len(san_line) >= max_plies:
            break
        try:
            mv = _parse_move_uci_or_san(sim, str(raw))
        except ValueError:
            break
        san_line.append(sim.san(mv))
        sim.push(mv)
        if sim.is_checkmate() or sim.is_stalemate():
            break

    return san_line


def _compute_wdl_loss(
    cp_loss: float | None,
    eval_before_cp: float | None,
    eval_after_cp: float | None,
) -> tuple[float | None, bool]:
    """Compute bounded ``[0.0, 1.0]`` Lichess expected-score loss."""
    if eval_before_cp is not None and eval_after_cp is not None:
        loss = max(
            0.0,
            cp_to_expected_score(float(eval_before_cp))
            - cp_to_expected_score(float(eval_after_cp)),
        )
        return round(loss, 4), False
    if cp_loss is not None:
        loss = max(
            0.0,
            cp_to_expected_score(0.0) - cp_to_expected_score(-abs(float(cp_loss))),
        )
        return round(loss, 4), True
    return None, False


def extract_tactical_facts(
    fen: str,
    student_move: str | None = None,
    best_move: str | None = None,
    pv: list[str] | None = None,
    *,
    cp_loss: float | None = None,
    eval_before_cp: float | None = None,
    eval_after_cp: float | None = None,
) -> dict[str, Any]:
    """Deterministically extract verified tactical facts from ``fen`` and moves.

    Args:
        fen: Valid chess FEN string.
        student_move: Optional student move in UCI or SAN format.
        best_move: Best move in UCI or SAN format (falls back to ``pv[0]`` if omitted).
        pv: Optional principal variation (list of UCI or SAN strings).
        cp_loss: Optional centipawn loss of ``student_move`` relative to ``best_move``.
        eval_before_cp: Optional engine centipawn evaluation of the best move.
        eval_after_cp: Optional engine centipawn evaluation after ``student_move``.

    Returns:
        Dictionary with all 29 verified tactical fact fields.
    """
    board = chess.Board(fen)

    resolved_best_str = best_move
    if not resolved_best_str and pv:
        resolved_best_str = str(pv[0])
    if not resolved_best_str:
        raise ValueError("best_move (or non-empty pv) is required to extract tactical facts")

    best_mv = _parse_move_uci_or_san(board, resolved_best_str)
    mover = board.turn
    enemy = not mover
    mover_color_str = _color_name(mover)
    enemy_color_str = _color_name(enemy)

    moving_piece = board.piece_at(best_mv.from_square)
    assert moving_piece is not None
    piece_moved_type = PIECE_TYPE_TO_NAME[moving_piece.piece_type]
    from_sq_name = chess.square_name(best_mv.from_square)
    to_sq_name = chess.square_name(best_mv.to_square)
    best_move_uci = best_mv.uci()
    best_move_san = board.san(best_mv)

    is_castling: str | None = None
    if board.is_kingside_castling(best_mv):
        is_castling = "kingside"
    elif board.is_queenside_castling(best_mv):
        is_castling = "queenside"

    is_en_passant = board.is_en_passant(best_mv)
    is_capture = board.is_capture(best_mv)
    promotion_piece = PIECE_TYPE_TO_NAME[best_mv.promotion] if best_mv.promotion else None

    captured_piece: str | None = None
    hanging_capture = False
    if is_capture:
        if is_en_passant:
            cap_sq = chess.square(
                chess.square_file(best_mv.to_square),
                chess.square_rank(best_mv.from_square),
            )
            captured_piece = f"{enemy_color_str} pawn on {chess.square_name(cap_sq)} (en passant)"
            if not board.is_attacked_by(enemy, cap_sq):
                hanging_capture = True
        else:
            cap_obj = board.piece_at(best_mv.to_square)
            if cap_obj is not None:
                captured_piece = (
                    f"{_color_name(cap_obj.color)} {PIECE_TYPE_TO_NAME[cap_obj.piece_type]} "
                    f"on {to_sq_name}"
                )
            if not board.is_attacked_by(enemy, best_mv.to_square):
                hanging_capture = True

    # Student move parsing (tolerant when None or invalid).
    student_move_uci: str | None = None
    student_move_san: str | None = None
    student_hanging_after_move: list[dict[str, Any]] = []
    if isinstance(student_move, str) and student_move.strip():
        try:
            stu_mv = _parse_move_uci_or_san(board, student_move)
            student_move_uci = stu_mv.uci()
            student_move_san = board.san(stu_mv)
            after_student = board.copy(stack=False)
            after_student.push(stu_mv)
            student_hanging_after_move = [
                h.to_dict() for h in _find_hanging_pieces(after_student, color_filter=mover)
            ]
        except ValueError:
            student_move_uci = None
            student_move_san = None
            student_hanging_after_move = []

    root_hanging = [h.to_dict() for h in _find_hanging_pieces(board)]
    has_center_tension = _has_center_tension(board)

    board_after = board.copy(stack=False)
    board_after.push(best_mv)

    gives_check = board_after.is_check()
    is_checkmate = board_after.is_checkmate()
    is_stalemate = board_after.is_stalemate()
    is_discovered_check = (
        gives_check
        and not is_castling
        and any(sq != best_mv.to_square for sq in board_after.checkers())
    )

    # Squares whose attacks/defenses belong to best_move (king + rook when castling).
    moved_dest_squares = [best_mv.to_square]
    rook_dest = _castling_rook_dest(board, best_mv)
    if rook_dest is not None and board_after.piece_at(rook_dest) is not None:
        moved_dest_squares.append(rook_dest)

    attacked_enemy_sqs: set[chess.Square] = set()
    defended_friendly_sqs: set[chess.Square] = set()
    for src_sq in moved_dest_squares:
        for sq in board_after.attacks(src_sq):
            occ = board_after.piece_at(sq)
            if occ is None:
                continue
            if occ.color == enemy:
                attacked_enemy_sqs.add(sq)
            elif occ.color == mover and occ.piece_type != chess.KING:
                defended_friendly_sqs.add(sq)

    discovered_attack_sqs = _detect_discovered_attacks(board, board_after, best_mv, mover)
    attacked_enemy_sqs.update(discovered_attack_sqs)

    pieces_attacked_by_best_move = [
        ref.label()
        for sq in sorted(attacked_enemy_sqs)
        if (ref := _piece_ref(board_after, sq)) is not None
    ]
    pieces_defended = [
        ref.label()
        for sq in sorted(defended_friendly_sqs)
        if (ref := _piece_ref(board_after, sq)) is not None
    ]

    # Motif detection (priority ordered).
    has_fork = any(_is_fork_after_move(board_after, sq, mover) for sq in moved_dest_squares)
    has_pin, has_skewer = _detect_pin_and_skewer(board_after, moved_dest_squares, mover)
    has_discovered_attack = any(
        (p := board_after.piece_at(sq)) is not None
        and (p.piece_type != chess.PAWN or not board_after.is_attacked_by(enemy, sq))
        for sq in discovered_attack_sqs
    )
    has_back_rank_threat = (not is_checkmate) and _detect_back_rank_threat(
        board_after, moved_dest_squares, mover
    )
    has_trapped = _detect_trapped_piece(board_after, mover)

    motifs: list[str] = []
    if is_checkmate:
        on_back, open_front = _is_back_rank_box_blocked(board_after, enemy)
        if on_back and open_front == 0:
            motifs.append("back_rank_mate")
        motifs.append("checkmate")
    if is_stalemate:
        motifs.append("stalemate")
    if is_discovered_check:
        motifs.append("discovered_check")
    if has_fork:
        motifs.append("fork")
    if has_pin:
        motifs.append("pin")
    if has_skewer:
        motifs.append("skewer")
    if has_discovered_attack and not is_discovered_check:
        motifs.append("discovered_attack")
    if has_trapped:
        motifs.append("trapped_piece")
    if has_back_rank_threat:
        motifs.append("back_rank_threat")
    if promotion_piece is not None:
        if best_mv.promotion != chess.QUEEN:
            motifs.append("underpromotion")
        motifs.append("promotion")
    if hanging_capture:
        motifs.append("hanging_piece_capture")
    if is_en_passant:
        motifs.append("en_passant")
    if is_castling is not None:
        motifs.append("castling")
        motifs.append("king_safety")
    if gives_check and not is_checkmate:
        motifs.append("check")
    if is_capture:
        motifs.append("capture")

    back_rank = 0 if mover == chess.WHITE else 7
    if (
        moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(best_mv.from_square) == back_rank
        and chess.square_rank(best_mv.to_square) != back_rank
    ):
        motifs.append("development")
    if best_mv.to_square in (chess.D4, chess.D5, chess.E4, chess.E5):
        motifs.append("center_control")
    if not motifs:
        motifs.append("positional_improvement")

    wdl_loss, wdl_loss_estimated = _compute_wdl_loss(cp_loss, eval_before_cp, eval_after_cp)
    best_pv_san = _build_verified_pv_san(board, best_mv, pv, max_plies=6)
    piece_moved_label = f"{mover_color_str} {piece_moved_type} on {from_sq_name} -> {to_sq_name}"

    facts = TacticalFacts(
        fen=fen,
        side_to_move=mover_color_str,
        piece_moved=piece_moved_label,
        piece_moved_type=piece_moved_type,
        from_square=from_sq_name,
        to_square=to_sq_name,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        student_move_uci=student_move_uci,
        student_move_san=student_move_san,
        captured_piece=captured_piece,
        promotion_piece=promotion_piece,
        is_castling=is_castling,
        is_en_passant=is_en_passant,
        gives_check=gives_check,
        is_checkmate=is_checkmate,
        is_stalemate=is_stalemate,
        has_center_tension=has_center_tension,
        board_zone=_board_zone(best_mv.to_square),
        hanging_pieces_with_attackers=root_hanging,
        student_hanging_after_move=student_hanging_after_move,
        pieces_attacked_by_best_move=pieces_attacked_by_best_move,
        pieces_defended=pieces_defended,
        tactical_motifs=motifs,
        tactical_motif=motifs[0],
        cp_loss=None if cp_loss is None else round(float(cp_loss), 2),
        wdl_loss=wdl_loss,
        wdl_loss_estimated=wdl_loss_estimated,
        best_pv_san=best_pv_san,
    )
    return facts.to_dict()


def format_tactical_facts_for_prompt(
    facts: dict[str, Any],
    *,
    task_type: str = "postmortem",
) -> str:
    """Format verified tactical facts as a compact, deterministic prompt block.

    Explicitly retains ``gives_check``, ``is_checkmate``, and
    ``has_center_tension`` booleans so coaching models do not hallucinate checks
    or center-tension boilerplate when they are ``False``.
    """
    compact: dict[str, Any] = {
        "side_to_move": facts.get("side_to_move"),
        "piece_moved": facts.get("piece_moved"),
        "best_move_san": facts.get("best_move_san"),
        "gives_check": bool(facts.get("gives_check", False)),
        "is_checkmate": bool(facts.get("is_checkmate", False)),
        "has_center_tension": bool(facts.get("has_center_tension", False)),
        "board_zone": facts.get("board_zone"),
        "tactical_motif": facts.get("tactical_motif"),
        "tactical_motifs": facts.get("tactical_motifs", [])[:4],
    }
    if facts.get("student_move_san"):
        compact["student_move_san"] = facts["student_move_san"]
    if facts.get("captured_piece"):
        compact["captured_piece"] = facts["captured_piece"]
    if facts.get("promotion_piece"):
        compact["promotion_piece"] = facts["promotion_piece"]
    if facts.get("is_castling"):
        compact["is_castling"] = facts["is_castling"]
    if facts.get("pieces_attacked_by_best_move"):
        compact["pieces_attacked_by_best_move"] = facts["pieces_attacked_by_best_move"][:5]
    if facts.get("pieces_defended"):
        compact["pieces_defended"] = facts["pieces_defended"][:5]
    if facts.get("hanging_pieces_with_attackers"):
        compact["hanging_pieces_with_attackers"] = [
            {
                "piece": h["piece"],
                "attackers": h["attackers"][:3],
                "reason": h["reason"],
            }
            for h in facts["hanging_pieces_with_attackers"][:4]
        ]
    if facts.get("student_hanging_after_move") and task_type in ("postmortem", "why_not"):
        compact["student_hanging_after_move"] = [
            {
                "piece": h["piece"],
                "attackers": h["attackers"][:3],
                "reason": h["reason"],
            }
            for h in facts["student_hanging_after_move"][:4]
        ]
    if facts.get("best_pv_san"):
        compact["best_pv_san"] = facts["best_pv_san"][:6]
    if facts.get("wdl_loss") is not None:
        compact["wdl_loss"] = facts["wdl_loss"]

    return f"Tactical Facts (Verified):\n{json.dumps(compact, ensure_ascii=False)}"


def _extract_target_squares(facts: dict[str, Any]) -> set[str]:
    """Collect concrete square names from attacked, defended, and hanging facts."""
    squares: set[str] = set()
    for key in ("pieces_attacked_by_best_move", "pieces_defended"):
        for item in facts.get(key) or []:
            for m in _SQUARE_TOKEN_RE.finditer(str(item)):
                squares.add(m.group(1).lower())
    for key in ("hanging_pieces_with_attackers", "student_hanging_after_move"):
        for entry in facts.get(key) or []:
            if isinstance(entry, dict) and entry.get("square"):
                squares.add(str(entry["square"]).lower())
            elif isinstance(entry, dict) and entry.get("piece"):
                for m in _SQUARE_TOKEN_RE.finditer(str(entry["piece"])):
                    squares.add(m.group(1).lower())
    cap = facts.get("captured_piece")
    if isinstance(cap, str):
        for m in _SQUARE_TOKEN_RE.finditer(cap):
            squares.add(m.group(1).lower())
    return squares


def _extract_enemy_piece_types(facts: dict[str, Any]) -> set[str]:
    """Collect enemy piece type names referenced by captures, attacks, or hanging facts."""
    piece_types: set[str] = set()
    side_to_move = str(facts.get("side_to_move") or "").lower()
    enemy = "black" if side_to_move == "white" else "white"
    all_names = ("queen", "rook", "bishop", "knight", "pawn", "king")

    cap = str(facts.get("captured_piece") or "").lower()
    for name in all_names:
        if name in cap:
            piece_types.add(name)
    for item in facts.get("pieces_attacked_by_best_move") or []:
        lower = str(item).lower()
        for name in all_names:
            if name in lower:
                piece_types.add(name)
    for entry in facts.get("hanging_pieces_with_attackers") or []:
        if isinstance(entry, dict) and str(entry.get("color", "")).lower() == enemy:
            pt = str(entry.get("piece_type", "")).lower()
            if pt:
                piece_types.add(pt)
    return piece_types


def _mentions_any_motif(lower_text: str, facts: dict[str, Any]) -> bool:
    motifs = facts.get("tactical_motifs") or []
    if not motifs and facts.get("tactical_motif"):
        motifs = [facts["tactical_motif"]]
    for motif in motifs:
        for kw in _MOTIF_SYNONYMS.get(str(motif), (str(motif).replace("_", " "),)):
            if kw in lower_text:
                return True
    if facts.get("gives_check") and "check" in lower_text:
        return True
    if facts.get("captured_piece") and any(
        w in lower_text for w in ("captur", "take", "takes", "taking", "win", "wins", "free")
    ):
        return True
    if (facts.get("hanging_pieces_with_attackers") or facts.get("student_hanging_after_move")) and (
        any(w in lower_text for w in ("hanging", "undefended", "unprotected", "loose", "en prise"))
    ):
        return True
    return False


def _mentions_move(text: str, san: str | None, uci: str | None) -> bool:
    candidates: set[str] = set()
    if san:
        candidates.add(san)
        candidates.add(san.rstrip("+#"))
    if uci:
        candidates.add(uci)
    return any(bool(_token_regex(c).search(text)) for c in candidates if c)


def compute_tactical_fact_coverage(
    text: str,
    facts: dict[str, Any],
    task_type: str = "postmortem",
) -> float:
    """Compute a deterministic, task-aware, spoiler-safe coverage score in ``[0.0, 1.0]``.

    On ``socratic_l1`` and ``socratic_l2``, any response that leaks
    ``best_move_uci`` receives ``0.0`` so coverage can never incentivize spoilers.
    """
    if not text or not text.strip() or not facts:
        return 0.0

    fen = facts.get("fen")
    best_move_uci = str(facts.get("best_move_uci") or "")
    if task_type in ("socratic_l1", "socratic_l2") and fen and best_move_uci:
        try:
            if contains_move_spoiler(text, best_move_uci, fen):
                return 0.0
        except ValueError:
            pass

    lower = text.lower()
    zone = str(facts.get("board_zone") or "")
    zone_kws = _ZONE_KEYWORDS.get(zone, ())
    piece_moved_type = str(facts.get("piece_moved_type") or "").lower()
    enemy_piece_types = _extract_enemy_piece_types(facts)
    target_squares = _extract_target_squares(facts)

    if task_type == "socratic_l1":
        anchors = 0
        hits = 0
        # Anchor 1: moving piece type OR motif/strategic theme.
        anchors += 1
        if (piece_moved_type and _token_regex(piece_moved_type).search(lower)) or (
            _mentions_any_motif(lower, facts)
        ):
            hits += 1
        # Anchor 2: board zone OR enemy target piece type (no squares required).
        if zone_kws or enemy_piece_types:
            anchors += 1
            zone_hit = any(kw in lower for kw in zone_kws)
            enemy_hit = any(bool(_token_regex(pt).search(lower)) for pt in enemy_piece_types)
            if zone_hit or enemy_hit:
                hits += 1
        return round(hits / max(1, anchors), 4)

    if task_type == "socratic_l2":
        anchors = 0
        hits = 0
        from_sq = str(facts.get("from_square") or "").lower()
        to_sq = str(facts.get("to_square") or "").lower()
        safe_squares = set(target_squares)
        safe_squares.discard(from_sq)
        if piece_moved_type == "pawn" or facts.get("is_castling"):
            safe_squares.discard(to_sq)

        # Anchor 1: piece type OR motif synonym.
        anchors += 1
        if (piece_moved_type and _token_regex(piece_moved_type).search(lower)) or (
            _mentions_any_motif(lower, facts)
        ):
            hits += 1
        # Anchor 2: board zone OR safe target/hanging square.
        if zone_kws or safe_squares:
            anchors += 1
            zone_hit = any(kw in lower for kw in zone_kws)
            sq_hit = any(bool(_token_regex(sq).search(lower)) for sq in safe_squares)
            if zone_hit or sq_hit:
                hits += 1
        # Anchor 3: enemy target piece type (if present).
        if enemy_piece_types:
            anchors += 1
            if any(bool(_token_regex(pt).search(lower)) for pt in enemy_piece_types):
                hits += 1
        return round(hits / max(1, anchors), 4)

    # Concrete tasks: postmortem, why_not, socratic_l3.
    anchors = 0
    hits = 0

    # Anchor 1: best move mentioned.
    if facts.get("best_move_san") or best_move_uci:
        anchors += 1
        if _mentions_move(text, facts.get("best_move_san"), best_move_uci):
            hits += 1

    # Anchor 2: student move (postmortem/why_not) OR verified PV follow-up (socratic_l3).
    if task_type in ("postmortem", "why_not") and (
        facts.get("student_move_san") or facts.get("student_move_uci")
    ):
        anchors += 1
        if _mentions_move(text, facts.get("student_move_san"), facts.get("student_move_uci")):
            hits += 1
    elif task_type == "socratic_l3":
        pv_san = facts.get("best_pv_san") or []
        if len(pv_san) >= 2:
            anchors += 1
            if _mentions_move(text, str(pv_san[1]), None):
                hits += 1

    # Anchor 3: tactical motif / capture / check / hanging concept.
    anchors += 1
    if _mentions_any_motif(lower, facts):
        hits += 1

    # Anchor 4: concrete target / defended / hanging square.
    if target_squares:
        anchors += 1
        if any(bool(_token_regex(sq).search(lower)) for sq in target_squares):
            hits += 1

    return round(hits / max(1, anchors), 4)
