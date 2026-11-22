"""Game-phase classification, stratified sampling, leakage checks, and Lichess puzzle parsing.

Pure ``python-chess`` utilities for building balanced, leakage-free chess training
and evaluation splits across openings, middlegames, and endgames.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import chess

GamePhase = Literal["opening", "middlegame", "endgame"]

_PHASES: tuple[GamePhase, GamePhase, GamePhase] = ("opening", "middlegame", "endgame")
_MINOR_MAJOR_TYPES: tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
)


def normalize_board_fen(fen: str) -> str:
    """Normalize a FEN string to its position-defining first 4 fields.

    Drops halfmove clock and fullmove number so transposed or move-numbered
    duplicates of the same board state collapse to one canonical key.
    """
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN (expected at least 4 fields): {fen!r}")
    return " ".join(parts[:4])


def classify_game_phase(fen: str) -> GamePhase:
    """Deterministically classify a FEN into ``opening``, ``middlegame``, or ``endgame``.

    Rules:
    - ``endgame``: at most 6 non-pawn pieces (including kings, i.e., <= 4 minor/major
      pieces) OR no queens on the board with at most 6 minor/major pieces.
    - ``opening``: fullmove number <= 10 and at least 10 minor/major pieces on board.
    - ``middlegame``: all other positions.
    """
    board = chess.Board(fen)

    minor_major = sum(
        len(board.pieces(piece_type, color))
        for piece_type in _MINOR_MAJOR_TYPES
        for color in (chess.WHITE, chess.BLACK)
    )
    kings = len(board.pieces(chess.KING, chess.WHITE)) + len(board.pieces(chess.KING, chess.BLACK))
    non_pawn_pieces = minor_major + kings
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )

    if non_pawn_pieces <= 6 or (queens == 0 and minor_major <= 6):
        return "endgame"
    if board.fullmove_number <= 10 and minor_major >= 10:
        return "opening"
    return "middlegame"


def phase_distribution(fens: list[str]) -> dict[str, float]:
    """Compute the fraction of positions in each game phase."""
    counts: dict[str, int] = {phase: 0 for phase in _PHASES}
    if not fens:
        return {phase: 0.0 for phase in _PHASES}

    for fen in fens:
        counts[classify_game_phase(fen)] += 1

    total = len(fens)
    return {phase: counts[phase] / total for phase in _PHASES}


def _allocate_quotas(total: int, ratios: tuple[float, float, float]) -> dict[GamePhase, int]:
    """Allocate integer quotas summing to ``total`` via largest-remainder rounding."""
    ratio_sum = sum(ratios)
    exact = [total * (r / ratio_sum) for r in ratios]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(ratios)), key=lambda i: (exact[i] - floors[i], ratios[i]), reverse=True
    )
    for idx in order[:remainder]:
        floors[idx] += 1
    return dict(zip(_PHASES, floors, strict=True))


def sample_balanced_positions(
    fens: list[str],
    total: int,
    ratios: tuple[float, float, float] = (0.25, 0.55, 0.20),
    seed: int = 42,
) -> list[str]:
    """Deduplicate by board FEN and return a phase-stratified sample.

    Parameters
    ----------
    fens:
        Candidate FEN strings.
    total:
        Target number of positions to return (capped at the number of unique
        board positions available).
    ratios:
        Desired ``(opening, middlegame, endgame)`` proportions.
    seed:
        Deterministic RNG seed.
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios must be 3 non-negative numbers with positive sum, got {ratios}")
    if total == 0 or not fens:
        return []

    # Deduplicate by first 4 FEN fields, keeping the first occurrence per board key.
    unique_by_key: dict[str, str] = {}
    for fen in fens:
        key = normalize_board_fen(fen)
        if key not in unique_by_key:
            unique_by_key[key] = fen.strip()

    rng = random.Random(seed)
    buckets: dict[GamePhase, list[str]] = {phase: [] for phase in _PHASES}
    for key in sorted(unique_by_key):
        fen = unique_by_key[key]
        buckets[classify_game_phase(fen)].append(fen)

    for phase in _PHASES:
        rng.shuffle(buckets[phase])

    target_total = min(total, len(unique_by_key))
    quotas = _allocate_quotas(target_total, ratios)

    selected: list[str] = []
    leftovers: dict[GamePhase, list[str]] = {}
    for phase in _PHASES:
        take = min(quotas[phase], len(buckets[phase]))
        selected.extend(buckets[phase][:take])
        leftovers[phase] = buckets[phase][take:]

    # Redistribute any shortfall from under-populated buckets deterministically.
    shortfall = target_total - len(selected)
    if shortfall > 0:
        remaining: list[str] = []
        for phase in _PHASES:
            remaining.extend(leftovers[phase])
        rng.shuffle(remaining)
        selected.extend(remaining[:shortfall])

    rng.shuffle(selected)
    return selected


def verify_zero_leakage(
    train_fens: list[str],
    val_fens: list[str],
    test_fens: list[str],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Verify and report position-level overlap across train/val/test splits.

    Positions are compared using their normalized 4-field board FEN so identical
    positions with different move counters are still caught as leakage.
    """
    train_set = {normalize_board_fen(f) for f in train_fens}
    val_set = {normalize_board_fen(f) for f in val_fens}
    test_set = {normalize_board_fen(f) for f in test_fens}

    train_val = sorted(train_set & val_set)
    train_test = sorted(train_set & test_set)
    val_test = sorted(val_set & test_set)
    total_overlap = len(train_val) + len(train_test) + len(val_test)
    zero_leakage = total_overlap == 0

    report: dict[str, Any] = {
        "zero_leakage": zero_leakage,
        "total_overlap": total_overlap,
        "counts": {
            "train": len(train_fens),
            "val": len(val_fens),
            "test": len(test_fens),
        },
        "unique_counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "overlap_counts": {
            "train_val": len(train_val),
            "train_test": len(train_test),
            "val_test": len(val_test),
        },
        "overlaps": {
            "train_val": train_val,
            "train_test": train_test,
            "val_test": val_test,
        },
    }

    if strict and not zero_leakage:
        raise ValueError(
            f"Position leakage detected across splits: "
            f"train/val={len(train_val)}, train/test={len(train_test)}, val/test={len(val_test)}"
        )

    return report


def parse_lichess_puzzle_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a CC0 Lichess puzzle CSV/JSON row into a normalized puzzle dict.

    In Lichess puzzle exports, ``FEN`` is the board position *before* the
    opponent's blunder/setup move, and ``Moves[0]`` is that setup move. This
    function validates and applies ``Moves[0]`` so the returned ``fen`` is the
    actual puzzle starting position and ``best_move`` is ``Moves[1]``.

    Returns ``None`` if the row is missing required fields or contains an
    invalid FEN / illegal move sequence.
    """
    raw_fen = row.get("FEN") or row.get("fen")
    raw_moves = row.get("Moves") if row.get("Moves") is not None else row.get("moves")
    if not raw_fen or not isinstance(raw_fen, str) or raw_moves is None:
        return None

    if isinstance(raw_moves, str):
        moves = [m.strip().lower() for m in raw_moves.split() if m.strip()]
    elif isinstance(raw_moves, list):
        moves = [str(m).strip().lower() for m in raw_moves if str(m).strip()]
    else:
        return None

    if len(moves) < 2:
        return None

    try:
        board = chess.Board(raw_fen.strip())
    except ValueError:
        return None

    # Apply the opponent's setup move (Moves[0]) to obtain the puzzle start FEN.
    try:
        setup_move = chess.Move.from_uci(moves[0])
    except ValueError:
        return None
    if setup_move not in board.legal_moves:
        return None
    board.push(setup_move)
    puzzle_fen = board.fen()

    # Verify the entire solution line (Moves[1:]) is legal sequentially.
    solution_line = moves[1:]
    verify_board = board.copy(stack=False)
    for uci in solution_line:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if mv not in verify_board.legal_moves:
            return None
        verify_board.push(mv)

    raw_rating = row.get("Rating") if row.get("Rating") is not None else row.get("rating", 0)
    try:
        rating = int(raw_rating) if raw_rating != "" else 0
    except (TypeError, ValueError):
        return None

    raw_themes = row.get("Themes") if row.get("Themes") is not None else row.get("themes", "")
    if isinstance(raw_themes, str):
        themes = [t.strip() for t in raw_themes.split() if t.strip()]
    elif isinstance(raw_themes, list):
        themes = [str(t).strip() for t in raw_themes if str(t).strip()]
    else:
        themes = []

    return {
        "fen": puzzle_fen,
        "best_move": solution_line[0],
        "line": solution_line,
        "rating": rating,
        "themes": themes,
        "phase": classify_game_phase(puzzle_fen),
    }
