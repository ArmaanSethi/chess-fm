#!/usr/bin/env python3
"""ACT-Eval-inspired deterministic Zero-Hallucination Claim Verifier for ChessFM Coach.

Extracts atomic verifiable chess claims (moves, piece-on-square assertions, and
check/checkmate assertions) from free-form coaching prose and verifies each
claim deterministically against ``python-chess`` (and optionally a Stockfish
engine for blunder detection).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import chess
import chess.engine

from chessfm.config import DEFAULT_STOCKFISH_DEPTH, MAX_CENTIPAWN_LOSS

# ---------------------------------------------------------------------------
# Constants & Regex Patterns
# ---------------------------------------------------------------------------

PIECE_NAME_TO_TYPE: dict[str, chess.PieceType] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

PIECE_TYPE_TO_NAME: dict[chess.PieceType, str] = {v: k for k, v in PIECE_NAME_TO_TYPE.items()}

# Piece-on-square assertions, e.g.:
#   "knight on f6", "white bishop on c4", "Black's rook on a8", "queen at d1"
_PIECE_ON_SQUARE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:(?P<color>white|black)(?:'s)?\s+)?"
    r"(?P<piece>king|queen|rook|bishop|knight|pawn)\s+"
    r"(?:on|at|sitting\s+on|placed\s+on|stationed\s+on)\s+"
    r"(?P<square>[a-h][1-8])"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# UCI moves: 4 coordinates + optional promotion piece (e.g. e2e4, e7e8q).
_UCI_MOVE_RE = re.compile(r"(?<![A-Za-z0-9])([a-h][1-8][a-h][1-8][qrbnQRBN]?)(?![A-Za-z0-9])")

# Unambiguous SAN moves:
#   - Castling: O-O-O, O-O, 0-0-0, 0-0 (with optional +/#)
#   - Piece moves: Nf3, Bxc6+, Nbd2, R1e2, Qxf7#, Qh4e1
#   - Pawn captures: exd5, axb8=Q#
#   - Pawn promotions: e8=Q+, a1=N
#   - Pawn pushes with explicit check/mate: e4+, g7#
_SAN_MOVE_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"(?:O-O-O|O-O|0-0-0|0-0)[+#]?"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h][18]=[QRBN][+#]?"
    r"|[a-h][1-8][+#]"
    r")(?![A-Za-z0-9])"
)

# Bare pawn push moves (e.g. "1. e4", "play d4", "after c5", "e4 e5") require
# move context so square references like "controls e4" are not misclassified.
_CONTEXTUAL_PAWN_MOVE_RE = re.compile(
    r"(?:"
    r"\b\d+\.(?:\.\.)?\s*"
    r"|\b(?:play|plays|playing|played|push|pushes|pushing|move|moves|moving|"
    r"try|tries|trying|consider|after|reply|replies|with|best\s+move\s+is|"
    r"instead\s+of|then|or)\s+"
    r")(?P<move>[a-h][2-7])(?![A-Za-z0-9=+#])",
    re.IGNORECASE,
)

# Continuation pawn move immediately following another move token, e.g. "1. e4 e5"
_FOLLOWUP_PAWN_MOVE_RE = re.compile(r"^\s+(?P<move>[a-h][2-7])(?![A-Za-z0-9=+#])")

# Check and checkmate prose claims.
_CHECKMATE_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:delivers|gives|is|achieves|scores|forces)\s+(?:a\s+)?(?:checkmate|mate)"
    r"(?:\s+on\s+(?P<sq1>[a-h][1-8]))?"
    r"|(?:checkmate|mate)\s+on\s+(?P<sq2>[a-h][1-8])"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_CHECK_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:delivers|gives|is|comes\s+with|with)\s+(?:a\s+)?check(?!mate)"
    r"(?:\s+on\s+(?P<sq1>[a-h][1-8]))?"
    r"|check(?!mate)\s+on\s+(?P<sq2>[a-h][1-8])"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_NEGATION_WINDOW_RE = re.compile(
    r"\b(?:not|no|never|without|n't|avoid|avoids|avoiding|prevent|prevents|"
    r"preventing|stop|stops|stopping|defend|defends|defending|threaten|"
    r"threatens|threatening)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)

_AVOIDANCE_WINDOW_RE = re.compile(
    r"\b(?:avoid|avoids|avoiding|not|never|don't\s+play|instead\s+of|"
    r"blunder|mistake|bad\s+move)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    """Return True if ``span`` overlaps any interval in ``occupied``."""
    start, end = span
    return any(not (end <= s or start >= e) for s, e in occupied)


def _is_negated(text: str, start: int) -> bool:
    """Return True if the phrase starting at ``start`` is immediately negated."""
    prefix = text[max(0, start - 40) : start]
    return bool(_NEGATION_WINDOW_RE.search(prefix))


def _same_sentence(text: str, pos_a: int, pos_b: int) -> bool:
    """Return True if no sentence boundary separates ``pos_a`` and ``pos_b``."""
    lo, hi = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
    segment = text[lo:hi]
    # Ignore move-number periods like "1. e4" or "12...Nf6".
    cleaned = re.sub(r"\b\d+\.(?:\.\.)?", "", segment)
    return not bool(re.search(r"[.!?\n]", cleaned))


# ---------------------------------------------------------------------------
# Claim Extraction
# ---------------------------------------------------------------------------


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Extract atomic verifiable chess claims from coaching prose.

    Supported claim types:
    - ``piece_on_square``: e.g. ``"knight on f6"``, ``"white bishop on c4"``.
    - ``move``: UCI moves (``e2e4``, ``e7e8q``) and SAN moves (``Nf3``,
      ``Bxc6+``, ``O-O``, ``Qxf7#``, ``exd5``, contextual pawn pushes ``1. e4``).
    - ``check_claim``: prose assertions like ``"gives check"``,
      ``"delivers checkmate"``, ``"checkmate on f7"``.

    Returns claims sorted by their character position in ``text``.
    """
    if not text:
        return []

    claims: list[dict[str, Any]] = []
    occupied_move_spans: list[tuple[int, int]] = []

    # 1. Piece-on-square claims (reserve their spans so their square tokens are
    #    not misread as contextual pawn moves).
    piece_spans: list[tuple[int, int]] = []
    for m in _PIECE_ON_SQUARE_RE.finditer(text):
        span = (m.start(), m.end())
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

    # 2. UCI move claims.
    for m in _UCI_MOVE_RE.finditer(text):
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

    # 3. Unambiguous SAN move claims.
    for m in _SAN_MOVE_RE.finditer(text):
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

    # 4. Contextual bare pawn moves (e.g. "1. e4 e5", "play d4").
    for m in _CONTEXTUAL_PAWN_MOVE_RE.finditer(text):
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

    # Also capture immediate follow-up pawn moves in variations (e.g. "1. e4 e5").
    # Iterate until fixed point for multi-ply pawn variations like "1. e4 e5 2. d4 d5".
    added = True
    while added:
        added = False
        move_ends = sorted(end for _, end in occupied_move_spans)
        for end_pos in move_ends:
            fm = _FOLLOWUP_PAWN_MOVE_RE.match(text[end_pos:])
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

    # 5. Check and checkmate claims (checkmate first so "checkmate" is not also
    #    matched as "check").
    check_spans: list[tuple[int, int]] = []
    for m in _CHECKMATE_CLAIM_RE.finditer(text):
        span = (m.start(), m.end())
        if _is_negated(text, m.start()):
            continue
        check_spans.append(span)
        sq = (m.group("sq1") or m.group("sq2") or "").lower() or None
        claims.append(
            {
                "type": "check_claim",
                "kind": "checkmate",
                "square": sq,
                "raw": m.group(0),
                "span": span,
            }
        )

    for m in _CHECK_CLAIM_RE.finditer(text):
        span = (m.start(), m.end())
        if _overlaps(span, check_spans) or _is_negated(text, m.start()):
            continue
        check_spans.append(span)
        sq = (m.group("sq1") or m.group("sq2") or "").lower() or None
        claims.append(
            {
                "type": "check_claim",
                "kind": "check",
                "square": sq,
                "raw": m.group(0),
                "span": span,
            }
        )

    claims.sort(key=lambda c: c["span"][0])
    return claims


# ---------------------------------------------------------------------------
# Claim Verification
# ---------------------------------------------------------------------------


def _parse_move_on_board(
    board: chess.Board, move_str: str, notation: str
) -> tuple[chess.Move | None, str | None]:
    """Attempt to parse a move on ``board``.

    Returns ``(move, error_reason)`` where ``error_reason`` is ``None`` on
    success.
    """
    if notation == "uci":
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            return None, f"Invalid UCI move syntax: '{move_str}'"
        if move not in board.legal_moves:
            return None, f"Illegal UCI move '{move_str}' in position"
        return move, None

    try:
        move = board.parse_san(move_str)
    except chess.AmbiguousMoveError:
        return None, f"Ambiguous SAN move '{move_str}'"
    except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError):
        return None, f"Illegal SAN move '{move_str}' in position"
    return move, None


def _verify_move_check_suffix(
    board_before: chess.Board, move: chess.Move, move_str: str
) -> str | None:
    """Verify explicit ``+`` or ``#`` suffixes on SAN moves.

    ``python-chess``'s ``parse_san`` ignores check/checkmate suffix accuracy,
    so we check them explicitly to catch hallucinated checks/mates.
    """
    if move_str.endswith("#"):
        board_after = board_before.copy()
        board_after.push(move)
        if not board_after.is_checkmate():
            return f"Move '{move_str}' does not deliver checkmate"
    elif move_str.endswith("+"):
        if not board_before.gives_check(move):
            return f"Move '{move_str}' does not give check"
    return None


def _verify_piece_on_square(board: chess.Board, claim: dict[str, Any]) -> str | None:
    """Verify a ``piece_on_square`` claim against ``board``."""
    sq = chess.parse_square(claim["square"])
    piece = board.piece_at(sq)
    expected_type = PIECE_NAME_TO_TYPE[claim["piece"]]
    expected_color_str = claim["color"]

    if piece is None:
        return f"No piece on {claim['square']} (claimed {claim['raw']})"

    actual_color_str = "white" if piece.color == chess.WHITE else "black"
    actual_piece_str = PIECE_TYPE_TO_NAME[piece.piece_type]

    if piece.piece_type != expected_type:
        return (
            f"Expected {claim['piece']} on {claim['square']}, "
            f"but found {actual_color_str} {actual_piece_str}"
        )

    if expected_color_str is not None and actual_color_str != expected_color_str:
        return (
            f"Expected {expected_color_str} {claim['piece']} on {claim['square']}, "
            f"but found {actual_color_str} {actual_piece_str}"
        )

    return None


def _verify_check_claim(
    root_board: chess.Board,
    claim: dict[str, Any],
    linked_move_ctx: tuple[chess.Board, chess.Move, chess.Board] | None,
) -> str | None:
    """Verify a ``check`` or ``checkmate`` prose claim."""
    kind = claim["kind"]
    target_sq_name: str | None = claim["square"]
    target_sq = chess.parse_square(target_sq_name) if target_sq_name else None

    if linked_move_ctx is not None:
        board_before, move, board_after = linked_move_ctx
        if target_sq is not None and move.to_square != target_sq:
            return (
                f"Claimed {kind} on {target_sq_name}, but move "
                f"{board_before.san(move)} lands on {chess.square_name(move.to_square)}"
            )
        if kind == "checkmate":
            if not board_after.is_checkmate():
                return f"Move {board_before.san(move)} does not deliver checkmate"
        else:
            if not board_before.gives_check(move):
                return f"Move {board_before.san(move)} does not give check"
        return None

    # No preceding move in the same sentence: evaluate against the position.
    if target_sq is not None:
        for legal_move in root_board.legal_moves:
            if legal_move.to_square != target_sq:
                continue
            if kind == "checkmate":
                after = root_board.copy()
                after.push(legal_move)
                if after.is_checkmate():
                    return None
            else:
                if root_board.gives_check(legal_move):
                    return None
        return f"No legal move delivers {kind} on {target_sq_name}"

    if kind == "checkmate":
        if root_board.is_checkmate():
            return None
        for legal_move in root_board.legal_moves:
            after = root_board.copy()
            after.push(legal_move)
            if after.is_checkmate():
                return None
        return "Position is not checkmate and no legal move delivers checkmate"

    # kind == "check"
    if root_board.is_check():
        return None
    for legal_move in root_board.legal_moves:
        if root_board.gives_check(legal_move):
            return None
    return "King is not in check and no legal move gives check"


def _score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """Convert a Stockfish PovScore into centipawns from ``turn``'s perspective."""
    return int(score.pov(turn).score(mate_score=10000))


def _check_engine_blunder(
    root_board: chess.Board,
    move: chess.Move,
    engine: Any,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    max_cp_loss: int = MAX_CENTIPAWN_LOSS,
) -> str | None:
    """Use an optional Stockfish engine to flag severe blunders."""
    limit = chess.engine.Limit(depth=depth)
    best_info = engine.analyse(root_board, limit)
    best_score = _score_to_cp(best_info["score"], root_board.turn)

    after = root_board.copy()
    after.push(move)
    move_info = engine.analyse(after, limit)
    move_score = _score_to_cp(move_info["score"], root_board.turn)

    cp_loss = best_score - move_score
    if cp_loss > max_cp_loss:
        return (
            f"Recommended move {root_board.san(move)} loses {cp_loss} cp "
            f"(exceeds {max_cp_loss} cp threshold)"
        )
    return None


def verify_claims(
    fen: str,
    text: str,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Deterministically verify all extracted chess claims against ``fen``.

    Args:
        fen: Position FEN string.
        text: Coach prose to verify.
        engine: Optional ``chess.engine.SimpleEngine`` (or compatible mock)
            used to flag recommended root moves that blunder more than
            ``MAX_CENTIPAWN_LOSS`` centipawns.

    Returns:
        Dictionary with keys ``passed``, ``factuality_score``,
        ``total_claims``, ``verified_claims``, and ``violations``.
    """
    root_board = chess.Board(fen)
    var_board = root_board.copy()

    claims = extract_claims(text)
    violations: list[dict[str, Any]] = []
    verified_claims = 0

    # Track the most recent verified move and its position context so check
    # claims in the same sentence bind to that move.
    last_move_span: tuple[int, int] | None = None
    last_move_ctx: tuple[chess.Board, chess.Move, chess.Board] | None = None
    root_moves_to_engine_check: list[tuple[dict[str, Any], chess.Move]] = []

    for claim in claims:
        ctype = claim["type"]

        if ctype == "piece_on_square":
            err = _verify_piece_on_square(root_board, claim)
            if err is not None:
                violations.append({"type": "piece_on_square", "claim": claim, "reason": err})
            else:
                verified_claims += 1

        elif ctype == "move":
            move_str = claim["move"]
            notation = claim["notation"]

            # Try continuing the current variation first, then fall back to root
            # (handles both variations like "1. e4 e5 2. Nf3" and alternatives
            # like "Consider Bc4 or Bb5").
            move, err = _parse_move_on_board(var_board, move_str, notation)
            board_used = var_board
            used_root = var_board.fen() == root_board.fen()

            if move is None and not used_root:
                move, err = _parse_move_on_board(root_board, move_str, notation)
                if move is not None:
                    var_board = root_board.copy()
                    board_used = var_board
                    used_root = True

            if move is None:
                violations.append(
                    {
                        "type": "illegal_move",
                        "claim": claim,
                        "reason": err or f"Illegal move '{move_str}'",
                    }
                )
                last_move_span = None
                last_move_ctx = None
                continue

            # Check explicit + / # suffix accuracy for SAN moves.
            suffix_err = (
                _verify_move_check_suffix(board_used, move, move_str) if notation == "san" else None
            )
            board_before = board_used.copy()
            var_board.push(move)
            board_after = var_board.copy()
            last_move_span = claim["span"]
            last_move_ctx = (board_before, move, board_after)

            if suffix_err is not None:
                violations.append(
                    {
                        "type": "invalid_check_annotation",
                        "claim": claim,
                        "reason": suffix_err,
                    }
                )
            else:
                verified_claims += 1
                if used_root and engine is not None:
                    prefix = text[max(0, claim["span"][0] - 30) : claim["span"][0]]
                    if not _AVOIDANCE_WINDOW_RE.search(prefix):
                        root_moves_to_engine_check.append((claim, move))

        elif ctype == "check_claim":
            linked_ctx: tuple[chess.Board, chess.Move, chess.Board] | None = None
            if (
                last_move_span is not None
                and last_move_ctx is not None
                and _same_sentence(text, last_move_span[1], claim["span"][0])
            ):
                linked_ctx = last_move_ctx

            err = _verify_check_claim(root_board, claim, linked_ctx)
            if err is not None:
                violations.append(
                    {
                        "type": f"fake_{claim['kind']}",
                        "claim": claim,
                        "reason": err,
                    }
                )
            else:
                verified_claims += 1

    # Optional engine blunder check on recommended root moves.
    if engine is not None:
        for claim, move in root_moves_to_engine_check:
            blunder_err = _check_engine_blunder(root_board, move, engine)
            if blunder_err is not None:
                verified_claims = max(0, verified_claims - 1)
                violations.append(
                    {
                        "type": "blunder",
                        "claim": claim,
                        "reason": blunder_err,
                    }
                )

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
# Repair Loop
# ---------------------------------------------------------------------------


def verify_and_repair(
    fen: str,
    text: str,
    repair_fn: Callable[[str, list[dict[str, Any]]], str],
    max_retries: int = 2,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Verify ``text`` against ``fen`` and iteratively repair any violations.

    Calls ``repair_fn(current_text, violations)`` up to ``max_retries`` times
    until ``verify_claims`` passes.

    Returns:
        Dictionary with keys ``text``, ``verification``, and ``retries_used``.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    current_text = text
    verification = verify_claims(fen, current_text, engine=engine)
    retries_used = 0

    while not verification["passed"] and retries_used < max_retries:
        current_text = repair_fn(current_text, verification["violations"])
        retries_used += 1
        verification = verify_claims(fen, current_text, engine=engine)

    return {
        "text": current_text,
        "verification": verification,
        "retries_used": retries_used,
    }
