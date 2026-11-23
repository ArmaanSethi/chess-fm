#!/usr/bin/env python3
"""ACT-Eval-inspired deterministic Zero-Hallucination Claim Verifier (V2) for ChessFM Coach.

Extracts atomic verifiable chess claims (moves, piece-on-square assertions,
hyphenated/square-first piece references, attack/ray geometry claims,
check/checkmate/forcing-king-move assertions, side-to-move/actor polarity, and
hollow center-tension boilerplate) from free-form coaching prose or chain-of-thought
``<think>`` traces, and verifies each claim deterministically against
``python-chess`` (and optionally a Stockfish engine for blunder detection).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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

_CENTER_SQUARES: tuple[chess.Square, ...] = (chess.D4, chess.D5, chess.E4, chess.E5)

_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_end|endoftext|im_start)\|>")
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK_TAG_RE = re.compile(r"<think>", re.IGNORECASE)

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

# Hyphenated square-first piece references, e.g.:
#   "e8-pawn", "f7-pawn", "white's c4-bishop", "black g5-knight"
# Intentionally does NOT match "[a-h][1-8]-square" (e.g. "e6-square").
_HYPHENATED_PIECE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:(?P<color>white|black)(?:'s)?\s+)?"
    r"(?P<square>[a-h][1-8])-(?P<piece>king|queen|rook|bishop|knight|pawn)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Determiner-anchored space-separated square-first piece references, e.g.:
#   "the e8 pawn", "white's f7 pawn", "enemy g5 knight"
_DETERMINER_SQ_PIECE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:(?:(?P<color>white|black)(?:'s)?)|(?:the|your|their|opponent(?:'s)?|enemy))\s+"
    r"(?P<square>[a-h][1-8])\s+"
    r"(?P<piece>king|queen|rook|bishop|knight|pawn)"
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
    r"try|tries|trying|consider|after|reply|replies|with|best\s+move(?:\s+is|:)|"
    r"choice\s+is|instead(?:\s+of|,)|then|or|"
    r"(?:expected|key|main|best)?\s*(?:continuation|line|variation):)\s+"
    r")(?P<move>[a-h][2-7])(?![A-Za-z0-9=+#])",
    re.IGNORECASE,
)

# Continuation pawn move immediately following another move token, e.g. "1. e4 e5"
_FOLLOWUP_PAWN_MOVE_RE = re.compile(r"^\s+(?P<move>[a-h][2-7])(?![A-Za-z0-9=+#])")

# Text separators between two moves that signal a fresh variation from the root position.
_VARIATION_RESET_RE = re.compile(
    r"(?:(?<!\d)[.!?;]\s+|\b(?:continuation|line|variation|instead|alternatively)\s*[:,\s])",
    re.IGNORECASE,
)

# Check and checkmate prose claims.
_CHECKMATE_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:delivers|delivering|gives|giving|is|achieves|achieving|scores|scoring|forces|forcing)"
    r"\s+(?:a\s+)?(?:checkmate|mate)"
    r"(?:\s+on\s+(?P<sq1>[a-h][1-8]))?"
    r"|(?:checkmate|mate)\s+on\s+(?P<sq2>[a-h][1-8])"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_KING_OWNER_1 = (
    r"(?:the\s+)?(?:(?P<king_color1>white|black)(?:'s)?\s+"
    r"|(?:your\s+opponent(?:'s)?|the\s+opponent(?:'s)?|opponent(?:'s)?|your|their|enemy)\s+)?"
)
_KING_OWNER_2 = (
    r"(?:the\s+)?(?:(?P<king_color2>white|black)(?:'s)?\s+"
    r"|(?:your\s+opponent(?:'s)?|the\s+opponent(?:'s)?|opponent(?:'s)?|your|their|enemy)\s+)?"
)
_KING_OWNER_3 = (
    r"(?:the\s+)?(?:(?P<king_color3>white|black)(?:'s)?\s+"
    r"|(?:your\s+opponent(?:'s)?|the\s+opponent(?:'s)?|opponent(?:'s)?|your|their|enemy)\s+)?"
)

_CHECK_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:delivers|delivering|gives|giving|is|comes\s+with|with)\s+(?:a\s+)?check(?!mate)"
    r"(?:\s+on\s+(?P<sq1>[a-h][1-8]))?"
    r"|check(?!mate)\s+on\s+(?P<sq2>[a-h][1-8])"
    r"|(?:puts|putting|places|placing|leaves|leaving)\s+"
    + _KING_OWNER_1
    + r"king\s+in\s+check(?!mate)"
    r"|(?:checks?|checking)\s+" + _KING_OWNER_2 + r"king(?!['’]s\b)"
    r"|(?:forces?|forcing)\s+"
    + _KING_OWNER_3
    + r"king\s+to\s+(?:move|retreat|flee|step|run|relocate)"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Attack / ray geometry claims:
# Allows up to 3 intervening determiners/adjectives (excluding clause-joining words).
_INTERVENING_ADJ = (
    r"(?:(?!(?:and|or|but|to|by|with|from|for|that|which|while|if|when|after|before|in|on|at)\b)"
    r"[A-Za-z]+(?:'s)?\s+){0,3}"
)

_ATTACK_CLAIM_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<verb>attacks?|attacking|challenges?|challenging|threatens?|threatening|"
    r"targets?|targeting|hits?|hitting|pressures?|pressuring)\s+" + _INTERVENING_ADJ + r"(?:"
    r"(?:king|queen|rook|bishop|knight|pawn)\s+(?:on|at)\s+(?P<sq1>[a-h][1-8])"
    r"|(?P<sq2>[a-h][1-8])(?:-(?:pawn|knight|bishop|rook|queen|king|square)"
    r"|\s+(?:pawn|knight|bishop|rook|queen|king|square))"
    r"|(?P<sq3>[a-h][1-8])(?:\s+square)?"
    r")"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Explicit side-to-move prose assertions (e.g. "White is to move", "Black to move").
_SIDE_TO_MOVE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:it\s+is\s+|with\s+)?"
    r"(?P<color>white|black)\s+(?:is\s+)?to\s+move"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Explicit color attribution immediately preceding a move token.
_ACTOR_MOVE_PREFIX_RE = re.compile(
    r"(?:"
    r"\b(?P<actor1>white|black)\s+(?:plays|play|played|moves|move|can\s+play|could\s+play|"
    r"should\s+play|must\s+play|responds\s+with|replies\s+with|captures\s+with|"
    r"pushes|tries|chooses)"
    r"|\b(?P<actor2>white|black)(?:'s)?\s+(?:best|strongest|top|only|main|key|winning|correct)\s+"
    r"move(?:\s+is|:)?"
    r")\s*[`*'\"(\[]*\s*$",
    re.IGNORECASE,
)

# Opponent pronoun attribution immediately preceding a root-turn move token.
_OPPONENT_PRONOUN_PREFIX_RE = re.compile(
    r"\bif\s+(?:the\s+|your\s+)?(?:opponent|enemy|they)\s+"
    r"(?:plays?|moves?|responds?\s+with|replies?\s+with|captures?\s+with|pushes|tries|chooses)"
    r"\s*[`*'\"(\[]*\s*$",
    re.IGNORECASE,
)

# Hollow center-tension boilerplate claim.
_CENTER_TENSION_RE = re.compile(
    r"\b(?:tension\s+in\s+the\s+center|central\s+tension)\b",
    re.IGNORECASE,
)

_NEGATION_WINDOW_RE = re.compile(
    r"(?:n['’]t\b|\b(?:not|cannot|no|never|without|avoid|avoids|avoiding|prevent|prevents|"
    r"preventing|stop|stops|stopping|defend|defends|defending|threaten|"
    r"threatens|threatening)\b)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)

_FUTURE_HEDGE_RE = re.compile(
    r"\b(?:later|eventually|prepares?\s+to|plans?\s+to|aiming\s+to|"
    r"can\s+later|could\s+eventually|intends?\s+to|hopes?\s+to|"
    r"looking\s+to|will\s+later)\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)

# Object-taking verbs immediately preceding a piece_on_square claim (e.g.
# "ignoring the bishop on e6", "attacking the bishop on e6"). Such target pieces
# must not be promoted to attacker subjects for subsequent attack claims.
_OBJECT_VERB_PREFIX_RE = re.compile(
    r"\b(?:attack(?:s|ed|ing)?|challeng(?:e|es|ed|ing)|threaten(?:s|ed|ing)?|"
    r"target(?:s|ed|ing)?|hit(?:s|ting)?|pressur(?:e|es|ed|ing)|"
    r"ignor(?:e|es|ed|ing)|captur(?:e|es|ed|ing)|tak(?:e|es|ing)|took|"
    r"win(?:s|ning)?|won|pin(?:s|ned|ning)?|fork(?:s|ed|ing)?|"
    r"defend(?:s|ed|ing)?|protect(?:s|ed|ing)?|guard(?:s|ed|ing)?|"
    r"block(?:s|ed|ing)?|trad(?:e|es|ed|ing)|exchang(?:e|es|ed|ing))\s+" + _INTERVENING_ADJ + r"$",
    re.IGNORECASE,
)

# Coordinating conjunction between a piece mention and a participle attack verb.
_COORD_CONJ_RE = re.compile(r"\b(?:and|or)\b", re.IGNORECASE)

# Infinitive relocation-purpose guard for static piece_on_square attack claims
# (e.g. "reposition the knight on c7 to challenge your dominant rook on g7").
_RELOCATION_VERB_RE = re.compile(
    r"\b(?P<verb>reposition(?:s|ed|ing)?|relocat(?:e|es|ed|ing)|mov(?:e|es|ed|ing)|"
    r"jump(?:s|ed|ing)?|retreat(?:s|ed|ing)?|rerout(?:e|es|ed|ing)|"
    r"maneuver(?:s|ed|ing)?|bring(?:s|ing)?|shift(?:s|ed|ing)?|"
    r"transfer(?:s|red|ring)?|swing(?:s|ing)?|slid(?:e|es|ing))\b",
    re.IGNORECASE,
)
_TO_END_RE = re.compile(r"\bto\s+$", re.IGNORECASE)
_PREV_WORD_RE = re.compile(r"([A-Za-z]+)\s+$")
_NOUN_MOVE_MODIFIERS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "the",
        "a",
        "an",
        "your",
        "their",
        "my",
        "its",
        "his",
        "her",
        "our",
        "best",
        "strongest",
        "good",
        "bad",
        "weak",
        "next",
        "last",
        "played",
        "previous",
        "candidate",
        "only",
        "key",
        "winning",
        "losing",
        "quiet",
        "forcing",
        "tactical",
        "natural",
    }
)

# Missed/hypothetical opportunity guard for check-claim move lookback
# (e.g. "Why not Re4? ... it misses a tactical opportunity ... with a check.").
_MISSED_OPPORTUNITY_RE = re.compile(
    r"\b(?:miss(?:es|ed|ing)?|overlook(?:s|ed|ing)?|fail(?:s|ed|ing)?\s+to|"
    r"give(?:s|n)?\s+up|giving\s+up|instead\s+of|rather\s+than)\b",
    re.IGNORECASE,
)

_AVOIDANCE_WINDOW_RE = re.compile(
    r"\b(?:avoid|avoids|avoiding|not|never|don['’]t\s+play|instead\s+of|"
    r"blunder|mistake|bad\s+move)\s+(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    """Return True if ``span`` overlaps any interval in ``occupied``."""
    start, end = span
    return any(not (end <= s or start >= e) for s, e in occupied)


def _is_negated(text: str, start: int) -> bool:
    """Return True if the phrase starting at ``start`` is immediately negated."""
    prefix = text[max(0, start - 40) : start]
    return bool(_NEGATION_WINDOW_RE.search(prefix))


def _is_future_hedged(text: str, start: int) -> bool:
    """Return True if an attack claim is hedged as a future/indirect plan."""
    prefix = text[max(0, start - 45) : start]
    return bool(_FUTURE_HEDGE_RE.search(prefix))


def _is_relocation_infinitive(text: str, start: int) -> bool:
    """Return True if an attack claim is the infinitive purpose of relocating a piece.

    Excludes noun uses of ``move``/``moves`` preceded by determiners or adjectives
    (e.g. ``"After this move, the bishop on c4 tries to attack h8."``).
    """
    prefix = text[max(0, start - 80) : start]
    to_match = _TO_END_RE.search(prefix)
    if to_match is None:
        return False
    before_to = prefix[: to_match.start()]
    sent_tail = re.split(r"[.!?\n]+", before_to)[-1]
    window = sent_tail[-60:]
    offset = len(sent_tail) - len(window)
    for m in _RELOCATION_VERB_RE.finditer(window):
        verb = m.group("verb").lower()
        if verb in ("move", "moves"):
            prev_m = _PREV_WORD_RE.search(sent_tail[: offset + m.start()])
            if prev_m and prev_m.group(1).lower() in _NOUN_MOVE_MODIFIERS:
                continue
        return True
    return False


def _sentence_distance(text: str, pos_a: int, pos_b: int) -> int:
    """Count sentence boundaries separating ``pos_a`` and ``pos_b``."""
    lo, hi = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
    segment = text[lo:hi]
    # Ignore move-number periods like "1. e4" or "12...Nf6".
    cleaned = re.sub(r"\b\d+\.(?:\.\.)?", "", segment)
    return len(re.findall(r"[.!?\n]+", cleaned))


def _same_sentence(text: str, pos_a: int, pos_b: int) -> bool:
    """Return True if no sentence boundary separates ``pos_a`` and ``pos_b``."""
    return _sentence_distance(text, pos_a, pos_b) == 0


def _has_center_tension(board: chess.Board) -> bool:
    """Return True if central squares (d4/d5/e4/e5) have genuine tension.

    Central tension exists when any piece occupying d4, d5, e4, or e5 is
    attacked by the opposing side, or when both sides occupy at least one
    central square.
    """
    white_occupies = False
    black_occupies = False
    for sq in _CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.color == chess.WHITE:
            white_occupies = True
        else:
            black_occupies = True
        if board.is_attacked_by(not piece.color, sq):
            return True
    return white_occupies and black_occupies


# ---------------------------------------------------------------------------
# Claim Extraction
# ---------------------------------------------------------------------------


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Extract atomic verifiable chess claims from coaching prose.

    Supported claim types:
    - ``piece_on_square``: e.g. ``"knight on f6"``, ``"white bishop on c4"``,
      ``"e8-pawn"``, ``"the f7 pawn"``.
    - ``move``: UCI moves (``e2e4``, ``e7e8q``) and SAN moves (``Nf3``,
      ``Bxc6+``, ``O-O``, ``Qxf7#``, ``exd5``, contextual pawn pushes ``1. e4``).
    - ``check_claim``: prose assertions like ``"gives check"``,
      ``"puts the king in check"``, ``"forces the white king to move"``,
      ``"delivers checkmate"``, ``"checkmate on f7"``.
    - ``attack_claim``: geometric attack claims like ``"attacks f7"``,
      ``"challenge the dominant rook on g7"``, ``"pressure the e6-square"``.
    - ``side_to_move``: explicit turn assertions like ``"White is to move"``.

    Returns claims sorted by their character position in ``text``.
    """
    if not text:
        return []

    text = _SPECIAL_TOKEN_RE.sub("", text)
    claims: list[dict[str, Any]] = []
    occupied_move_spans: list[tuple[int, int]] = []

    # 1. Piece-on-square claims (reserve their spans so their square tokens are
    #    not misread as contextual pawn moves).
    piece_spans: list[tuple[int, int]] = []
    for pattern in (_PIECE_ON_SQUARE_RE, _HYPHENATED_PIECE_RE, _DETERMINER_SQ_PIECE_RE):
        for m in pattern.finditer(text):
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
                "king_color": None,
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
        king_color = (
            m.group("king_color1") or m.group("king_color2") or m.group("king_color3") or ""
        ).lower() or None
        claims.append(
            {
                "type": "check_claim",
                "kind": "check",
                "square": sq,
                "king_color": king_color,
                "raw": m.group(0),
                "span": span,
            }
        )

    # 6. Attack / ray geometry claims.
    attack_spans: list[tuple[int, int]] = []
    for m in _ATTACK_CLAIM_RE.finditer(text):
        span = (m.start(), m.end())
        if _overlaps(span, attack_spans):
            continue
        if _is_negated(text, m.start()) or _is_future_hedged(text, m.start()):
            continue
        target_sq = (m.group("sq1") or m.group("sq2") or m.group("sq3") or "").lower()
        if not target_sq:
            continue
        attack_spans.append(span)
        claims.append(
            {
                "type": "attack_claim",
                "verb": m.group("verb").lower(),
                "target": target_sq,
                "raw": m.group(0),
                "span": span,
            }
        )

    # 7. Explicit side-to-move assertions.
    for m in _SIDE_TO_MOVE_RE.finditer(text):
        if _is_negated(text, m.start()):
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


def _verify_attack_claim(
    root_board: chess.Board,
    claim: dict[str, Any],
    linked_move_ctx: tuple[chess.Board, chess.Move, chess.Board] | None,
    linked_piece_sq: chess.Square | None,
) -> str | None:
    """Verify an ``attack_claim`` using ``python-chess`` ray/attack geometry."""
    target_name: str = claim["target"]
    target_sq = chess.parse_square(target_name)

    if linked_move_ctx is not None:
        board_before, move, board_after = linked_move_ctx
        mover = board_before.turn
        direct_attack = target_sq in board_after.attacks(move.to_square)
        discovered_or_castled = len(board_after.attackers(mover, target_sq)) > len(
            board_before.attackers(mover, target_sq)
        )
        if direct_attack or discovered_or_castled:
            return None
        return (
            f"Move {board_before.san(move)} does not attack or add pressure to {target_name} "
            f"(claimed '{claim['raw']}')"
        )

    if linked_piece_sq is not None:
        if target_sq in root_board.attacks(linked_piece_sq):
            return None
        return (
            f"Piece on {chess.square_name(linked_piece_sq)} does not attack {target_name} "
            f"(claimed '{claim['raw']}')"
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
    king_color: str | None = claim.get("king_color")

    if linked_move_ctx is not None:
        board_before, move, board_after = linked_move_ctx
        if target_sq is not None and move.to_square != target_sq:
            return (
                f"Claimed {kind} on {target_sq_name}, but move "
                f"{board_before.san(move)} lands on {chess.square_name(move.to_square)}"
            )
        if king_color is not None:
            opponent_color = "black" if board_before.turn == chess.WHITE else "white"
            if king_color != opponent_color:
                return (
                    f"Move {board_before.san(move)} cannot check {king_color}'s king "
                    f"(opponent is {opponent_color})"
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
    root_mover_color = "white" if root_board.turn == chess.WHITE else "black"
    if root_board.is_check():
        if king_color is None or king_color == root_mover_color:
            return None
    if king_color is not None and king_color == root_mover_color:
        return (
            f"{root_mover_color.title()} to move cannot check or force "
            f"{king_color}'s own king to move"
        )
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
    *,
    check_boilerplate: bool = True,
) -> dict[str, Any]:
    """Deterministically verify all extracted chess claims against ``fen``.

    Args:
        fen: Position FEN string.
        text: Coach prose to verify.
        engine: Optional ``chess.engine.SimpleEngine`` (or compatible mock)
            used to flag recommended root moves that blunder more than
            ``MAX_CENTIPAWN_LOSS`` centipawns.
        check_boilerplate: When ``True`` (default), flags hollow center-tension
            boilerplate claims when the center squares (d4/d5/e4/e5) have no
            genuine tension.

    Returns:
        Dictionary with keys ``passed``, ``factuality_score``,
        ``total_claims``, ``verified_claims``, and ``violations``.
    """
    root_board = chess.Board(fen)
    var_board = root_board.copy()
    cleaned_text = _SPECIAL_TOKEN_RE.sub("", text or "")

    claims = extract_claims(cleaned_text)
    attack_claim_spans: list[tuple[int, int]] = [
        c["span"] for c in claims if c["type"] == "attack_claim"
    ]
    violations: list[dict[str, Any]] = []
    verified_claims = 0

    # Track the most recent verified move and piece-on-square so subsequent
    # claims in the same sentence bind to them.
    last_move_span: tuple[int, int] | None = None
    last_move_ctx: tuple[chess.Board, chess.Move, chess.Board] | None = None
    last_verified_piece_span: tuple[int, int] | None = None
    last_verified_piece_sq: chess.Square | None = None
    root_moves_to_engine_check: list[tuple[dict[str, Any], chess.Move]] = []

    for claim in claims:
        ctype = claim["type"]

        if ctype == "piece_on_square":
            if claim["piece"] == "pawn" and claim["square"][1] in ("1", "8"):
                violations.append(
                    {
                        "type": "impossible_pawn_rank",
                        "claim": claim,
                        "reason": (
                            f"Impossible pawn on rank {claim['square'][1]} "
                            f"({claim['square']}): pawns cannot occupy rank 1 or 8"
                        ),
                    }
                )
                continue

            err = _verify_piece_on_square(root_board, claim)
            verified_on_root = err is None
            if err is not None:
                if (
                    last_move_span is not None
                    and last_move_ctx is not None
                    and _same_sentence(cleaned_text, last_move_span[1], claim["span"][0])
                ):
                    err_after = _verify_piece_on_square(last_move_ctx[2], claim)
                    if err_after is None:
                        err = None

            if err is not None:
                violations.append({"type": "piece_on_square", "claim": claim, "reason": err})
            else:
                verified_claims += 1
                if verified_on_root:
                    obj_prefix = cleaned_text[max(0, claim["span"][0] - 35) : claim["span"][0]]
                    is_object_piece = _overlaps(claim["span"], attack_claim_spans) or bool(
                        _OBJECT_VERB_PREFIX_RE.search(obj_prefix)
                    )
                    if not is_object_piece:
                        last_verified_piece_span = claim["span"]
                        last_verified_piece_sq = chess.parse_square(claim["square"])

        elif ctype == "move":
            move_str = claim["move"]
            notation = claim["notation"]
            prev_move_span = last_move_span

            if last_move_span is not None:
                between = cleaned_text[last_move_span[1] : claim["span"][0]]
                if _VARIATION_RESET_RE.search(between):
                    var_board = root_board.copy()

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

            # Check explicit color actor attribution and root-turn pronoun polarity.
            prefix = cleaned_text[max(0, claim["span"][0] - 55) : claim["span"][0]]
            actual_color = "white" if board_used.turn == chess.WHITE else "black"
            polarity_err: str | None = None

            actor_match = _ACTOR_MOVE_PREFIX_RE.search(prefix)
            if actor_match is not None:
                actor = (actor_match.group("actor1") or actor_match.group("actor2") or "").lower()
                if actor and actor != actual_color:
                    polarity_err = (
                        f"Attributed move '{move_str}' to {actor.title()}, "
                        f"but it is {actual_color.title()}'s move"
                    )
            elif used_root and (
                prev_move_span is None
                or not _same_sentence(cleaned_text, prev_move_span[1], claim["span"][0])
            ):
                if _OPPONENT_PRONOUN_PREFIX_RE.search(prefix):
                    polarity_err = (
                        f"Attributed root-turn move '{move_str}' to the opponent, "
                        f"but it is {actual_color.title()}'s turn to move"
                    )

            # Check explicit + / # suffix accuracy for SAN moves.
            suffix_err = (
                _verify_move_check_suffix(board_used, move, move_str) if notation == "san" else None
            )
            board_before = board_used.copy()
            var_board.push(move)
            board_after = var_board.copy()
            last_move_span = claim["span"]
            last_move_ctx = (board_before, move, board_after)

            move_has_violation = False
            if polarity_err is not None:
                move_has_violation = True
                violations.append(
                    {
                        "type": "color_polarity_violation",
                        "claim": claim,
                        "reason": polarity_err,
                    }
                )
            if suffix_err is not None:
                move_has_violation = True
                violations.append(
                    {
                        "type": "invalid_check_annotation",
                        "claim": claim,
                        "reason": suffix_err,
                    }
                )

            if not move_has_violation:
                verified_claims += 1
                if used_root and engine is not None:
                    avoid_prefix = cleaned_text[max(0, claim["span"][0] - 30) : claim["span"][0]]
                    if not _AVOIDANCE_WINDOW_RE.search(avoid_prefix):
                        root_moves_to_engine_check.append((claim, move))

        elif ctype == "check_claim":
            linked_ctx: tuple[chess.Board, chess.Move, chess.Board] | None = None
            if last_move_span is not None and last_move_ctx is not None:
                sent_dist = _sentence_distance(cleaned_text, last_move_span[1], claim["span"][0])
                if sent_dist == 0:
                    between = cleaned_text[last_move_span[1] : claim["span"][0]]
                    if claim["square"] is not None or not _MISSED_OPPORTUNITY_RE.search(between):
                        linked_ctx = last_move_ctx
                elif sent_dist == 1 and claim["square"] is None:
                    between = cleaned_text[last_move_span[1] : claim["span"][0]]
                    curr_sent_prefix = re.split(r"[.!?\n]+", between)[-1]
                    if not _MISSED_OPPORTUNITY_RE.search(curr_sent_prefix):
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

        elif ctype == "attack_claim":
            linked_move_ctx: tuple[chess.Board, chess.Move, chess.Board] | None = None
            linked_piece_sq: chess.Square | None = None
            move_in_sent = (
                last_move_span is not None
                and last_move_ctx is not None
                and _same_sentence(cleaned_text, last_move_span[1], claim["span"][0])
            )
            piece_in_sent = (
                last_verified_piece_span is not None
                and last_verified_piece_sq is not None
                and _same_sentence(cleaned_text, last_verified_piece_span[1], claim["span"][0])
            )
            if move_in_sent and piece_in_sent:
                assert last_move_span is not None and last_verified_piece_span is not None
                if last_verified_piece_span[1] > last_move_span[1]:
                    between_piece_and_attack = cleaned_text[
                        last_verified_piece_span[1] : claim["span"][0]
                    ]
                    is_conj_participle = claim["verb"].endswith("ing") and bool(
                        _COORD_CONJ_RE.search(between_piece_and_attack)
                    )
                    if is_conj_participle:
                        linked_move_ctx = last_move_ctx
                    else:
                        linked_piece_sq = last_verified_piece_sq
                else:
                    linked_move_ctx = last_move_ctx
            elif move_in_sent:
                linked_move_ctx = last_move_ctx
            elif piece_in_sent:
                linked_piece_sq = last_verified_piece_sq

            if linked_move_ctx is None and linked_piece_sq is not None:
                if _is_relocation_infinitive(cleaned_text, claim["span"][0]):
                    linked_piece_sq = None

            err = _verify_attack_claim(root_board, claim, linked_move_ctx, linked_piece_sq)
            if err is not None:
                violations.append(
                    {
                        "type": "invalid_attack_claim",
                        "claim": claim,
                        "reason": err,
                    }
                )
            else:
                verified_claims += 1

        elif ctype == "side_to_move":
            if (
                last_move_span is not None
                and last_move_ctx is not None
                and _same_sentence(cleaned_text, last_move_span[1], claim["span"][0])
            ):
                ref_board = last_move_ctx[2]
            else:
                ref_board = root_board

            actual_turn = "white" if ref_board.turn == chess.WHITE else "black"
            if claim["color"] != actual_turn:
                violations.append(
                    {
                        "type": "color_polarity_violation",
                        "claim": claim,
                        "reason": (
                            f"Claimed {claim['color'].title()} to move, "
                            f"but it is {actual_turn.title()} to move"
                        ),
                    }
                )
            else:
                verified_claims += 1

    # Per-row hollow center-tension boilerplate check.
    extra_claims = 0
    if check_boilerplate and cleaned_text:
        for m in _CENTER_TENSION_RE.finditer(cleaned_text):
            if _is_negated(cleaned_text, m.start()):
                continue
            if not _has_center_tension(root_board):
                extra_claims += 1
                bp_claim = {
                    "type": "hollow_boilerplate",
                    "raw": m.group(0),
                    "span": (m.start(), m.end()),
                }
                violations.append(
                    {
                        "type": "hollow_boilerplate",
                        "claim": bp_claim,
                        "reason": (
                            "Claimed center tension, but center squares (d4/d5/e4/e5) "
                            "have no contested tension"
                        ),
                    }
                )
                break

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

    total_claims = len(claims) + extra_claims
    factuality_score = verified_claims / total_claims if total_claims > 0 else 1.0

    return {
        "passed": len(violations) == 0,
        "factuality_score": factuality_score,
        "total_claims": total_claims,
        "verified_claims": verified_claims,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Corpus Distinct-4-Gram / Anti-Slop Boilerplate Detector
# ---------------------------------------------------------------------------


def detect_boilerplate_slop(
    texts: list[str],
    *,
    max_doc_freq_ratio: float = 0.20,
    min_doc_count: int = 3,
    min_distinct_4gram_ratio: float = 0.50,
) -> dict[str, Any]:
    """Measure corpus-wide 4-gram lexical diversity and flag boilerplate slop.

    Strips special chat tokens (``<|im_end|>``, ``<|endoftext|>``) before
    tokenizing so raw model completions and teacher completions are scored on
    an apples-to-apples basis.

    Returns:
        Dictionary with ``total_texts``, ``total_4grams``, ``unique_4grams``,
        ``distinct_4gram_ratio``, ``exact_duplicate_rate``,
        ``top_repeated_4grams``, ``flagged_indices``, ``flagged_count``,
        ``flagged_rate``, and ``is_slop_corpus``.
    """
    n_texts = len(texts)
    if n_texts == 0:
        return {
            "total_texts": 0,
            "total_4grams": 0,
            "unique_4grams": 0,
            "distinct_4gram_ratio": 1.0,
            "exact_duplicate_rate": 0.0,
            "top_repeated_4grams": [],
            "flagged_indices": [],
            "flagged_count": 0,
            "flagged_rate": 0.0,
            "is_slop_corpus": False,
        }

    total_4grams = 0
    unique_4grams_set: set[tuple[str, str, str, str]] = set()
    doc_freq: Counter[tuple[str, str, str, str]] = Counter()
    per_doc_4grams: list[set[tuple[str, str, str, str]]] = []
    norm_texts: list[str] = []

    for raw in texts:
        cleaned = _SPECIAL_TOKEN_RE.sub("", raw or "").strip()
        tokens = re.findall(r"[a-z0-9]+", cleaned.lower())
        norm_texts.append(" ".join(tokens))

        doc_grams: set[tuple[str, str, str, str]] = set()
        if len(tokens) >= 4:
            for i in range(len(tokens) - 3):
                ng = (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
                total_4grams += 1
                unique_4grams_set.add(ng)
                doc_grams.add(ng)
        for ng in doc_grams:
            doc_freq[ng] += 1
        per_doc_4grams.append(doc_grams)

    unique_4grams = len(unique_4grams_set)
    distinct_4gram_ratio = unique_4grams / total_4grams if total_4grams > 0 else 1.0
    exact_duplicate_rate = 1.0 - (len(set(norm_texts)) / n_texts)

    top_repeated = [
        {
            "ngram": " ".join(ng),
            "doc_count": count,
            "doc_freq": round(count / n_texts, 4),
        }
        for ng, count in sorted(doc_freq.items(), key=lambda kv: (-kv[1], " ".join(kv[0])))[:10]
    ]

    slop_grams = {
        ng
        for ng, count in doc_freq.items()
        if count >= min_doc_count and (count / n_texts) >= max_doc_freq_ratio
    }

    norm_text_counts = Counter(t for t in norm_texts if t)
    slop_exact_texts = {
        t
        for t, count in norm_text_counts.items()
        if count >= min_doc_count and (count / n_texts) >= max_doc_freq_ratio
    }

    flagged_indices = [
        i
        for i, (grams, norm) in enumerate(zip(per_doc_4grams, norm_texts, strict=True))
        if (grams & slop_grams) or (norm in slop_exact_texts)
    ]
    flagged_count = len(flagged_indices)
    flagged_rate = flagged_count / n_texts
    is_slop_corpus = (
        distinct_4gram_ratio < min_distinct_4gram_ratio or flagged_rate >= max_doc_freq_ratio
    )

    return {
        "total_texts": n_texts,
        "total_4grams": total_4grams,
        "unique_4grams": unique_4grams,
        "distinct_4gram_ratio": round(distinct_4gram_ratio, 4),
        "exact_duplicate_rate": round(exact_duplicate_rate, 4),
        "top_repeated_4grams": top_repeated,
        "flagged_indices": flagged_indices,
        "flagged_count": flagged_count,
        "flagged_rate": round(flagged_rate, 4),
        "is_slop_corpus": is_slop_corpus,
    }


# ---------------------------------------------------------------------------
# Chain-of-Thought Faithfulness Verification
# ---------------------------------------------------------------------------


@dataclass
class CoTVerificationResult:
    """Result of deterministic claim verification on a ``<think>...</think>`` trace."""

    claims_total: int
    claims_verified: int
    claim_accuracy: float
    zero_violation: bool
    violations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize verification result to a JSON-compatible dictionary."""
        return {
            "claims_total": self.claims_total,
            "claims_verified": self.claims_verified,
            "claim_accuracy": self.claim_accuracy,
            "zero_violation": self.zero_violation,
            "violations": self.violations,
        }


def verify_think_trace(fen: str, think_text: str) -> CoTVerificationResult:
    """Verify chess claims inside a model's chain-of-thought ``<think>`` trace.

    If ``<think>...</think>`` tags are present, extracts and verifies the inner
    reasoning content; otherwise verifies ``think_text`` directly. Disables the
    coaching-prose center-tension boilerplate check while keeping all piece,
    move, ray-attack, check, and color-polarity checks active.
    """
    raw = think_text or ""
    blocks = _THINK_TAG_RE.findall(raw)
    if blocks:
        cleaned = "\n".join(b.strip() for b in blocks if b.strip())
    else:
        open_match = _OPEN_THINK_TAG_RE.search(raw)
        if open_match is not None:
            cleaned = raw[open_match.end() :].strip()
        else:
            cleaned = raw.strip()
    cleaned = _SPECIAL_TOKEN_RE.sub("", cleaned).strip()

    res = verify_claims(fen, cleaned, engine=None, check_boilerplate=False)
    return CoTVerificationResult(
        claims_total=res["total_claims"],
        claims_verified=res["verified_claims"],
        claim_accuracy=res["factuality_score"],
        zero_violation=res["passed"],
        violations=res["violations"],
    )


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
