#!/usr/bin/env python3
"""Gemma 4 multimodal prompt builder and high-level CoachOrchestrator facade.

Provides:
- ``build_gemma4_coach_messages``: builds Gemma 4 multimodal ``messages`` using
  ``"user"`` / ``"model"`` roles, ``{"type": "image", ...}`` content parts when
  an image is attached, audio transcript grounding, and authoritative Stockfish
  + Maia-2 context so the LLM never guesses board state.
- ``verify_chess_claims``: deterministic claim verifier that validates UCI/SAN
  moves and piece-on-square assertions in coach prose against ``chess.Board``.
- ``CoachOrchestrator``: high-level facade tying together Stockfish analysis,
  Maia-2 human trap detection, 3-level Socratic hints, Claim Verification, and
  optional LLM generation (``llm_fn``) with a deterministic CPU-only fallback.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import chess
import chess.engine

from chessfm.coach.game_review import (
    PIECE_VALUES,
    _analyze_position,
    _static_eval,
    cp_to_win_prob,
    review_pgn_game,
)
from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.coach.verifier import verify_and_repair, verify_claims
from chessfm.config import render_ascii_board
from chessfm.rl.engine import _cp_from_white, close_engine

_COACH_SYSTEM_PREAMBLE = (
    "You are ChessFM Coach, a patient and precise chess instructor. "
    "Never guess or invent the board state: rely strictly on the verified "
    "FEN, ASCII board, Stockfish evaluation, and Maia human-move grounding "
    "provided below. Every move you recommend must be legal in the given position. "
    "Tailor your language to the student's Elo rating."
)

# Regex patterns for deterministic claim verification.
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
_SAN_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    r"O-O-O|O-O"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"
    r")(?![A-Za-z0-9_])"
)
_PIECE_ON_SQ_RE = re.compile(
    r"\b(?:(white|black)\s+)?"
    r"(pawn|knight|bishop|rook|queen|king)\s+"
    r"(?:on|at)\s+([a-h][1-8])\b",
    re.IGNORECASE,
)

_PIECE_NAME_TO_TYPE: dict[str, chess.PieceType] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

_PIECE_TYPE_TO_NAME: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_ENGINE_ERRORS = (chess.engine.EngineError, OSError)


def _format_grounding_block(
    board: chess.Board,
    context: dict[str, Any],
) -> str:
    """Format authoritative board + Stockfish + Maia grounding text."""
    side = "White" if board.turn == chess.WHITE else "Black"
    status_flags: list[str] = []
    if board.is_check():
        status_flags.append("in check")
    if board.is_checkmate():
        status_flags.append("checkmate")
    elif board.is_stalemate():
        status_flags.append("stalemate")
    status_str = f" ({', '.join(status_flags)})" if status_flags else ""

    lines = [
        _COACH_SYSTEM_PREAMBLE,
        "",
        "### Grounded Board State",
        f"- FEN: {board.fen()}",
        f"- Side to move: {side}{status_str}",
        f"- Legal moves count: {board.legal_moves.count()}",
        "```",
        render_ascii_board(board.fen()),
        "```",
    ]

    if "student_elo" in context:
        lines.append(f"- Student Elo: {context['student_elo']}")
    if context.get("hint_level") is not None:
        lines.append(
            f"- Requested Socratic hint level: {context['hint_level']} "
            "(1=conceptual nudge, 2=piece/zone hint, 3=concrete move & line)"
        )

    sf = context.get("stockfish")
    if isinstance(sf, dict) and sf:
        lines.append("")
        lines.append("### Stockfish Grounding")
        if sf.get("best_move"):
            san_part = f" ({sf['best_san']})" if sf.get("best_san") else ""
            lines.append(f"- Best move: {sf['best_move']}{san_part}")
        if sf.get("eval_cp") is not None:
            lines.append(f"- Evaluation (mover perspective): {sf['eval_cp']:+.1f} cp")
        if sf.get("win_prob") is not None:
            lines.append(f"- Win probability: {sf['win_prob']:.1f}%")
        if sf.get("pv"):
            pv_str = " ".join(sf["pv"][:6])
            lines.append(f"- Principal variation: {pv_str}")
        if sf.get("top_moves"):
            top_fmt = ", ".join(
                f"{m.get('san') or m.get('uci')} ({m.get('cp', 0.0):+.0f} cp)"
                for m in sf["top_moves"][:3]
            )
            lines.append(f"- Top candidates: {top_fmt}")

    maia = context.get("maia")
    if isinstance(maia, dict) and maia:
        lines.append("")
        lines.append("### Maia-2 Human Move & Trap Grounding")
        if maia.get("predicted_human_move"):
            lines.append(f"- Most likely human move: {maia['predicted_human_move']}")
        if maia.get("human_moves"):
            hm_fmt = ", ".join(
                f"{m['move']} ({m['prob'] * 100:.0f}%)" for m in maia["human_moves"][:3]
            )
            lines.append(f"- Human move distribution: {hm_fmt}")
        traps = maia.get("traps") or []
        if traps:
            lines.append("- Detected human traps (tempting moves that lose significant eval):")
            for trap in traps[:3]:
                lines.append(
                    f"  * {trap.get('san') or trap.get('uci')}: "
                    f"-{trap.get('cp_loss', 0):.0f} cp ({trap.get('reason', 'tactical blunder')})"
                )

    return "\n".join(lines)


def build_gemma4_coach_messages(
    fen: str,
    user_question: str,
    context: dict[str, Any] | None = None,
    image_path: str | None = None,
    image_data_url: str | None = None,
    audio_transcript: str | None = None,
) -> list[dict[str, Any]]:
    """Construct Gemma 4 multimodal ``messages`` for chess coaching.

    Uses strict Gemma 4 roles (``"user"`` and ``"model"``), attaches a
    ``{"type": "image", "image": ...}`` part when ``image_data_url`` or
    ``image_path`` is provided, includes optional voice transcript text, and
    embeds Stockfish / Maia grounding so the LLM never guesses board state.

    Args:
        fen: Valid FEN string for the current chess position.
        user_question: Student's question or coaching request.
        context: Optional grounding dictionary (may include ``stockfish``,
            ``maia``, ``student_elo``, ``hint_level``, and prior ``history``).
        image_path: Optional path or URI to a board image for multimodal input.
        image_data_url: Optional base64 ``data:image/...`` URI for inline
            multimodal image input (takes precedence over ``image_path``).
        audio_transcript: Optional ASR transcript of a student's spoken query.

    Returns:
        List of Gemma 4 chat message dicts with ``"role"`` in
        ``{"user", "model"}`` and typed ``"content"`` parts.
    """
    board = chess.Board(fen)
    ctx = dict(context or {})

    messages: list[dict[str, Any]] = []

    # Normalize any prior conversation turns to Gemma 4's "user" / "model" roles.
    for turn in ctx.get("history") or []:
        raw_role = str(turn.get("role", "user")).strip().lower()
        role = "model" if raw_role in ("assistant", "model") else "user"
        raw_content = turn.get("content", "")
        if isinstance(raw_content, list):
            content_parts = list(raw_content)
        else:
            content_parts = [{"type": "text", "text": str(raw_content)}]
        messages.append({"role": role, "content": content_parts})

    user_parts: list[dict[str, Any]] = []
    if image_data_url:
        user_parts.append({"type": "image", "image": str(image_data_url)})
    elif image_path:
        user_parts.append({"type": "image", "image": str(image_path)})

    grounding_text = _format_grounding_block(board, ctx)
    query_sections = [grounding_text, ""]
    if audio_transcript and audio_transcript.strip():
        query_sections.append(f"### Student Voice Transcript\n{audio_transcript.strip()}\n")
    query_sections.append(f"### Student Question\n{user_question.strip()}")

    user_parts.append({"type": "text", "text": "\n".join(query_sections)})
    messages.append({"role": "user", "content": user_parts})
    return messages


def verify_chess_claims(
    fen: str,
    response_text: str,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically verify UCI/SAN moves and piece-on-square claims in prose.

    Every UCI move, SAN move, and explicit ``"<piece> on <square>"`` mention in
    ``response_text`` is checked against ``chess.Board(fen)``. Any illegal move
    or nonexistent piece claim marks ``verified=False``.
    """
    board = chess.Board(fen)
    legal_ucis = {m.uci() for m in board.legal_moves}

    illegal_moves: list[str] = []
    verified_moves: list[str] = []
    hallucinated_pieces: list[str] = []
    issues: list[str] = []

    seen_uci: set[str] = set()
    for match in _UCI_RE.finditer(response_text):
        uci = match.group(1)
        if uci in seen_uci:
            continue
        seen_uci.add(uci)
        if uci in legal_ucis:
            verified_moves.append(uci)
        else:
            illegal_moves.append(uci)
            issues.append(f"Illegal UCI move mentioned: {uci}")

    seen_san: set[str] = set()
    for match in _SAN_RE.finditer(response_text):
        san = match.group(1)
        if san in seen_san:
            continue
        seen_san.add(san)
        try:
            move = board.parse_san(san)
            verified_moves.append(move.uci())
        except ValueError:
            illegal_moves.append(san)
            issues.append(f"Illegal SAN move mentioned: {san}")

    for match in _PIECE_ON_SQ_RE.finditer(response_text):
        color_word, piece_word, sq_name = match.group(1), match.group(2), match.group(3)
        sq = chess.parse_square(sq_name.lower())
        actual = board.piece_at(sq)
        expected_type = _PIECE_NAME_TO_TYPE[piece_word.lower()]
        claim_str = match.group(0)
        if actual is None:
            hallucinated_pieces.append(claim_str)
            issues.append(f"No piece on {sq_name.lower()} (claimed '{claim_str}')")
            continue
        if actual.piece_type != expected_type:
            hallucinated_pieces.append(claim_str)
            actual_name = _PIECE_TYPE_TO_NAME[actual.piece_type]
            issues.append(
                f"Square {sq_name.lower()} has a {actual_name}, not a {piece_word.lower()}"
            )
            continue
        if color_word:
            expected_color = chess.WHITE if color_word.lower() == "white" else chess.BLACK
            if actual.color != expected_color:
                hallucinated_pieces.append(claim_str)
                actual_color = "white" if actual.color == chess.WHITE else "black"
                issues.append(
                    f"Piece on {sq_name.lower()} is {actual_color}, not {color_word.lower()}"
                )

    best_move = (analysis or {}).get("stockfish", {}).get("best_move")
    best_move_mentioned = bool(best_move and best_move in verified_moves)

    return {
        "verified": len(issues) == 0,
        "illegal_moves": illegal_moves,
        "hallucinated_pieces": hallucinated_pieces,
        "verified_moves": list(dict.fromkeys(verified_moves)),
        "best_move_mentioned": best_move_mentioned,
        "issues": issues,
    }


def _detect_human_traps_heuristic(
    board: chess.Board,
    best_uci: str | None,
    best_cp: float,
    engine: Any | None,
    depth: int,
    student_elo: int,
) -> dict[str, Any]:
    """Deterministic Maia-2 fallback: estimate tempting human moves and traps.

    Scores legal moves by human-salient heuristics (captures, checks, center
    pushes, development) and checks whether any tempting candidate drops >= 120
    centipawns compared to ``best_cp``.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {
            "student_elo": student_elo,
            "predicted_human_move": None,
            "human_moves": [],
            "traps": [],
        }

    mover = board.turn
    scored_human: list[tuple[float, chess.Move]] = []
    for move in legal_moves:
        salience = 1.0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            salience += 2.5 + (PIECE_VALUES[captured.piece_type] / 400.0 if captured else 0.5)
        if board.gives_check(move):
            salience += 1.8
        if move.promotion is not None:
            salience += 3.0
        if move.uci() == best_uci:
            # Higher-Elo students are more likely to find the engine best move.
            salience += 1.0 + max(0.0, (student_elo - 800) / 600.0)
        scored_human.append((salience, move))

    scored_human.sort(key=lambda item: item[0], reverse=True)
    top_human = scored_human[:5]
    total_w = sum(w for w, _ in top_human) or 1.0
    human_moves = [
        {"move": m.uci(), "san": board.san(m), "prob": round(w / total_w, 3)} for w, m in top_human
    ]

    traps: list[dict[str, Any]] = []
    for _, move in top_human[:4]:
        if move.uci() == best_uci:
            continue
        # Only flag tempting moves (captures, checks, or top-2 human candidates).
        is_tempting = board.is_capture(move) or board.gives_check(move) or move == top_human[0][1]
        if not is_tempting:
            continue
        san = board.san(move)
        board.push(move)
        if board.is_checkmate():
            cp_after = 10000.0
        elif engine is not None:
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=max(4, depth - 2)))
                cp_after = _cp_from_white(info, mover)
            except _ENGINE_ERRORS:
                cp_after = _static_eval(board, mover)
        else:
            cp_after = _static_eval(board, mover)
        board.pop()

        cp_loss = max(0.0, best_cp - cp_after)
        if cp_loss >= 120.0:
            reason = (
                "loses material to a recapture or counter-tactic"
                if board.is_capture(move)
                else "allows a strong opponent tactical reply"
            )
            traps.append(
                {
                    "uci": move.uci(),
                    "san": san,
                    "cp_loss": round(cp_loss, 1),
                    "reason": reason,
                }
            )

    traps.sort(key=lambda t: t["cp_loss"], reverse=True)
    return {
        "student_elo": student_elo,
        "predicted_human_move": human_moves[0]["move"] if human_moves else None,
        "human_moves": human_moves,
        "traps": traps[:3],
    }


def _describe_zone(square: chess.Square) -> str:
    """Describe a square's board zone without naming the exact square."""
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if file_idx <= 2:
        wing = "queenside"
    elif file_idx >= 5:
        wing = "kingside"
    else:
        wing = "center"
    if rank_idx >= 5:
        depth_band = "upper ranks"
    elif rank_idx <= 2:
        depth_band = "lower ranks"
    else:
        depth_band = "central ranks"
    return f"{wing} ({depth_band})"


def _default_socratic_hint(
    board: chess.Board,
    level: int,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Generate a deterministic 3-level Socratic hint from grounded analysis."""
    if level not in (1, 2, 3):
        raise ValueError(f"hint_level must be 1, 2, or 3; got {level!r}.")

    sf = analysis.get("stockfish") or {}
    best_uci = sf.get("best_move")
    best_san = sf.get("best_san")
    if not best_uci:
        return {
            "level": level,
            "hint": "This position has no legal moves remaining.",
            "reveals_move": False,
        }

    move = chess.Move.from_uci(best_uci)
    piece = board.piece_at(move.from_square)
    piece_name = _PIECE_TYPE_TO_NAME[piece.piece_type] if piece else "piece"
    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)
    is_castle = board.is_castling(move)

    if level == 1:
        if gives_check:
            nudge = "Look for a forcing check that puts immediate pressure on the opponent's king."
        elif is_capture:
            nudge = "Check all captures and undefended targets—there is a concrete material opportunity."
        elif is_castle:
            nudge = "Prioritize king safety and connect your rooks before launching an attack."
        else:
            nudge = "Focus on improving piece activity and controlling key central squares."
        return {"level": 1, "hint": nudge, "reveals_move": False}

    if level == 2:
        zone = _describe_zone(move.to_square)
        action = "capture" if is_capture else ("deliver check" if gives_check else "reposition")
        nudge = (
            f"Consider moving your {piece_name} to {action} toward the {zone}. "
            "Calculate the opponent's most forcing reply before committing."
        )
        return {"level": 2, "hint": nudge, "reveals_move": False}

    # Level 3: concrete move + justification
    pv = sf.get("pv") or [best_uci]
    pv_tail = f" (main line: {' '.join(pv[:4])})" if len(pv) > 1 else ""
    display_move = f"{best_san} ({best_uci})" if best_san else best_uci
    traps = (analysis.get("maia") or {}).get("traps") or []
    trap_note = (
        f" Avoid the tempting trap {traps[0]['san']} ({traps[0]['uci']}), "
        f"which drops ~{traps[0]['cp_loss']:.0f} cp."
        if traps
        else ""
    )
    nudge = f"Play {display_move}{pv_tail}.{trap_note}"
    return {"level": 3, "hint": nudge, "reveals_move": True, "best_move": best_uci}


def _format_verifier_result(
    raw: dict[str, Any],
    *,
    retries_used: int = 0,
) -> dict[str, Any]:
    """Normalize ``verify_claims`` output with backward-compatible keys."""
    violations = list(raw.get("violations") or [])
    passed = bool(raw.get("passed", len(violations) == 0))
    issues = [str(v.get("reason", "")) for v in violations]
    illegal_moves = [
        str(v["claim"].get("move") or v["claim"].get("raw") or "")
        for v in violations
        if v.get("type") == "illegal_move" and isinstance(v.get("claim"), dict)
    ]
    hallucinated_pieces = [
        str(v["claim"].get("raw") or "")
        for v in violations
        if v.get("type") == "piece_on_square" and isinstance(v.get("claim"), dict)
    ]
    return {
        "passed": passed,
        "verified": passed,
        "factuality_score": float(raw.get("factuality_score", 1.0 if passed else 0.0)),
        "total_claims": int(raw.get("total_claims", 0)),
        "verified_claims": int(raw.get("verified_claims", 0)),
        "violations": violations,
        "issues": issues,
        "illegal_moves": illegal_moves,
        "hallucinated_pieces": hallucinated_pieces,
        "retries_used": retries_used,
    }


def _build_repair_messages(
    base_messages: list[dict[str, Any]],
    current_text: str,
    violations: list[dict[str, Any]],
    fen: str,
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construct targeted Gemma 4 repair messages citing exact claim violations."""
    sf = analysis.get("stockfish") or {}
    best_label = (
        f"{sf['best_san']} ({sf['best_move']})"
        if sf.get("best_san") and sf.get("best_move")
        else sf.get("best_move") or "N/A"
    )
    violation_bullets = "\n".join(
        f"- {v.get('reason', 'Unverified chess claim')}" for v in violations
    )
    repair_instruction = (
        "Your previous coaching response contained factual chess claim violations "
        f"for FEN `{fen}`:\n{violation_bullets}\n\n"
        f"Verified engine best move: {best_label}.\n"
        "Please rewrite your coaching explanation so every move, piece-on-square "
        "reference, and check/checkmate claim is strictly legal and accurate for "
        "this exact position."
    )
    return [
        *base_messages,
        {"role": "model", "content": [{"type": "text", "text": current_text}]},
        {"role": "user", "content": [{"type": "text", "text": repair_instruction}]},
    ]


class CoachOrchestrator:
    """High-level facade tying together Stockfish, Maia-2, Socratic hints,
    ACT-Eval Claim Verification + Repair, and optional Gemma 4 multimodal LLM
    generation.

    Works fully end-to-end on CPU when ``llm_fn is None`` (via deterministic
    template-backed coaching prose) and/or when ``engine is None`` (via
    heuristic fallback). External components can be injected via ``engine``,
    ``maia_sidecar``, ``analyzer_fn``, ``maia_fn``, ``hint_fn``,
    ``verifier_fn``, and ``llm_fn``.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        maia_sidecar: MaiaSidecar | None = None,
        llm_fn: Callable[[list[dict[str, Any]]], str] | None = None,
        analyzer_fn: Callable[[str, int], dict[str, Any]] | None = None,
        maia_fn: Callable[[str, int], dict[str, Any]] | None = None,
        hint_fn: Callable[[str, int, dict[str, Any]], dict[str, Any]] | None = None,
        verifier_fn: Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None = None,
        max_repair_retries: int = 2,
        owns_engine: bool = False,
    ) -> None:
        self.engine = engine
        self.maia_sidecar = maia_sidecar
        self.llm_fn = llm_fn
        self.analyzer_fn = analyzer_fn
        self.maia_fn = maia_fn
        self.hint_fn = hint_fn
        self.verifier_fn = verifier_fn
        self.max_repair_retries = max(0, int(max_repair_retries))
        self._owns_engine = owns_engine

    def close(self) -> None:
        """Close the Stockfish engine if owned by this orchestrator."""
        if self._owns_engine and self.engine is not None:
            close_engine(self.engine)
            self.engine = None

    def __enter__(self) -> CoachOrchestrator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _analyze_with_maia_sidecar(
        self,
        board: chess.Board,
        best_uci: str | None,
        best_cp: float,
        student_elo: int,
        depth: int,
    ) -> dict[str, Any]:
        """Use a shared ``MaiaSidecar`` instance for human moves and trap detection."""
        assert self.maia_sidecar is not None
        fen = board.fen()
        pred = self.maia_sidecar.predict_human_moves(fen, student_elo=student_elo, top_k=5)
        top_moves: list[tuple[str, float]] = list(pred.get("top_moves") or [])

        human_moves: list[dict[str, Any]] = []
        for uci, prob in top_moves:
            try:
                mv = chess.Move.from_uci(uci)
                san = board.san(mv) if mv in board.legal_moves else uci
            except ValueError:
                san = uci
            human_moves.append({"move": uci, "san": san, "prob": round(float(prob), 3)})

        mover = board.turn
        traps: list[dict[str, Any]] = []
        for idx, (uci, prob) in enumerate(top_moves[:4]):
            if uci == best_uci:
                continue
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            is_tempting = (
                float(prob) >= 0.15 or board.is_capture(move) or board.gives_check(move) or idx == 0
            )
            if not is_tempting:
                continue

            san = board.san(move)
            board.push(move)
            if board.is_checkmate():
                cp_after = 10000.0
            elif self.engine is not None:
                try:
                    info = self.engine.analyse(board, chess.engine.Limit(depth=max(4, depth - 2)))
                    cp_after = _cp_from_white(info, mover)
                except _ENGINE_ERRORS:
                    cp_after = _static_eval(board, mover)
            else:
                cp_after = _static_eval(board, mover)
            board.pop()

            cp_loss = max(0.0, best_cp - cp_after)
            if cp_loss >= 120.0:
                reason = (
                    "loses material to a recapture or counter-tactic"
                    if board.is_capture(move)
                    else "allows a strong opponent tactical reply"
                )
                traps.append(
                    {
                        "uci": uci,
                        "san": san,
                        "cp_loss": round(cp_loss, 1),
                        "reason": reason,
                    }
                )

        traps.sort(key=lambda t: t["cp_loss"], reverse=True)
        return {
            "student_elo": student_elo,
            "backend": pred.get("backend", self.maia_sidecar.backend),
            "elo_bracket": pred.get("elo_bracket"),
            "predicted_human_move": human_moves[0]["move"] if human_moves else None,
            "human_moves": human_moves,
            "traps": traps[:3],
        }

    def analyze_position(
        self,
        fen: str,
        student_elo: int = 1200,
        depth: int = 8,
    ) -> dict[str, Any]:
        """Run Stockfish + Maia-2 grounding analysis for ``fen``."""
        board = chess.Board(fen)

        if self.analyzer_fn is not None:
            sf_data = dict(self.analyzer_fn(fen, depth))
        else:
            sf_data = self._analyze_stockfish(board, depth)

        if self.maia_fn is not None:
            maia_data = dict(self.maia_fn(fen, student_elo))
        elif self.maia_sidecar is not None:
            maia_data = self._analyze_with_maia_sidecar(
                board=board,
                best_uci=sf_data.get("best_move"),
                best_cp=float(sf_data.get("eval_cp", 0.0)),
                student_elo=student_elo,
                depth=depth,
            )
        else:
            maia_data = _detect_human_traps_heuristic(
                board=board,
                best_uci=sf_data.get("best_move"),
                best_cp=float(sf_data.get("eval_cp", 0.0)),
                engine=self.engine,
                depth=depth,
                student_elo=student_elo,
            )

        return {
            "fen": board.fen(),
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "student_elo": student_elo,
            "stockfish": sf_data,
            "maia": maia_data,
        }

    def _analyze_stockfish(self, board: chess.Board, depth: int) -> dict[str, Any]:
        """Analyze ``board`` with Stockfish MultiPV or heuristic fallback."""
        mover = board.turn
        if self.engine is not None:
            try:
                infos = self.engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=3,
                )
                if isinstance(infos, dict):
                    infos = [infos]
                top_moves: list[dict[str, Any]] = []
                for info in infos:
                    pv = info.get("pv") or []
                    if not pv:
                        continue
                    uci = pv[0].uci()
                    san = board.san(pv[0]) if pv[0] in board.legal_moves else uci
                    cp = _cp_from_white(info, mover)
                    top_moves.append(
                        {
                            "uci": uci,
                            "san": san,
                            "cp": round(cp, 1),
                            "pv": [m.uci() for m in pv[:6]],
                        }
                    )
                if top_moves:
                    best = top_moves[0]
                    return {
                        "source": "stockfish",
                        "best_move": best["uci"],
                        "best_san": best["san"],
                        "eval_cp": best["cp"],
                        "win_prob": round(cp_to_win_prob(best["cp"]), 1),
                        "pv": best["pv"],
                        "top_moves": top_moves,
                    }
            except _ENGINE_ERRORS:
                pass

        best_uci, best_cp, _ = _analyze_position(board, None, depth, mover)
        best_san: str | None = None
        if best_uci is not None:
            move = chess.Move.from_uci(best_uci)
            if move in board.legal_moves:
                best_san = board.san(move)
        return {
            "source": "heuristic",
            "best_move": best_uci,
            "best_san": best_san,
            "eval_cp": round(best_cp, 1),
            "win_prob": round(cp_to_win_prob(best_cp), 1),
            "pv": [best_uci] if best_uci else [],
            "top_moves": (
                [{"uci": best_uci, "san": best_san, "cp": round(best_cp, 1), "pv": [best_uci]}]
                if best_uci
                else []
            ),
        }

    def generate_socratic_hint(
        self,
        fen: str,
        level: int = 1,
        analysis: dict[str, Any] | None = None,
        student_elo: int = 1200,
        depth: int = 8,
    ) -> dict[str, Any]:
        """Generate a 3-level Socratic hint for ``fen``."""
        board = chess.Board(fen)
        grounded = (
            analysis
            if analysis is not None
            else self.analyze_position(fen, student_elo=student_elo, depth=depth)
        )
        if self.hint_fn is not None:
            return dict(self.hint_fn(fen, level, grounded))
        return _default_socratic_hint(board, level, grounded)

    def verify_response(
        self,
        fen: str,
        response_text: str,
        analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify factual chess claims in ``response_text`` against ``fen``."""
        if self.verifier_fn is not None:
            return dict(self.verifier_fn(fen, response_text, analysis))
        raw = verify_claims(fen, response_text, engine=None)
        return _format_verifier_result(raw, retries_used=0)

    def _template_fallback_response(
        self,
        board: chess.Board,
        user_question: str,
        analysis: dict[str, Any],
        hint: dict[str, Any] | None,
        audio_transcript: str | None = None,
    ) -> str:
        """Deterministic CPU-friendly coaching response grounded in analysis."""
        side = "White" if board.turn == chess.WHITE else "Black"
        sf = analysis["stockfish"]
        maia = analysis["maia"]
        eval_cp = sf.get("eval_cp", 0.0)

        if eval_cp >= 150:
            eval_phrase = f"{side} has a clear advantage ({eval_cp:+.0f} cp)"
        elif eval_cp <= -150:
            eval_phrase = f"{side} is under pressure ({eval_cp:+.0f} cp)"
        else:
            eval_phrase = f"The position is roughly balanced ({eval_cp:+.0f} cp)"

        parts = [f"{eval_phrase} with {side} to move."]

        if audio_transcript and audio_transcript.strip():
            parts.append(f'Addressing your spoken question ("{audio_transcript.strip()}"):')
        elif user_question.strip():
            parts.append(f'Regarding "{user_question.strip()}":')

        if hint is not None:
            parts.append(f"Hint (Level {hint['level']}): {hint['hint']}")
        else:
            best_san = sf.get("best_san")
            best_uci = sf.get("best_move")
            if best_san and best_uci:
                parts.append(f"The strongest move is {best_san} ({best_uci}).")
            traps = maia.get("traps") or []
            if traps:
                trap = traps[0]
                parts.append(
                    f"Watch out for the common trap {trap['san']} ({trap['uci']}), "
                    f"which loses about {trap['cp_loss']:.0f} cp ({trap['reason']})."
                )

        return " ".join(parts)

    def coach(
        self,
        fen: str,
        user_question: str,
        *,
        student_elo: int = 1200,
        hint_level: int | None = None,
        image_path: str | None = None,
        image_data_url: str | None = None,
        audio_transcript: str | None = None,
        history: list[dict[str, Any]] | None = None,
        depth: int = 8,
    ) -> dict[str, Any]:
        """Run the full coaching pipeline for a position and student query.

        Steps:
        1. Analyze position with Stockfish (or heuristic fallback) + Maia-2 trap detector.
        2. Generate a Socratic hint if ``hint_level`` is specified (1, 2, or 3).
        3. Construct Gemma 4 multimodal ``messages`` with full grounding context.
        4. Call ``llm_fn(messages)`` if available, otherwise use the deterministic
           CPU template fallback.
        5. Run the ACT-Eval Claim Verifier on the response; when ``llm_fn`` is
           provided and violations are detected, run up to
           ``max_repair_retries`` targeted repair passes via
           :func:`chessfm.coach.verifier.verify_and_repair`. If violations still
           remain after repair, append a verified ground-truth correction note.
        """
        board = chess.Board(fen)
        analysis = self.analyze_position(fen, student_elo=student_elo, depth=depth)

        hint: dict[str, Any] | None = None
        if hint_level is not None:
            hint = self.generate_socratic_hint(
                fen, level=hint_level, analysis=analysis, student_elo=student_elo, depth=depth
            )

        context: dict[str, Any] = {
            "student_elo": student_elo,
            "hint_level": hint_level,
            "stockfish": analysis["stockfish"],
            "maia": analysis["maia"],
            "history": history or [],
        }
        messages = build_gemma4_coach_messages(
            fen=fen,
            user_question=user_question,
            context=context,
            image_path=image_path,
            image_data_url=image_data_url,
            audio_transcript=audio_transcript,
        )

        if self.llm_fn is not None:
            mode = "llm"
            raw_response = str(self.llm_fn(messages))
        else:
            mode = "template_fallback"
            raw_response = self._template_fallback_response(
                board=board,
                user_question=user_question,
                analysis=analysis,
                hint=hint,
                audio_transcript=audio_transcript,
            )

        if mode == "llm" and self.llm_fn is not None and self.verifier_fn is None:
            llm_callable = self.llm_fn

            def _repair_fn(current_text: str, violations: list[dict[str, Any]]) -> str:
                repair_msgs = _build_repair_messages(
                    base_messages=messages,
                    current_text=current_text,
                    violations=violations,
                    fen=board.fen(),
                    analysis=analysis,
                )
                return str(llm_callable(repair_msgs))

            repaired = verify_and_repair(
                fen=board.fen(),
                text=raw_response,
                repair_fn=_repair_fn,
                max_retries=self.max_repair_retries,
                engine=None,
            )
            final_response = repaired["text"]
            verification = _format_verifier_result(
                repaired["verification"],
                retries_used=repaired["retries_used"],
            )
        else:
            final_response = raw_response
            verification = self.verify_response(fen, raw_response, analysis)

        if not verification["verified"] and mode == "llm":
            sf = analysis["stockfish"]
            best_label = (
                f"{sf['best_san']} ({sf['best_move']})"
                if sf.get("best_san") and sf.get("best_move")
                else sf.get("best_move") or "N/A"
            )
            correction = (
                f"\n\n[Coach Verifier Note: Some move or piece claims could not be verified "
                f"({'; '.join(verification['issues'])}). "
                f"Verified engine recommendation: {best_label}.]"
            )
            final_response = final_response + correction

        return {
            "fen": board.fen(),
            "mode": mode,
            "response": final_response,
            "hint": hint,
            "analysis": analysis,
            "verification": verification,
            "messages": messages,
        }

    def review_game(
        self,
        pgn_text: str,
        *,
        student_color: str = "white",
        student_elo: int = 1200,
        depth: int = 8,
        max_critical: int = 5,
    ) -> dict[str, Any]:
        """Review a full PGN game using this orchestrator's configured engine."""
        return review_pgn_game(
            pgn_text=pgn_text,
            engine=self.engine,
            student_color=student_color,
            student_elo=student_elo,
            depth=depth,
            max_critical=max_critical,
        )
