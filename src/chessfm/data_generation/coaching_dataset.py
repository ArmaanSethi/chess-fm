"""Deterministic chess motif detection, best-line SFT construction, coaching dataset builder, and audit.

Generates 100% engine-verified SFT and coaching examples at $0 API cost using
``python-chess`` ground truth, ready for Gemma 4 E4B or Qwen3-4B fine-tuning.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import chess

from chessfm.config import (
    CHATML_ASSISTANT_CLOSE,
    CHATML_ASSISTANT_OPEN,
    CHATML_USER_CLOSE,
    CHATML_USER_OPEN,
    MAX_CENTIPAWN_LOSS,
    build_chat_prompt,
    render_ascii_board,
)
from chessfm.data_generation.phase_sampler import classify_game_phase

CoachingMode = Literal["postmortem", "socratic_l1", "socratic_l2", "socratic_l3", "why_not"]
VALID_COACHING_MODES: frozenset[str] = frozenset(
    {"postmortem", "socratic_l1", "socratic_l2", "socratic_l3", "why_not"}
)

_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

_PIECE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_SLIDER_RAYS: dict[chess.PieceType, tuple[tuple[int, int], ...]] = {
    chess.BISHOP: ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    chess.ROOK: ((1, 0), (-1, 0), (0, 1), (0, -1)),
    chess.QUEEN: ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)),
}

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def _parse_legal_move(board: chess.Board, move_uci: str, label: str = "move") -> chess.Move:
    """Parse ``move_uci`` and verify it is legal on ``board``."""
    try:
        move = chess.Move.from_uci(move_uci.strip().lower())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid UCI {label}: {move_uci!r}") from exc
    if move not in board.legal_moves:
        raise ValueError(f"Illegal {label} {move_uci!r} for FEN {board.fen()!r}")
    return move


def _validate_and_format_pv(board: chess.Board, pv_ucis: list[str]) -> list[str]:
    """Validate every UCI move in ``pv_ucis`` sequentially and return SAN strings."""
    sim = board.copy(stack=False)
    sans: list[str] = []
    for idx, uci in enumerate(pv_ucis):
        mv = _parse_legal_move(sim, uci, label=f"pv[{idx}]")
        sans.append(sim.san(mv))
        sim.push(mv)
    return sans


def _is_fork(board_after: chess.Board, to_square: chess.Square, mover: chess.Color) -> bool:
    """Return True if the piece on ``to_square`` forks at least two valuable enemy targets."""
    moved_piece = board_after.piece_at(to_square)
    if moved_piece is None:
        return False

    opponent = not mover
    mover_val = _PIECE_VALUES[moved_piece.piece_type]
    attacked_targets = 0

    for sq in board_after.attacks(to_square):
        target = board_after.piece_at(sq)
        if target is None or target.color != opponent or target.piece_type == chess.PAWN:
            continue

        if moved_piece.piece_type in (chess.PAWN, chess.KNIGHT):
            attacked_targets += 1
        else:
            target_val = _PIECE_VALUES[target.piece_type]
            is_undefended = not board_after.is_attacked_by(opponent, sq)
            if target.piece_type == chess.KING or target_val > mover_val or is_undefended:
                attacked_targets += 1

    return attacked_targets >= 2


def _creates_pin(board_after: chess.Board, to_square: chess.Square, mover: chess.Color) -> bool:
    """Return True if the moved sliding piece on ``to_square`` creates an absolute or relative pin."""
    moved_piece = board_after.piece_at(to_square)
    if moved_piece is None or moved_piece.piece_type not in _SLIDER_RAYS:
        return False

    opponent = not mover

    # 1. Absolute pin against the enemy king.
    for sq in board_after.attacks(to_square):
        target = board_after.piece_at(sq)
        if target is not None and target.color == opponent and target.piece_type != chess.KING:
            if board_after.is_pinned(opponent, sq) and to_square in board_after.pin(opponent, sq):
                return True

    # 2. Relative pin (lower-value enemy piece shielded in front of a higher-value enemy piece).
    start_file = chess.square_file(to_square)
    start_rank = chess.square_rank(to_square)
    for df, dr in _SLIDER_RAYS[moved_piece.piece_type]:
        f, r = start_file + df, start_rank + dr
        first_enemy: chess.Piece | None = None
        while 0 <= f < 8 and 0 <= r < 8:
            sq = chess.square(f, r)
            piece = board_after.piece_at(sq)
            if piece is not None:
                if piece.color == mover:
                    break
                if first_enemy is None:
                    if piece.piece_type == chess.KING:
                        break
                    first_enemy = piece
                else:
                    if _PIECE_VALUES[piece.piece_type] > _PIECE_VALUES[first_enemy.piece_type]:
                        return True
                    break
            f += df
            r += dr

    return False


def detect_tactical_motifs(fen: str, move_uci: str) -> list[str]:
    """Deterministically detect tactical motifs for ``move_uci`` from ``fen``.

    Supported motifs:
    ``check``, ``checkmate``, ``capture``, ``promotion``, ``castling``,
    ``fork``, ``pin``, ``discovered_check``, ``hanging_piece_capture``.
    """
    board = chess.Board(fen)
    move = _parse_legal_move(board, move_uci, label="move")
    mover = board.turn
    opponent = not mover

    is_capture = board.is_capture(move)
    is_castling = board.is_castling(move)
    is_en_passant = board.is_en_passant(move)

    hanging_capture = False
    if is_capture:
        captured_sq = (
            chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            if is_en_passant
            else move.to_square
        )
        if not board.is_attacked_by(opponent, captured_sq):
            hanging_capture = True

    board_after = board.copy(stack=False)
    board_after.push(move)

    motifs: list[str] = []
    if is_capture:
        motifs.append("capture")
    if hanging_capture:
        motifs.append("hanging_piece_capture")
    if move.promotion is not None:
        motifs.append("promotion")
    if is_castling:
        motifs.append("castling")
    if board_after.is_check():
        motifs.append("check")
        if not is_castling and any(sq != move.to_square for sq in board_after.checkers()):
            motifs.append("discovered_check")
    if board_after.is_checkmate():
        motifs.append("checkmate")
    if _is_fork(board_after, move.to_square, mover):
        motifs.append("fork")
    if _creates_pin(board_after, move.to_square, mover):
        motifs.append("pin")

    return motifs


def _format_eval_pawns(eval_cp: float) -> str:
    """Format centipawn evaluation in signed pawns (e.g. ``+1.25``)."""
    pawns = eval_cp / 100.0
    sign = "+" if pawns >= 0 else ""
    return f"{sign}{pawns:.2f}"


def build_best_line_sft_example(
    fen: str,
    best_move_uci: str,
    pv_ucis: list[str],
    eval_cp: float,
    include_ascii: bool = True,
) -> dict[str, Any]:
    """Build an engine-verified Best-Line SFT row with grounded ``<think>`` reasoning.

    Every move in ``best_move_uci`` and ``pv_ucis`` is strictly validated for
    legality against ``fen``.
    """
    board = chess.Board(fen)
    best_move_uci = best_move_uci.strip().lower()
    best_move = _parse_legal_move(board, best_move_uci, label="best_move")
    best_san = board.san(best_move)

    pv_clean = [m.strip().lower() for m in pv_ucis] if pv_ucis else [best_move_uci]
    if pv_clean[0] != best_move_uci:
        raise ValueError(
            f"pv_ucis must start with best_move_uci ({best_move_uci!r}), got {pv_clean[0]!r}"
        )
    pv_sans = _validate_and_format_pv(board, pv_clean)

    phase = classify_game_phase(fen)
    motifs = detect_tactical_motifs(fen, best_move_uci)
    color = "White" if board.turn == chess.WHITE else "Black"

    think_lines = [
        f"Position assessment: {color} to move in the {phase} (engine eval {_format_eval_pawns(eval_cp)}).",
        f"Best move: {best_san} ({best_move_uci}).",
    ]
    if motifs:
        think_lines.append(f"Tactical motifs: {', '.join(motifs)}.")
    if len(pv_sans) > 1:
        think_lines.append(f"Expected continuation: {' '.join(pv_sans)}.")
    else:
        think_lines.append(f"Key move {best_san} improves {color}'s position.")

    reasoning = "\n".join(think_lines)
    prompt = build_chat_prompt(fen, include_ascii=include_ascii)
    completion = f"<think>\n{reasoning}\n</think>\n{best_move_uci}{CHATML_ASSISTANT_CLOSE}"

    return {
        "prompt": prompt,
        "completion": completion,
        "fen": fen,
        "move": best_move_uci,
        "eval_cp": float(eval_cp),
        "centipawn_loss": 0.0,
        "stockfish_best": best_move_uci,
        "pv": pv_clean,
        "motifs": motifs,
        "phase": phase,
        "is_reasonable": True,
        "source": "engine_best_line",
    }


def _classify_severity(cp_loss: float) -> str:
    """Classify centipawn loss into a standard coaching severity label."""
    if cp_loss < 20:
        return "accurate"
    if cp_loss < 80:
        return "inaccuracy"
    if cp_loss <= MAX_CENTIPAWN_LOSS:
        return "mistake"
    return "blunder"


def _elo_tier(student_elo: int) -> str:
    """Map numeric Elo to a concise pedagogical tier."""
    if student_elo < 1000:
        return "beginner"
    if student_elo < 1600:
        return "club"
    return "advanced"


def _validate_coaching_pv(
    board: chess.Board, student_move: chess.Move, pv_ucis: list[str]
) -> list[str]:
    """Validate ``pv_ucis`` from either ``board`` (best line) or after ``student_move`` (refutation)."""
    if not pv_ucis:
        return []
    try:
        return _validate_and_format_pv(board, pv_ucis)
    except ValueError:
        after_student = board.copy(stack=False)
        after_student.push(student_move)
        return _validate_and_format_pv(after_student, pv_ucis)


def build_coaching_example(
    fen: str,
    student_move_uci: str,
    best_move_uci: str,
    pv_ucis: list[str],
    cp_loss: float,
    student_elo: int = 1200,
    maia_prob: float | None = None,
    mode: CoachingMode = "postmortem",
) -> dict[str, Any]:
    """Build a grounded chess coaching SFT pair for Gemma 4 E4B or Qwen3-4B.

    Supports modes:
    - ``postmortem``: explains why the student's move loses centipawns and why the best move works.
    - ``socratic_l1``: gentle conceptual nudge without revealing the best move or target piece.
    - ``socratic_l2``: narrows attention to the relevant piece type and motif without spoiling the move.
    - ``socratic_l3``: concrete tactical hint pointing to the piece square and follow-up idea.
    - ``why_not``: directly contrasts ``student_move_uci`` against ``best_move_uci``.
    """
    if mode not in VALID_COACHING_MODES:
        raise ValueError(
            f"Unsupported coaching mode {mode!r}; expected one of {sorted(VALID_COACHING_MODES)}"
        )
    if cp_loss < 0:
        raise ValueError(f"cp_loss must be non-negative, got {cp_loss}")
    if maia_prob is not None and not (0.0 <= maia_prob <= 1.0):
        raise ValueError(f"maia_prob must be in [0.0, 1.0], got {maia_prob}")

    board = chess.Board(fen)
    student_move_uci = student_move_uci.strip().lower()
    best_move_uci = best_move_uci.strip().lower()
    student_move = _parse_legal_move(board, student_move_uci, label="student_move")
    best_move = _parse_legal_move(board, best_move_uci, label="best_move")

    student_san = board.san(student_move)
    best_san = board.san(best_move)
    pv_clean = [m.strip().lower() for m in pv_ucis]
    pv_sans = _validate_coaching_pv(board, student_move, pv_clean)

    phase = classify_game_phase(fen)
    student_motifs = detect_tactical_motifs(fen, student_move_uci)
    best_motifs = detect_tactical_motifs(fen, best_move_uci)
    severity = _classify_severity(cp_loss)
    tier = _elo_tier(student_elo)
    color = "White" if board.turn == chess.WHITE else "Black"

    best_piece = board.piece_at(best_move.from_square)
    best_piece_name = _PIECE_NAMES[best_piece.piece_type] if best_piece else "piece"
    from_sq_name = chess.square_name(best_move.from_square)

    maia_note = ""
    if maia_prob is not None:
        maia_note = (
            f" This is a natural human candidate at ~{student_elo} Elo "
            f"(Maia human-move probability: {maia_prob * 100:.0f}%)."
        )

    pv_clause = f" Key continuation: {' '.join(pv_sans)}." if pv_sans else ""
    best_motif_clause = (
        f" It creates tactical themes ({', '.join(best_motifs)})." if best_motifs else ""
    )

    board_ascii = render_ascii_board(fen)
    header = (
        f"Position (FEN): {fen}\n"
        f"Side to move: {color}\n"
        f"Student rating: {student_elo} Elo ({tier})\n"
        f"Board:\n{board_ascii}\n\n"
    )

    if mode == "postmortem":
        user_msg = (
            f"{header}"
            f"The student played {student_san} ({student_move_uci}). "
            f"Give a concise postmortem coaching explanation tailored to a {student_elo}-Elo player."
        )
        assistant_msg = (
            f"Your move {student_san} ({student_move_uci}) is a {severity} in this {phase} "
            f"(~{cp_loss:.0f} cp loss).{maia_note} "
            f"The stronger choice is {best_san} ({best_move_uci}).{best_motif_clause}{pv_clause}"
        )
    elif mode == "socratic_l1":
        user_msg = (
            f"{header}"
            f"The student is considering {student_san} ({student_move_uci}). "
            f"Give a Level 1 Socratic hint without revealing the best move."
        )
        focus = (
            f"forcing ideas like {', '.join(best_motifs)}"
            if best_motifs
            else "piece activity, king safety, and undefended targets"
        )
        assistant_msg = (
            f"Pause before playing {student_san}.{maia_note} "
            f"In this {phase}, check all checks, captures, and threats first—can you spot "
            f"a stronger resource involving {focus}?"
        )
    elif mode == "socratic_l2":
        user_msg = (
            f"{header}"
            f"The student played {student_san} ({student_move_uci}) and asked for a deeper hint. "
            f"Give a Level 2 Socratic hint pointing to the right piece/motif without giving away the exact move."
        )
        motif_hint = (
            f" to create a {'/'.join(best_motifs)}" if best_motifs else " to gain initiative"
        )
        assistant_msg = (
            f"Look closely at your {best_piece_name}. Instead of {student_san}, "
            f"how can that {best_piece_name} enter the action{motif_hint}?"
        )
    elif mode == "socratic_l3":
        user_msg = (
            f"{header}"
            f"The student still needs a concrete hint after {student_san} ({student_move_uci}). "
            f"Give a Level 3 Socratic hint identifying the piece square and tactical idea."
        )
        motif_hint = f" ({', '.join(best_motifs)})" if best_motifs else ""
        assistant_msg = (
            f"Focus specifically on your {best_piece_name} on {from_sq_name}. "
            f"Moving it decisively{motif_hint} avoids the ~{cp_loss:.0f} cp drop of {student_san} "
            f"and leads to the best line ({best_san}).{pv_clause}"
        )
    else:  # why_not
        user_msg = f"{header}Why not {student_san} ({student_move_uci}) here?"
        assistant_msg = (
            f"{student_san} ({student_move_uci}) is a {severity} costing about {cp_loss:.0f} cp.{maia_note} "
            f"Instead, {best_san} ({best_move_uci}) keeps the advantage.{best_motif_clause}{pv_clause}"
        )

    prompt = f"{CHATML_USER_OPEN}{user_msg}{CHATML_USER_CLOSE}{CHATML_ASSISTANT_OPEN}"
    completion = f"{assistant_msg}{CHATML_ASSISTANT_CLOSE}"

    return {
        "prompt": prompt,
        "completion": completion,
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "mode": mode,
        "fen": fen,
        "student_move": student_move_uci,
        "best_move": best_move_uci,
        "pv": pv_clean,
        "cp_loss": float(cp_loss),
        "student_elo": int(student_elo),
        "maia_prob": maia_prob,
        "student_motifs": student_motifs,
        "best_motifs": best_motifs,
        "phase": phase,
        "severity": severity,
    }


def _check_think_tags(completion: str, require_think: bool) -> str | None:
    """Return an error detail string if ``<think>`` tags are malformed, else ``None``."""
    opens = completion.count("<think>")
    closes = completion.count("</think>")
    if opens == 0 and closes == 0:
        return "missing <think>...</think> tags" if require_think else None
    if opens != 1 or closes != 1:
        return f"unbalanced/nested think tags (opens={opens}, closes={closes})"
    open_idx = completion.index("<think>")
    close_idx = completion.index("</think>")
    if close_idx < open_idx:
        return "</think> appears before <think>"
    body = completion[open_idx + len("<think>") : close_idx].strip()
    if not body:
        return "empty <think> block"
    return None


def _has_duplicate_trailing_move(completion: str, move: str | None) -> bool:
    """Return True if ``completion`` ends with a duplicated UCI move token."""
    text = completion.strip()
    if text.endswith(CHATML_ASSISTANT_CLOSE):
        text = text[: -len(CHATML_ASSISTANT_CLOSE)].rstrip()

    tail = text.split("</think>")[-1].strip() if "</think>" in text else text
    tokens = tail.split()

    if len(tokens) >= 2 and tokens[-1].lower() == tokens[-2].lower():
        if _UCI_RE.match(tokens[-1]):
            return True

    if move:
        move_l = move.strip().lower()
        if len(tokens) >= 2 and tokens[-1].lower() == move_l and tokens[-2].lower() == move_l:
            return True
        if "</think>" in text:
            before_close = text.split("</think>")[0].rstrip()
            if before_close.lower().endswith(move_l) and tail.lower() == move_l:
                # Check if the move was pasted verbatim as a standalone trailing line inside </think>
                last_line = before_close.splitlines()[-1].strip().lower()
                if last_line == move_l:
                    return True

    return False


def audit_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit SFT and coaching dataset rows for critical data-quality invariants.

    Verifies:
    1. 0 illegal moves (across ``move``, ``student_move``, ``best_move``, and completion tail).
    2. 0 duplicate trailing moves in ``completion``.
    3. 0 malformed ``<think>`` tags.
    4. 0 blunders (> ``MAX_CENTIPAWN_LOSS`` = 200 cp) labeled ``is_reasonable=True``.
    """
    illegal_moves = 0
    duplicate_trailing_moves = 0
    malformed_think_tags = 0
    blunders_labeled_reasonable = 0
    issues: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        fen = row.get("fen")
        board: chess.Board | None = None
        if not fen or not isinstance(fen, str):
            illegal_moves += 1
            issues.append(
                {"index": idx, "type": "illegal_move", "detail": "missing or invalid fen"}
            )
        else:
            try:
                board = chess.Board(fen)
            except ValueError as exc:
                illegal_moves += 1
                issues.append(
                    {"index": idx, "type": "illegal_move", "detail": f"invalid fen: {exc}"}
                )

        if board is not None:
            for field in ("move", "student_move", "best_move"):
                val = row.get(field)
                if val is not None:
                    try:
                        _parse_legal_move(board, str(val), label=field)
                    except ValueError as exc:
                        illegal_moves += 1
                        issues.append({"index": idx, "type": "illegal_move", "detail": str(exc)})

        completion = row.get("completion", "")
        if isinstance(completion, str):
            require_think = "move" in row and "mode" not in row
            think_err = _check_think_tags(completion, require_think=require_think)
            if think_err is not None:
                malformed_think_tags += 1
                issues.append({"index": idx, "type": "malformed_think_tags", "detail": think_err})

            if _has_duplicate_trailing_move(completion, row.get("move")):
                duplicate_trailing_moves += 1
                issues.append(
                    {
                        "index": idx,
                        "type": "duplicate_trailing_move",
                        "detail": "completion ends with duplicated move",
                    }
                )

            # For move-prediction rows, also verify the trailing completion move is legal.
            if board is not None and "move" in row and "mode" not in row:
                stripped = completion.strip()
                if stripped.endswith(CHATML_ASSISTANT_CLOSE):
                    stripped = stripped[: -len(CHATML_ASSISTANT_CLOSE)].rstrip()
                tail_tokens = (
                    stripped.split("</think>")[-1].strip().split()
                    if "</think>" in stripped
                    else stripped.split()
                )
                if tail_tokens:
                    tail_move = tail_tokens[-1]
                    try:
                        _parse_legal_move(board, tail_move, label="completion_move")
                    except ValueError as exc:
                        illegal_moves += 1
                        issues.append({"index": idx, "type": "illegal_move", "detail": str(exc)})

        cp_val = row.get("centipawn_loss")
        if cp_val is None:
            cp_val = row.get("cp_loss")
        if row.get("is_reasonable") is True and cp_val is not None:
            if float(cp_val) > MAX_CENTIPAWN_LOSS:
                blunders_labeled_reasonable += 1
                issues.append(
                    {
                        "index": idx,
                        "type": "blunder_labeled_reasonable",
                        "detail": f"cp_loss={cp_val} > {MAX_CENTIPAWN_LOSS} with is_reasonable=True",
                    }
                )

    passed = (
        illegal_moves == 0
        and duplicate_trailing_moves == 0
        and malformed_think_tags == 0
        and blunders_labeled_reasonable == 0
    )

    return {
        "total_rows": len(rows),
        "passed": passed,
        "illegal_moves": illegal_moves,
        "duplicate_trailing_moves": duplicate_trailing_moves,
        "malformed_think_tags": malformed_think_tags,
        "blunders_labeled_reasonable": blunders_labeled_reasonable,
        "issues": issues,
    }
