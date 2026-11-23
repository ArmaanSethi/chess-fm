#!/usr/bin/env python3
"""Build frozen Phase II benchmarks, 80K/20K SFT corpora, and 40K RL positions.

Generates all Phase II artifacts under ``data_generation/full/phase2/``
(gitignored) with verified 0.00% normalized 4-field FEN overlap across all
training, RL, and frozen test splits:

Frozen Test Suites (mutually disjoint from each other and from train/RL):
  - ``test_openings_20.txt``: 20 diverse ECO opening FENs spanning A/B/C/D/E.
  - ``test_fens_1000.jsonl``: 1,000 phase-balanced elite game FENs (SF d10, mpv3).
  - ``test_puzzles_900.jsonl``: 900 Lichess puzzles (500 x 20 themes + 400 x 4 bands).
  - ``test_chessqa_1000.jsonl``: 1,000 items across all 5 ChessQA categories (200 each).
  - ``test_coach_300.jsonl``: 300 stratified coaching items (60 x 5 modes, SF d10).
  - ``test_english_100.jsonl``: 100-item general-English forgetting sanity benchmark.

Training & RL Corpora (disjoint from all frozen tests and from each other):
  - ``sft_phase2_80k.jsonl``: 80,000 balanced SFT rows
      (20K games + 40K puzzles + 20K 5-category ChessQA).
  - ``sft_phase2_20k.jsonl``: 20,000 deterministic stratified subset of 80K
      (5K games + 10K puzzles + 5K ChessQA).
  - ``rl_positions_40k.txt``: 40,000 fresh phase-balanced RL positions
      (10K opening / 22K middlegame / 8K endgame).
  - ``leakage_audit.json``: full pairwise normalized-FEN overlap audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessfm.coach.classifier import build_partial_strategy_tree
from chessfm.coach.openings import OPENING_BOOK, identify_opening
from chessfm.config import CHATML_ASSISTANT_CLOSE, build_chat_prompt
from chessfm.data_generation.coaching_dataset import (
    build_best_line_sft_example,
    build_coaching_example,
    detect_tactical_motifs,
)
from chessfm.data_generation.phase_sampler import (
    classify_game_phase,
    normalize_board_fen,
    parse_lichess_puzzle_row,
    phase_distribution,
)
from chessfm.eval.forgetting import export_forgetting_jsonl
from chessfm.rl.engine import close_engine, open_engine

logger = logging.getLogger("chessfm.build_phase2_data")

MAJOR_PUZZLE_THEMES: tuple[str, ...] = (
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "sacrifice",
    "deflection",
    "attraction",
    "backRankMate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "hangingPiece",
    "trappedPiece",
    "xRayAttack",
    "interference",
    "clearance",
    "promotion",
    "zugzwang",
    "doubleCheck",
    "smotheredMate",
)

DIFFICULTY_BANDS: tuple[tuple[str, int, int], ...] = (
    ("<1200", 0, 1199),
    ("1200-1599", 1200, 1599),
    ("1600-1999", 1600, 1999),
    (">=2000", 2000, 10000),
)

CHESSQA_CATEGORIES: tuple[str, ...] = (
    "Structural",
    "Motifs",
    "Short Tactics",
    "Position Judgment",
    "Semantic",
)

COACHING_MODES: tuple[str, ...] = (
    "postmortem",
    "socratic_l1",
    "socratic_l2",
    "socratic_l3",
    "why_not",
)

_EVAL_TAG_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
_COMMENT_RE = re.compile(r"\{([^}]*)\}")
_MOVE_NUM_RE = re.compile(r"^\d+\.+$")
_RESULT_TOKENS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})

_PIECE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

_CHESSQA_SYSTEM_PROMPT: str = (
    "You are a precise chess board-reading and analysis assistant. "
    "Answer the question concisely and accurately."
)


def is_valid_standard_fen(fen: str, *, require_not_game_over: bool = True) -> bool:
    """Return True iff ``fen`` is a valid standard-chess board (rejecting Chess960)."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    if not board.is_valid():
        return False
    if require_not_game_over and board.is_game_over():
        return False
    return True


# ---------------------------------------------------------------------------
# 1. ECO Opening Selection (20 diverse openings spanning A/B/C/D/E)
# ---------------------------------------------------------------------------


def build_diverse_eco_openings(count: int = 20) -> list[dict[str, Any]]:
    """Select ``count`` diverse ECO opening FENs balanced across A/B/C/D/E codes."""
    if count <= 0:
        return []

    by_letter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[str] = set()

    for entry in OPENING_BOOK:
        board = chess.Board()
        for san in entry.moves:
            board.push_san(san)
        fen = board.fen()
        key = normalize_board_fen(fen)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        letter = entry.eco[0].upper()
        by_letter[letter].append(
            {
                "eco": entry.eco,
                "name": entry.name,
                "fen": fen,
                "key": key,
            }
        )

    selected: list[dict[str, Any]] = []
    letters = sorted(by_letter.keys())
    idx = 0
    while len(selected) < count and any(by_letter[lt] for lt in letters):
        lt = letters[idx % len(letters)]
        if by_letter[lt]:
            selected.append(by_letter[lt].pop(0))
        idx += 1

    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique ECO openings available, requested {count}")
    return selected


# ---------------------------------------------------------------------------
# 2. Fast Movetext + [%eval] Extraction from Lichess Tournament Games
# ---------------------------------------------------------------------------


def parse_eval_to_cp(eval_str: str) -> float | None:
    """Convert a Lichess PGN ``[%eval ...]`` token to centipawns (White view)."""
    token = eval_str.strip()
    if not token:
        return None
    if token.startswith("#"):
        mate_str = token[1:]
        try:
            mate_n = int(mate_str)
        except ValueError:
            return None
        return 10000.0 if mate_n > 0 else -10000.0
    try:
        return float(token) * 100.0
    except ValueError:
        return None


def classify_position_judgment(eval_cp_white: float) -> str | None:
    """Map White-perspective centipawn score to a clear 3-way judgment label.

    Returns ``None`` for ambiguous borderline positions (``50 < |cp| < 150``)
    so every judgment item has an unambiguous ground-truth label.
    """
    if eval_cp_white >= 150.0:
        return "White"
    if eval_cp_white <= -150.0:
        return "Black"
    if -50.0 <= eval_cp_white <= 50.0:
        return "Roughly Equal"
    return None


def extract_positions_from_movetext(
    movetext: str,
    *,
    eco: str | None = None,
    opening_name: str | None = None,
    min_fullmove: int = 3,
    max_fullmove: int = 60,
    stride: int = 2,
) -> list[dict[str, Any]]:
    """Extract legal FENs and optional ``[%eval]`` centipawns from PGN movetext."""
    if not movetext or "(" in movetext:
        # Strip variations if any exist to stay strictly on the mainline.
        movetext = re.sub(r"\([^()]*\)", " ", movetext or "")

    # Tokenize into SAN moves and {...} comment blocks.
    tokens = re.findall(r"\{[^}]*\}|\S+", movetext)
    board = chess.Board()
    extracted: list[dict[str, Any]] = []
    ply = 0

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok.startswith("{") or tok.startswith("$") or tok in _RESULT_TOKENS:
            continue
        if _MOVE_NUM_RE.match(tok):
            continue
        # Strip trailing NAG annotations like ! ? !?
        clean_san = tok.rstrip("!?")
        if not clean_san:
            continue
        try:
            move = board.parse_san(clean_san)
        except ValueError:
            break
        board.push(move)
        ply += 1

        # Peek at immediate comment block for [%eval ...].
        eval_cp: float | None = None
        if i < len(tokens) and tokens[i].startswith("{"):
            comment_match = _EVAL_TAG_RE.search(tokens[i])
            if comment_match:
                eval_cp = parse_eval_to_cp(comment_match.group(1))
            i += 1

        fullmove = board.fullmove_number
        if fullmove < min_fullmove or fullmove > max_fullmove:
            continue
        if board.is_game_over() or not board.is_valid():
            continue
        if stride > 1 and (ply % stride != 0):
            continue

        fen = board.fen()
        extracted.append(
            {
                "fen": fen,
                "phase": classify_game_phase(fen),
                "eval_cp_white": eval_cp,
                "eco": eco,
                "opening": opening_name,
            }
        )

    return extracted


# ---------------------------------------------------------------------------
# 3. ChessQA 5-Category Builders (Structural, Motifs, Short Tactics,
#    Position Judgment, Semantic)
# ---------------------------------------------------------------------------


def _sha_bytes(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _build_qa_messages(question: str, answer: str) -> list[dict[str, str]]:
    """Build chat messages list for ChessQA rows so train_sft preserves QA prompts."""
    return [
        {"role": "system", "content": _CHESSQA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _format_qa_chat(question: str, answer: str) -> tuple[str, str]:
    """Build ChatML prompt/completion pair for ChessQA SFT rows."""
    prompt = (
        "<|im_start|>system\n"
        f"{_CHESSQA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    completion = f"{answer}{CHATML_ASSISTANT_CLOSE}"
    return prompt, completion


def _find_deterministic_illegal_move(board: chess.Board, seed_byte: int) -> str:
    """Return a deterministic pseudo-UCI move that is illegal on ``board``."""
    legal = {m.uci() for m in board.legal_moves}
    squares = list(chess.SQUARES)
    start_idx = seed_byte % 64
    for offset in range(64):
        src = squares[(start_idx + offset) % 64]
        for dst_offset in range(1, 64):
            dst = squares[(src + dst_offset * 7) % 64]
            if src == dst:
                continue
            cand = chess.square_name(src) + chess.square_name(dst)
            if cand not in legal:
                return cand
    return "a1a1"


def build_structural_chessqa_item(
    fen: str,
    subtype: str,
    *,
    force_yes: bool | None = None,
) -> dict[str, Any]:
    """Build one Structural ChessQA item (piece_lookup, legal_moves, attacks, check_status)."""
    board = chess.Board(fen)
    digest = _sha_bytes(f"{fen}|{subtype}")

    if subtype == "piece_lookup":
        occupied = sorted(board.piece_map().keys())
        sq = occupied[digest[0] % len(occupied)]
        piece = board.piece_at(sq)
        assert piece is not None
        color_str = "white" if piece.color == chess.WHITE else "black"
        sq_name = chess.square_name(sq)
        question = (
            f"FEN: {fen}\n"
            f"What piece is on square {sq_name}? "
            "Answer as '<color> <piece>' (e.g. 'white knight') or 'empty'."
        )
        answer = f"{color_str} {_PIECE_NAMES[piece.piece_type]}"
    elif subtype == "legal_moves":
        legal_ucis = sorted(m.uci() for m in board.legal_moves)
        want_yes = bool(digest[1] % 2 == 0) if force_yes is None else force_yes
        if want_yes and legal_ucis:
            move_uci = legal_ucis[digest[2] % len(legal_ucis)]
            answer = "yes"
        else:
            move_uci = _find_deterministic_illegal_move(board, digest[3])
            answer = "no"
        question = (
            f"FEN: {fen}\nIs the UCI move {move_uci} legal in this position? Answer 'yes' or 'no'."
        )
    elif subtype == "attacks":
        occupied = sorted(board.piece_map().keys())
        sq = occupied[digest[4] % len(occupied)]
        sq_name = chess.square_name(sq)
        targets = sorted(chess.square_name(t) for t in board.attacks(sq))
        answer = ", ".join(targets) if targets else "none"
        question = (
            f"FEN: {fen}\n"
            f"List all squares attacked by the piece on {sq_name} in alphabetical order "
            "separated by ', ' (or 'none' if it attacks no squares)."
        )
    elif subtype == "check_status":
        answer = "yes" if board.is_check() else "no"
        question = f"FEN: {fen}\nIs the side to move currently in check? Answer 'yes' or 'no'."
    else:
        raise ValueError(f"Unknown Structural subtype: {subtype!r}")

    prompt, completion = _format_qa_chat(question, answer)
    messages = _build_qa_messages(question, answer)
    return {
        "fen": fen,
        "category": "Structural",
        "subcategory": subtype,
        "question": question,
        "answer": answer,
        "phase": classify_game_phase(fen),
        "source": "chessqa",
        "task_type": "chessqa",
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


def build_motifs_chessqa_item(puzzle: dict[str, Any]) -> dict[str, Any]:
    """Build a Motifs ChessQA item verified via ``detect_tactical_motifs`` & Lichess themes."""
    fen = puzzle["fen"]
    best_move = puzzle["best_move"]
    detected = detect_tactical_motifs(fen, best_move)
    theme_set = set(puzzle.get("themes", []))
    matched_major = [t for t in MAJOR_PUZZLE_THEMES if t in theme_set]

    # Prefer deterministic board-verified motifs; supplement with primary Lichess tactical theme.
    labels: list[str] = []
    for m in detected:
        if m not in labels:
            labels.append(m)
    for t in matched_major:
        if t not in labels and len(labels) < 3:
            labels.append(t)
    if not labels:
        labels = ["tactical_gain"]

    answer = ", ".join(labels)
    question = (
        f"FEN: {fen}\n"
        f"What tactical motif(s) characterize the winning move {best_move} in this position? "
        "List the motif names separated by ', '."
    )
    prompt, completion = _format_qa_chat(question, answer)
    messages = _build_qa_messages(question, answer)
    return {
        "fen": fen,
        "category": "Motifs",
        "subcategory": "tactical_motif",
        "best_move": best_move,
        "question": question,
        "answer": answer,
        "motifs": labels,
        "phase": puzzle.get("phase") or classify_game_phase(fen),
        "source": "chessqa",
        "task_type": "chessqa",
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


def build_short_tactics_chessqa_item(puzzle: dict[str, Any]) -> dict[str, Any]:
    """Build a Short Tactics ChessQA item from an unseen Lichess puzzle."""
    fen = puzzle["fen"]
    best_move = puzzle["best_move"]
    line = list(puzzle.get("line", [best_move]))
    question = (
        f"FEN: {fen}\nFind the best tactical move in UCI format (e.g. e2e4) for the side to move."
    )
    prompt, completion = _format_qa_chat(question, best_move)
    messages = _build_qa_messages(question, best_move)
    return {
        "fen": fen,
        "category": "Short Tactics",
        "subcategory": "best_tactical_move",
        "best_move": best_move,
        "solution": line,
        "rating": int(puzzle.get("rating", 0)),
        "question": question,
        "answer": best_move,
        "phase": puzzle.get("phase") or classify_game_phase(fen),
        "source": "chessqa",
        "task_type": "chessqa",
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


def build_position_judgment_chessqa_item(
    fen: str,
    eval_cp_white: float,
) -> dict[str, Any]:
    """Build a Position Judgment ChessQA item (White / Black / Roughly Equal)."""
    label = classify_position_judgment(eval_cp_white)
    if label is None:
        raise ValueError(f"Ambiguous eval_cp_white={eval_cp_white} for Position Judgment")
    question = (
        f"FEN: {fen}\n"
        "Who has the clear advantage in this position? "
        "Answer exactly one of: 'White', 'Black', or 'Roughly Equal'."
    )
    prompt, completion = _format_qa_chat(question, label)
    messages = _build_qa_messages(question, label)
    return {
        "fen": fen,
        "category": "Position Judgment",
        "subcategory": "advantage_3way",
        "eval_cp_white": round(float(eval_cp_white), 1),
        "question": question,
        "answer": label,
        "phase": classify_game_phase(fen),
        "source": "chessqa",
        "task_type": "chessqa",
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


def build_semantic_chessqa_item(
    fen: str,
    subtype: str,
    *,
    opening_name: str | None = None,
) -> dict[str, Any]:
    """Build a Semantic ChessQA item (game_phase, castling_rights, side_to_move, material_balance)."""
    board = chess.Board(fen)

    if subtype == "game_phase":
        answer = classify_game_phase(fen)
        question = (
            f"FEN: {fen}\n"
            "Classify the game phase of this position as 'opening', 'middlegame', or 'endgame'."
        )
    elif subtype == "castling_rights":
        rights = board.castling_xfen()
        answer = "none" if rights == "-" else rights
        question = (
            f"FEN: {fen}\n"
            "What are the remaining castling rights in FEN notation (e.g. 'KQkq', 'Kq', or 'none')?"
        )
    elif subtype == "side_to_move":
        answer = "white" if board.turn == chess.WHITE else "black"
        question = f"FEN: {fen}\nWhich side is to move? Answer 'white' or 'black'."
    elif subtype == "material_balance":
        weights = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        w_mat = sum(len(board.pieces(pt, chess.WHITE)) * w for pt, w in weights.items())
        b_mat = sum(len(board.pieces(pt, chess.BLACK)) * w for pt, w in weights.items())
        diff = w_mat - b_mat
        if diff > 0:
            answer = f"White +{diff}"
        elif diff < 0:
            answer = f"Black +{abs(diff)}"
        else:
            answer = "Equal"
        question = (
            f"FEN: {fen}\n"
            "Using standard piece values (P=1, N=3, B=3, R=5, Q=9), what is the material balance? "
            "Answer 'White +N', 'Black +N', or 'Equal'."
        )
    elif subtype == "opening_family":
        detected = identify_opening(fen=fen)
        name = detected.name if detected is not None else (opening_name or "Unknown")
        answer = name
        question = f"FEN: {fen}\nIdentify the ECO opening name for this position."
    else:
        raise ValueError(f"Unknown Semantic subtype: {subtype!r}")

    prompt, completion = _format_qa_chat(question, answer)
    messages = _build_qa_messages(question, answer)
    return {
        "fen": fen,
        "category": "Semantic",
        "subcategory": subtype,
        "question": question,
        "answer": answer,
        "phase": classify_game_phase(fen),
        "source": "chessqa",
        "task_type": "chessqa",
        "messages": messages,
        "prompt": prompt,
        "completion": completion,
    }


# ---------------------------------------------------------------------------
# 4. Puzzle Best-Line <think> SFT Builder (0 Stockfish calls)
# ---------------------------------------------------------------------------


def build_puzzle_sft_example(
    puzzle: dict[str, Any],
    *,
    include_ascii: bool = True,
) -> dict[str, Any]:
    """Format a validated Lichess puzzle as a grounded best-line ``<think>`` SFT row."""
    fen = puzzle["fen"]
    best_move_uci = puzzle["best_move"].strip().lower()
    line = [m.strip().lower() for m in puzzle["line"]]
    board = chess.Board(fen)

    pv_sans: list[str] = []
    sim = board.copy(stack=False)
    for uci in line[:6]:
        mv = chess.Move.from_uci(uci)
        if mv not in sim.legal_moves:
            break
        pv_sans.append(sim.san(mv))
        sim.push(mv)

    if not pv_sans:
        raise ValueError(f"Illegal puzzle line for FEN {fen!r}: {line!r}")

    phase = puzzle.get("phase") or classify_game_phase(fen)
    motifs = detect_tactical_motifs(fen, best_move_uci)
    themes = list(puzzle.get("themes", []))
    color = "White" if board.turn == chess.WHITE else "Black"
    best_san = pv_sans[0]

    think_lines = [
        f"Position assessment: {color} to move in a tactical {phase} position.",
        f"Best move: {best_san} ({best_move_uci}).",
    ]
    if motifs:
        think_lines.append(f"Tactical motifs: {', '.join(motifs)}.")
    if themes:
        think_lines.append(f"Themes: {', '.join(themes[:4])}.")
    if len(pv_sans) > 1:
        think_lines.append(f"Forced tactical line: {' '.join(pv_sans)}.")
    else:
        think_lines.append(f"Decisive move {best_san} resolves the tactic immediately.")

    reasoning = "\n".join(think_lines)
    prompt = build_chat_prompt(fen, include_ascii=include_ascii)
    completion = f"<think>\n{reasoning}\n</think>\n{best_move_uci}{CHATML_ASSISTANT_CLOSE}"

    return {
        "prompt": prompt,
        "completion": completion,
        "fen": fen,
        "move": best_move_uci,
        "best_move": best_move_uci,
        "pv": line[: len(pv_sans)],
        "motifs": motifs,
        "themes": themes,
        "rating": int(puzzle.get("rating", 0)),
        "phase": phase,
        "is_reasonable": True,
        "source": "lichess_puzzle",
        "task_type": "chess_move",
    }


# ---------------------------------------------------------------------------
# 5. Stockfish SIGFPE-Resilient Parallel Workers
# ---------------------------------------------------------------------------


def _safe_analyse(
    engine_box: list[Any],
    board: chess.Board,
    *,
    depth: int,
    multipv: int = 1,
    max_retries: int = 3,
) -> Any | None:
    """Run Stockfish ``analyse`` with explicit SIGFPE / OSError auto-restart."""
    for attempt in range(max_retries):
        engine = engine_box[0]
        if engine is None:
            engine = open_engine(threads=1, hash_mb=16, max_restarts=3)
            engine_box[0] = engine
            if engine is None:
                return None
        try:
            kwargs: dict[str, Any] = {}
            if multipv > 1:
                kwargs["multipv"] = multipv
            return engine.analyse(board, chess.engine.Limit(depth=depth), **kwargs)
        except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError) as exc:
            logger.warning(
                "Stockfish error (%s) on attempt %d/%d; restarting engine.",
                exc,
                attempt + 1,
                max_retries,
            )
            close_engine(engine_box[0])
            engine_box[0] = open_engine(threads=1, hash_mb=16, max_restarts=3)
    return None


def _score_to_cp_side_to_move(info: dict[str, Any], turn: chess.Color) -> float:
    score = info["score"].white()
    if score.is_mate():
        cp = 10000.0 if (score == chess.engine.MateGiven or (score.mate() or 0) > 0) else -10000.0
    else:
        cp = float(score.score() or 0)
    return cp if turn == chess.WHITE else -cp


def _worker_annotate_sft_games(
    fens: list[str],
    depth: int,
    include_ascii: bool,
) -> list[dict[str, Any]]:
    """Worker: Stockfish depth-``depth`` multipv=1 -> best-line ``<think>`` SFT rows."""
    engine_box: list[Any] = [open_engine(threads=1, hash_mb=16, max_restarts=3)]
    rows: list[dict[str, Any]] = []
    try:
        for fen in fens:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            info = _safe_analyse(engine_box, board, depth=depth, multipv=1)
            if not info or not info.get("pv"):
                continue
            pv_ucis = [m.uci() for m in info["pv"][:4]]
            eval_cp = _score_to_cp_side_to_move(info, board.turn)
            row = build_best_line_sft_example(
                fen=fen,
                best_move_uci=pv_ucis[0],
                pv_ucis=pv_ucis,
                eval_cp=eval_cp,
                include_ascii=include_ascii,
            )
            row["best_move"] = row["move"]
            row["task_type"] = "chess_move"
            rows.append(row)
    finally:
        close_engine(engine_box[0])
    return rows


def _worker_annotate_eval_fens(fens: list[str], depth: int) -> list[dict[str, Any]]:
    """Worker: Stockfish depth-``depth`` multipv=3 -> frozen eval benchmark rows."""
    engine_box: list[Any] = [open_engine(threads=1, hash_mb=16, max_restarts=3)]
    rows: list[dict[str, Any]] = []
    try:
        for fen in fens:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            for attempt in range(3):
                engine = engine_box[0]
                if engine is None:
                    engine = open_engine(threads=1, hash_mb=16, max_restarts=3)
                    engine_box[0] = engine
                if engine is None:
                    break
                try:
                    pst = build_partial_strategy_tree(
                        fen=fen,
                        engine=engine,
                        depth=depth,
                        multipv=3,
                        pv_plies=4,
                    )
                    top_lines = pst.get("top_lines", [])
                    if top_lines and pst.get("best_move"):
                        best_line = top_lines[0]
                        top_moves = [
                            {
                                "move": line["move"],
                                "san": line["san"],
                                "eval_cp": round(float(line["eval_cp"]), 1),
                                "pv_uci": list(line["pv_uci"]),
                            }
                            for line in top_lines
                        ]
                        rows.append(
                            {
                                "fen": fen,
                                "phase": classify_game_phase(fen),
                                "best_move": best_line["move"],
                                "top_moves": top_moves,
                                "eval_cp": round(float(best_line["eval_cp"]), 1),
                            }
                        )
                    break
                except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError):
                    close_engine(engine_box[0])
                    engine_box[0] = open_engine(threads=1, hash_mb=16, max_restarts=3)
                    if attempt == 2:
                        logger.warning("Skipping eval FEN after 3 Stockfish crashes: %s", fen)
    finally:
        close_engine(engine_box[0])
    return rows


def _worker_annotate_coach_items(
    items: list[tuple[str, str, int]],
    depth: int,
) -> list[dict[str, Any]]:
    """Worker: Stockfish depth-``depth`` multipv=3 -> stratified coaching rows."""
    engine_box: list[Any] = [open_engine(threads=1, hash_mb=16, max_restarts=3)]
    rows: list[dict[str, Any]] = []
    try:
        for fen, mode, student_elo in items:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if board.is_game_over() or len(legal_moves) < 2:
                continue
            infos = _safe_analyse(engine_box, board, depth=depth, multipv=min(3, len(legal_moves)))
            if not infos:
                continue
            if isinstance(infos, dict):
                infos = [infos]
            best_info = infos[0]
            if not best_info.get("pv"):
                continue
            best_pv = [m.uci() for m in best_info["pv"][:4]]
            best_uci = best_pv[0]
            best_cp = _score_to_cp_side_to_move(best_info, board.turn)

            # Pick a legal suboptimal student move from lower MultiPV or fallback legal move.
            student_uci = None
            cp_loss = 85.0
            for alt in infos[1:]:
                alt_pv = alt.get("pv")
                if alt_pv and alt_pv[0].uci() != best_uci:
                    student_uci = alt_pv[0].uci()
                    alt_cp = _score_to_cp_side_to_move(alt, board.turn)
                    cp_loss = max(25.0, min(600.0, best_cp - alt_cp))
                    break
            if student_uci is None:
                for mv in legal_moves:
                    if mv.uci() != best_uci:
                        student_uci = mv.uci()
                        break
            if student_uci is None:
                continue

            row = build_coaching_example(
                fen=fen,
                student_move_uci=student_uci,
                best_move_uci=best_uci,
                pv_ucis=best_pv,
                cp_loss=round(cp_loss, 1),
                student_elo=student_elo,
                mode=mode,  # type: ignore[arg-type]
            )
            rows.append(row)
    finally:
        close_engine(engine_box[0])
    return rows


# ---------------------------------------------------------------------------
# 6. Stratification & Leakage Audit Helpers
# ---------------------------------------------------------------------------


def stratify_puzzles_c1(
    puzzles: Iterable[dict[str, Any]],
    *,
    themes: Sequence[str] = MAJOR_PUZZLE_THEMES,
    per_theme: int = 25,
    per_band: int = 100,
    used_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select C1-4B benchmark puzzles: ``per_theme`` x 20 themes + ``per_band`` x 4 bands."""
    claimed = set() if used_keys is None else used_keys
    theme_buckets: dict[str, list[dict[str, Any]]] = {t: [] for t in themes}
    band_buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in DIFFICULTY_BANDS}

    def all_full() -> bool:
        return all(len(v) >= per_theme for v in theme_buckets.values()) and all(
            len(v) >= per_band for v in band_buckets.values()
        )

    for p in puzzles:
        if all_full():
            break
        key = normalize_board_fen(p["fen"])
        if key in claimed:
            continue

        placed = False
        # Fill rarest unfinished matching theme first so rarer themes (e.g. smotheredMate) fill quickly.
        matching_themes = [
            t
            for t in p.get("themes", [])
            if t in theme_buckets and len(theme_buckets[t]) < per_theme
        ]
        if matching_themes:
            matching_themes.sort(key=lambda t: len(theme_buckets[t]))
            chosen = matching_themes[0]
            item = dict(p)
            item["split_group"] = "theme"
            item["primary_theme"] = chosen
            theme_buckets[chosen].append(item)
            claimed.add(key)
            placed = True

        if not placed:
            rating = int(p.get("rating", 0))
            for band_name, lo, hi in DIFFICULTY_BANDS:
                if lo <= rating <= hi and len(band_buckets[band_name]) < per_band:
                    item = dict(p)
                    item["split_group"] = "difficulty"
                    item["difficulty_band"] = band_name
                    band_buckets[band_name].append(item)
                    claimed.add(key)
                    break

    result: list[dict[str, Any]] = []
    for t in themes:
        result.extend(theme_buckets[t])
    for band_name, _, _ in DIFFICULTY_BANDS:
        result.extend(band_buckets[band_name])
    return result


def build_stratified_20k_subset(
    rows_80k: Sequence[dict[str, Any]],
    *,
    game_count: int = 5000,
    puzzle_count: int = 10000,
    chessqa_count: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically select a 20K stratified subset of the 80K SFT corpus."""
    rng = random.Random(seed)
    games_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    puzzles: list[dict[str, Any]] = []
    chessqa_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows_80k:
        src = row.get("source")
        if src == "engine_best_line":
            games_by_phase[row.get("phase", "middlegame")].append(row)
        elif src == "lichess_puzzle":
            puzzles.append(row)
        elif src == "chessqa":
            chessqa_by_cat[row.get("category", "Structural")].append(row)

    selected: list[dict[str, Any]] = []

    # 25% opening / 55% middlegame / 20% endgame for games
    game_quotas = {
        "opening": int(round(game_count * 0.25)),
        "middlegame": int(round(game_count * 0.55)),
        "endgame": game_count - int(round(game_count * 0.25)) - int(round(game_count * 0.55)),
    }
    for phase, q in game_quotas.items():
        pool = list(games_by_phase[phase])
        rng.shuffle(pool)
        selected.extend(pool[:q])

    # Puzzles
    p_pool = list(puzzles)
    rng.shuffle(p_pool)
    selected.extend(p_pool[:puzzle_count])

    # ChessQA evenly across 5 categories
    per_cat = chessqa_count // len(CHESSQA_CATEGORIES)
    for cat in CHESSQA_CATEGORIES:
        c_pool = list(chessqa_by_cat[cat])
        rng.shuffle(c_pool)
        selected.extend(c_pool[:per_cat])

    rng.shuffle(selected)
    return selected


def compute_phase2_leakage_audit(
    splits: dict[str, Sequence[str]],
    *,
    subset_pairs: Sequence[tuple[str, str]] = (("sft_phase2_20k", "sft_phase2_80k"),),
    strict: bool = True,
) -> dict[str, Any]:
    """Audit normalized 4-field FEN overlap across all Phase II splits.

    ``subset_pairs`` lists ``(child, parent)`` pairs where ``child`` is expected
    to be a strict subset of ``parent`` (e.g. ``sft_phase2_20k`` ⊆ ``sft_phase2_80k``);
    all other pairs MUST have 0 overlap when ``strict=True``.
    """
    subset_set = set(subset_pairs) | {(b, a) for a, b in subset_pairs}
    names = sorted(splits.keys())
    norm_sets: dict[str, set[str]] = {
        name: {normalize_board_fen(fen) for fen in fens} for name, fens in splits.items()
    }

    counts = {name: len(splits[name]) for name in names}
    unique_counts = {name: len(norm_sets[name]) for name in names}
    phases = {name: phase_distribution(list(splits[name])) for name in names}

    pairwise_overlaps: dict[str, int] = {}
    disjoint_violations: dict[str, int] = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = len(norm_sets[a] & norm_sets[b])
            pair_key = f"{a}__vs__{b}"
            pairwise_overlaps[pair_key] = overlap
            if (a, b) not in subset_set and overlap > 0:
                disjoint_violations[pair_key] = overlap

    subset_checks: dict[str, dict[str, Any]] = {}
    subset_violations: list[str] = []
    for child, parent in subset_pairs:
        if child in norm_sets and parent in norm_sets:
            extra = len(norm_sets[child] - norm_sets[parent])
            ok = extra == 0
            subset_checks[f"{child}__in__{parent}"] = {
                "child_unique": len(norm_sets[child]),
                "parent_unique": len(norm_sets[parent]),
                "outside_parent": extra,
                "is_strict_subset": ok,
            }
            if not ok:
                subset_violations.append(f"{child} has {extra} FENs not in {parent}")

    invalid_by_split: dict[str, int] = {}
    for name in names:
        bad = sum(
            1 for fen in splits[name] if not is_valid_standard_fen(fen, require_not_game_over=False)
        )
        if bad > 0:
            invalid_by_split[name] = bad
    invalid_standard_boards = sum(invalid_by_split.values())

    zero_leakage = (len(disjoint_violations) == 0) and (len(subset_violations) == 0)
    audit = {
        "zero_leakage": zero_leakage,
        "invalid_standard_boards": invalid_standard_boards,
        "invalid_standard_boards_by_split": invalid_by_split,
        "counts": counts,
        "unique_counts": unique_counts,
        "phase_distributions": phases,
        "pairwise_overlaps": pairwise_overlaps,
        "disjoint_violations": disjoint_violations,
        "subset_checks": subset_checks,
        "subset_violations": subset_violations,
    }

    if strict and (not zero_leakage or invalid_standard_boards > 0):
        raise ValueError(
            f"Phase II leakage audit failed: disjoint_violations={disjoint_violations}, "
            f"subset_violations={subset_violations}, "
            f"invalid_standard_boards={invalid_standard_boards} ({invalid_by_split})"
        )
    return audit


# ---------------------------------------------------------------------------
# 7. I/O & Pool Collection Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.strip() + "\n")


def _load_local_fens(paths: Sequence[Path]) -> list[str]:
    fens: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and is_valid_standard_fen(s):
                    fens.append(s)
    return fens


def _load_existing_sft_games(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if "fen" in row and "move" in row and is_valid_standard_fen(row["fen"]):
                row.setdefault("best_move", row["move"])
                row.setdefault("task_type", "chess_move")
                row.setdefault("source", "engine_best_line")
                rows.append(row)
    return rows


def _stream_hf_puzzles(
    *,
    need_test_900: bool,
    need_test_motifs: int,
    need_test_short_tactics: int,
    need_sft_puzzles: int,
    need_sft_motifs: int,
    need_sft_short_tactics: int,
    used_keys: set[str],
    max_stream_rows: int = 120_000,
) -> dict[str, list[dict[str, Any]]]:
    """Stream ``Lichess/chess-puzzles`` once to fill all frozen & SFT puzzle splits."""
    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    # Targets
    theme_buckets: dict[str, list[dict[str, Any]]] = {t: [] for t in MAJOR_PUZZLE_THEMES}
    band_buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in DIFFICULTY_BANDS}
    test_motifs: list[dict[str, Any]] = []
    test_short_tactics: list[dict[str, Any]] = []

    sft_theme_cap = max(1, need_sft_puzzles // len(MAJOR_PUZZLE_THEMES))
    sft_theme_counts: Counter[str] = Counter()
    sft_puzzles: list[dict[str, Any]] = []
    sft_overflow_puzzles: list[dict[str, Any]] = []
    sft_motifs: list[dict[str, Any]] = []
    sft_short_tactics: list[dict[str, Any]] = []

    def test_900_done() -> bool:
        if not need_test_900:
            return True
        return all(len(v) >= 25 for v in theme_buckets.values()) and all(
            len(v) >= 100 for v in band_buckets.values()
        )

    def all_done() -> bool:
        return (
            test_900_done()
            and len(test_motifs) >= need_test_motifs
            and len(test_short_tactics) >= need_test_short_tactics
            and len(sft_puzzles) >= need_sft_puzzles
            and len(sft_motifs) >= need_sft_motifs
            and len(sft_short_tactics) >= need_sft_short_tactics
        )

    streamed = 0
    t0 = time.time()
    for raw in ds:
        streamed += 1
        if streamed > max_stream_rows:
            break
        if streamed % 25000 == 0:
            logger.info(
                "Streamed %d puzzles (%.1fs): test_t=%d/500 test_d=%d/400 sft_p=%d/%d",
                streamed,
                time.time() - t0,
                sum(len(v) for v in theme_buckets.values()),
                sum(len(v) for v in band_buckets.values()),
                len(sft_puzzles),
                need_sft_puzzles,
            )

        parsed = parse_lichess_puzzle_row(raw)
        if parsed is None or not is_valid_standard_fen(parsed["fen"]):
            continue
        key = normalize_board_fen(parsed["fen"])
        if key in used_keys:
            continue

        # 1) Frozen test_puzzles_900 (theme-balanced 500 + difficulty-balanced 400)
        if not test_900_done():
            matching = [
                t for t in parsed["themes"] if t in theme_buckets and len(theme_buckets[t]) < 25
            ]
            if matching:
                matching.sort(key=lambda t: len(theme_buckets[t]))
                chosen = matching[0]
                parsed["split_group"] = "theme"
                parsed["primary_theme"] = chosen
                theme_buckets[chosen].append(parsed)
                used_keys.add(key)
                continue

            rating = parsed["rating"]
            placed_band = False
            for band_name, lo, hi in DIFFICULTY_BANDS:
                if lo <= rating <= hi and len(band_buckets[band_name]) < 100:
                    parsed["split_group"] = "difficulty"
                    parsed["difficulty_band"] = band_name
                    band_buckets[band_name].append(parsed)
                    used_keys.add(key)
                    placed_band = True
                    break
            if placed_band:
                continue

        # 2) Frozen ChessQA puzzle categories
        if len(test_motifs) < need_test_motifs:
            test_motifs.append(parsed)
            used_keys.add(key)
            continue
        if len(test_short_tactics) < need_test_short_tactics:
            test_short_tactics.append(parsed)
            used_keys.add(key)
            continue

        # 3) SFT ChessQA puzzle categories
        if len(sft_motifs) < need_sft_motifs:
            sft_motifs.append(parsed)
            used_keys.add(key)
            continue
        if len(sft_short_tactics) < need_sft_short_tactics:
            sft_short_tactics.append(parsed)
            used_keys.add(key)
            continue

        # 4) SFT theme-balanced puzzles
        if len(sft_puzzles) < need_sft_puzzles:
            matching_sft = [
                t
                for t in parsed["themes"]
                if t in theme_buckets and sft_theme_counts[t] < sft_theme_cap
            ]
            if matching_sft:
                matching_sft.sort(key=lambda t: sft_theme_counts[t])
                sft_theme_counts[matching_sft[0]] += 1
                sft_puzzles.append(parsed)
                used_keys.add(key)
            elif len(sft_overflow_puzzles) < need_sft_puzzles:
                sft_overflow_puzzles.append(parsed)
                used_keys.add(key)

        if all_done():
            break

    # Top up sft_puzzles from overflow if rare themes didn't hit exact cap.
    if len(sft_puzzles) < need_sft_puzzles and sft_overflow_puzzles:
        need = need_sft_puzzles - len(sft_puzzles)
        sft_puzzles.extend(sft_overflow_puzzles[:need])
        # Release any unused overflow keys back from used_keys.
        for unused in sft_overflow_puzzles[need:]:
            used_keys.discard(normalize_board_fen(unused["fen"]))
    else:
        for unused in sft_overflow_puzzles:
            used_keys.discard(normalize_board_fen(unused["fen"]))

    test_puzzles_900: list[dict[str, Any]] = []
    for t in MAJOR_PUZZLE_THEMES:
        test_puzzles_900.extend(theme_buckets[t])
    for band_name, _, _ in DIFFICULTY_BANDS:
        test_puzzles_900.extend(band_buckets[band_name])

    return {
        "test_puzzles_900": test_puzzles_900,
        "test_motifs": test_motifs,
        "test_short_tactics": test_short_tactics,
        "sft_puzzles": sft_puzzles,
        "sft_motifs": sft_motifs,
        "sft_short_tactics": sft_short_tactics,
    }


def _collect_game_pools(
    *,
    local_paths: Sequence[Path],
    used_keys: set[str],
    need_opening: int,
    need_middlegame: int,
    need_endgame: int,
    need_judgment_per_label: int,
    max_hf_games: int = 15_000,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, float]]]]:
    """Collect unique game FENs by phase and ``[%eval]`` positions by 3-way judgment."""
    by_phase: dict[str, list[str]] = {"opening": [], "middlegame": [], "endgame": []}
    by_judgment: dict[str, list[tuple[str, float]]] = {
        "White": [],
        "Black": [],
        "Roughly Equal": [],
    }
    pool_keys: set[str] = set()

    # 1) Load local FEN pools first (fast, already on disk, standard chess only).
    for fen in _load_local_fens(local_paths):
        try:
            key = normalize_board_fen(fen)
        except ValueError:
            continue
        if key in used_keys or key in pool_keys:
            continue
        phase = classify_game_phase(fen)
        pool_keys.add(key)
        by_phase[phase].append(fen)

    logger.info(
        "Local game pool loaded: opening=%d, middlegame=%d, endgame=%d",
        len(by_phase["opening"]),
        len(by_phase["middlegame"]),
        len(by_phase["endgame"]),
    )

    def pools_satisfied() -> bool:
        return (
            len(by_phase["opening"]) >= need_opening
            and len(by_phase["middlegame"]) >= need_middlegame
            and len(by_phase["endgame"]) >= need_endgame
            and all(len(v) >= need_judgment_per_label for v in by_judgment.values())
        )

    if pools_satisfied():
        return by_phase, by_judgment

    # 2) Stream unseen elite GM/master games from HF Lichess/tournament-chess-games.
    from datasets import load_dataset

    ds = load_dataset("Lichess/tournament-chess-games", split="train", streaming=True)
    games_seen = 0
    t0 = time.time()

    for row in ds:
        if row.get("Variant") not in (None, "", "Standard"):
            continue
        if row.get("FEN") not in (None, "", chess.STARTING_FEN):
            continue
        movetext = row.get("movetext")
        if not movetext:
            continue
        games_seen += 1
        if games_seen > max_hf_games:
            break

        positions = extract_positions_from_movetext(
            movetext,
            eco=row.get("ECO"),
            opening_name=row.get("Opening"),
            min_fullmove=3,
            max_fullmove=60,
            stride=2,
        )
        for pos in positions:
            fen = pos["fen"]
            key = normalize_board_fen(fen)
            if key in used_keys or key in pool_keys:
                continue

            # Prioritize filling Position Judgment buckets when [%eval] is present.
            ev = pos["eval_cp_white"]
            if ev is not None:
                label = classify_position_judgment(ev)
                if label is not None and len(by_judgment[label]) < need_judgment_per_label:
                    by_judgment[label].append((fen, ev))
                    pool_keys.add(key)
                    continue

            phase = pos["phase"]
            phase_target = {
                "opening": need_opening,
                "middlegame": need_middlegame,
                "endgame": need_endgame,
            }[phase]
            if len(by_phase[phase]) < phase_target:
                by_phase[phase].append(fen)
                pool_keys.add(key)

        if games_seen % 500 == 0:
            logger.info(
                "Streamed %d HF tournament games (%.1fs): open=%d/%d mid=%d/%d end=%d/%d judg=%s",
                games_seen,
                time.time() - t0,
                len(by_phase["opening"]),
                need_opening,
                len(by_phase["middlegame"]),
                need_middlegame,
                len(by_phase["endgame"]),
                need_endgame,
                {k: len(v) for k, v in by_judgment.items()},
            )

        if pools_satisfied():
            break

    return by_phase, by_judgment


def _pop_phase_balanced(
    by_phase: dict[str, list[str]],
    used_keys: set[str],
    *,
    opening: int,
    middlegame: int,
    endgame: int,
    rng: random.Random,
    min_legal_moves: int = 1,
) -> list[str]:
    """Pop exact phase counts from ``by_phase``, marking each key in ``used_keys``."""
    selected: list[str] = []
    quotas = {"opening": opening, "middlegame": middlegame, "endgame": endgame}
    for phase, target in quotas.items():
        taken = 0
        bucket = by_phase[phase]
        deferred: list[str] = []
        while bucket and taken < target:
            fen = bucket.pop()
            key = normalize_board_fen(fen)
            if key in used_keys:
                continue
            if not is_valid_standard_fen(fen):
                continue
            if min_legal_moves > 1 and len(list(chess.Board(fen).legal_moves)) < min_legal_moves:
                deferred.append(fen)
                continue
            used_keys.add(key)
            selected.append(fen)
            taken += 1
        if deferred:
            bucket.extend(deferred)
        if taken < target:
            raise RuntimeError(f"Insufficient {phase} FENs: needed {target}, only found {taken}")
    rng.shuffle(selected)
    return selected


def _pop_any_fens(
    by_phase: dict[str, list[str]],
    used_keys: set[str],
    count: int,
    *,
    require_check: bool | None = None,
) -> list[str]:
    """Pop ``count`` unused FENs across buckets, optionally filtering by ``is_check()``."""
    selected: list[str] = []
    # Scan middlegame -> endgame -> opening to preserve scarce opening FENs.
    for phase in ("middlegame", "endgame", "opening"):
        if len(selected) >= count:
            break
        kept: list[str] = []
        for fen in by_phase[phase]:
            if len(selected) >= count:
                kept.append(fen)
                continue
            key = normalize_board_fen(fen)
            if key in used_keys:
                continue
            if require_check is not None:
                if chess.Board(fen).is_check() != require_check:
                    kept.append(fen)
                    continue
            used_keys.add(key)
            selected.append(fen)
        by_phase[phase] = kept
    if len(selected) < count:
        raise RuntimeError(
            f"Insufficient FENs for require_check={require_check}: needed {count}, found {len(selected)}"
        )
    return selected


def _build_structural_batch(
    by_phase: dict[str, list[str]],
    used_keys: set[str],
    total: int,
) -> list[dict[str, Any]]:
    """Build ``total`` balanced Structural ChessQA items (4 subtypes, yes/no balanced)."""
    per_sub = total // 4
    items: list[dict[str, Any]] = []

    # piece_lookup
    for fen in _pop_any_fens(by_phase, used_keys, per_sub):
        items.append(build_structural_chessqa_item(fen, "piece_lookup"))

    # legal_moves (50% yes, 50% no)
    half_lm = per_sub // 2
    for idx, fen in enumerate(_pop_any_fens(by_phase, used_keys, per_sub)):
        items.append(build_structural_chessqa_item(fen, "legal_moves", force_yes=(idx < half_lm)))

    # attacks
    for fen in _pop_any_fens(by_phase, used_keys, per_sub):
        items.append(build_structural_chessqa_item(fen, "attacks"))

    # check_status (try 50% in-check / 50% not-in-check; gracefully backfill if fewer checks exist)
    half_chk = per_sub // 2
    in_check: list[str] = []
    try:
        in_check = _pop_any_fens(by_phase, used_keys, half_chk, require_check=True)
    except RuntimeError:
        # Take whatever in-check positions exist and fill remainder with non-check.
        for phase in ("middlegame", "endgame", "opening"):
            kept = []
            for fen in by_phase[phase]:
                key = normalize_board_fen(fen)
                if (
                    len(in_check) < half_chk
                    and key not in used_keys
                    and chess.Board(fen).is_check()
                ):
                    used_keys.add(key)
                    in_check.append(fen)
                else:
                    kept.append(fen)
            by_phase[phase] = kept
    not_check = _pop_any_fens(by_phase, used_keys, per_sub - len(in_check), require_check=False)
    for fen in in_check + not_check:
        items.append(build_structural_chessqa_item(fen, "check_status"))

    return items


def _build_semantic_batch(
    by_phase: dict[str, list[str]],
    used_keys: set[str],
    total: int,
) -> list[dict[str, Any]]:
    """Build ``total`` balanced Semantic ChessQA items across 4 subtypes."""
    subtypes = ("game_phase", "castling_rights", "side_to_move", "material_balance")
    per_sub = total // len(subtypes)
    items: list[dict[str, Any]] = []
    for sub in subtypes:
        for fen in _pop_any_fens(by_phase, used_keys, per_sub):
            items.append(build_semantic_chessqa_item(fen, sub))
    return items


def _build_judgment_batch(
    by_judgment: dict[str, list[tuple[str, float]]],
    used_keys: set[str],
    total: int,
) -> list[dict[str, Any]]:
    """Build ``total`` balanced Position Judgment ChessQA items across 3 labels."""
    labels = ("White", "Black", "Roughly Equal")
    base = total // 3
    rem = total - base * 3
    items: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        quota = base + (1 if idx < rem else 0)
        taken = 0
        bucket = by_judgment[label]
        while bucket and taken < quota:
            fen, ev = bucket.pop()
            key = normalize_board_fen(fen)
            if key in used_keys:
                continue
            used_keys.add(key)
            items.append(build_position_judgment_chessqa_item(fen, ev))
            taken += 1
        if taken < quota:
            raise RuntimeError(
                f"Insufficient Position Judgment items for {label!r}: needed {quota}, found {taken}"
            )
    return items


def _run_parallel_sft_games(
    fens: list[str],
    *,
    depth: int,
    include_ascii: bool,
    workers: int,
    checkpoint_path: Path,
    chunk_size: int = 250,
) -> list[dict[str, Any]]:
    """Annotate SFT game FENs in parallel with incremental JSONL checkpointing."""
    done_rows: list[dict[str, Any]] = []
    done_keys: set[str] = set()
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                row = json.loads(s)
                key = normalize_board_fen(row["fen"])
                if key not in done_keys:
                    done_keys.add(key)
                    done_rows.append(row)
        logger.info("Resumed %d SFT game rows from %s", len(done_rows), checkpoint_path)

    remaining = [f for f in fens if normalize_board_fen(f) not in done_keys]
    if not remaining:
        return done_rows

    chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]
    t0 = time.time()
    completed_fens = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        checkpoint_path.open("a", encoding="utf-8") as ckpt_fh,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [
            pool.submit(_worker_annotate_sft_games, ch, depth, include_ascii) for ch in chunks
        ]
        for fut in as_completed(futures):
            batch = fut.result()
            for row in batch:
                ckpt_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done_rows.append(row)
            ckpt_fh.flush()
            completed_fens += len(batch)
            elapsed = time.time() - t0
            rate = completed_fens / elapsed if elapsed > 0 else 0.0
            eta = (len(remaining) - completed_fens) / rate if rate > 0 else 0.0
            logger.info(
                "SFT games Stockfish progress: %d/%d new (total %d) | %.1f FEN/s | ETA %.1f min",
                completed_fens,
                len(remaining),
                len(done_rows),
                rate,
                eta / 60.0,
            )

    return done_rows


def _run_parallel_chunks(
    worker_fn,
    items: list[Any],
    *,
    workers: int,
    chunk_size: int,
    extra_args: tuple[Any, ...],
) -> list[dict[str, Any]]:
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_fn, ch, *extra_args) for ch in chunks]
        for fut in as_completed(futures):
            results.extend(fut.result())
    return results


# ---------------------------------------------------------------------------
# 8. Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


def build_phase2_artifacts(
    output_dir: Path,
    *,
    seed: int = 42,
    workers: int = 2,
    sft_depth: int = 8,
    eval_depth: int = 10,
    include_ascii: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the full Phase II benchmark + 80K/20K SFT + 40K RL generation pipeline."""
    root = repo_root or Path(__file__).resolve().parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    used_keys: set[str] = set()

    # 1. Frozen test_openings_20.txt
    logger.info("[1/9] Building test_openings_20.txt ...")
    openings = build_diverse_eco_openings(count=20)
    opening_fens = [o["fen"] for o in openings]
    for fen in opening_fens:
        used_keys.add(normalize_board_fen(fen))
    _write_lines(output_dir / "test_openings_20.txt", opening_fens)

    # 2. Frozen test_english_100.jsonl
    logger.info("[2/9] Building test_english_100.jsonl ...")
    n_english = export_forgetting_jsonl(output_dir / "test_english_100.jsonl")
    logger.info("Exported %d English forgetting items.", n_english)

    # 3. Stream Lichess puzzles for frozen test_puzzles_900 + ChessQA + 40K SFT puzzles
    logger.info("[3/9] Streaming Lichess/chess-puzzles from HuggingFace ...")
    puzzle_splits = _stream_hf_puzzles(
        need_test_900=True,
        need_test_motifs=200,
        need_test_short_tactics=200,
        need_sft_puzzles=40_000,
        need_sft_motifs=4_000,
        need_sft_short_tactics=4_000,
        used_keys=used_keys,
    )
    test_puzzles_900 = puzzle_splits["test_puzzles_900"]
    assert len(test_puzzles_900) == 900, f"Expected 900 test puzzles, got {len(test_puzzles_900)}"
    _write_jsonl(output_dir / "test_puzzles_900.jsonl", test_puzzles_900)
    logger.info("Wrote test_puzzles_900.jsonl (%d rows).", len(test_puzzles_900))

    # 4. Collect elite game FENs (local + HF Lichess/tournament-chess-games)
    logger.info("[4/9] Collecting elite game FEN pools & [%eval] positions ...")
    # Reserve reusable rows from sft_bestline_2500.jsonl first if not colliding with puzzles/openings.
    existing_sft_path = root / "data_generation" / "full" / "sft_bestline_2500.jsonl"
    reused_sft_games: list[dict[str, Any]] = []
    reused_phase_counts: Counter[str] = Counter()
    sft_game_targets = {"opening": 5000, "middlegame": 11000, "endgame": 4000}
    for row in _load_existing_sft_games(existing_sft_path):
        key = normalize_board_fen(row["fen"])
        phase = row.get("phase") or classify_game_phase(row["fen"])
        if key not in used_keys and reused_phase_counts[phase] < sft_game_targets[phase]:
            row["phase"] = phase
            reused_sft_games.append(row)
            reused_phase_counts[phase] += 1
            used_keys.add(key)
    logger.info(
        "Reused %d non-overlapping SFT best-line rows (%s).",
        len(reused_sft_games),
        dict(reused_phase_counts),
    )

    rem_sft_open = sft_game_targets["opening"] - reused_phase_counts["opening"]
    rem_sft_mid = sft_game_targets["middlegame"] - reused_phase_counts["middlegame"]
    rem_sft_end = sft_game_targets["endgame"] - reused_phase_counts["endgame"]

    # Total needed from game pools (with buffer for Structural/Semantic + safety margin):
    need_open = 250 + 75 + rem_sft_open + 10_000 + 600
    need_mid = 550 + 165 + rem_sft_mid + 22_000 + 9_500
    need_end = 200 + 60 + rem_sft_end + 8_000 + 1_200
    # Need 200 test + 4000 SFT Position Judgment = 4,200 total -> 1,400 per label (+ buffer)
    need_judg_per_label = 1_450

    local_pools = [
        root / "data_generation" / "full" / "positions_inclusive.txt",
        root / "data_generation" / "positions.txt",
    ]
    by_phase, by_judgment = _collect_game_pools(
        local_paths=local_pools,
        used_keys=used_keys,
        need_opening=need_open,
        need_middlegame=need_mid,
        need_endgame=need_end,
        need_judgment_per_label=need_judg_per_label,
    )
    for bucket in by_phase.values():
        rng.shuffle(bucket)
    for bucket in by_judgment.values():
        rng.shuffle(bucket)

    # 5. Frozen test_fens_1000.jsonl (SF depth=10, multipv=3)
    logger.info("[5/9] Annotating test_fens_1000.jsonl (Stockfish d=%d, mpv=3) ...", eval_depth)
    test_1000_fens = _pop_phase_balanced(
        by_phase, used_keys, opening=250, middlegame=550, endgame=200, rng=rng
    )
    test_fens_1000 = _run_parallel_chunks(
        _worker_annotate_eval_fens,
        test_1000_fens,
        workers=workers,
        chunk_size=50,
        extra_args=(eval_depth,),
    )
    _write_jsonl(output_dir / "test_fens_1000.jsonl", test_fens_1000)
    logger.info("Wrote test_fens_1000.jsonl (%d rows).", len(test_fens_1000))

    # 6. Frozen test_coach_300.jsonl (60 x 5 modes, SF depth=10)
    logger.info(
        "[6/9] Annotating test_coach_300.jsonl (Stockfish d=%d, 60 x 5 modes) ...", eval_depth
    )
    coach_fens = _pop_phase_balanced(
        by_phase,
        used_keys,
        opening=75,
        middlegame=165,
        endgame=60,
        rng=rng,
        min_legal_moves=2,
    )
    coach_specs: list[tuple[str, str, int]] = []
    elo_cycle = (900, 1200, 1500, 1800)
    for idx, fen in enumerate(coach_fens):
        mode = COACHING_MODES[idx % len(COACHING_MODES)]
        coach_specs.append((fen, mode, elo_cycle[idx % len(elo_cycle)]))
    test_coach_300 = _run_parallel_chunks(
        _worker_annotate_coach_items,
        coach_specs,
        workers=workers,
        chunk_size=30,
        extra_args=(eval_depth,),
    )
    # Backfill if Stockfish skipped any position so every mode has exact 60 rows.
    mode_counts = Counter(r["mode"] for r in test_coach_300)
    missing_specs: list[tuple[str, str, int]] = []
    for m_idx, mode in enumerate(COACHING_MODES):
        deficit = 60 - mode_counts[mode]
        if deficit > 0:
            repl_fens = _pop_phase_balanced(
                by_phase,
                used_keys,
                opening=0,
                middlegame=deficit,
                endgame=0,
                rng=rng,
                min_legal_moves=2,
            )
            for j, fen in enumerate(repl_fens):
                missing_specs.append((fen, mode, elo_cycle[(m_idx + j) % len(elo_cycle)]))
    if missing_specs:
        test_coach_300.extend(_worker_annotate_coach_items(missing_specs, eval_depth))
    assert len(test_coach_300) == 300, f"Expected exact 300 coach rows, got {len(test_coach_300)}"
    _write_jsonl(output_dir / "test_coach_300.jsonl", test_coach_300)
    logger.info("Wrote test_coach_300.jsonl (%d rows).", len(test_coach_300))

    # 7. Frozen test_chessqa_1000.jsonl (200 x 5 categories)
    logger.info("[7/9] Building test_chessqa_1000.jsonl (200 x 5 categories) ...")
    test_chessqa_1000: list[dict[str, Any]] = []
    test_chessqa_1000.extend(_build_structural_batch(by_phase, used_keys, 200))
    test_chessqa_1000.extend(build_motifs_chessqa_item(p) for p in puzzle_splits["test_motifs"])
    test_chessqa_1000.extend(
        build_short_tactics_chessqa_item(p) for p in puzzle_splits["test_short_tactics"]
    )
    test_chessqa_1000.extend(_build_judgment_batch(by_judgment, used_keys, 200))
    test_chessqa_1000.extend(_build_semantic_batch(by_phase, used_keys, 200))
    rng.shuffle(test_chessqa_1000)
    _write_jsonl(output_dir / "test_chessqa_1000.jsonl", test_chessqa_1000)
    logger.info("Wrote test_chessqa_1000.jsonl (%d rows).", len(test_chessqa_1000))

    # 8. RL Positions (40,000 fresh phase-balanced FENs)
    logger.info("[8/9] Building rl_positions_40k.txt (10K open / 22K mid / 8K end) ...")
    rl_positions_40k = _pop_phase_balanced(
        by_phase, used_keys, opening=10_000, middlegame=22_000, endgame=8_000, rng=rng
    )
    _write_lines(output_dir / "rl_positions_40k.txt", rl_positions_40k)
    logger.info("Wrote rl_positions_40k.txt (%d lines).", len(rl_positions_40k))

    # 9. SFT 80K (20K games + 40K puzzles + 20K ChessQA) + 20K subset
    logger.info(
        "[9/9] Building sft_phase2_80k.jsonl (annotating %d new game FENs at SF d=%d) ...",
        rem_sft_open + rem_sft_mid + rem_sft_end,
        sft_depth,
    )
    new_sft_game_fens = _pop_phase_balanced(
        by_phase,
        used_keys,
        opening=rem_sft_open,
        middlegame=rem_sft_mid,
        endgame=rem_sft_end,
        rng=rng,
    )
    ckpt_path = output_dir / "_sft_games_ckpt.jsonl"
    annotated_sft_games = _run_parallel_sft_games(
        new_sft_game_fens,
        depth=sft_depth,
        include_ascii=include_ascii,
        workers=workers,
        checkpoint_path=ckpt_path,
    )
    all_sft_games = reused_sft_games + annotated_sft_games

    sft_puzzle_rows = [
        build_puzzle_sft_example(p, include_ascii=include_ascii)
        for p in puzzle_splits["sft_puzzles"]
    ]

    sft_chessqa_rows: list[dict[str, Any]] = []
    sft_chessqa_rows.extend(_build_structural_batch(by_phase, used_keys, 4_000))
    sft_chessqa_rows.extend(build_motifs_chessqa_item(p) for p in puzzle_splits["sft_motifs"])
    sft_chessqa_rows.extend(
        build_short_tactics_chessqa_item(p) for p in puzzle_splits["sft_short_tactics"]
    )
    sft_chessqa_rows.extend(_build_judgment_batch(by_judgment, used_keys, 4_000))
    sft_chessqa_rows.extend(_build_semantic_batch(by_phase, used_keys, 4_000))

    sft_80k = all_sft_games + sft_puzzle_rows + sft_chessqa_rows
    rng.shuffle(sft_80k)
    _write_jsonl(output_dir / "sft_phase2_80k.jsonl", sft_80k)
    logger.info("Wrote sft_phase2_80k.jsonl (%d rows).", len(sft_80k))

    sft_20k = build_stratified_20k_subset(
        sft_80k,
        game_count=5_000,
        puzzle_count=10_000,
        chessqa_count=5_000,
        seed=seed,
    )
    _write_jsonl(output_dir / "sft_phase2_20k.jsonl", sft_20k)
    logger.info("Wrote sft_phase2_20k.jsonl (%d rows).", len(sft_20k))

    # Clean up temporary checkpoint file once 80K is materialized.
    if ckpt_path.exists():
        ckpt_path.unlink()

    # Strict Leakage Audit
    audit = compute_phase2_leakage_audit(
        {
            "test_openings_20": opening_fens,
            "test_fens_1000": [r["fen"] for r in test_fens_1000],
            "test_puzzles_900": [r["fen"] for r in test_puzzles_900],
            "test_chessqa_1000": [r["fen"] for r in test_chessqa_1000],
            "test_coach_300": [r["fen"] for r in test_coach_300],
            "sft_phase2_80k": [r["fen"] for r in sft_80k],
            "sft_phase2_20k": [r["fen"] for r in sft_20k],
            "rl_positions_40k": rl_positions_40k,
        },
        subset_pairs=(("sft_phase2_20k", "sft_phase2_80k"),),
        strict=True,
    )
    audit["english_forgetting_count"] = n_english
    audit["sft_80k_source_breakdown"] = dict(Counter(r["source"] for r in sft_80k))
    audit["sft_20k_source_breakdown"] = dict(Counter(r["source"] for r in sft_20k))
    audit["chessqa_test_category_breakdown"] = dict(
        Counter(r["category"] for r in test_chessqa_1000)
    )
    audit["coach_test_mode_breakdown"] = dict(Counter(r["mode"] for r in test_coach_300))

    audit_path = output_dir / "leakage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    logger.info("Leakage audit PASSED (zero_leakage=%s): %s", audit["zero_leakage"], audit_path)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase II frozen benchmarks, 80K/20K SFT corpora, and 40K RL positions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_generation/full/phase2"),
        help="Destination directory (default: data_generation/full/phase2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 2))
    parser.add_argument("--sft-depth", type=int, default=8)
    parser.add_argument("--eval-depth", type=int, default=10)
    parser.add_argument("--no-ascii", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    build_phase2_artifacts(
        output_dir=args.output_dir,
        seed=args.seed,
        workers=args.workers,
        sft_depth=args.sft_depth,
        eval_depth=args.eval_depth,
        include_ascii=not args.no_ascii,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
