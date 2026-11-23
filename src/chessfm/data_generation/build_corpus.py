#!/usr/bin/env python3
"""Engine-verified coaching corpus builder and SFT dataset cleaner.

Constructs 100% Stockfish-verified coaching and best-line SFT corpora across
Opening, Middlegame, and Endgame positions using Partial Strategy Trees,
Maia-2 human-Elo calibration, 3-level 0%-spoiler Socratic hints, and ACT-Eval
claim verification. Also provides a deterministic cleaner/verifier for
existing ``sft_train.jsonl`` files.

Usage:
    python -m chessfm.data_generation.build_corpus --output-dir data_generation --depth 8
    python -m chessfm.data_generation.build_corpus --output-dir data_generation --depth 8 --clean-sft
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import chess
import chess.engine

from chessfm.coach.classifier import build_partial_strategy_tree
from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.coach.socratic import contains_move_spoiler, generate_socratic_hint
from chessfm.coach.verifier import verify_claims
from chessfm.config import CHATML_ASSISTANT_CLOSE, MAX_CENTIPAWN_LOSS, build_chat_prompt
from chessfm.data_generation.coaching_dataset import (
    CoachingMode,
    audit_dataset_rows,
    build_best_line_sft_example,
    build_coaching_example,
)
from chessfm.data_generation.convert_to_training import _strip_trailing_move, fix_think_tags
from chessfm.data_generation.phase_sampler import (
    classify_game_phase,
    normalize_board_fen,
    phase_distribution,
    sample_balanced_positions,
    verify_zero_leakage,
)
from chessfm.fens import read_fens, read_jsonl
from chessfm.rl.engine import best_cp, best_move, close_engine, eval_after_move, open_engine

# ---------------------------------------------------------------------------
# Curated Phase-Balanced Seed Positions (40 unique FENs: 25% / 50% / 25%)
# ---------------------------------------------------------------------------

OPENING_SEED_FENS: tuple[str, ...] = (
    # Italian Game (Giuoco Piano)
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Ruy Lopez (Morphy Defense)
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    # Sicilian Najdorf
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # Queen's Gambit Declined
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # French Defense (Advance)
    "rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/2P5/PP3PPP/RNBQKBNR b KQkq - 0 4",
    # Caro-Kann Classical
    "rn1qkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 1 5",
    # King's Indian Defense
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 3 6",
    # English Opening (Symmetrical)
    "r1bqkbnr/pp1ppppp/2n5/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3",
    # London System
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R b KQkq - 0 4",
    # Two Knights Defense
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
)

MIDDLEGAME_SEED_FENS: tuple[str, ...] = (
    # Sicilian Dragon Yugoslav Attack
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP1Q1PP/2KR1B1R b - - 2 11",
    # Ruy Lopez Chigorin
    "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 11",
    # Queen's Gambit Carlsbad structure
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 2 11",
    # French Winawer complex middlegame
    "r1b1k2r/pp3ppp/2n1pq2/2ppP3/3P4/P1P2N2/2P2PPP/R1BQKB1R w KQkq - 0 11",
    # King's Indian Mar del Plata
    "r1bq1rk1/ppp1n1bp/3p2p1/3Ppp2/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 11",
    # Italian middlegame center tension
    "r1bq1rk1/pp1p1ppp/2n2n2/2b1p3/2B1P3/2PP1N2/PP3PPP/RNBQ1RK1 b - - 0 11",
    # Isolated Queen's Pawn (IQP) middlegame
    "r2q1rk1/pp2bppp/2n1pn2/8/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 1 12",
    # Tactical knight outpost & central activity
    "r1b2rk1/ppp2ppp/2n2q2/3np3/2B5/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 11",
    # Pin on f6 & kingside pressure
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP4/PPP2PPP/R2Q1RK1 b - - 2 11",
    # Sicilian Scheveningen open c-file battle
    "r1b2rk1/1p2bppp/p1nppn2/8/3NP3/2N1B3/PPP1BPPP/R2Q1RK1 w - - 4 12",
    # Central pawn tension
    "r2q1rk1/ppp1bppp/2n2n2/3pp3/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 0 11",
    # Opposite-side castling attack
    "2kr1bnr/ppp2ppp/2np4/4p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 11",
    # Kingside attacking structure (French e5 chain)
    "r1bq1rk1/ppp1nppp/4p3/3pP3/3P4/2PB1N2/PP3PPP/R1BQ1RK1 w - - 1 12",
    # Queen and rook battery
    "r2qr1k1/pp1nbppp/2p2n2/3p4/3P4/2NBPN2/PPQ2PPP/R3R1K1 w - - 4 13",
    # Caro-Kann e5 outpost
    "r2q1rk1/pp2bppp/2n1pn2/3pN3/3P4/2N1P3/PP2BPPP/R1BQ1RK1 b - - 3 12",
    # Benoni pawn structure
    "r1bq1rk1/pp3pbp/3p1np1/2pPp3/4P3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 11",
    # Nimzo-Indian doubled c-pawns
    "r1bq1rk1/pp1p1ppp/2n1pn2/8/2PP4/2P1PN2/P4PPP/R1BQKB1R b KQ - 0 11",
    # Active heavy-piece middlegame
    "r4rk1/ppp2ppp/2np1q2/2b1p3/2B1P3/2NP1Q2/PPP2PPP/R4RK1 w - - 2 13",
    # Semi-Slav middlegame
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 b - - 2 11",
    # Grunfeld center battle
    "r1bq1rk1/pp2ppbp/2n3p1/2pp4/3PP3/2N1BN2/PP3PPP/R2QKB1R w KQ - 0 11",
)

ENDGAME_SEED_FENS: tuple[str, ...] = (
    # King + Pawn opposition
    "8/8/4k3/8/4P3/4K3/8/8 w - - 0 45",
    # King + Pawn breakthrough
    "6k1/ppp5/8/PPP5/8/8/6K1/8 w - - 0 42",
    # King + Pawn outside passed pawn
    "8/5pp1/4k2p/p7/P6P/5PP1/6K1/8 w - - 0 40",
    # Rook endgame (Lucena-style)
    "1K6/1P1k4/8/8/8/8/r7/4R3 w - - 0 55",
    # Rook endgame (Philidor defense)
    "8/8/3k4/8/3P4/3K4/r7/4R3 b - - 0 50",
    # Rook + pawns endgame
    "8/5pk1/6p1/p2R3p/P6P/6P1/5PK1/3r4 w - - 0 38",
    # Knight vs Bishop endgame
    "8/5pk1/6p1/2b1p2p/4P2P/2N3P1/5PK1/8 w - - 0 41",
    # Same-colored bishops endgame
    "8/5pk1/3b2p1/p3p2p/P3P2P/3B2P1/5PK1/8 w - - 0 43",
    # Opposite-colored bishops endgame
    "8/5pk1/4b1p1/p3p2p/P3P2P/3B2P1/5PK1/8 b - - 0 44",
    # Knight endgame
    "8/5pk1/4n1p1/p6p/P6P/4N1P1/5PK1/8 w - - 0 46",
)

CURATED_SEED_FENS: list[str] = [
    *OPENING_SEED_FENS,
    *MIDDLEGAME_SEED_FENS,
    *ENDGAME_SEED_FENS,
]

COACHING_MODES: tuple[CoachingMode, ...] = (
    "postmortem",
    "socratic_l1",
    "socratic_l2",
    "socratic_l3",
    "why_not",
)

_DEFAULT_ELOS: tuple[int, ...] = (1000, 1200, 1500)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _select_student_move(
    board: chess.Board,
    best_move_uci: str,
    top_lines: list[dict[str, Any]],
    maia_prediction: dict[str, Any],
) -> tuple[str | None, float | None]:
    """Select a realistic non-best human candidate move for coaching pairs."""
    move_probs: dict[str, float] = maia_prediction.get("move_probs", {})
    for candidate, prob in maia_prediction.get("top_moves", []):
        if candidate != best_move_uci:
            return candidate, float(prob)

    for line in top_lines[1:]:
        candidate = line["move"]
        if candidate != best_move_uci:
            return candidate, float(move_probs.get(candidate, 0.0))

    for move in board.legal_moves:
        uci = move.uci()
        if uci != best_move_uci:
            return uci, float(move_probs.get(uci, 0.0))

    return None, None


def _compute_student_cp_loss(
    board: chess.Board,
    student_move_uci: str,
    best_eval_cp: float,
    top_lines: list[dict[str, Any]],
    engine: Any,
    depth: int,
) -> float:
    """Compute non-negative centipawn loss for ``student_move_uci``."""
    matched = next((line for line in top_lines if line["move"] == student_move_uci), None)
    if matched is not None:
        return max(0.0, float(best_eval_cp) - float(matched["eval_cp"]))

    after_cp = eval_after_move(engine, board, student_move_uci, depth=depth)
    if after_cp is None:
        return 0.0
    return max(0.0, float(best_eval_cp) - float(after_cp))


def _ensure_spoiler_free(
    row: dict[str, Any],
    fen: str,
    best_move_uci: str,
    level: int,
) -> dict[str, Any]:
    """Guarantee Socratic L1/L2 rows contain zero move spoilers."""
    assistant_msg = row["messages"][-1]["content"]
    if not contains_move_spoiler(assistant_msg, best_move_uci, fen):
        return row

    safe = generate_socratic_hint(fen, best_move_uci, level=level)
    safe_text = safe["hint"]
    row["messages"][-1]["content"] = safe_text
    row["completion"] = f"{safe_text}{CHATML_ASSISTANT_CLOSE}"
    return row


def _extract_reasoning_from_completion(completion: str, move_uci: str) -> tuple[str, bool, bool]:
    """Repair ``<think>`` tags and strip duplicate trailing move tokens.

    Returns ``(clean_reasoning, fixed_think, fixed_trailing)``.
    """
    raw = completion.strip()
    if raw.endswith(CHATML_ASSISTANT_CLOSE):
        raw = raw[: -len(CHATML_ASSISTANT_CLOSE)].rstrip()

    # Separate body before the trailing move token.
    stripped_before = _strip_trailing_move(raw, move_uci)
    fixed_trailing = stripped_before != raw

    # Remove any additional trailing UCI-like duplicate token after </think>.
    if "</think>" in stripped_before:
        head, tail = stripped_before.rsplit("</think>", 1)
        if tail.strip():
            fixed_trailing = True
            stripped_before = f"{head}</think>"

    repaired_think = fix_think_tags(stripped_before)
    fixed_think = repaired_think != stripped_before

    # Ensure <think>...</think> exists and is non-empty.
    if "<think>" not in repaired_think and "</think>" not in repaired_think:
        body = repaired_think.strip() or f"Selected legal move {move_uci}."
        repaired_think = f"<think>\n{body}\n</think>"
        fixed_think = True

    # Strip standalone duplicate move line immediately before </think>.
    if "</think>" in repaired_think:
        open_idx = repaired_think.index("<think>") + len("<think>")
        close_idx = repaired_think.index("</think>")
        inner = repaired_think[open_idx:close_idx].strip()
        lines = inner.splitlines()
        while lines and lines[-1].strip().lower() == move_uci.lower():
            lines.pop()
            fixed_trailing = True
        inner_clean = "\n".join(lines).strip()
        if not inner_clean:
            inner_clean = f"Selected legal move {move_uci}."
            fixed_think = True
        repaired_think = f"<think>\n{inner_clean}\n</think>"

    # Collapse stray leading "think\n" artifact sometimes emitted by teacher models.
    repaired_think = re.sub(r"<think>\s*\nthink\n", "<think>\n", repaired_think, count=1)

    return repaired_think, fixed_think, fixed_trailing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_verified_coaching_corpus(
    fens: list[str],
    engine: Any,
    depth: int = 8,
    include_ascii: bool = True,
) -> dict[str, Any]:
    """Build an engine-verified coaching & best-line SFT corpus plus benchmark set.

    For each unique, non-terminal FEN, queries Stockfish via
    ``build_partial_strategy_tree`` and Maia-2 (``MaiaSidecar``) to produce:
    - One ``best_line`` SFT row (``build_best_line_sft_example``)
    - Five coaching rows (``postmortem``, ``socratic_l1``, ``socratic_l2``,
      ``socratic_l3``, ``why_not``) via ``build_coaching_example``
    - One ground-truth ``eval_benchmark`` record (``fen``, ``phase``,
      ``best_move``, ``top_moves``, ``eval_cp``)

    Audits all generated rows with ``audit_dataset_rows``,
    ``contains_move_spoiler`` (for Socratic L1/L2), and ``verify_claims``.
    """
    maia = MaiaSidecar(use_neural=False)
    rows: list[dict[str, Any]] = []
    eval_benchmark: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    unique_fens: list[str] = []

    mode_counts: dict[str, int] = {
        "best_line": 0,
        **{mode: 0 for mode in COACHING_MODES},
    }
    spoiler_violations = 0
    claim_violations = 0

    for idx, raw_fen in enumerate(fens):
        try:
            fen = raw_fen.strip()
            key = normalize_board_fen(fen)
            board = chess.Board(fen)
        except (AttributeError, ValueError):
            continue

        if key in seen_keys or not board.is_valid() or board.is_game_over():
            continue
        seen_keys.add(key)

        tree = build_partial_strategy_tree(
            fen=fen,
            engine=engine,
            depth=depth,
            multipv=3,
            pv_plies=4,
        )
        top_lines = tree.get("top_lines", [])
        if not top_lines or not tree.get("best_move"):
            continue

        unique_fens.append(fen)
        phase = classify_game_phase(fen)
        best_line = top_lines[0]
        best_move_uci = best_line["move"]
        pv_ucis = best_line["pv_uci"]
        best_eval_cp = float(best_line["eval_cp"])

        eval_benchmark.append(
            {
                "fen": fen,
                "phase": phase,
                "best_move": best_move_uci,
                "top_moves": [
                    {
                        "move": line["move"],
                        "san": line["san"],
                        "eval_cp": line["eval_cp"],
                        "pv_uci": line["pv_uci"],
                    }
                    for line in top_lines
                ],
                "eval_cp": best_eval_cp,
            }
        )

        # 1. Best-line SFT row
        best_row = build_best_line_sft_example(
            fen=fen,
            best_move_uci=best_move_uci,
            pv_ucis=pv_ucis,
            eval_cp=best_eval_cp,
            include_ascii=include_ascii,
        )
        rows.append(best_row)
        mode_counts["best_line"] += 1

        # 2. Coaching rows across all 5 pedagogical modes
        student_elo = _DEFAULT_ELOS[idx % len(_DEFAULT_ELOS)]
        maia_pred = maia.predict_human_moves(fen, student_elo=student_elo, top_k=5)
        student_move_uci, maia_prob = _select_student_move(
            board=board,
            best_move_uci=best_move_uci,
            top_lines=top_lines,
            maia_prediction=maia_pred,
        )
        if student_move_uci is None:
            continue

        cp_loss = _compute_student_cp_loss(
            board=board,
            student_move_uci=student_move_uci,
            best_eval_cp=best_eval_cp,
            top_lines=top_lines,
            engine=engine,
            depth=depth,
        )

        for mode in COACHING_MODES:
            coach_row = build_coaching_example(
                fen=fen,
                student_move_uci=student_move_uci,
                best_move_uci=best_move_uci,
                pv_ucis=pv_ucis,
                cp_loss=cp_loss,
                student_elo=student_elo,
                maia_prob=maia_prob,
                mode=mode,
            )

            if mode == "socratic_l1":
                coach_row = _ensure_spoiler_free(coach_row, fen, best_move_uci, level=1)
                if contains_move_spoiler(coach_row["messages"][-1]["content"], best_move_uci, fen):
                    spoiler_violations += 1
            elif mode == "socratic_l2":
                coach_row = _ensure_spoiler_free(coach_row, fen, best_move_uci, level=2)
                if contains_move_spoiler(coach_row["messages"][-1]["content"], best_move_uci, fen):
                    spoiler_violations += 1

            claim_check = verify_claims(fen, coach_row["messages"][-1]["content"])
            if not claim_check["passed"]:
                claim_violations += 1

            rows.append(coach_row)
            mode_counts[mode] += 1

    audit = audit_dataset_rows(rows)
    passed = bool(audit["passed"] and spoiler_violations == 0 and claim_violations == 0)

    return {
        "rows": rows,
        "eval_benchmark": eval_benchmark,
        "audit": audit,
        "spoiler_violations": spoiler_violations,
        "claim_violations": claim_violations,
        "phase_distribution": phase_distribution(unique_fens),
        "mode_counts": mode_counts,
        "total_fens": len(unique_fens),
        "total_rows": len(rows),
        "passed": passed,
    }


def clean_and_verify_sft_file(
    input_path: Path,
    output_path: Path,
    engine: Any | None = None,
    depth: int = 8,
) -> dict[str, Any]:
    """Repair formatting, drop illegal/blunder rows, and verify an SFT JSONL file.

    Applies ``fix_think_tags``, removes duplicate trailing moves, deduplicates
    by normalized board FEN, and—when ``engine`` is provided—re-evaluates every
    row with Stockfish, dropping moves whose centipawn loss exceeds
    ``MAX_CENTIPAWN_LOSS``.
    """
    raw_rows = read_jsonl(Path(input_path))
    dropped_invalid = 0
    dropped_blunders = 0
    dropped_duplicates = 0
    repaired_think_tags = 0
    repaired_trailing_moves = 0

    best_by_key: dict[str, dict[str, Any]] = {}

    for row in raw_rows:
        fen = row.get("fen")
        move_raw = row.get("move")
        if not fen or not isinstance(fen, str) or not move_raw or not isinstance(move_raw, str):
            dropped_invalid += 1
            continue

        move_uci = move_raw.strip().lower()
        try:
            board = chess.Board(fen.strip())
            move_obj = chess.Move.from_uci(move_uci)
        except ValueError:
            dropped_invalid += 1
            continue

        if move_obj not in board.legal_moves:
            dropped_invalid += 1
            continue

        cp_loss = row.get("centipawn_loss")
        sf_best = row.get("stockfish_best")

        if engine is not None:
            best_eval = best_cp(engine, board, depth=depth)
            sf_best = best_move(engine, board, depth=depth) or sf_best
            after_eval = eval_after_move(engine, board, move_uci, depth=depth)
            if best_eval is not None and after_eval is not None:
                cp_loss = (
                    0.0
                    if sf_best and move_uci == sf_best.lower()
                    else max(0.0, float(best_eval) - float(after_eval))
                )
        else:
            if row.get("is_reasonable") is False:
                dropped_blunders += 1
                continue

        if cp_loss is not None and float(cp_loss) > MAX_CENTIPAWN_LOSS:
            dropped_blunders += 1
            continue

        completion = str(row.get("completion", ""))
        clean_reasoning, fixed_think, fixed_trailing = _extract_reasoning_from_completion(
            completion, move_uci
        )
        if fixed_think:
            repaired_think_tags += 1
        if fixed_trailing:
            repaired_trailing_moves += 1

        cleaned = dict(row)
        cleaned["fen"] = fen.strip()
        cleaned["move"] = move_uci
        cleaned["prompt"] = row.get("prompt") or build_chat_prompt(cleaned["fen"])
        cleaned["completion"] = f"{clean_reasoning}\n{move_uci}{CHATML_ASSISTANT_CLOSE}"
        cleaned["is_reasonable"] = True
        if cp_loss is not None:
            cleaned["centipawn_loss"] = round(max(0.0, float(cp_loss)), 2)
        if sf_best is not None:
            cleaned["stockfish_best"] = sf_best

        key = normalize_board_fen(cleaned["fen"])
        prev = best_by_key.get(key)
        if prev is not None:
            dropped_duplicates += 1
            prev_cp = float(prev.get("centipawn_loss") or 0.0)
            cur_cp = float(cleaned.get("centipawn_loss") or 0.0)
            if cur_cp < prev_cp:
                best_by_key[key] = cleaned
        else:
            best_by_key[key] = cleaned

    cleaned_rows = list(best_by_key.values())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fout:
        for item in cleaned_rows:
            fout.write(json.dumps(item) + "\n")

    audit = audit_dataset_rows(cleaned_rows)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_input": len(raw_rows),
        "total_output": len(cleaned_rows),
        "dropped_invalid": dropped_invalid,
        "dropped_blunders": dropped_blunders,
        "dropped_duplicates": dropped_duplicates,
        "repaired_think_tags": repaired_think_tags,
        "repaired_trailing_moves": repaired_trailing_moves,
        "engine_verified": engine is not None,
        "audit": audit,
        "passed": audit["passed"],
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records to ``path``, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write newline-delimited strings to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")


def _chunk_list(items: list[str], num_chunks: int) -> list[list[str]]:
    """Split ``items`` into at most ``num_chunks`` balanced non-empty slices."""
    if not items:
        return []
    n = max(1, min(num_chunks, len(items)))
    chunk_size = (len(items) + n - 1) // n
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def split_disjoint_balanced_fens(
    candidate_fens: list[str],
    *,
    eval_count: int = 200,
    sft_count: int = 2500,
    grpo_count: int = 5000,
    ratios: tuple[float, float, float] = (0.25, 0.55, 0.20),
    seed: int = 42,
) -> dict[str, Any]:
    """Allocate strictly disjoint, phase-balanced eval/SFT/GRPO FEN splits.

    Deduplicates ``candidate_fens`` by 4-field ``normalize_board_fen``, drops
    invalid or game-over positions, samples ``eval -> sft -> grpo`` without
    replacement, and enforces ``verify_zero_leakage(..., strict=True)``.
    """
    unique_valid: list[str] = []
    seen_keys: set[str] = set()
    for raw_fen in candidate_fens:
        if not isinstance(raw_fen, str):
            continue
        fen = raw_fen.strip()
        if not fen:
            continue
        try:
            key = normalize_board_fen(fen)
            board = chess.Board(fen)
        except ValueError:
            continue
        if key in seen_keys or not board.is_valid() or board.is_game_over():
            continue
        seen_keys.add(key)
        unique_valid.append(fen)

    eval_fens = sample_balanced_positions(unique_valid, eval_count, ratios=ratios, seed=seed)
    eval_keys = {normalize_board_fen(f) for f in eval_fens}

    rem_after_eval = [f for f in unique_valid if normalize_board_fen(f) not in eval_keys]
    sft_fens = sample_balanced_positions(rem_after_eval, sft_count, ratios=ratios, seed=seed + 1)
    sft_keys = {normalize_board_fen(f) for f in sft_fens}

    excluded = eval_keys | sft_keys
    rem_after_sft = [f for f in unique_valid if normalize_board_fen(f) not in excluded]
    grpo_fens = sample_balanced_positions(rem_after_sft, grpo_count, ratios=ratios, seed=seed + 2)

    leakage = verify_zero_leakage(sft_fens, grpo_fens, eval_fens, strict=True)
    return {
        "eval_fens": eval_fens,
        "sft_fens": sft_fens,
        "grpo_fens": grpo_fens,
        "leakage": leakage,
        "phase_distributions": {
            "eval": phase_distribution(eval_fens),
            "sft": phase_distribution(sft_fens),
            "grpo": phase_distribution(grpo_fens),
        },
        "total_valid_unique": len(unique_valid),
    }


def _analyze_sft_chunk(
    fens_chunk: list[str],
    depth: int,
    include_ascii: bool,
) -> list[dict[str, Any]]:
    """Worker: run single-PV Stockfish analysis and build best-line SFT rows."""
    engine = open_engine(threads=1, hash_mb=16)
    if engine is None:
        raise RuntimeError(
            "Stockfish binary not found. Install Stockfish (e.g. `apt-get install stockfish`) "
            "before running chessfm.data_generation.build_corpus."
        )
    rows: list[dict[str, Any]] = []
    try:
        for fen in fens_chunk:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            pv = info.get("pv", [])
            if not pv:
                continue
            pv_ucis = [m.uci() for m in pv[:4]]
            score_obj = info.get("score")
            eval_cp = (
                float(score_obj.pov(board.turn).score(mate_score=10000))
                if score_obj is not None
                else 0.0
            )
            rows.append(
                build_best_line_sft_example(
                    fen=fen,
                    best_move_uci=pv_ucis[0],
                    pv_ucis=pv_ucis,
                    eval_cp=eval_cp,
                    include_ascii=include_ascii,
                )
            )
    finally:
        close_engine(engine)
    return rows


def _analyze_eval_chunk(fens_chunk: list[str], depth: int) -> list[dict[str, Any]]:
    """Worker: run MultiPV=3 Stockfish analysis for held-out eval benchmark rows."""
    engine = open_engine(threads=1, hash_mb=16)
    if engine is None:
        raise RuntimeError(
            "Stockfish binary not found. Install Stockfish (e.g. `apt-get install stockfish`) "
            "before running chessfm.data_generation.build_corpus."
        )
    rows: list[dict[str, Any]] = []
    try:
        for fen in fens_chunk:
            tree = build_partial_strategy_tree(
                fen=fen,
                engine=engine,
                depth=depth,
                multipv=3,
                pv_plies=4,
            )
            top_lines = tree.get("top_lines", [])
            if not top_lines or not tree.get("best_move"):
                continue
            best_line = top_lines[0]
            rows.append(
                {
                    "fen": fen,
                    "phase": classify_game_phase(fen),
                    "best_move": best_line["move"],
                    "top_moves": [
                        {
                            "move": line["move"],
                            "san": line["san"],
                            "eval_cp": line["eval_cp"],
                            "pv_uci": line["pv_uci"],
                        }
                        for line in top_lines
                    ],
                    "eval_cp": float(best_line["eval_cp"]),
                }
            )
    finally:
        close_engine(engine)
    return rows


def build_p1_splits(
    positions_path: Path,
    sft_out: Path,
    grpo_out: Path,
    eval_out: Path,
    openings_out: Path | None = None,
    *,
    sft_count: int = 2500,
    grpo_count: int = 5000,
    eval_count: int = 200,
    openings_count: int = 25,
    depth: int = 10,
    workers: int = 2,
    include_ascii: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Build disjoint phase-balanced P1 SFT, GRPO, Eval, and paired-opening files."""
    candidate_fens = read_fens(Path(positions_path))
    splits = split_disjoint_balanced_fens(
        candidate_fens,
        eval_count=eval_count,
        sft_count=sft_count,
        grpo_count=grpo_count,
        seed=seed,
    )
    eval_fens: list[str] = splits["eval_fens"]
    sft_fens: list[str] = splits["sft_fens"]
    grpo_fens: list[str] = splits["grpo_fens"]

    sft_chunks = _chunk_list(sft_fens, max(1, workers))
    eval_chunks = _chunk_list(eval_fens, max(1, workers))

    if workers <= 1:
        sft_rows = [
            row for chunk in sft_chunks for row in _analyze_sft_chunk(chunk, depth, include_ascii)
        ]
        eval_rows = [row for chunk in eval_chunks for row in _analyze_eval_chunk(chunk, depth)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            sft_futures = [
                pool.submit(_analyze_sft_chunk, chunk, depth, include_ascii) for chunk in sft_chunks
            ]
            sft_rows = [row for fut in sft_futures for row in fut.result()]
            eval_futures = [pool.submit(_analyze_eval_chunk, chunk, depth) for chunk in eval_chunks]
            eval_rows = [row for fut in eval_futures for row in fut.result()]

    sft_audit = audit_dataset_rows(sft_rows)
    if not sft_audit["passed"]:
        raise RuntimeError(f"Generated SFT rows failed dataset audit: {sft_audit}")

    final_sft_fens = [r["fen"] for r in sft_rows]
    final_eval_fens = [r["fen"] for r in eval_rows]
    leakage = verify_zero_leakage(final_sft_fens, grpo_fens, final_eval_fens, strict=True)

    _write_jsonl(Path(sft_out), sft_rows)
    _write_lines(Path(grpo_out), grpo_fens)
    _write_jsonl(Path(eval_out), eval_rows)

    opening_fens = [r["fen"] for r in eval_rows if r["phase"] == "opening"][:openings_count]
    if openings_out is not None:
        _write_lines(Path(openings_out), opening_fens)

    return {
        "total_valid_unique": splits["total_valid_unique"],
        "sft_count": len(sft_rows),
        "grpo_count": len(grpo_fens),
        "eval_count": len(eval_rows),
        "openings_count": len(opening_fens),
        "sft_audit": sft_audit,
        "leakage": leakage,
        "phase_distributions": {
            "sft": phase_distribution(final_sft_fens),
            "grpo": phase_distribution(grpo_fens),
            "eval": phase_distribution(final_eval_fens),
        },
        "paths": {
            "sft_out": str(sft_out),
            "grpo_out": str(grpo_out),
            "eval_out": str(eval_out),
            "openings_out": str(openings_out) if openings_out is not None else None,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the corpus builder."""
    parser = argparse.ArgumentParser(
        description="Build Stockfish-verified ChessFM coaching corpus and eval benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_generation"),
        help="Directory for coaching_sample.jsonl and eval_benchmark.jsonl",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Stockfish search depth in plies (default: 8)",
    )
    parser.add_argument(
        "--include-ascii",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include ASCII board rendering in best-line SFT prompts (default: True)",
    )
    parser.add_argument(
        "--max-fens",
        type=int,
        default=None,
        help="Optional cap on number of seed FENs to process",
    )
    parser.add_argument(
        "--clean-sft",
        action="store_true",
        help="Also clean and Stockfish-verify sft_train.jsonl",
    )
    parser.add_argument(
        "--sft-input",
        type=Path,
        default=None,
        help="Input SFT JSONL file for --clean-sft (default: <output-dir>/sft_train.jsonl)",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=None,
        help="Output SFT JSONL file for --clean-sft (default: <output-dir>/sft_train.jsonl)",
    )
    parser.add_argument(
        "--build-p1-splits",
        action="store_true",
        help="Build disjoint phase-balanced P1 SFT, GRPO, Eval, and paired-opening splits",
    )
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=Path("data_generation/full/positions_inclusive.txt"),
        help="Input candidate FENs file for --build-p1-splits",
    )
    parser.add_argument(
        "--sft-out",
        type=Path,
        default=Path("data_generation/full/sft_bestline_2500.jsonl"),
        help="Output best-line SFT JSONL for --build-p1-splits",
    )
    parser.add_argument(
        "--grpo-out",
        type=Path,
        default=Path("data_generation/full/grpo_positions_5k.txt"),
        help="Output GRPO FENs text file for --build-p1-splits",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=Path("data_generation/eval_benchmark_200.jsonl"),
        help="Output 200-FEN MultiPV eval benchmark JSONL for --build-p1-splits",
    )
    parser.add_argument(
        "--openings-out",
        type=Path,
        default=Path("data_generation/full/eval_openings_25.txt"),
        help="Output paired opening FENs file for --build-p1-splits",
    )
    parser.add_argument("--sft-count", type=int, default=2500, help="Target SFT FEN count")
    parser.add_argument("--grpo-count", type=int, default=5000, help="Target GRPO FEN count")
    parser.add_argument("--eval-count", type=int, default=200, help="Target Eval FEN count")
    parser.add_argument(
        "--openings-count", type=int, default=25, help="Target paired opening FEN count"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Parallel Stockfish worker processes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building the verified coaching corpus."""
    args = parse_args(argv)
    if args.build_p1_splits:
        summary = build_p1_splits(
            positions_path=args.positions_file,
            sft_out=args.sft_out,
            grpo_out=args.grpo_out,
            eval_out=args.eval_out,
            openings_out=args.openings_out,
            sft_count=args.sft_count,
            grpo_count=args.grpo_count,
            eval_count=args.eval_count,
            openings_count=args.openings_count,
            depth=args.depth,
            workers=args.workers,
            include_ascii=args.include_ascii,
            seed=args.seed,
        )
        print(
            f"Built P1 splits: SFT={summary['sft_count']}, GRPO={summary['grpo_count']}, "
            f"Eval={summary['eval_count']}, Openings={summary['openings_count']} "
            f"(zero_leakage={summary['leakage']['zero_leakage']}, "
            f"total_overlap={summary['leakage']['total_overlap']})."
        )
        print(f"Phase distributions: {summary['phase_distributions']}")
        return 0

    engine = open_engine()
    if engine is None:
        raise RuntimeError(
            "Stockfish binary not found. Install Stockfish (e.g. `apt-get install stockfish`) "
            "before running chessfm.data_generation.build_corpus."
        )

    try:
        seed_fens = CURATED_SEED_FENS[: args.max_fens] if args.max_fens else CURATED_SEED_FENS
        result = build_verified_coaching_corpus(
            fens=seed_fens,
            engine=engine,
            depth=args.depth,
            include_ascii=args.include_ascii,
        )

        coaching_path = args.output_dir / "coaching_sample.jsonl"
        benchmark_path = args.output_dir / "eval_benchmark.jsonl"
        _write_jsonl(coaching_path, result["rows"])
        _write_jsonl(benchmark_path, result["eval_benchmark"])

        print(
            f"Built {result['total_rows']} verified rows from {result['total_fens']} FENs "
            f"(passed={result['passed']}, spoilers={result['spoiler_violations']}, "
            f"claim_violations={result['claim_violations']})."
        )
        print(f"Phase distribution: {result['phase_distribution']}")
        print(f"Coaching corpus:    {coaching_path}")
        print(f"Eval benchmark:     {benchmark_path}")

        if args.clean_sft:
            sft_in = args.sft_input or (args.output_dir / "sft_train.jsonl")
            sft_out = args.sft_output or sft_in
            if sft_in.exists():
                clean_summary = clean_and_verify_sft_file(
                    input_path=sft_in,
                    output_path=sft_out,
                    engine=engine,
                    depth=args.depth,
                )
                print(
                    f"Cleaned SFT: {clean_summary['total_output']}/{clean_summary['total_input']} "
                    f"kept (blunders={clean_summary['dropped_blunders']}, "
                    f"invalid={clean_summary['dropped_invalid']}, "
                    f"passed={clean_summary['passed']})."
                )
            else:
                print(f"Skipping --clean-sft: input file not found at {sft_in}")
    finally:
        close_engine(engine)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
