#!/usr/bin/env python3
"""Shared analysis helpers and Stockfish engine management.

Error-handling contract: engine/infrastructure failures are converted to
``None`` (callers decide policy) but *only* for genuine engine errors
(``chess.engine.EngineError``, ``OSError``). Any other exception propagates,
so a bug in calling code is never silently masked as "engine unavailable".
"""

import chess
import chess.engine

from chessfm.config import STOCKFISH_PATHS

_ENGINE_ERRORS = (chess.engine.EngineError, OSError)


def open_engine(threads: int = 1, hash_mb: int = 16) -> chess.engine.SimpleEngine | None:
    """Open a Stockfish engine from the configured paths, or None if unavailable."""
    for path in STOCKFISH_PATHS:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": threads, "Hash": hash_mb})
            return engine
        except _ENGINE_ERRORS:
            continue
    return None


def close_engine(engine) -> None:
    """Quit and close an engine if it is open.

    Used in ``finally``/cleanup paths, so it never raises: genuine engine
    errors are swallowed here deliberately (the process is going away anyway),
    and duck-typed stand-ins without ``quit``/``close`` are skipped.
    """
    if engine is None:
        return
    if hasattr(engine, "quit"):
        try:
            engine.quit()
        except _ENGINE_ERRORS:
            pass
    if hasattr(engine, "close"):
        try:
            engine.close()
        except _ENGINE_ERRORS:
            pass


def _cp_from_white(info, for_color: chess.Color) -> float:
    """Return a Stockfish score in centipawns from ``for_color``'s perspective."""
    score = info["score"].white()
    cp = 10000.0 if score.is_mate() and score.mate() > 0 else -10000.0
    if not score.is_mate():
        cp = float(score.score())
    return cp if for_color == chess.WHITE else -cp


def best_cp(
    engine, board: chess.Board, depth: int = 10, for_color: chess.Color | None = None
) -> float | None:
    """Best centipawn score for the side to move in ``board``, or None on error.

    Pass ``for_color`` to evaluate from a specific side's perspective rather
    than the side to move.
    """
    if engine is None:
        return None
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        return _cp_from_white(info, for_color if for_color is not None else board.turn)
    except _ENGINE_ERRORS:
        return None


def best_move(engine, board: chess.Board, depth: int = 10) -> str | None:
    """Best move in UCI for the side to move, or None if unavailable."""
    if engine is None:
        return None
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        pv = info.get("pv")
        return pv[0].uci() if pv else None
    except _ENGINE_ERRORS:
        return None


def eval_after_move(engine, board: chess.Board, move_uci: str, depth: int = 10) -> float | None:
    """Centipawn evaluation of ``board`` after ``move_uci``, from the mover's view.

    Returns None for illegal or unparseable moves, or when no engine is set.
    """
    if engine is None:
        return None
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None
        mover = board.turn
        board = board.copy()
        board.push(move)
        return best_cp(engine, board, depth, for_color=mover)
    except (ValueError, TypeError):
        return None
