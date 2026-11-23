#!/usr/bin/env python3
"""Shared analysis helpers and Stockfish engine management.

Error-handling contract: engine/infrastructure failures are converted to
``None`` (callers decide policy) but *only* for genuine engine errors
(``chess.engine.EngineError``, ``OSError``). Any other exception propagates,
so a bug in calling code is never silently masked as "engine unavailable".
"""

import logging

import chess
import chess.engine

from chessfm.config import STOCKFISH_PATHS

logger = logging.getLogger("chessfm.engine")

_ENGINE_ERRORS = (chess.engine.EngineError, OSError)
_CRASH_ERRORS = (chess.engine.EngineTerminatedError, BrokenPipeError)


class ResilientEngine:
    """Thin wrapper around ``chess.engine.SimpleEngine`` with bounded auto-restart.

    Stockfish 16 on Linux can occasionally terminate with ``SIGFPE`` (exit code
    ``-8``) during rapid shallow-depth ``analyse()`` bursts. Rather than leaving
    a dead engine handle that fails every subsequent call, this wrapper detects
    ``EngineTerminatedError`` / ``BrokenPipeError``, cleans up the dead process,
    respawns Stockfish with the exact same configuration, and retries up to
    ``max_restarts`` times per call.
    """

    def __init__(
        self,
        path: str,
        engine: chess.engine.SimpleEngine,
        *,
        threads: int = 1,
        hash_mb: int = 16,
        max_restarts: int = 12,
    ) -> None:
        self._path = path
        self._engine = engine
        self._threads = threads
        self._hash_mb = hash_mb
        self._max_restarts = max(0, int(max_restarts))
        self._options: dict = {"Threads": threads, "Hash": hash_mb}
        self.restart_count = 0

    def _restart(self) -> None:
        close_engine(self._engine)
        fresh = chess.engine.SimpleEngine.popen_uci(self._path)
        if self._options:
            fresh.configure(self._options)
        self._engine = fresh
        self.restart_count += 1

    def _call_with_restart(self, method_name: str, *args, **kwargs):
        attempt = 0
        while True:
            try:
                return getattr(self._engine, method_name)(*args, **kwargs)
            except _CRASH_ERRORS as exc:
                if attempt >= self._max_restarts:
                    raise
                attempt += 1
                logger.warning(
                    "Stockfish process terminated during %s (%s); restarting (%d/%d).",
                    method_name,
                    exc,
                    attempt,
                    self._max_restarts,
                )
                self._restart()

    def configure(self, options: dict) -> None:
        self._engine.configure(options)
        self._options.update(options)

    def analyse(self, *args, **kwargs):
        return self._call_with_restart("analyse", *args, **kwargs)

    def play(self, *args, **kwargs):
        return self._call_with_restart("play", *args, **kwargs)

    def quit(self) -> None:
        close_engine(self._engine)

    def close(self) -> None:
        close_engine(self._engine)

    def __enter__(self) -> "ResilientEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()

    def __getattr__(self, name: str):
        return getattr(self._engine, name)


def open_engine(
    threads: int = 1,
    hash_mb: int = 16,
    max_restarts: int = 12,
    stockfish_path: str | None = None,
) -> ResilientEngine | None:
    """Open a Stockfish engine from ``stockfish_path`` or configured paths, or None."""
    paths = [stockfish_path] if stockfish_path is not None else STOCKFISH_PATHS
    for path in paths:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": threads, "Hash": hash_mb})
            return ResilientEngine(
                path,
                engine,
                threads=threads,
                hash_mb=hash_mb,
                max_restarts=max_restarts,
            )
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
    if score.is_mate():
        if score == chess.engine.MateGiven or score.mate() > 0:
            cp = 10000.0
        else:
            cp = -10000.0
    else:
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
