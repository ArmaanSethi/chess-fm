#!/usr/bin/env python3
"""
Reward functions for Chess RL training.

Implements the staged reward system:

- Stage 0: Legality only (+1 legal, -1 illegal) + format/budget bonuses
- Stage 1: + Move quality vs Stockfish (chosen move's centipawn loss)
- Stage 2: + Opponent-reply-aware delta (realized outcome after Stockfish's
  best counterplay)

Design notes
------------

Stage 1 and Stage 2 measure *different* things:

  - Stage 1 move quality is *proximity*: how close the chosen move's immediate
    eval is to Stockfish's best move in the same position.
  - Stage 2 reply-aware delta is *robustness*: the evaluation after Stockfish's
    best response to the chosen move, compared with the pre-move evaluation.
    A move can look near-best at shallow depth yet concede a strong refutation;
    this component measures what actually happens once the opponent plays its
    best reply. See ``combined_reward`` for per-stage engine costs.

Error-handling contract
-----------------------

An illegal move is the model's fault and is penalized. An engine failure is
infrastructure: :class:`EngineAnalysisError` propagates out of every reward
function so callers can distinguish it from a legitimately low score and never
silently fold it into the training signal.

Win/lose outcome rewards are not used in GRPO here: training runs on single
positions (one completion per FEN), not full rollouts, so per-completion
outcomes cannot be computed. ``reward_outcome`` is kept for full-game
evaluation (see ``chessfm.rl.evaluate``).
"""

import logging
import math
import threading
from collections import OrderedDict

import chess
import chess.engine

from chessfm.config import DEFAULT_STOCKFISH_DEPTH
from chessfm.rl.engine import best_cp, best_move

logger = logging.getLogger("chessfm.rewards")

# Component weights, one row per stage (legality, format, budget, quality, delta).
# Minimum/maximum reasoning word counts for the budget reward. These are WORD
# counts, not token counts (see reward_budget).
MIN_THINK_WORDS = 5
MAX_THINK_WORDS = 250

_ILLEGAL_REWARD = -1.0
_MAX_QUALITY_LOSS = 200.0
_LICHESS_CP_COEFF = 0.00368208
_WDL_QUALITY_WEIGHT = 2.0

# Anti-hack sharp WDL reward constants (eliminates passive edge-pawn reward hacking).
_SHARP_WDL_ILLEGAL_REWARD = -0.5
_SHARP_WDL_THRESHOLD = 0.15
_SHARP_WDL_QUALITY_WEIGHT = 1.5
_SHARP_WDL_EXACT_BONUS = 0.5
_SHARP_WDL_NEAR_BEST_CP = 15.0
_SHARP_WDL_FORMAT_BONUS = 0.1

_CACHE_ENGINE_ERRORS = (chess.engine.EngineError, OSError)


class EngineAnalysisError(RuntimeError):
    """Raised when Stockfish fails or returns no evaluation for a legal move.

    Distinct from an illegal move: an illegal move is the model's fault and is
    penalized; an engine failure is infrastructure and must propagate so the
    caller can abort or flag it instead of corrupting the training signal.
    """


def cp_to_expected_score(cp: float) -> float:
    """Convert a centipawn score to expected game score in ``[0.0, 1.0]``.

    Uses the Lichess logistic curve ``1 / (1 + exp(-0.00368208 * cp))``,
    clamping the exponent to ``[-500.0, 500.0]`` so extreme mate scores
    (±10000 cp) and ``±inf`` never overflow ``math.exp``.

    Raises:
        ValueError: if ``cp`` is ``NaN``.
    """
    val = float(cp)
    if math.isnan(val):
        raise ValueError("cp must not be NaN")
    exponent = max(-500.0, min(500.0, -_LICHESS_CP_COEFF * val))
    return 1.0 / (1.0 + math.exp(exponent))


def _white_cp_from_info(info: dict) -> float | None:
    """Extract a White-perspective centipawn float from a python-chess info dict."""
    pov_score = info.get("score") if isinstance(info, dict) else None
    if pov_score is None:
        return None
    try:
        score = pov_score.white()
        if score.is_mate():
            if score == chess.engine.MateGiven or (score.mate() is not None and score.mate() > 0):
                return 10000.0
            return -10000.0
        cp = score.score()
    except (AttributeError, TypeError, ValueError):
        return None
    if isinstance(cp, bool) or not isinstance(cp, (int, float)):
        return None
    cp_val = float(cp)
    if math.isnan(cp_val):
        return None
    return cp_val


class StockfishEvalCache:
    """Thread-safe bounded LRU cache for Stockfish per-FEN and per-(FEN, move) evals.

    Stores White-perspective centipawn scores so lookups work for both sides
    to move, and caches ``(white_cp, best_move_uci)`` from a single engine
    search. Engine failures (``None``) are never cached.
    """

    def __init__(self, max_size: int = 32768) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._fen_cache: OrderedDict[tuple[str, int], tuple[float, str | None]] = OrderedDict()
        self._move_cache: OrderedDict[tuple[str, str, int], float] = OrderedDict()
        self._fen_hits = 0
        self._fen_misses = 0
        self._move_hits = 0
        self._move_misses = 0

    def _put_lru(self, store: OrderedDict, key, value) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > self.max_size:
            store.popitem(last=False)

    def get_best(
        self,
        engine,
        board: chess.Board,
        depth: int,
        for_color: chess.Color | None = None,
    ) -> tuple[float | None, str | None]:
        """Return ``(cp, best_move_uci)`` for ``board`` at ``depth`` using the cache."""
        target = board.turn if for_color is None else for_color
        key = (board.fen(), depth)
        with self._lock:
            cached = self._fen_cache.get(key)
            if cached is not None:
                self._fen_cache.move_to_end(key)
                self._fen_hits += 1
                white_cp, pv_uci = cached
                cp = white_cp if target == chess.WHITE else -white_cp
                return cp, pv_uci
            self._fen_misses += 1

        try:
            with self._engine_lock:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
            white_cp = _white_cp_from_info(info)
            if white_cp is None:
                return None, None
            pv = info.get("pv") if isinstance(info, dict) else None
            pv_uci: str | None = None
            if isinstance(pv, (list, tuple)) and pv and hasattr(pv[0], "uci"):
                try:
                    raw_uci = pv[0].uci()
                    if isinstance(raw_uci, str):
                        pv_uci = raw_uci
                except (AttributeError, TypeError):
                    pv_uci = None
        except _CACHE_ENGINE_ERRORS:
            return None, None

        with self._lock:
            self._put_lru(self._fen_cache, key, (white_cp, pv_uci))

        cp = white_cp if target == chess.WHITE else -white_cp
        return cp, pv_uci

    def get_after_move_cp(
        self,
        engine,
        board: chess.Board,
        move_uci: str,
        depth: int,
        for_color: chess.Color | None = None,
    ) -> float | None:
        """Return centipawn eval after pushing ``move_uci`` on ``board`` using the cache."""
        target = board.turn if for_color is None else for_color
        key = (board.fen(), move_uci, depth)
        with self._lock:
            cached_white_cp = self._move_cache.get(key)
            if cached_white_cp is not None:
                self._move_cache.move_to_end(key)
                self._move_hits += 1
                return cached_white_cp if target == chess.WHITE else -cached_white_cp
            self._move_misses += 1

        after_board = board.copy()
        after_board.push(chess.Move.from_uci(move_uci))
        after_white_cp, _ = self.get_best(engine, after_board, depth, for_color=chess.WHITE)
        if after_white_cp is None:
            return None

        with self._lock:
            self._put_lru(self._move_cache, key, after_white_cp)
        return after_white_cp if target == chess.WHITE else -after_white_cp

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counters and current entry counts."""
        with self._lock:
            return {
                "fen_hits": self._fen_hits,
                "fen_misses": self._fen_misses,
                "move_hits": self._move_hits,
                "move_misses": self._move_misses,
                "fen_size": len(self._fen_cache),
                "move_size": len(self._move_cache),
            }

    def clear(self) -> None:
        """Clear all cached entries and reset hit/miss counters."""
        with self._lock:
            self._fen_cache.clear()
            self._move_cache.clear()
            self._fen_hits = 0
            self._fen_misses = 0
            self._move_hits = 0
            self._move_misses = 0


def _evaluate_pair(
    engine,
    board: chess.Board,
    move_uci: str,
    depth: int,
    cache: StockfishEvalCache | None = None,
):
    """Evaluate the position before and after ``move_uci`` in one shot.

    Returns ``(best_before, eval_after)`` in centipawns, both from the mover's
    perspective. Returns ``(None, None)`` only when ``move_uci`` is illegal;
    raises :class:`EngineAnalysisError` when the engine is missing or an
    analysis fails, so callers can tell the two cases apart.
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except (ValueError, TypeError):
        return None, None
    if move not in board.legal_moves:
        return None, None

    if engine is None:
        raise EngineAnalysisError("no engine provided")
    mover = board.turn
    if cache is not None:
        best_before, _ = cache.get_best(engine, board, depth, for_color=mover)
        if best_before is None:
            raise EngineAnalysisError(f"analysis failed for {board.fen()} -> {move_uci}")
        eval_after = cache.get_after_move_cp(engine, board, move_uci, depth, for_color=mover)
    else:
        best_before = best_cp(engine, board, depth, for_color=mover)
        if best_before is None:
            raise EngineAnalysisError(f"analysis failed for {board.fen()} -> {move_uci}")
        after_board = board.copy()
        after_board.push(move)
        eval_after = best_cp(engine, after_board, depth, for_color=mover)
    if eval_after is None:
        raise EngineAnalysisError(f"analysis failed for {board.fen()} -> {move_uci}")
    return best_before, eval_after


def _evaluate_with_reply(engine, board: chess.Board, move_uci: str, depth: int):
    """Evaluate ``move_uci`` accounting for Stockfish's best counterplay.

    Returns ``(best_before, eval_after_reply)`` from the mover's perspective,
    where ``eval_after_reply`` is the evaluation after Stockfish's best reply
    to ``move_uci``. This is the Stage-2 signal: it measures the realized
    outcome of the move against best defense, not just its proximity to the
    pre-move best move.

    Illegal moves return ``(None, None)``; engine failures raise
    :class:`EngineAnalysisError`. If no reply is available (e.g. the move ends
    the game), the post-move evaluation is used directly.
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except (ValueError, TypeError):
        return None, None
    if move not in board.legal_moves:
        return None, None

    if engine is None:
        raise EngineAnalysisError("no engine provided")
    mover = board.turn
    best_before = best_cp(engine, board, depth, for_color=mover)
    after_board = board.copy()
    after_board.push(move)

    if after_board.is_game_over():
        # No reply exists; the move itself decided the game. Use the raw
        # post-move evaluation (mate scores are already clamped in best_cp).
        eval_after_reply = best_cp(engine, after_board, depth, for_color=mover)
    else:
        reply = best_move(engine, after_board, depth)
        if reply is None:
            raise EngineAnalysisError(
                f"no reply analysis for {after_board.fen()} (after {move_uci})"
            )
        reply_board = after_board.copy()
        reply_board.push(chess.Move.from_uci(reply))
        eval_after_reply = best_cp(engine, reply_board, depth, for_color=mover)

    if best_before is None or eval_after_reply is None:
        raise EngineAnalysisError(f"reply analysis failed for {board.fen()} -> {move_uci}")
    return best_before, eval_after_reply


def reward_legality(fen: str, move_uci: str) -> float:
    """Stage 0 reward: pure legality check (+1 legal, -1 illegal/invalid)."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return 1.0
        return _ILLEGAL_REWARD
    except (ValueError, TypeError):
        return _ILLEGAL_REWARD


def reward_outcome(result: str, our_color: str) -> float:
    """Game-outcome reward (+1 win, +0.3 draw, -0.5 loss).

    Only used by full-game evaluation, not GRPO. Unknown or unfinished
    results (e.g. "*") score 0.0.
    """
    if result == "1-0":
        return 1.0 if our_color == "white" else -0.5
    if result == "0-1":
        return 1.0 if our_color == "black" else -0.5
    if result == "1/2-1/2":
        return 0.3
    return 0.0


def reward_move_quality(
    fen: str,
    move_uci: str,
    engine,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    max_loss: float = _MAX_QUALITY_LOSS,
    cache: StockfishEvalCache | None = None,
) -> float:
    """How close the chosen move is to Stockfish's best (in [0, 1], or -1 illegal).

    ``max_loss`` is the centipawn threshold at which reward reaches ~0.
    Raises :class:`EngineAnalysisError` on engine failure — callers must not
    conflate infrastructure failures with a bad move.
    """
    board = _board_or_raise(fen)
    move = _move_or_illegal(board, move_uci)
    if move is None:
        return _ILLEGAL_REWARD
    best, chosen = _evaluate_pair(engine, board, move_uci, depth, cache=cache)
    return _quality_from_pair(best, chosen)


def _wdl_delta_from_pair(best: float, chosen: float) -> float:
    """Expected-score delta ``E(chosen) - E(best)`` clamped to ``[-1.0, 0.0]``."""
    delta = cp_to_expected_score(chosen) - cp_to_expected_score(best)
    return max(-1.0, min(0.0, delta))


def wdl_loss_reward(
    fen: str,
    move_uci: str,
    engine,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    cache: StockfishEvalCache | None = None,
) -> float:
    """Expected-score (WDL) loss for ``move_uci`` in ``[-1.0, 0.0]`` (or ``-1.0`` illegal).

    Computes ``clamp(E(after_cp) - E(best_cp), -1.0, 0.0)`` where ``E(cp)`` is
    the Lichess expected-score logistic curve. Best moves score ``0.0``; blunders
    in equal positions incur a large negative penalty while blunders in already
    decided positions saturate gracefully.

    Raises :class:`EngineAnalysisError` on engine failure or malformed FEN.
    """
    board = _board_or_raise(fen)
    move = _move_or_illegal(board, move_uci)
    if move is None:
        return _ILLEGAL_REWARD
    best, chosen = _evaluate_pair(engine, board, move_uci, depth, cache=cache)
    return _wdl_delta_from_pair(best, chosen)


reward_wdl_loss = wdl_loss_reward


def reward_opponent_reply_delta(
    fen: str,
    move_uci: str,
    engine,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
) -> float:
    """Realized eval change after Stockfish's best reply, in [-1, 1] (or -1 illegal).

    tanh((eval_after_reply - eval_before) / 100), both from the mover's
    perspective. Raises :class:`EngineAnalysisError` on engine failure.
    """
    board = _board_or_raise(fen)
    move = _move_or_illegal(board, move_uci)
    if move is None:
        return _ILLEGAL_REWARD
    best, after_reply = _evaluate_with_reply(engine, board, move_uci, depth)
    return _delta_from_pair(best, after_reply)


def reward_stockfish_delta(
    fen_before: str,
    fen_after: str,
    engine,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
) -> float:
    """Stockfish-eval improvement between two FENs, tanh(delta / 100) in [-1, 1].

    Generic utility (used outside GRPO). Both evaluations are computed from the
    side-to-move-in-``fen_before`` perspective. Returns 0.0 on engine failure;
    raises ``ValueError`` on malformed input FENs.
    """
    board_before = chess.Board(fen_before)
    board_after = chess.Board(fen_after)
    mover = board_before.turn

    before = best_cp(engine, board_before, depth, for_color=mover)
    after = best_cp(engine, board_after, depth, for_color=mover)
    if before is None or after is None:
        return 0.0

    return math.tanh((after - before) / 100.0)


def reward_budget(output: str) -> float:
    """Penalty for reasoning that is too short or too long.

    The bounds are expressed in *word counts* (see MIN_THINK_WORDS and
    MAX_THINK_WORDS); they are a rough proxy for verbosity, not token counts.
    """
    if "<think>" not in output or "</think>" not in output:
        return 0.0
    think_text = output.split("<think>")[1].split("</think>")[0].strip()
    num_words = len(think_text.split())
    if num_words < MIN_THINK_WORDS:
        return -0.1
    if num_words > MAX_THINK_WORDS:
        return -0.05
    return 0.0


def reward_format(output: str) -> float:
    """Reward for the <think>...</think> format: +0.1 when well formed, else -0.1.

    Missing, inverted, or empty reasoning tags are penalized, not ignored:
    without a penalty the model is free to collapse to a bare move and lose
    nothing, which contradicts the reasoning-first objective of the project.
    """
    open_idx = output.find("<think>")
    if open_idx == -1:
        return -0.1
    close_idx = output.find("</think>")
    if close_idx == -1 or close_idx <= open_idx:
        return -0.1
    if not output[open_idx + len("<think>") : close_idx].strip():
        return -0.1
    return 0.1


def _board_or_raise(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except (ValueError, TypeError) as e:
        raise EngineAnalysisError(f"malformed FEN: {fen[:100]}") from e


def _move_or_illegal(board: chess.Board, move_uci: str) -> chess.Move | None:
    """Parse ``move_uci``; None marks an illegal/unparseable (model-fault) move."""
    try:
        move = chess.Move.from_uci(move_uci)
    except (ValueError, TypeError):
        return None
    return move if move in board.legal_moves else None


def _quality_from_pair(best: float | None, chosen: float | None) -> float:
    """Move-quality score (in [0, 1]) from a shared (best, chosen) eval pair."""
    if best is None or chosen is None:
        return 0.0
    loss = max(0.0, best - chosen)
    return max(0.0, 1.0 - (loss / _MAX_QUALITY_LOSS))


def _delta_from_pair(best: float | None, after_reply: float | None) -> float:
    """Reply-aware delta score (in [-1, 1]) from a shared (best, reply) pair."""
    if best is None or after_reply is None:
        return 0.0
    return math.tanh((after_reply - best) / 100.0)


def combined_reward(
    fen: str,
    move_uci: str,
    output: str,
    stage: int | str = 0,
    engine=None,
    game_result: str | None = None,
    our_color: str = "white",
    move_quality_depth: int = DEFAULT_STOCKFISH_DEPTH,
    reward_mode: str = "legacy",
    cache: StockfishEvalCache | None = None,
) -> dict:
    """Combined reward across all stages and reward modes.

    Each component already carries its intended magnitude (format +-0.1,
    budget -0.1/0, quality [0, 1], delta tanh, wdl_quality [0, 2]), so the
    total is a plain sum. Per-stage ranges (for a legal move):

    - Stage 0: [0.8, 1.1]  (legality + format + budget)
    - Stage 1: [0.8, 2.1]  (+ move quality)
    - Stage 2: [-0.9, 3.1] (+ opponent-reply delta; note it is usually negative
      because Stockfish's best reply claws back some of the mover's gain)
    - WDL mode (``reward_mode="wdl_loss"`` or ``stage in (3, "wdl", "wdl_loss")``):
      [0.8, 3.1] (``r_legal + r_format + r_budget + 2.0 * (1.0 + wdl_delta)``)

    GRPO normalizes rewards within a group before applying the policy update,
    so the differing absolute ranges across stages are not a correctness issue;
    they are documented here to keep the reward design explicit.

    Cost: Stage 1 / WDL costs two Stockfish analyses per completion (amortized
    via ``cache`` when provided); Stage 2 adds two more (the reply search and
    the post-reply evaluation) and *reuses* the Stage-1 position evaluation
    rather than recomputing it.

    - Illegal moves short-circuit to ``_ILLEGAL_REWARD``.
    - Stages >= 1 and WDL mode require an engine; if missing the run is a
      config error, not a silent downgrade.
    - :class:`EngineAnalysisError` **propagates** on engine failure. It is the
      caller's job (see ``train_grpo.make_reward_func``) to treat that as
      neutral-and-counted rather than as a bad move.
    """
    if reward_mode not in ("legacy", "wdl_loss", "sharp_wdl"):
        raise ValueError(
            f"Unsupported reward_mode {reward_mode!r}; expected 'legacy', 'wdl_loss', or 'sharp_wdl'."
        )
    is_sharp_wdl = reward_mode == "sharp_wdl"
    is_wdl = (stage in (3, "wdl", "wdl_loss")) or (reward_mode == "wdl_loss")
    if not is_wdl and not is_sharp_wdl and stage not in (0, 1, 2):
        raise ValueError(f"Unsupported stage {stage!r}; expected 0, 1, 2, 3, 'wdl', or 'wdl_loss'.")

    components: dict[str, float] = {}

    r_legal = reward_legality(fen, move_uci)
    if is_sharp_wdl:
        if r_legal < 0:
            components["legality"] = _SHARP_WDL_ILLEGAL_REWARD
            return {"total": _SHARP_WDL_ILLEGAL_REWARD, "components": components, "legal": False}
        if engine is None:
            raise ValueError(
                f"Stage {stage} (reward_mode={reward_mode!r}) reward requires a Stockfish engine, "
                "but engine=None. This is a configuration error, not a runnable state."
            )
        board = chess.Board(fen)
        mover = board.turn
        if cache is not None:
            best_before, best_mv = cache.get_best(
                engine, board, move_quality_depth, for_color=mover
            )
            if best_before is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            eval_after = cache.get_after_move_cp(
                engine, board, move_uci, move_quality_depth, for_color=mover
            )
        else:
            try:
                info_before = engine.analyse(board, chess.engine.Limit(depth=move_quality_depth))
            except _CACHE_ENGINE_ERRORS as exc:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}") from exc
            white_cp = _white_cp_from_info(info_before)
            if white_cp is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            best_before = white_cp if mover == chess.WHITE else -white_cp
            pv = info_before.get("pv") if isinstance(info_before, dict) else None
            best_mv = pv[0].uci() if pv else None
            after_board = board.copy()
            after_board.push(chess.Move.from_uci(move_uci))
            eval_after = best_cp(engine, after_board, move_quality_depth, for_color=mover)
        if eval_after is None:
            raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")

        wdl_loss = max(0.0, cp_to_expected_score(best_before) - cp_to_expected_score(eval_after))
        r_sharp_quality = (
            _SHARP_WDL_QUALITY_WEIGHT * max(0.0, 1.0 - (wdl_loss / _SHARP_WDL_THRESHOLD)) ** 2
        )
        cp_loss = max(0.0, best_before - eval_after)
        is_near_best = (cp_loss <= _SHARP_WDL_NEAR_BEST_CP) or (
            best_mv is not None and move_uci.strip().lower() == best_mv.strip().lower()
        )
        r_exact = _SHARP_WDL_EXACT_BONUS if is_near_best else 0.0
        r_format = _SHARP_WDL_FORMAT_BONUS if reward_format(output) > 0 else 0.0
        r_budget = reward_budget(output)

        components["legality"] = 0.0
        components["format"] = r_format
        components["sharp_wdl_quality"] = r_sharp_quality
        components["exact_bonus"] = r_exact
        if r_budget:
            components["budget"] = r_budget
        total = r_format + r_budget + r_sharp_quality + r_exact
        if game_result:
            r_outcome = reward_outcome(game_result, our_color)
            components["outcome"] = r_outcome
            total += r_outcome
        return {"total": total, "components": components, "legal": True}

    components["legality"] = r_legal
    if r_legal < 0:
        return {"total": r_legal, "components": components, "legal": False}

    needs_engine = is_wdl or (isinstance(stage, int) and stage >= 1)
    if needs_engine and engine is None:
        raise ValueError(
            f"Stage {stage} (reward_mode={reward_mode!r}) reward requires a Stockfish engine, "
            "but engine=None. This is a configuration error, not a runnable state."
        )

    r_format = reward_format(output)
    r_budget = reward_budget(output)
    components["format"] = r_format
    if r_budget:
        components["budget"] = r_budget

    total = r_legal + r_format + r_budget

    if is_wdl and engine is not None:
        board = chess.Board(fen)
        mover = board.turn
        if cache is not None:
            best_before, _ = cache.get_best(engine, board, move_quality_depth, for_color=mover)
            if best_before is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            eval_after = cache.get_after_move_cp(
                engine, board, move_uci, move_quality_depth, for_color=mover
            )
        else:
            best_before = best_cp(engine, board, move_quality_depth, for_color=mover)
            if best_before is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            after_board = board.copy()
            after_board.push(chess.Move.from_uci(move_uci))
            eval_after = best_cp(engine, after_board, move_quality_depth, for_color=mover)
        if eval_after is None:
            raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")

        wdl_delta = _wdl_delta_from_pair(best_before, eval_after)
        r_wdl_quality = _WDL_QUALITY_WEIGHT * (1.0 + wdl_delta)
        components["wdl_quality"] = r_wdl_quality
        total += r_wdl_quality
    elif isinstance(stage, int) and stage >= 1 and engine is not None:
        # Legality was already established above, so the move parses and is
        # legal; the only failure mode left is infrastructure.
        board = chess.Board(fen)
        mover = board.turn
        after_board = board.copy()
        after_board.push(chess.Move.from_uci(move_uci))
        if cache is not None:
            best_before, _ = cache.get_best(engine, board, move_quality_depth, for_color=mover)
            if best_before is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            eval_after = cache.get_after_move_cp(
                engine, board, move_uci, move_quality_depth, for_color=mover
            )
        else:
            best_before = best_cp(engine, board, move_quality_depth, for_color=mover)
            if best_before is None:
                raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")
            eval_after = best_cp(engine, after_board, move_quality_depth, for_color=mover)
        if eval_after is None:
            raise EngineAnalysisError(f"analysis failed for {fen} -> {move_uci}")

        r_quality = _quality_from_pair(best_before, eval_after)
        components["move_quality"] = r_quality
        total += r_quality

        if stage >= 2:
            after_reply = _best_reply_eval(
                engine, after_board, mover, move_quality_depth, cache=cache
            )
            r_delta = _delta_from_pair(best_before, after_reply)
            components["opponent_reply_delta"] = r_delta
            total += r_delta

    if game_result:
        r_outcome = reward_outcome(game_result, our_color)
        components["outcome"] = r_outcome
        total += r_outcome

    return {"total": total, "components": components, "legal": True}


def _best_reply_eval(
    engine,
    after_board: chess.Board,
    mover: chess.Color,
    depth: int,
    cache: StockfishEvalCache | None = None,
) -> float:
    """Evaluation after Stockfish's best reply, from ``mover``'s perspective."""
    if after_board.is_game_over():
        # No reply exists; the move itself decided the game.
        if cache is not None:
            eval_after_reply, _ = cache.get_best(engine, after_board, depth, for_color=mover)
        else:
            eval_after_reply = best_cp(engine, after_board, depth, for_color=mover)
    else:
        if cache is not None:
            _, reply = cache.get_best(engine, after_board, depth)
        else:
            reply = best_move(engine, after_board, depth)
        if reply is None:
            raise EngineAnalysisError(f"no reply analysis for {after_board.fen()}")
        if cache is not None:
            eval_after_reply = cache.get_after_move_cp(
                engine, after_board, reply, depth, for_color=mover
            )
        else:
            reply_board = after_board.copy()
            reply_board.push(chess.Move.from_uci(reply))
            eval_after_reply = best_cp(engine, reply_board, depth, for_color=mover)

    if eval_after_reply is None:
        raise EngineAnalysisError(f"reply evaluation failed for {after_board.fen()}")
    return eval_after_reply
