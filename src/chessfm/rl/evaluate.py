#!/usr/bin/env python3
"""
Evaluation script for ChessFM.

Runs simulated games between a fine-tuned ChessFM checkpoint and Stockfish (or
a random bot) to measure legal-move rate, format adherence, win/loss/draw, and
an estimated Elo rating.

The evaluator is the project's primary measurement harness. Games are recorded
as PGNs under ``results/pgns/``, and every run writes a JSON report (with
reproducibility metadata) under ``results/``.

USAGE:
    python -m chessfm.rl.evaluate --model checkpoints/stage0_final --games 50
    python -m chessfm.rl.evaluate --model Qwen/Qwen2.5-3B-Instruct --games 20 --stockfish-level 1
    python -m chessfm.rl.evaluate --model checkpoints/final --openings data_generation/openings.txt --games 100
"""

import argparse
import contextlib
import json
import logging
import math
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OPPONENT_DEPTH,
    MAX_GAME_MOVES,
    PGNS_DIR,
    RESULTS_DIR,
    STOCKFISH_LEVEL_ELO,
    build_chat_prompt,
)
from chessfm.fens import read_fens
from chessfm.load_model import load_model
from chessfm.rl.chess_env import ChessEnv
from chessfm.rl.move_utils import parse_move_from_output
from chessfm.training.manifest import _get_git_sha

logger = logging.getLogger("chessfm.evaluate")

MIN_STOCKFISH_ELO = 1320
MATCH_ENGINE_MAX_RESTARTS = 12


def git_revision() -> str:
    """Return the current git revision for reproducibility metadata."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def estimated_elo(wins: int, losses: int, draws: int, opponent_elo: int) -> float:
    """Estimate model Elo from results via the standard expected-score model.

    E = 1 / (1 + 10^((Elo_opp - Elo_model)/400)), solved for Elo_model. The
    score is clamped to avoid +/-inf at perfect or zero scores.
    """
    games = wins + losses + draws
    if games == 0:
        return 0.0
    score = (wins + 0.5 * draws) / games
    score = min(0.999, max(0.001, score))
    return opponent_elo + 400 * math.log10(score / (1 - score))


def wilson_ci(wins: int, losses: int, draws: int, z: float = 1.96) -> tuple[float, float]:
    """Return the 95% Wilson score confidence interval ``(low, high)`` for win-score."""
    games = wins + losses + draws
    if games == 0:
        return (0.0, 0.0)
    p = (wins + 0.5 * draws) / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / games + z2 / (4.0 * games * games))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_openings(path: Path | None) -> list[str]:
    """Load opening FENs (one per line). Returns [] if no file given."""
    if path is None or not path.exists():
        return []
    try:
        return read_fens(path)
    except FileNotFoundError:
        return []


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    decoding_mode: str = "strict",
    fen: str | None = None,
    require_think_close: bool = True,
) -> str:
    """Generate a completion for ``prompt`` and return ONLY the newly generated text.

    ``max_new_tokens`` defaults to the training-time completion budget
    (``config.DEFAULT_MAX_NEW_TOKENS``) so the model is never truncated to a
    tighter budget at inference than it was trained on.

    When ``decoding_mode == "constrained"``, attaches a
    :class:`~chessfm.rl.constrained_decoding.LegalMoveLogitsProcessor` for ``fen``
    so only legal UCI moves can be emitted after ``</think>`` (or immediately
    when ``require_think_close=False`` for Think-OFF ablations).
    """
    if decoding_mode not in ("strict", "constrained"):
        raise ValueError(
            f"Unsupported decoding_mode {decoding_mode!r}; expected 'strict' or 'constrained'."
        )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    input_len = inputs.input_ids.shape[1]
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if decoding_mode == "constrained":
        if fen is None:
            raise ValueError("fen must be provided when decoding_mode='constrained'.")
        from chessfm.rl.constrained_decoding import LegalMoveLogitsProcessor

        proc = LegalMoveLogitsProcessor(
            fen,
            tokenizer,
            prompt_length=input_len,
            require_think_close=require_think_close,
        )
        try:
            from transformers import LogitsProcessorList

            gen_kwargs["logits_processor"] = LogitsProcessorList([proc])
        except ImportError:
            gen_kwargs["logits_processor"] = [proc]
    outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


class BatchedLegalMoveLogitsProcessor:
    """Per-row legal-move logits processor for left-padded batched generation.

    Composes one :class:`~chessfm.rl.constrained_decoding.LegalMoveLogitsProcessor`
    per row in ``fens`` and masks each row independently so there is zero
    cross-game leakage and NaN-free EOS fallback when rows finish at different
    steps.
    """

    def __init__(
        self,
        fens: list[str],
        tokenizer,
        prompt_length: int = 0,
        require_think_close: bool = True,
    ) -> None:
        from chessfm.rl.constrained_decoding import LegalMoveLogitsProcessor

        self.fens = list(fens)
        self.prompt_length = prompt_length
        self.require_think_close = require_think_close
        self.processors = [
            LegalMoveLogitsProcessor(
                fen,
                tokenizer,
                prompt_length=prompt_length,
                require_think_close=require_think_close,
            )
            for fen in self.fens
        ]

    def __call__(self, input_ids, scores):
        batch_size = scores.shape[0]
        if batch_size != len(self.processors):
            raise ValueError(
                f"Batch size mismatch: scores has {batch_size} rows, "
                f"expected {len(self.processors)}."
            )
        masked_scores = scores.clone()
        for row, proc in enumerate(self.processors):
            row_out = proc(input_ids[row : row + 1], scores[row : row + 1])
            masked_scores[row] = row_out[0]
        return masked_scores


def generate_completions_batched(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    decoding_mode: str = "strict",
    fens: list[str] | None = None,
    require_think_close: bool = True,
) -> list[str]:
    """Generate completions for a batch of prompts using left-padded decoding.

    Saves and restores ``tokenizer.padding_side``, ``tokenizer.pad_token_id``,
    and ``tokenizer.pad_token`` in a ``try ... finally`` block so caller state
    is never mutated. When ``decoding_mode == "constrained"``, attaches a
    :class:`BatchedLegalMoveLogitsProcessor` aligned to ``padded_prompt_len``.
    """
    if decoding_mode not in ("strict", "constrained"):
        raise ValueError(
            f"Unsupported decoding_mode {decoding_mode!r}; expected 'strict' or 'constrained'."
        )
    if not prompts:
        return []
    if decoding_mode == "constrained":
        if fens is None or len(fens) != len(prompts):
            raise ValueError(
                "fens must be provided with the same length as prompts when "
                "decoding_mode='constrained'."
            )

    has_padding_side = hasattr(tokenizer, "padding_side")
    orig_padding_side = getattr(tokenizer, "padding_side", None)
    has_pad_token_id = hasattr(tokenizer, "pad_token_id")
    orig_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    has_pad_token = hasattr(tokenizer, "pad_token")
    orig_pad_token = getattr(tokenizer, "pad_token", None)

    try:
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                tokenizer.pad_token_id = eos_id
            eos_tok = getattr(tokenizer, "eos_token", None)
            if getattr(tokenizer, "pad_token", None) is None and eos_tok is not None:
                tokenizer.pad_token = eos_tok

        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        input_ids = (
            inputs["input_ids"]
            if isinstance(inputs, dict) or hasattr(inputs, "__getitem__")
            else inputs.input_ids
        )
        padded_prompt_len = int(input_ids.shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id

        if decoding_mode == "constrained":
            assert fens is not None
            proc = BatchedLegalMoveLogitsProcessor(
                fens,
                tokenizer,
                prompt_length=padded_prompt_len,
                require_think_close=require_think_close,
            )
            try:
                from transformers import LogitsProcessorList

                gen_kwargs["logits_processor"] = LogitsProcessorList([proc])
            except ImportError:
                gen_kwargs["logits_processor"] = [proc]

        try:
            import torch

            no_grad_ctx = torch.no_grad()
        except ImportError:
            no_grad_ctx = contextlib.nullcontext()

        with no_grad_ctx:
            outputs = model.generate(**inputs, **gen_kwargs)

        completions: list[str] = []
        for row in range(len(prompts)):
            generated = outputs[row][padded_prompt_len:]
            completions.append(tokenizer.decode(generated, skip_special_tokens=True))
        return completions
    finally:
        if has_padding_side:
            tokenizer.padding_side = orig_padding_side
        elif hasattr(tokenizer, "padding_side"):
            delattr(tokenizer, "padding_side")

        if has_pad_token_id:
            tokenizer.pad_token_id = orig_pad_token_id
        elif hasattr(tokenizer, "pad_token_id"):
            delattr(tokenizer, "pad_token_id")

        if has_pad_token:
            tokenizer.pad_token = orig_pad_token
        elif hasattr(tokenizer, "pad_token"):
            delattr(tokenizer, "pad_token")


def _count_result(stats: dict, result: str, model_color: str) -> None:
    """Tally a finished game from the model's perspective, regardless of color."""
    if result == "1-0":
        if model_color == "white":
            stats["wins"] += 1
        else:
            stats["losses"] += 1
    elif result == "0-1":
        if model_color == "black":
            stats["wins"] += 1
        else:
            stats["losses"] += 1
    else:
        stats["draws"] += 1


def _sample_maia_move(
    maia_sidecar: MaiaSidecar,
    fen: str,
    maia_elo: int,
    fallback_legal: list[str],
) -> str:
    """Sample a legal opponent move from ``maia_sidecar`` at ``maia_elo``."""
    pred = maia_sidecar.predict_human_moves(fen, student_elo=maia_elo)
    move_probs: dict[str, float] = pred.get("move_probs", {})
    if move_probs:
        moves = list(move_probs.keys())
        weights = list(move_probs.values())
        return random.choices(moves, weights=weights, k=1)[0]
    return random.choice(fallback_legal)


def _play_engine_move(
    engine,
    board: chess.Board,
    opponent_depth: int,
    max_retries: int = MATCH_ENGINE_MAX_RESTARTS,
) -> str:
    """Request an opponent move from ``engine`` with bounded crash recovery.

    Stockfish 16 on Linux can occasionally terminate with ``SIGFPE`` (exit code
    ``-8``) in consecutive bursts under ``UCI_LimitStrength`` / ``UCI_Elo``.
    ``ResilientEngine.play`` handles restarts internally up to
    ``_max_restarts``; this helper provides a bounded safety net for callers
    that invoke :func:`play_game` or :func:`play_games_batched` directly with
    an engine whose restart budget has not been raised.
    """
    retries = max(0, int(max_retries))
    attempt = 0
    while True:
        try:
            result = engine.play(board, chess.engine.Limit(depth=opponent_depth))
            return result.move.uci()
        except (chess.engine.EngineTerminatedError, BrokenPipeError) as exc:
            if attempt >= retries:
                raise
            attempt += 1
            logger.warning(
                "Opponent engine terminated during play (%s); retrying (%d/%d).",
                exc,
                attempt,
                retries,
            )
            restart_fn = getattr(engine, "_restart", None)
            if callable(restart_fn):
                restart_fn()


def play_game(
    model,
    tokenizer,
    env: ChessEnv,
    model_color: str,
    use_random_opponent: bool = False,
    opponent_depth: int = DEFAULT_OPPONENT_DEPTH,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    maia_sidecar: MaiaSidecar | None = None,
    maia_elo: int | None = None,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
) -> tuple:
    """Play one game between the model and the opponent in ``env``.

    Returns ``(game, stats)`` where ``game`` is a ``chess.pgn.Game`` (move tree
    populated, headers left for the caller) and ``stats`` is a dict of
    per-game counters. Win/loss/draw are always from the model's perspective.

    The Stockfish opponent searches to a fixed depth (not wall-clock time) so
    its strength is reproducible across runs and machines. When ``maia_sidecar``
    and ``maia_elo`` are supplied, the opponent samples moves from Maia's
    Elo-calibrated move distribution.

    A model forfeit on an illegal move is recorded truthfully in ``stats``.
    """
    stats = {
        "total_moves": 0,
        "legal_moves": 0,
        "formatted_outputs": 0,
        "illegal_by_model": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "result": "*",
    }
    game = chess.pgn.Game()
    if env.get_fen() != chess.STARTING_FEN:
        game.setup(env.board.copy())
    node = game
    done = False

    while not done:
        # Refresh the FEN every ply so the model is always prompted with the
        # actual current board, never a stale one.
        fen = env.get_fen()
        if env.get_turn() == model_color:
            prompt = build_chat_prompt(
                fen,
                tokenizer=tokenizer,
                include_ascii=include_ascii,
                enable_thinking=enable_thinking,
                include_piece_list=include_piece_list,
                include_legal_moves=include_legal_moves,
            )
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                decoding_mode=decoding_mode,
                fen=fen,
                require_think_close=enable_thinking,
            )
            move_uci = parse_move_from_output(completion) or "xxxx"
            stats["total_moves"] += 1

            if "<think>" in completion and "</think>" in completion:
                stats["formatted_outputs"] += 1

            step_res = env.step(move_uci)
            if step_res.legal:
                stats["legal_moves"] += 1
                node = node.add_variation(chess.Move.from_uci(move_uci))
            else:
                # Illegal move: the model loses by forfeit.
                stats["illegal_by_model"] += 1
                stats["losses"] += 1
                stats["result"] = "0-1" if model_color == "white" else "1-0"
                done = True
                break

            if step_res.done:
                done = True
                stats["result"] = step_res.info.get("result", "")
                _count_result(stats, stats["result"], model_color)
        else:
            # Opponent: Maia sidecar, Stockfish, or random.
            if maia_sidecar is not None and maia_elo is not None:
                move_uci = _sample_maia_move(maia_sidecar, fen, maia_elo, env.get_legal_moves())
            elif env.engine and not use_random_opponent:
                move_uci = _play_engine_move(env.engine, env.board, opponent_depth)
            else:
                move_uci = random.choice(env.get_legal_moves())

            step_res = env.step(move_uci)
            if step_res.legal:
                node = node.add_variation(chess.Move.from_uci(move_uci))
            if step_res.done:
                done = True
                stats["result"] = step_res.info.get("result", "")
                _count_result(stats, stats["result"], model_color)

    return game, stats


@dataclass
class _BatchedGameState:
    """Per-game mutable state for rolling batched match evaluation."""

    index: int
    model_color: str
    board: chess.Board
    game: chess.pgn.Game
    node: Any
    stats: dict
    move_count: int = 0
    max_moves: int = MAX_GAME_MOVES
    done: bool = False


def _new_game_stats() -> dict:
    """Return a fresh per-game statistics dictionary."""
    return {
        "total_moves": 0,
        "legal_moves": 0,
        "formatted_outputs": 0,
        "illegal_by_model": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "result": "*",
    }


def _step_board(state: _BatchedGameState, move_uci: str) -> tuple[bool, bool, dict]:
    """Advance ``state.board`` by ``move_uci`` mirroring :meth:`ChessEnv.step`."""
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return False, True, {"error": "invalid_uci", "move": move_uci}

    if move not in state.board.legal_moves:
        return False, True, {"error": "illegal_move", "move": move_uci}

    state.board.push(move)
    state.move_count += 1

    if state.board.is_game_over(claim_draw=True):
        return (
            True,
            True,
            {"result": state.board.result(claim_draw=True), "moves": state.move_count},
        )

    if state.move_count >= state.max_moves:
        return True, True, {"result": "max_moves", "moves": state.move_count}

    return True, False, {"moves": state.move_count}


def _play_opponent_turn(
    state: _BatchedGameState,
    *,
    engine,
    use_random_opponent: bool,
    opponent_depth: int,
    maia_sidecar: MaiaSidecar | None,
    maia_elo: int | None,
) -> None:
    """Execute one opponent ply on ``state`` using Maia, Stockfish, or Random."""
    fen = state.board.fen()
    legal_ucis = [m.uci() for m in state.board.legal_moves]
    if maia_sidecar is not None and maia_elo is not None:
        move_uci = _sample_maia_move(maia_sidecar, fen, maia_elo, legal_ucis)
    elif engine is not None and not use_random_opponent:
        move_uci = _play_engine_move(engine, state.board, opponent_depth)
    else:
        move_uci = random.choice(legal_ucis)

    legal, done, info = _step_board(state, move_uci)
    if legal:
        state.node = state.node.add_variation(chess.Move.from_uci(move_uci))
    if done:
        state.done = True
        state.stats["result"] = info.get("result", "")
        _count_result(state.stats, state.stats["result"], state.model_color)


def play_games_batched(
    model,
    tokenizer,
    game_specs: list[tuple[int, str, str | None]],
    engine,
    batch_size: int = 16,
    use_random_opponent: bool = False,
    opponent_depth: int = DEFAULT_OPPONENT_DEPTH,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    maia_sidecar: MaiaSidecar | None = None,
    maia_elo: int | None = None,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
) -> list[tuple[chess.pgn.Game, dict]]:
    """Play multiple games concurrently using a rolling active pool.

    Maintains up to ``batch_size`` active games sharing a single ``engine``
    instance. As soon as any game finishes, a pending game refills its slot,
    opponent turns advance on CPU, and all active model-turn boards are
    evaluated in a single :func:`generate_completions_batched` call.

    Returns ``[(game, stats), ...]`` ordered strictly by the original game
    indices ``0..len(game_specs) - 1``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total_games = len(game_specs)
    pending: deque[tuple[int, str, str | None]] = deque(game_specs)
    active: list[_BatchedGameState] = []
    completed: dict[int, tuple[chess.pgn.Game, dict]] = {}

    def _finalize(state: _BatchedGameState) -> None:
        completed[state.index] = (state.game, state.stats)
        forfeit_note = (
            f" (model forfeited via illegal move: {state.stats['result']})"
            if state.stats["illegal_by_model"]
            else ""
        )
        print(
            f"Completed game {state.index + 1}/{total_games} ({state.model_color}): "
            f"{state.stats['result']}{forfeit_note}",
            flush=True,
        )

    while active or pending:
        # Refill the active pool and advance all opponent-turn boards until
        # every active board is waiting on the model.
        while True:
            while len(active) < batch_size and pending:
                idx, g_color, start_fen = pending.popleft()
                board = chess.Board(start_fen) if start_fen else chess.Board()
                game = chess.pgn.Game()
                if board.fen() != chess.STARTING_FEN:
                    game.setup(board.copy())
                state = _BatchedGameState(
                    index=idx,
                    model_color=g_color,
                    board=board,
                    game=game,
                    node=game,
                    stats=_new_game_stats(),
                )
                if board.is_game_over(claim_draw=True):
                    state.done = True
                    state.stats["result"] = board.result(claim_draw=True)
                    _count_result(state.stats, state.stats["result"], state.model_color)
                    _finalize(state)
                else:
                    active.append(state)

            still_active: list[_BatchedGameState] = []
            finished_any = False
            for state in active:
                while not state.done:
                    turn_color = "white" if state.board.turn == chess.WHITE else "black"
                    if turn_color == state.model_color:
                        break
                    _play_opponent_turn(
                        state,
                        engine=engine,
                        use_random_opponent=use_random_opponent,
                        opponent_depth=opponent_depth,
                        maia_sidecar=maia_sidecar,
                        maia_elo=maia_elo,
                    )
                if state.done:
                    _finalize(state)
                    finished_any = True
                else:
                    still_active.append(state)

            active = still_active
            if not finished_any or not pending:
                break

        if not active:
            break

        fens = [state.board.fen() for state in active]
        prompts = [
            build_chat_prompt(
                fen,
                tokenizer=tokenizer,
                include_ascii=include_ascii,
                enable_thinking=enable_thinking,
                include_piece_list=include_piece_list,
                include_legal_moves=include_legal_moves,
            )
            for fen in fens
        ]
        completions = generate_completions_batched(
            model,
            tokenizer,
            prompts,
            decoding_mode=decoding_mode,
            fens=fens if decoding_mode == "constrained" else None,
            require_think_close=enable_thinking,
        )

        next_active: list[_BatchedGameState] = []
        for state, completion in zip(active, completions, strict=True):
            move_uci = parse_move_from_output(completion) or "xxxx"
            state.stats["total_moves"] += 1
            if "<think>" in completion and "</think>" in completion:
                state.stats["formatted_outputs"] += 1

            legal, done, info = _step_board(state, move_uci)
            if legal:
                state.stats["legal_moves"] += 1
                state.node = state.node.add_variation(chess.Move.from_uci(move_uci))
                if done:
                    state.done = True
                    state.stats["result"] = info.get("result", "")
                    _count_result(state.stats, state.stats["result"], state.model_color)
            else:
                state.stats["illegal_by_model"] += 1
                state.stats["losses"] += 1
                state.stats["result"] = "0-1" if state.model_color == "white" else "1-0"
                state.done = True

            if state.done:
                _finalize(state)
            else:
                next_active.append(state)

        active = next_active

    return [completed[idx] for idx, _, _ in game_specs]


def write_pgn(games: list, path: Path) -> None:
    """Serialize a list of games to a PGN file."""
    PGNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for game in games:
            f.write(str(game) + "\n\n")


def write_report(results: dict, path: Path) -> None:
    """Write the results report as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _color_split_summary(bucket: dict[str, int]) -> dict:
    """Format a per-color W-D-L bucket with score."""
    games = bucket["wins"] + bucket["losses"] + bucket["draws"]
    score = (bucket["wins"] + 0.5 * bucket["draws"]) / games if games > 0 else 0.0
    return {
        "games": games,
        "wins": bucket["wins"],
        "draws": bucket["draws"],
        "losses": bucket["losses"],
        "score": round(score, 4),
    }


def _detect_hardware() -> str:
    """Return CUDA device name if available, otherwise ``'cpu'``."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def evaluate_model(
    model,
    tokenizer,
    model_name: str,
    num_games: int = 50,
    stockfish_level: int = 1,
    model_color: str = "white",
    alternate_colors: bool = True,
    opponent_depth: int = DEFAULT_OPPONENT_DEPTH,
    use_random_opponent: bool = False,
    seed: int = 42,
    openings: Path | None = None,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    maia_elo: int | None = None,
    manifest_output: Path | None = None,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    stockfish_elo: int | None = None,
    batch_size: int = 1,
) -> dict:
    """Run ``num_games`` games of ``model`` (with ``tokenizer``) and write reports.

    The model is expected to already be loaded (see ``main``); keeping the
    model I/O out of this function makes the whole evaluation loop testable
    with lightweight fakes. ``model_name`` is only used for metadata/headers.

    Colors alternate between games by default so the win/loss/Elo estimates do
    not over-weight a single color. When ``openings`` is supplied with
    ``alternate_colors=True``, each opening FEN is played twice in a row (once
    as White, once as Black) via ``opening_fens[(g // 2) % len(opening_fens)]``.

    The Stockfish strength setting (``UCI_Elo`` or ``Skill Level``) MUST be
    applied successfully when playing Stockfish: a silently unapplied strength
    setting would make every reported number meaningless, so a configuration
    failure aborts the run.
    """
    if decoding_mode not in ("strict", "constrained"):
        raise ValueError(
            f"Unsupported decoding_mode {decoding_mode!r}; expected 'strict' or 'constrained'."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if maia_elo is not None and maia_elo <= 0:
        raise ValueError(f"maia_elo must be a positive integer, got {maia_elo}")
    if stockfish_elo is not None and stockfish_elo < MIN_STOCKFISH_ELO:
        raise ValueError(
            f"stockfish_elo must be >= {MIN_STOCKFISH_ELO} "
            f"(Stockfish 16 UCI_Elo minimum), got {stockfish_elo}"
        )
    explicit_opponents = sum(
        [maia_elo is not None, bool(use_random_opponent), stockfish_elo is not None]
    )
    if explicit_opponents > 1:
        raise ValueError(
            "Specify at most one opponent override among maia_elo, "
            "use_random_opponent, and stockfish_elo."
        )

    eval_start = time.monotonic()
    random.seed(seed)

    env = ChessEnv()
    if (
        env.engine is not None
        and hasattr(env.engine, "_max_restarts")
        and isinstance(env.engine._max_restarts, int)
        and not isinstance(env.engine._max_restarts, bool)
    ):
        env.engine._max_restarts = max(env.engine._max_restarts, MATCH_ENGINE_MAX_RESTARTS)
    try:
        maia_sidecar: MaiaSidecar | None = None
        if maia_elo is not None:
            maia_sidecar = MaiaSidecar(use_neural=True, engine=env.engine)
            opponent_backend = maia_sidecar.backend
            opponent = f"Maia-{maia_elo} ({opponent_backend})"
        elif use_random_opponent:
            opponent_backend = "random"
            opponent = "Random bot"
        elif stockfish_elo is not None:
            opponent_backend = "stockfish"
            opponent = f"Stockfish Elo {stockfish_elo} (depth {opponent_depth})"
            if env.engine is None:
                raise SystemExit(
                    "Stockfish engine not found; it is required for evaluation "
                    "unless --random-opponent or --maia-elo is passed."
                )
            try:
                env.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(stockfish_elo)})
            except chess.engine.EngineError as e:
                raise SystemExit(f"Stockfish rejected UCI_Elo {stockfish_elo}: {e}") from e
        else:
            opponent_backend = "stockfish"
            opponent = f"Stockfish L{stockfish_level} (depth {opponent_depth})"
            if env.engine is None:
                raise SystemExit(
                    "Stockfish engine not found; it is required for evaluation "
                    "unless --random-opponent or --maia-elo is passed."
                )
            try:
                env.engine.configure({"Skill Level": stockfish_level})
            except chess.engine.EngineError as e:
                raise SystemExit(f"Stockfish rejected Skill Level {stockfish_level}: {e}") from e

        print("=" * 60, flush=True)
        print(f"ChessFM Evaluation: {model_name} vs {opponent}", flush=True)
        color_desc = f"Alternating (starting {model_color})" if alternate_colors else model_color
        print(f"Model color: {color_desc}", flush=True)
        print(f"Decoding mode: {decoding_mode} | Batch size: {batch_size}", flush=True)
        print(
            f"Include ASCII: {include_ascii} | Piece list: {include_piece_list} | "
            f"Legal moves: {include_legal_moves} | Enable thinking: {enable_thinking}",
            flush=True,
        )
        print(f"Seed: {seed}", flush=True)
        print("=" * 60, flush=True)

        opening_fens = load_openings(openings)
        if opening_fens:
            print(f"Using {len(opening_fens)} opening positions from {openings}", flush=True)

        total_moves = 0
        legal_moves = 0
        formatted_outputs = 0
        wins = 0
        losses = 0
        draws = 0
        illegal_by_model = 0
        color_buckets = {
            "white": {"wins": 0, "losses": 0, "draws": 0},
            "black": {"wins": 0, "losses": 0, "draws": 0},
        }
        games_list = []

        other_color = "black" if model_color == "white" else "white"
        game_specs: list[tuple[int, str, str | None]] = []
        for g in range(num_games):
            if alternate_colors:
                game_color = model_color if g % 2 == 0 else other_color
            else:
                game_color = model_color
            if opening_fens:
                opening_idx = (
                    (g // 2) % len(opening_fens) if alternate_colors else g % len(opening_fens)
                )
                start_fen: str | None = opening_fens[opening_idx]
            else:
                start_fen = None
            game_specs.append((g, game_color, start_fen))

        if batch_size == 1:
            played_games: list[tuple[chess.pgn.Game, dict]] = []
            for g, game_color, start_fen in game_specs:
                if start_fen is not None:
                    env.reset(start_fen)
                else:
                    env.reset()
                print(f"\nGame {g + 1}/{num_games} ({game_color})", flush=True)

                game, stats = play_game(
                    model,
                    tokenizer,
                    env,
                    game_color,
                    use_random_opponent=use_random_opponent,
                    opponent_depth=opponent_depth,
                    decoding_mode=decoding_mode,
                    include_ascii=include_ascii,
                    enable_thinking=enable_thinking,
                    maia_sidecar=maia_sidecar,
                    maia_elo=maia_elo,
                    include_piece_list=include_piece_list,
                    include_legal_moves=include_legal_moves,
                )
                if stats["illegal_by_model"]:
                    print(
                        f"   (model forfeited via illegal move: {stats['result']})",
                        flush=True,
                    )
                played_games.append((game, stats))
        else:
            played_games = play_games_batched(
                model,
                tokenizer,
                game_specs,
                engine=env.engine,
                batch_size=batch_size,
                use_random_opponent=use_random_opponent,
                opponent_depth=opponent_depth,
                decoding_mode=decoding_mode,
                include_ascii=include_ascii,
                enable_thinking=enable_thinking,
                maia_sidecar=maia_sidecar,
                maia_elo=maia_elo,
                include_piece_list=include_piece_list,
                include_legal_moves=include_legal_moves,
            )

        for (g, game_color, _), (game, stats) in zip(game_specs, played_games, strict=True):
            game.headers["Event"] = f"ChessFM Eval game {g + 1}"
            game.headers["Model"] = model_name
            game.headers["Opponent"] = opponent
            game.headers["ModelColor"] = game_color
            game.headers["Seed"] = str(seed)
            game.headers["Result"] = stats["result"]

            games_list.append(game)
            total_moves += stats["total_moves"]
            legal_moves += stats["legal_moves"]
            formatted_outputs += stats["formatted_outputs"]
            wins += stats["wins"]
            losses += stats["losses"]
            draws += stats["draws"]
            illegal_by_model += stats["illegal_by_model"]
            color_buckets[game_color]["wins"] += stats["wins"]
            color_buckets[game_color]["losses"] += stats["losses"]
            color_buckets[game_color]["draws"] += stats["draws"]
    finally:
        env.close()
    wall_clock_seconds = round(time.monotonic() - eval_start, 2)

    now_utc = datetime.now(timezone.utc)
    run_ts = now_utc.strftime("%Y%m%d_%H%M%S")
    timestamp_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    pgn_path = PGNS_DIR / f"eval_{run_ts}.pgn"
    out_path = RESULTS_DIR / f"eval_{run_ts}.json"

    legal_rate = 100 * legal_moves / max(1, total_moves)
    format_rate = 100 * formatted_outputs / max(1, total_moves)
    ci_low, ci_high = wilson_ci(wins, losses, draws)
    white_split = _color_split_summary(color_buckets["white"])
    black_split = _color_split_summary(color_buckets["black"])
    color_balanced = num_games > 0 and white_split["games"] == black_split["games"]

    print("\n" + "=" * 60, flush=True)
    print("Evaluation Results Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Games Played:   {num_games}", flush=True)
    print(f"Legal Move Rate:      {legal_moves}/{total_moves} ({legal_rate:.1f}%)", flush=True)
    print(
        f"Format Adherence:     {formatted_outputs}/{total_moves} ({format_rate:.1f}%)", flush=True
    )
    print(f"Record (W-L-D):       {wins}-{losses}-{draws}", flush=True)
    print(
        f"White / Black split:  W {white_split['wins']}-{white_split['losses']}-{white_split['draws']}"
        f" | B {black_split['wins']}-{black_split['losses']}-{black_split['draws']}",
        flush=True,
    )
    print(f"Score 95% Wilson CI:  [{ci_low:.3f}, {ci_high:.3f}]", flush=True)
    print(f"Illegal moves by AI:  {illegal_by_model}", flush=True)

    elo = None
    opponent_elo = None
    if maia_elo is not None:
        opponent_elo = int(maia_elo)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
    elif stockfish_elo is not None:
        opponent_elo = int(stockfish_elo)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
        print(
            f"Elo (rough indicator): ~{elo:.0f} vs Stockfish UCI_Elo={stockfish_elo} "
            f"(depth {opponent_depth}, {num_games} games).",
            flush=True,
        )
    elif not use_random_opponent:
        opponent_elo = STOCKFISH_LEVEL_ELO.get(stockfish_level, 1000)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
        print(
            f"Elo (rough indicator): ~{elo:.0f} — NOT a rating. Derived from a "
            f"{num_games}-game sample vs Stockfish L{stockfish_level} searching to a fixed "
            "depth; the skill-level Elo table is an informal community calibration. "
            "Treat W-D-L and legal-move rate as the primary metrics.",
            flush=True,
        )

    git_sha = _get_git_sha()
    results = {
        "model": model_name,
        "num_games": num_games,
        "legal_rate": round(legal_rate, 2),
        "format_rate": round(format_rate, 2),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "white": white_split,
        "black": black_split,
        "color_balanced": color_balanced,
        "score_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "decoding_mode": decoding_mode,
        "include_ascii": include_ascii,
        "include_piece_list": include_piece_list,
        "include_legal_moves": include_legal_moves,
        "enable_thinking": enable_thinking,
        "estimated_elo": None if elo is None else round(elo, 1),
        "opponent": opponent,
        "opponent_backend": opponent_backend,
        "maia_elo": maia_elo,
        "stockfish_level": stockfish_level,
        "stockfish_elo": stockfish_elo,
        "batch_size": batch_size,
        "opponent_depth": opponent_depth,
        "model_color": model_color,
        "alternate_colors": alternate_colors,
        "seed": seed,
        "openings": str(openings) if openings else None,
        "pgn_file": str(pgn_path),
        "git_revision": git_revision(),
        "git_sha": git_sha,
        "timestamp": now_utc.isoformat(),
        "timestamp_utc": timestamp_utc,
        "wall_clock_seconds": wall_clock_seconds,
    }
    write_report(results, out_path)
    print(f"\nResults saved to {out_path}", flush=True)

    if manifest_output is not None:
        manifest = {
            "schema_version": "1.0",
            "git_sha": git_sha,
            "timestamp_utc": timestamp_utc,
            "model_id": model_name,
            "decoding_mode": decoding_mode,
            "enable_thinking": bool(enable_thinking),
            "include_ascii": bool(include_ascii),
            "include_piece_list": bool(include_piece_list),
            "include_legal_moves": bool(include_legal_moves),
            "hardware": _detect_hardware(),
            "wall_clock_seconds": wall_clock_seconds,
            "tiers": {
                "tier1_static": {},
                "tier2_probes": {},
                "tier3_act_eval": {},
                "tier4_match": {
                    "opponent_name": opponent,
                    "opponent_backend": opponent_backend,
                    "opponent_elo": opponent_elo,
                    "stockfish_elo": stockfish_elo,
                    "overall": _color_split_summary(
                        {"wins": wins, "losses": losses, "draws": draws}
                    ),
                    "white": white_split,
                    "black": black_split,
                    "color_balanced": color_balanced,
                    "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "estimated_elo": None if elo is None else round(elo, 1),
                    "legal_rate": round(legal_rate, 2),
                    "format_rate": round(format_rate, 2),
                },
            },
        }
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Eval manifest saved to {manifest_output}", flush=True)

    write_pgn(games_list, pgn_path)
    print(f"Games saved to {pgn_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChessFM model")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL, help="Model checkpoint to evaluate"
    )
    parser.add_argument("--games", type=int, default=10, help="Number of evaluation games")
    parser.add_argument(
        "--stockfish-level", type=int, default=1, help="Stockfish skill level (0-20)"
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=None,
        help=(
            f"Play against Stockfish with UCI_LimitStrength=True and UCI_Elo=N "
            f"(>= {MIN_STOCKFISH_ELO}; overrides --stockfish-level)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Concurrent games evaluated per batched model.generate() call (default: 1)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Color the model plays (games alternate colors by default)",
    )
    parser.add_argument(
        "--alternate-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Alternate model color every game for balanced win/loss stats",
    )
    parser.add_argument(
        "--opponent-depth",
        type=int,
        default=DEFAULT_OPPONENT_DEPTH,
        help="Fixed search depth for the Stockfish opponent (reproducible strength)",
    )
    parser.add_argument(
        "--random-opponent",
        action="store_true",
        help="Play against a random mover instead of Stockfish",
    )
    parser.add_argument(
        "--maia-elo",
        type=int,
        default=None,
        help="Play against MaiaSidecar at the specified Elo rating instead of Stockfish",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for opponent + color assignment"
    )
    parser.add_argument(
        "--openings",
        type=Path,
        default=None,
        help="Path to opening FENs (one per line) to vary game starts",
    )
    parser.add_argument(
        "--decoding-mode",
        type=str,
        default="strict",
        choices=["strict", "constrained"],
        help="Decoding mode: 'strict' (unconstrained Pass@1) or 'constrained' (100%%-legal trie)",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include an ASCII board rendering in prompts (must match training prompt format)",
    )
    parser.add_argument(
        "--include-piece-list",
        action="store_true",
        help="Include a lossless piece-by-square listing in prompts (must match training prompt format)",
    )
    parser.add_argument(
        "--include-legal-moves",
        action="store_true",
        help="Include sorted UCI legal moves in prompts (isolated comparison mode only)",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable <think> reasoning prompt mode (pass --no-enable-thinking for Think-OFF ablation)",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional path to write standardized eval_manifest.json (schema_version 1.0)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.maia_elo is not None and args.maia_elo <= 0:
        parser.error(f"--maia-elo must be a positive integer, got {args.maia_elo}")
    if args.stockfish_elo is not None and args.stockfish_elo < MIN_STOCKFISH_ELO:
        parser.error(
            f"--stockfish-elo must be >= {MIN_STOCKFISH_ELO} "
            f"(Stockfish 16 UCI_Elo minimum), got {args.stockfish_elo}"
        )
    explicit_opponents = sum(
        [args.maia_elo is not None, bool(args.random_opponent), args.stockfish_elo is not None]
    )
    if explicit_opponents > 1:
        parser.error(
            "Specify at most one opponent override among --maia-elo, "
            "--random-opponent, and --stockfish-elo."
        )

    logging.basicConfig(level=logging.INFO)

    model, tokenizer = load_model(args.model, for_inference=True)

    evaluate_model(
        model,
        tokenizer,
        args.model,
        num_games=args.games,
        stockfish_level=args.stockfish_level,
        model_color=args.color,
        alternate_colors=args.alternate_colors,
        opponent_depth=args.opponent_depth,
        use_random_opponent=args.random_opponent,
        seed=args.seed,
        openings=args.openings,
        decoding_mode=args.decoding_mode,
        include_ascii=args.include_ascii,
        enable_thinking=args.enable_thinking,
        maia_elo=args.maia_elo,
        manifest_output=args.manifest_output,
        include_piece_list=args.include_piece_list,
        include_legal_moves=args.include_legal_moves,
        stockfish_elo=args.stockfish_elo,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
