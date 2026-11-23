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
import json
import logging
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OPPONENT_DEPTH,
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
                result = env.engine.play(env.board, chess.engine.Limit(depth=opponent_depth))
                move_uci = result.move.uci()
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
) -> dict:
    """Run ``num_games`` games of ``model`` (with ``tokenizer``) and write reports.

    The model is expected to already be loaded (see ``main``); keeping the
    model I/O out of this function makes the whole evaluation loop testable
    with lightweight fakes. ``model_name`` is only used for metadata/headers.

    Colors alternate between games by default so the win/loss/Elo estimates do
    not over-weight a single color. When ``openings`` is supplied with
    ``alternate_colors=True``, each opening FEN is played twice in a row (once
    as White, once as Black) via ``opening_fens[(g // 2) % len(opening_fens)]``.

    The Stockfish skill level MUST be applied successfully when playing
    Stockfish: a silently unapplied skill level would make every reported number
    meaningless, so a configuration failure aborts the run.
    """
    if decoding_mode not in ("strict", "constrained"):
        raise ValueError(
            f"Unsupported decoding_mode {decoding_mode!r}; expected 'strict' or 'constrained'."
        )
    if maia_elo is not None and maia_elo <= 0:
        raise ValueError(f"maia_elo must be a positive integer, got {maia_elo}")

    eval_start = time.monotonic()
    random.seed(seed)

    env = ChessEnv()
    try:
        maia_sidecar: MaiaSidecar | None = None
        if maia_elo is not None:
            maia_sidecar = MaiaSidecar(use_neural=True, engine=env.engine)
            opponent_backend = maia_sidecar.backend
            opponent = f"Maia-{maia_elo} ({opponent_backend})"
        elif use_random_opponent:
            opponent_backend = "random"
            opponent = "Random bot"
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

        print("=" * 60)
        print(f"ChessFM Evaluation: {model_name} vs {opponent}")
        color_desc = f"Alternating (starting {model_color})" if alternate_colors else model_color
        print(f"Model color: {color_desc}")
        print(f"Decoding mode: {decoding_mode}")
        print(f"Include ASCII: {include_ascii} | Enable thinking: {enable_thinking}")
        print(f"Seed: {seed}")
        print("=" * 60)

        opening_fens = load_openings(openings)
        if opening_fens:
            print(f"Using {len(opening_fens)} opening positions from {openings}")

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
        for g in range(num_games):
            if alternate_colors:
                game_color = model_color if g % 2 == 0 else other_color
            else:
                game_color = model_color
            if opening_fens:
                opening_idx = (
                    (g // 2) % len(opening_fens) if alternate_colors else g % len(opening_fens)
                )
                env.reset(opening_fens[opening_idx])
            else:
                env.reset()
            print(f"\nGame {g + 1}/{num_games} ({game_color})")

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
            )

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

            if stats["illegal_by_model"]:
                print(f"   (model forfeited via illegal move: {stats['result']})")
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

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Games Played:   {num_games}")
    print(f"Legal Move Rate:      {legal_moves}/{total_moves} ({legal_rate:.1f}%)")
    print(f"Format Adherence:     {formatted_outputs}/{total_moves} ({format_rate:.1f}%)")
    print(f"Record (W-L-D):       {wins}-{losses}-{draws}")
    print(
        f"White / Black split:  W {white_split['wins']}-{white_split['losses']}-{white_split['draws']}"
        f" | B {black_split['wins']}-{black_split['losses']}-{black_split['draws']}"
    )
    print(f"Score 95% Wilson CI:  [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Illegal moves by AI:  {illegal_by_model}")

    elo = None
    opponent_elo = None
    if maia_elo is not None:
        opponent_elo = int(maia_elo)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
    elif not use_random_opponent:
        opponent_elo = STOCKFISH_LEVEL_ELO.get(stockfish_level, 1000)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
        print(
            f"Elo (rough indicator): ~{elo:.0f} — NOT a rating. Derived from a "
            f"{num_games}-game sample vs Stockfish L{stockfish_level} searching to a fixed "
            "depth; the skill-level Elo table is an informal community calibration. "
            "Treat W-D-L and legal-move rate as the primary metrics."
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
        "enable_thinking": enable_thinking,
        "estimated_elo": None if elo is None else round(elo, 1),
        "opponent": opponent,
        "opponent_backend": opponent_backend,
        "maia_elo": maia_elo,
        "stockfish_level": stockfish_level,
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
    print(f"\nResults saved to {out_path}")

    if manifest_output is not None:
        manifest = {
            "schema_version": "1.0",
            "git_sha": git_sha,
            "timestamp_utc": timestamp_utc,
            "model_id": model_name,
            "decoding_mode": decoding_mode,
            "enable_thinking": bool(enable_thinking),
            "include_ascii": bool(include_ascii),
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
        print(f"Eval manifest saved to {manifest_output}")

    write_pgn(games_list, pgn_path)
    print(f"Games saved to {pgn_path}")
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
    if args.maia_elo is not None and args.maia_elo <= 0:
        parser.error(f"--maia-elo must be a positive integer, got {args.maia_elo}")

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
    )


if __name__ == "__main__":
    main()
