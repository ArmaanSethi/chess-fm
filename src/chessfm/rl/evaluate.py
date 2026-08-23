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
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

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


def load_openings(path: Path | None) -> list[str]:
    """Load opening FENs (one per line). Returns [] if no file given."""
    if path is None or not path.exists():
        return []
    try:
        return read_fens(path)
    except FileNotFoundError:
        return []


def generate_completion(
    model, tokenizer, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> str:
    """Generate a completion for ``prompt`` and return ONLY the newly generated text.

    ``max_new_tokens`` defaults to the training-time completion budget
    (``config.DEFAULT_MAX_NEW_TOKENS``) so the model is never truncated to a
    tighter budget at inference than it was trained on.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs.input_ids.shape[1]
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


def play_game(
    model,
    tokenizer,
    env: ChessEnv,
    model_color: str,
    use_random_opponent: bool = False,
    opponent_depth: int = DEFAULT_OPPONENT_DEPTH,
) -> tuple:
    """Play one game between the model and the opponent in ``env``.

    Returns ``(game, stats)`` where ``game`` is a ``chess.pgn.Game`` (move tree
    populated, headers left for the caller) and ``stats`` is a dict of
    per-game counters. Win/loss/draw are always from the model's perspective.

    The Stockfish opponent searches to a fixed depth (not wall-clock time) so
    its strength is reproducible across runs and machines.

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
    node = game
    done = False

    while not done:
        # Refresh the FEN every ply so the model is always prompted with the
        # actual current board, never a stale one.
        fen = env.get_fen()
        if env.get_turn() == model_color:
            completion = generate_completion(model, tokenizer, build_chat_prompt(fen))
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
            # Opponent (Stockfish or random).
            if env.engine and not use_random_opponent:
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
):
    """Run ``num_games`` games of ``model`` (with ``tokenizer``) and write reports.

    The model is expected to already be loaded (see ``main``); keeping the
    model I/O out of this function makes the whole evaluation loop testable
    with lightweight fakes. ``model_name`` is only used for metadata/headers.

    Colors alternate between games by default so the win/loss/Elo estimates do
    not over-weight a single color. Pass ``alternate_colors=False`` to pin the
    model to ``model_color`` for every game.

    The Stockfish skill level MUST be applied successfully: a silently
    unapplied skill level would make every reported number meaningless, so a
    configuration failure aborts the run.
    """
    print("=" * 60)
    opponent = (
        "Random bot"
        if use_random_opponent
        else (f"Stockfish L{stockfish_level} (depth {opponent_depth})")
    )
    print(f"ChessFM Evaluation: {model_name} vs {opponent}")
    color_desc = f"Alternating (starting {model_color})" if alternate_colors else model_color
    print(f"Model color: {color_desc}")
    print(f"Seed: {seed}")
    print("=" * 60)

    random.seed(seed)

    env = ChessEnv()
    if not use_random_opponent:
        if env.engine is None:
            env.close()
            raise SystemExit(
                "Stockfish engine not found; it is required for evaluation "
                "unless --random-opponent is passed."
            )
        try:
            env.engine.configure({"Skill Level": stockfish_level})
        except chess.engine.EngineError as e:
            env.close()
            raise SystemExit(f"Stockfish rejected Skill Level {stockfish_level}: {e}") from e

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
    games_list = []

    for g in range(num_games):
        if alternate_colors:
            game_color = "white" if g % 2 == 0 else "black"
        else:
            game_color = model_color
        if opening_fens:
            env.reset(opening_fens[g % len(opening_fens)])
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

        if stats["illegal_by_model"]:
            print(f"   (model forfeited via illegal move: {stats['result']})")

    env.close()

    pgn_path = PGNS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
    write_pgn(games_list, pgn_path)
    print(f"\nGames saved to {pgn_path}")

    legal_rate = 100 * legal_moves / max(1, total_moves)
    format_rate = 100 * formatted_outputs / max(1, total_moves)

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Games Played:   {num_games}")
    print(f"Legal Move Rate:      {legal_moves}/{total_moves} ({legal_rate:.1f}%)")
    print(f"Format Adherence:     {formatted_outputs}/{total_moves} ({format_rate:.1f}%)")
    print(f"Record (W-L-D):       {wins}-{losses}-{draws}")
    print(f"Illegal moves by AI:  {illegal_by_model}")
    elo = None
    if not use_random_opponent:
        opponent_elo = STOCKFISH_LEVEL_ELO.get(stockfish_level, 1000)
        elo = estimated_elo(wins, losses, draws, opponent_elo)
        print(
            f"Elo (rough indicator): ~{elo:.0f} — NOT a rating. Derived from a "
            f"{num_games}-game sample vs Stockfish L{stockfish_level} searching to a fixed "
            "depth; the skill-level Elo table is an informal community calibration. "
            "Treat W-D-L and legal-move rate as the primary metrics."
        )

    results = {
        "model": model_name,
        "num_games": num_games,
        "legal_rate": round(legal_rate, 2),
        "format_rate": round(format_rate, 2),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "estimated_elo": None if elo is None else round(elo, 1),
        "opponent": opponent,
        "stockfish_level": stockfish_level,
        "opponent_depth": opponent_depth,
        "model_color": model_color,
        "alternate_colors": alternate_colors,
        "seed": seed,
        "openings": str(openings) if openings else None,
        "pgn_file": str(pgn_path),
        "git_revision": git_revision(),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_report(results, out_path)
    print(f"\nResults saved to {out_path}")


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
        "--seed", type=int, default=42, help="Random seed for opponent + color assignment"
    )
    parser.add_argument(
        "--openings",
        type=Path,
        default=None,
        help="Path to opening FENs (one per line) to vary game starts",
    )
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
