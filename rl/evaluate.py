#!/usr/bin/env python3
"""
Evaluation Script for ChessFM

Runs simulated games between a fine-tuned ChessFM checkpoint and Stockfish /
a random bot to measure legal move rate, format adherence, win rate, and an
estimated Elo rating.

USAGE:
    python rl/evaluate.py --model checkpoints/stage0_final --games 50
    python rl/evaluate.py --model Qwen/Qwen2.5-3B-Instruct --games 20 --stockfish-level 1
"""

import argparse
import sys
import math
from pathlib import Path

import chess

# Ensure project root is in sys.path so `config`, `rl.*` are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_BASE_MODEL, STOCKFISH_PATHS, make_prompt
from rl.chess_env import ChessEnv
from rl.move_utils import parse_move_from_output

# torch is imported lazily so `--help` works without a GPU stack.
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# Approximate Elo for Stockfish skill levels (0-20), used for a rough estimate.
# Source: common community calibration tables (e.g. Lichess/CCRL-ish mapping).
STOCKFISH_LEVEL_ELO = {
    0: 800, 1: 900, 2: 1000, 3: 1100, 4: 1200,
    5: 1300, 6: 1400, 7: 1500, 8: 1600, 9: 1700,
    10: 1800, 11: 1900, 12: 2000, 13: 2100, 14: 2200,
    15: 2300, 16: 2400, 17: 2500, 18: 2600, 19: 2700, 20: 2800,
}


def load_model(model_path: str):
    """Load a ChessFM checkpoint for inference (unsloth or transformers fallback)."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        print(f"⚠️ Failed to load with unsloth ({e}), falling back to transformers...")
        torch = _get_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        return model, tokenizer


def estimated_elo(wins: int, losses: int, draws: int, opponent_elo: int) -> float:
    """
    Estimate model Elo from game results vs a fixed opponent using the
    standard Elo expected-score model: E = 1 / (1 + 10^((Elo_opp - Elo_model)/400)).
    Solved for Elo_model given the observed score fraction.
    """
    games = wins + losses + draws
    if games == 0:
        return 0.0
    score = (wins + 0.5 * draws) / games
    # Clamp to avoid +/-inf at perfect or zero scores.
    score = min(0.999, max(0.001, score))
    return opponent_elo + 400 * math.log10(score / (1 - score))


def evaluate_model(
    model_path: str,
    num_games: int = 50,
    stockfish_level: int = 1,
    model_color: str = "white",
    time_per_move: float = 0.05,
    use_random_opponent: bool = False,
):
    print("=" * 60)
    opponent = "Random bot" if use_random_opponent else f"Stockfish Level {stockfish_level}"
    print(f"ChessFM Evaluation: {model_path} vs {opponent}")
    print(f"Model color: {model_color}")
    print("=" * 60)

    env = ChessEnv()
    if env.engine is None:
        print("⚠️ Warning: Stockfish engine not loaded.")

    if env.engine is not None and not use_random_opponent:
        try:
            env.engine.configure({"Skill Level": stockfish_level})
        except Exception:
            pass

    model, tokenizer = load_model(model_path)
    torch = _get_torch()

    total_moves = 0
    legal_moves = 0
    formatted_outputs = 0
    wins = 0
    losses = 0
    draws = 0
    illegal_by_model = 0

    for g in range(num_games):
        fen = env.reset()
        done = False
        print(f"\n🎮 Game {g + 1}/{num_games}")

        while not done:
            turn = env.get_turn()
            if turn == model_color:
                # Model's turn
                prompt = (
                    f"<|im_start|>user\n{make_prompt(fen)}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(
                    getattr(model, "device", torch.device("cpu"))
                )
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Strip the prompt before parsing the model's completion
                completion = resp.split("<|im_start|>assistant\n")[-1]

                move_uci = parse_move_from_output(completion) or "xxxx"
                total_moves += 1

                if "<think>" in completion and "</think>" in completion:
                    formatted_outputs += 1

                step_res = env.step(move_uci)
                if step_res.legal:
                    legal_moves += 1
                else:
                    illegal_by_model += 1
                    losses += 1
                    done = True
                    break

                if step_res.done:
                    done = True
                    result = step_res.info.get("result", "")
                    if result == "1-0":
                        wins += 1
                    elif result == "0-1":
                        losses += 1
                    else:
                        draws += 1
            else:
                # Opponent's turn (Stockfish or Random)
                if env.engine and not use_random_opponent:
                    result = env.engine.play(env.board, chess.engine.Limit(time=time_per_move))
                    move_uci = result.move.uci()
                else:
                    import random
                    move_uci = random.choice(env.get_legal_moves())

                step_res = env.step(move_uci)
                if step_res.done:
                    done = True
                    res = step_res.info.get("result", "")
                    if res == "1-0":
                        wins += 1
                    elif res == "0-1":
                        losses += 1
                    else:
                        draws += 1

    env.close()

    legal_rate = 100 * legal_moves / max(1, total_moves)
    format_rate = 100 * formatted_outputs / max(1, total_moves)
    opponent_elo = 0 if use_random_opponent else STOCKFISH_LEVEL_ELO.get(stockfish_level, 1000)
    elo = estimated_elo(wins, losses, draws, opponent_elo)

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Games Played:   {num_games}")
    print(f"Legal Move Rate:      {legal_moves}/{total_moves} ({legal_rate:.1f}%)")
    print(f"Format Adherence:     {formatted_outputs}/{total_moves} ({format_rate:.1f}%)")
    print(f"Record (W-L-D):       {wins}-{losses}-{draws}")
    print(f"Win rate:             {100*wins/max(1,num_games):.1f}%")
    print(f"Illegal moves by AI:  {illegal_by_model}")
    if not use_random_opponent:
        print(f"Estimated Elo:        {elo:.0f} (vs SF level {stockfish_level}, ~{opponent_elo} Elo)")

    # Save results to a JSON file for easy tracking
    import json
    from datetime import datetime
    results = {
        "model": model_path,
        "num_games": num_games,
        "legal_rate": round(legal_rate, 2),
        "format_rate": round(format_rate, 2),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "estimated_elo": round(elo, 1),
        "opponent": opponent,
        "timestamp": datetime.now().isoformat(),
    }
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChessFM model")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Model checkpoint to evaluate")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of evaluation games")
    parser.add_argument("--stockfish-level", type=int, default=1,
                        help="Stockfish skill level (0-20)")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"],
                        help="Color the model plays")
    parser.add_argument("--time-per-move", type=float, default=0.05,
                        help="Seconds per move for the opponent")
    parser.add_argument("--random-opponent", action="store_true",
                        help="Play against a random mover instead of Stockfish")
    args = parser.parse_args()

    evaluate_model(
        args.model,
        num_games=args.games,
        stockfish_level=args.stockfish_level,
        model_color=args.color,
        time_per_move=args.time_per_move,
        use_random_opponent=args.random_opponent,
    )
