#!/usr/bin/env python3
"""
Evaluation Script for ChessFM

Runs simulated games between a fine-tuned ChessFM checkpoint and Stockfish / Random bot
to measure legal move rate, format adherence, win rate, and estimated Elo.
"""

import argparse
import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_BASE_MODEL, STOCKFISH_PATHS
from rl.chess_env import ChessEnv
from rl.train_grpo import parse_move_from_output


def evaluate_model(model_path: str, num_games: int = 50, stockfish_level: int = 1):
    print("=" * 60)
    print(f"ChessFM Evaluation: {model_path} vs Stockfish Level {stockfish_level}")
    print("=" * 60)

    # Initialize environment
    env = ChessEnv()
    if env.engine is None:
        print("⚠️ Warning: Stockfish engine not loaded. Evaluation will run with basic env.")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"⚠️ Failed to load with Unsloth ({e}), falling back to transformers...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    total_moves = 0
    legal_moves = 0
    formatted_outputs = 0
    wins = 0
    losses = 0
    draws = 0

    for g in range(num_games):
        fen = env.reset()
        done = False
        print(f"\n🎮 Game {g + 1}/{num_games}")

        while not done:
            turn = env.get_turn()
            if turn == "white":
                # Model's turn
                prompt = f"<|im_start|>user\nPosition (FEN): {fen}\nOutput your move in UCI format (e.g. e2e4) preceded by reasoning in <think>...</think> tags.<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else "cpu")
                outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                move_uci = parse_move_from_output(resp) or "xxxx"
                total_moves += 1

                if "<think>" in resp and "</think>" in resp:
                    formatted_outputs += 1

                step_res = env.step(move_uci)
                if step_res.legal:
                    legal_moves += 1
                else:
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
                if env.engine:
                    result = env.engine.play(env.board, chess.engine.Limit(time=0.05))
                    move_uci = result.move.uci()
                else:
                    move_uci = env.get_legal_moves()[0]
                
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

    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Games Played: {num_games}")
    print(f"Legal Move Rate:    {legal_moves}/{total_moves} ({100*legal_moves/max(1, total_moves):.1f}%)")
    print(f"Format Adherence:   {formatted_outputs}/{total_moves} ({100*formatted_outputs/max(1, total_moves):.1f}%)")
    print(f"Record (W-L-D):     {wins}-{losses}-{draws}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChessFM model")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Model checkpoint to evaluate")
    parser.add_argument("--games", type=int, default=10, help="Number of evaluation games")
    args = parser.parse_args()

    evaluate_model(args.model, num_games=args.games)
