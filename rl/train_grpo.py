#!/usr/bin/env python3
"""
GRPO Training Script for ChessFM

Train a language model to play chess using Group Relative Policy Optimization.
Supports staged training: legality → win/lose → Stockfish optimization.

USAGE:
    # Quick test (100 steps)
    python train_grpo.py --stage 0 --steps 100
    
    # Full Stage 0 training
    python train_grpo.py --stage 0 --steps 2000 --checkpoint-every 500
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
from tqdm import tqdm

from chess_env import ChessEnv
from rewards import reward_legality, combined_reward


def parse_move_from_output(output: str) -> Optional[str]:
    """
    Extract the move from model output.
    
    Expected format:
    <think>reasoning</think>
    e2e4
    
    Or just:
    e2e4
    """
    # Remove think tags if present
    text = output.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # Get the last word/token that looks like a move
    parts = text.split()
    if not parts:
        return None
    
    # Try to find a valid UCI move pattern (e.g., e2e4, a7a8q)
    for part in reversed(parts):
        part = part.strip().lower()
        if len(part) >= 4 and len(part) <= 5:
            if part[0] in "abcdefgh" and part[1] in "12345678":
                if part[2] in "abcdefgh" and part[3] in "12345678":
                    return part
    
    return parts[-1].strip() if parts else None


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    if prompt in response:
        response = response[len(prompt):].strip()
    return response


def create_prompt(fen: str, turn: str) -> str:
    """Create the prompt for the model."""
    return f"""You are a chess player. Given the position, output a legal move in UCI format.

Position (FEN): {fen}
Side to move: {turn}

Output your move (e.g., e2e4):"""


def load_positions(filepath: Path, max_positions: int = 500) -> list[str]:
    """Load training positions from file."""
    if not filepath.exists():
        print(f"Warning: {filepath} not found, using starting position")
        return ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            fen = line.strip()
            if fen:
                positions.append(fen)
            if len(positions) >= max_positions:
                break
    return positions


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with GRPO")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Training stage (0=legality, 1=win/lose, 2=stockfish)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of training steps")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model to train")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (number of responses per prompt)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--positions", type=Path, 
                        default=Path("../data_generation/positions.txt"),
                        help="Path to positions file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just test setup, don't train")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ChessFM RL Training (GRPO)")
    print("=" * 60)
    print(f"Stage:      {args.stage} ({'legality' if args.stage == 0 else 'win/lose' if args.stage == 1 else 'stockfish'})")
    print(f"Model:      {args.model}")
    print(f"Steps:      {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load positions
    positions = load_positions(args.positions)
    print(f"📂 Loaded {len(positions)} positions")
    
    # Initialize chess environment
    env = ChessEnv()
    print("♟️  Chess environment initialized")
    
    if args.dry_run:
        print("\n🧪 Dry run - testing setup...")
        
        # Test reward function
        fen = positions[0]
        legal_moves = env.reset(fen)
        moves = env.get_legal_moves()
        
        r_legal = reward_legality(fen, moves[0])
        r_illegal = reward_legality(fen, "xxxx")
        
        print(f"  Legal move reward: {r_legal}")
        print(f"  Illegal move reward: {r_illegal}")
        print("\n✅ Setup OK! Ready to train.")
        return
    
    # Load model
    print("\n🔌 Loading model...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ unsloth not installed!")
        print("   Run: pip install unsloth")
        return
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("  Model loaded with LoRA!")
    
    # Training loop
    print("\n🚀 Starting training...")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    legal_count = 0
    total_count = 0
    rewards_sum = 0.0
    
    # Log file
    log_file = args.output / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    for step in tqdm(range(args.steps), desc="Training"):
        # Sample a random position
        fen = random.choice(positions)
        env.reset(fen)
        turn = env.get_turn()
        
        # Create prompt
        prompt = create_prompt(fen, turn)
        
        # Generate multiple responses (GRPO uses groups)
        responses = []
        for _ in range(args.batch_size):
            output = generate_response(model, tokenizer, prompt)
            move = parse_move_from_output(output)
            reward = reward_legality(fen, move or "xxxx")
            responses.append({
                "output": output,
                "move": move,
                "reward": reward
            })
        
        # Compute metrics
        batch_legal = sum(1 for r in responses if r["reward"] > 0)
        batch_reward = sum(r["reward"] for r in responses) / len(responses)
        
        legal_count += batch_legal
        total_count += len(responses)
        rewards_sum += batch_reward
        
        # GRPO update would happen here
        # For now, we're just measuring baseline performance
        # TODO: Implement actual GRPO loss computation and backprop
        
        # Log
        if step % 10 == 0:
            log_entry = {
                "step": step,
                "legal_rate": legal_count / total_count if total_count > 0 else 0,
                "avg_reward": rewards_sum / (step + 1),
                "batch_legal": batch_legal,
                "batch_size": len(responses)
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Checkpoint
        if step > 0 and step % args.checkpoint_every == 0:
            checkpoint_path = args.output / f"stage{args.stage}_step_{step}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            print(f"   Legal rate: {100*legal_count/total_count:.1f}%")
    
    # Final save
    final_path = args.output / f"stage{args.stage}_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final legal move rate: {100*legal_count/total_count:.2f}%")
    print(f"Average reward: {rewards_sum/args.steps:.3f}")
    print(f"Model saved to: {final_path}")
    
    env.close()


if __name__ == "__main__":
    main()
