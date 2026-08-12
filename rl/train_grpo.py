#!/usr/bin/env python3
"""
GRPO Training Script for ChessFM

Train a language model to play chess using Group Relative Policy Optimization (TRL GRPOTrainer).
Supports staged training: legality → win/lose → Stockfish optimization.

USAGE:
    python train_grpo.py --stage 0 --steps 500
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_BASE_MODEL, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA

import torch
from datasets import Dataset
from chess_env import ChessEnv
from rewards import combined_reward, reward_legality, reward_format, reward_budget


def parse_move_from_output(output: str) -> Optional[str]:
    """
    Extract the move from model output.
    """
    text = output.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    parts = text.split()
    if not parts:
        return None
    
    for part in reversed(parts):
        part = part.strip().lower().rstrip(".,!?")
        if 4 <= len(part) <= 5:
            if part[0] in "abcdefgh" and part[1] in "12345678":
                if part[2] in "abcdefgh" and part[3] in "12345678":
                    return part
    
    return None


def extract_fen_from_prompt(prompt: str) -> str:
    """Extract FEN string from prompt text."""
    for line in prompt.split("\n"):
        if line.startswith("Position (FEN):"):
            return line.replace("Position (FEN):", "").strip()
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def make_reward_func(stage: int):
    def chess_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            fen = extract_fen_from_prompt(prompt)
            move = parse_move_from_output(completion) or "xxxx"
            r = combined_reward(fen, move, completion, stage=stage)
            rewards.append(r["total"])
        return rewards
    return chess_reward_fn


def create_prompt(fen: str) -> str:
    """Create standard ChatML formatted prompt."""
    return f"Position (FEN): {fen}\nOutput your move in UCI format (e.g. e2e4) preceded by reasoning in <think>...</think> tags."


def load_positions(filepath: Path, max_positions: int = 1000) -> List[Dict[str, str]]:
    """Load positions into a dataset format."""
    positions = []
    if filepath.exists():
        with open(filepath, 'r') as f:
            for line in f:
                fen = line.strip()
                if fen:
                    prompt = f"<|im_start|>user\n{create_prompt(fen)}<|im_end|>\n<|im_start|>assistant\n"
                    positions.append({"prompt": prompt})
                if len(positions) >= max_positions:
                    break
    if not positions:
        default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        prompt = f"<|im_start|>user\n{create_prompt(default_fen)}<|im_end|>\n<|im_start|>assistant\n"
        positions.append({"prompt": prompt})
    return positions


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with GRPO")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Training stage (0=legality, 1=win/lose, 2=stockfish)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of max training steps")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model to train")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--positions", type=Path, 
                        default=Path(__file__).parent.parent / "data_generation" / "positions.txt",
                        help="Path to positions file")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessFM RL Training (GRPO via TRL)")
    print("=" * 60)
    print(f"Stage:      {args.stage}")
    print(f"Model:      {args.model}")
    print(f"Max Steps:  {args.steps}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Load Dataset
    raw_positions = load_positions(args.positions)
    dataset = Dataset.from_list(raw_positions)
    print(f"📂 Loaded {len(dataset)} prompts/positions for training")

    # Load Model with Unsloth / PEFT
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    except ImportError:
        print("⚠️ unsloth not available, falling back to standard transformers/peft")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Configure GRPO Trainer
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("❌ 'trl' package is required for GRPO. Install via `pip install trl`.")
        return

    training_args = GRPOConfig(
        output_dir=str(args.output),
        learning_rate=args.lr,
        max_steps=args.steps,
        num_generations=args.batch_size,
        max_prompt_length=256,
        max_completion_length=256,
        logging_steps=10,
        save_steps=100,
        fp16=torch.cuda.is_available(),
    )

    reward_fn = make_reward_func(args.stage)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n🚀 Starting GRPO Training Loop...")
    trainer.train()

    final_path = args.output / f"stage{args.stage}_final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n✅ Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
