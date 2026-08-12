#!/usr/bin/env python3
"""
GRPO training script for ChessFM.

Trains a language model to play chess with Group Relative Policy Optimization
(TRL GRPOTrainer). Supports staged training:

    Stage 0: Legality only      (+1 legal, -1 illegal) + format/budget bonuses
    Stage 1: + Stockfish move quality (how close to Stockfish's best move)
    Stage 2: + Stockfish eval delta (dense per-move signal)

USAGE:
    python -m chessfm.rl.train_grpo --stage 0 --steps 500
    python -m chessfm.rl.train_grpo --stage 1 --steps 2000 --checkpoint-every 500
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from chessfm.config import DEFAULT_BASE_MODEL, build_chat_prompt
from chessfm.load_model import load_model
from chessfm.rl.engine import close_engine, open_engine
from chessfm.rl.move_utils import extract_fen_from_prompt, parse_move_from_output
from chessfm.rl.rewards import combined_reward

logger = logging.getLogger("chessfm.train_grpo")


def make_reward_func(stage: int, engine=None):
    """Build a TRL-compatible reward function.

    TRL calls reward funcs as ``reward_func(prompts=..., completions=..., **kwargs)``
    and expects a list of floats back, one per completion.
    """

    def chess_reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions, strict=True):
            try:
                fen = extract_fen_from_prompt(prompt)
                move = parse_move_from_output(completion) or "xxxx"
                r = combined_reward(fen, move, completion, stage=stage, engine=engine)
                rewards.append(r["total"])
            except Exception as e:
                # A bad reward should surface as a warning, not a silent -1.0
                # that could corrupt the training signal at scale.
                logger.warning("Reward computation failed, using -1.0: %s", e)
                rewards.append(-1.0)
        return rewards

    return chess_reward_fn


def load_positions(filepath: Path, max_positions: int = 1000) -> list[dict]:
    """Load positions into ChatML-formatted prompt dataset.

    Raises:
        SystemExit: if the positions file is missing or contains no positions.
    """
    if not filepath.exists():
        raise SystemExit(
            f"Positions file not found: {filepath}\n"
            "Generate it first: python -m chessfm.data_generation.fetch_elite_data "
            "(or pass --positions)."
        )
    positions = []
    with open(filepath) as f:
        for line in f:
            fen = line.strip()
            if fen:
                positions.append({"prompt": build_chat_prompt(fen)})
            if len(positions) >= max_positions:
                break
    if not positions:
        raise SystemExit(f"Positions file contains no positions: {filepath}")
    return positions


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with GRPO")
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Training stage (0=legality, 1=+move quality, 2=+eval delta)",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of max training steps")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Base model to train")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Number of generations per prompt for GRPO"
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=4, help="Per-device training batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path.cwd() / "data_generation" / "positions.txt",
        help="Path to positions file",
    )
    parser.add_argument(
        "--max-positions", type=int, default=1000, help="Maximum number of positions to sample"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100, help="Save a checkpoint every N steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ChessFM RL Training (GRPO via TRL)")
    print("=" * 60)
    print(f"Stage:            {args.stage}")
    print(f"Model:            {args.model}")
    print(f"Max Steps:        {args.steps}")
    print(f"Generations/grp:  {args.batch_size}")
    print(f"Checkpoint every: {args.checkpoint_every}")
    print()

    torch.manual_seed(args.seed)

    raw_positions = load_positions(args.positions, args.max_positions)
    dataset = Dataset.from_list(raw_positions)
    print(f"Loaded {len(dataset)} prompts/positions for training")

    model, tokenizer = load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    engine = None
    if args.stage >= 1:
        engine = open_engine()
        if engine is None:
            raise SystemExit(
                "Stockfish engine not found. Stages 1/2 require Stockfish for "
                "the move-quality reward. Install it (brew install stockfish / "
                "apt install stockfish) or run with --stage 0."
            )

    training_args = GRPOConfig(
        output_dir=str(args.output),
        learning_rate=args.lr,
        max_steps=args.steps,
        num_generations=args.batch_size,
        per_device_train_batch_size=args.per_device_batch_size,
        max_prompt_length=256,
        max_completion_length=256,
        logging_steps=10,
        save_steps=args.checkpoint_every,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    reward_fn = make_reward_func(args.stage, engine=engine)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    print("\nStarting GRPO Training Loop...")
    trainer.train()

    final_path = args.output / f"stage{args.stage}_final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nTraining complete! Model saved to {final_path}")

    close_engine(engine)


if __name__ == "__main__":
    main()
