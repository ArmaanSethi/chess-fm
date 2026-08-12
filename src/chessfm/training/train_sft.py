#!/usr/bin/env python3
"""
ChessFM SFT Training Script.

Train the base model on chess reasoning data using unsloth for efficiency.

USAGE:
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from chessfm.config import DEFAULT_BASE_MODEL, LORA_R, MAX_SEQ_LENGTH
from chessfm.load_model import load_model


def load_dataset(filepath: Path) -> list:
    """Load JSONL training data."""
    data = []
    with open(filepath) as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data.append(json.loads(line_str))
    return data


def fen_level_split(rows: list, val_fraction: float = 0.15, seed: int = 42):
    """Split rows by unique FEN so no position appears in both train and val.

    Returns ``(train_indices, val_indices, val_fens)``.
    """
    fens = sorted({r["fen"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(fens)
    val_fens = set(fens[: max(1, round(val_fraction * len(fens)))])
    val_indices = [i for i, r in enumerate(rows) if r["fen"] in val_fens]
    val_set = set(val_indices)
    train_indices = [i for i in range(len(rows)) if i not in val_set]
    return train_indices, val_indices, val_fens


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with SFT")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.cwd() / "data_generation" / "sft_train.jsonl",
        help="Path to sft_train.jsonl",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL, help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Just validate data, don't train")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessFM SFT Training")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA rank:  {LORA_R}")
    print()

    print("Loading training data...")
    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        print("Run: python -m chessfm.data_generation.convert_to_training first")
        return

    raw_data = load_dataset(args.data)
    print(f"   Loaded {len(raw_data)} samples")

    # Filter to valid, reasonable samples.
    valid_data = [
        s for s in raw_data if "prompt" in s and "completion" in s and s.get("is_reasonable", True)
    ]
    print(f"   Valid & reasonable samples: {len(valid_data)}/{len(raw_data)}")

    if not valid_data:
        print("No valid samples after filtering. Check your data.")
        return

    if args.dry_run:
        bad_tags = sum(
            1
            for s in valid_data
            if s["completion"].count("<think>") != s["completion"].count("</think>")
        )
        print(f"   Samples with unbalanced <think> tags: {bad_tags}")
        if bad_tags:
            print("   Run: python -m chessfm.data_generation.convert_to_training to fix these.")
        print("\nDry run complete. Data looks good!")
        return

    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nPreparing dataset with prompt/completion columns...")
    dataset = Dataset.from_list(valid_data)
    # Loss is computed on completion tokens only (TRL's documented behavior for
    # prompt/completion columns), so the user prompt is never trained on.
    dataset = dataset.select_columns(["prompt", "completion"])

    train_indices, val_indices, val_fens = fen_level_split(valid_data)
    train_ds = dataset.select(train_indices)
    eval_ds = dataset.select(val_indices)
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(eval_ds)} (unique FENs: {len(val_fens)})")
    print("   Split is by FEN: no position appears in both sets.")

    print("\nStarting training...")
    args.output.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        seed=42,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    trainer.train()

    print("\nSaving model...")
    model.save_pretrained(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")
    print(f"Training complete! Model saved to: {args.output / 'final'}")


if __name__ == "__main__":
    main()
