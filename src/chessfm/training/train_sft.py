#!/usr/bin/env python3
"""
ChessFM SFT Training Script.

Train the base model on chess reasoning data with TRL's SFTTrainer. Model
loading is handled by ``chessfm.load_model`` (unsloth-accelerated when the
``unsloth`` extra is installed, with a plain PEFT fallback otherwise).

USAGE:
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trl import SFTConfig

from chessfm.config import (
    CHATML_ASSISTANT_OPEN,
    DEFAULT_BASE_MODEL,
    LORA_R,
    MAX_SEQ_LENGTH,
    build_chat_prompt,
    get_assistant_role,
)
from chessfm.fens import read_jsonl
from chessfm.training.manifest import write_run_manifest
from chessfm.training.split import fen_level_split

GEMMA_ASSISTANT_OPEN = "<|turn>model\n"


def get_response_template(model_name: str = DEFAULT_BASE_MODEL) -> str:
    """Return the assistant turn-open marker for completion-only loss masking."""
    if get_assistant_role(model_name) == "model":
        return GEMMA_ASSISTANT_OPEN
    return CHATML_ASSISTANT_OPEN


def build_completion_collator(tokenizer, model_name: str = DEFAULT_BASE_MODEL):
    """Build a TRL collator that masks prompt tokens to -100 during SFT."""
    from trl import DataCollatorForCompletionOnlyLM

    response_template = get_response_template(model_name)
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )


def build_sft_config(args) -> SFTConfig:
    """Construct the TRL SFTConfig from parsed CLI args.

    Imported lazily so ``read_jsonl`` stays usable without the full torch/trl
    stack. Kept separate from ``main`` so the exact kwargs the training script
    passes are exercised by tests (guards against TRL API drift).
    """
    import torch
    from trl import SFTConfig

    use_gpu = torch.cuda.is_available()
    return SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="epoch",
        # bf16 is preferred on modern GPUs (matches Qwen2.5's native dtype);
        # fp16 is only a fallback for pre-Ampere hardware.
        bf16=use_gpu and torch.cuda.is_bf16_supported(),
        fp16=use_gpu and not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if use_gpu else "adamw_torch",
        seed=args.seed,
        max_length=MAX_SEQ_LENGTH,
    )


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
    parser.add_argument(
        "--merge-and-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights and save full checkpoint to <output>/final_merged",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include an ASCII board rendering in prompts when FEN is available",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessFM SFT Training")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed:       {args.seed}")
    print(f"LoRA rank:  {LORA_R}")
    print()

    print("Loading training data...")
    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        print("Run: python -m chessfm.data_generation.convert_to_training first")
        return

    raw_data = read_jsonl(args.data)
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
        if bad_tags:
            print(f"   ERROR: {bad_tags} sample(s) with unbalanced <think> tags.")
            print("   Fix the data before training: re-run")
            print("   python -m chessfm.data_generation.convert_to_training")
            raise SystemExit(1)
        print("   All completions have balanced <think> tags.")
        print("\nDry run complete. Data looks good!")
        return

    import torch
    from datasets import Dataset
    from trl import SFTTrainer

    from chessfm.load_model import load_model

    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nPreparing formatted dataset with completion-only loss masking...")
    formatted_rows = []
    for sample in valid_data:
        if args.include_ascii and "fen" in sample:
            prompt = build_chat_prompt(
                sample["fen"],
                tokenizer=tokenizer,
                include_ascii=True,
            )
        else:
            prompt = sample["prompt"]
        formatted_rows.append({"text": f"{prompt}{sample['completion']}"})
    dataset = Dataset.from_list(formatted_rows)

    train_indices, val_indices, val_fens = fen_level_split(valid_data)
    train_ds = dataset.select(train_indices)
    eval_ds = dataset.select(val_indices)
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(eval_ds)} (unique FENs: {len(val_fens)})")
    print("   Split is by FEN: no position appears in both sets.")

    print("\nStarting training...")
    args.output.mkdir(parents=True, exist_ok=True)

    training_args = build_sft_config(args)
    data_collator = build_completion_collator(tokenizer, model_name=args.model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        args=training_args,
    )

    train_start = time.monotonic()
    trainer.train()
    wall_clock_seconds = round(time.monotonic() - train_start, 2)

    print("\nSaving model...")
    final_dir = args.output / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete! Adapter saved to: {final_dir}")

    if args.merge_and_save and hasattr(model, "merge_and_unload"):
        merged_dir = args.output / "final_merged"
        print(f"Merging LoRA adapter into base weights -> {merged_dir}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    manifest_path = write_run_manifest(
        args.output,
        model_id=args.model,
        stage="sft",
        seed=args.seed,
        hyperparameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "include_ascii": args.include_ascii,
            "merge_and_save": args.merge_and_save,
        },
        dataset_path=args.data,
        gpu_type=gpu_type,
        wall_clock_seconds=wall_clock_seconds,
    )
    print(f"Run manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
