#!/usr/bin/env python3
"""
ChessFM SFT Training Script

Train the base model on chess reasoning data using unsloth for efficiency.

USAGE:
    python training/train_sft.py --data data_generation/sft_train.jsonl
    python training/train_sft.py --data data_generation/sft_train.jsonl --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path so `config` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_BASE_MODEL, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA

# torch is imported lazily so `--dry-run` works on machines without a GPU stack.
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def load_dataset(filepath: Path) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data.append(json.loads(line_str))
    return data


def load_model(model_name: str):
    """Load the base model with unsloth (4-bit + LoRA) or a PEFT fallback."""
    torch = _get_torch()
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    except ImportError:
        print("⚠️ unsloth not found, loading via transformers/peft...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with SFT")
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent.parent / "data_generation" / "sft_train.jsonl",
                        help="Path to sft_train.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model to fine-tune")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just validate data, don't train")
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

    print("📂 Loading training data...")
    if not args.data.exists():
        print(f"❌ Data file not found: {args.data}")
        print("   Run: python data_generation/convert_to_training.py first")
        return

    raw_data = load_dataset(args.data)
    print(f"   Loaded {len(raw_data)} samples")

    # Filter valid samples (drop weak/unreasonable moves)
    valid_data = [s for s in raw_data if "prompt" in s and "completion" in s
                  and s.get("is_reasonable", True)]
    print(f"   Valid & reasonable samples: {len(valid_data)}/{len(raw_data)}")

    if not valid_data:
        print("❌ No valid samples after filtering. Check your data.")
        return

    if args.dry_run:
        # Validate balanced <think> tags across the dataset.
        bad_tags = sum(
            1 for s in valid_data
            if s["completion"].count("<think>") != s["completion"].count("</think>")
        )
        print(f"   Samples with unbalanced <think> tags: {bad_tags}")
        if bad_tags:
            print("   ⚠️ Run convert_to_training.py to fix these.")
        print("\n✅ Dry run complete. Data looks good!")
        return

    print("\n🔌 Loading model...")
    model, tokenizer = load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n📊 Preparing dataset with prompt/completion columns...")
    from datasets import Dataset

    dataset = Dataset.from_list(valid_data)

    # Loss is computed on completion tokens only (TRL's documented behavior
    # for datasets with 'prompt'/'completion' columns), so the model never
    # trains loss on the user prompt.
    dataset = dataset.select_columns(["prompt", "completion"])

    # Train/Val split at the FEN level so identical positions never leak
    # between train and validation.
    fens = {s["fen"] for s in valid_data}
    import random
    rng = random.Random(42)
    fens = list(fens)
    rng.shuffle(fens)
    val_fens = set(fens[: max(1, round(0.15 * len(fens)))])
    val_indices = [i for i, s in enumerate(valid_data) if s["fen"] in val_fens]
    train_indices = [i for i in range(len(valid_data)) if i not in set(val_indices)]

    train_ds = dataset.select(train_indices)
    eval_ds = dataset.select(val_indices)
    print(f"   Train samples: {len(train_ds)} (unique FENs: {len(fens) - len(val_fens)})")
    print(f"   Val samples:   {len(eval_ds)} (unique FENs: {len(val_fens)})")
    print("   Split is by FEN: no position appears in both sets.")

    print("\n🚀 Starting training...")
    args.output.mkdir(parents=True, exist_ok=True)
    torch = _get_torch()

    # TRL >= 0.12 uses SFTConfig. With 'prompt'/'completion' columns, TRL
    # computes loss on completion tokens only.
    try:
        from trl import SFTConfig, SFTTrainer
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
            max_seq_length=MAX_SEQ_LENGTH,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
        )
    except ImportError:
        print("⚠️ 'trl' too old for SFTConfig, falling back to legacy SFTTrainer API...")
        from trl import SFTTrainer
        from transformers import TrainingArguments

        training_args = TrainingArguments(
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
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="completion",
            max_seq_length=MAX_SEQ_LENGTH,
            args=training_args,
        )

    trainer.train()

    print("\n💾 Saving model...")
    model.save_pretrained(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")
    print(f"✅ Training complete! Model saved to: {args.output / 'final'}")


if __name__ == "__main__":
    main()
