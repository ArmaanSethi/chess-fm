#!/usr/bin/env python3
"""
ChessFM SFT Training Script

Train the base model on chess reasoning data using unsloth for efficiency.

USAGE:
    python train_sft.py --data ../data_generation/data/sft_train.jsonl
    python train_sft.py --data ../data_generation/data/sft_train.jsonl --epochs 3 --output ./checkpoints

REQUIREMENTS:
    pip install unsloth transformers datasets accelerate
    # Or on RunPod:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

import argparse
import json
from pathlib import Path


def load_dataset(filepath: Path) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_for_training(sample: dict) -> dict:
    """Format sample for instruction-following fine-tuning."""
    return {
        "instruction": sample["instruction"],
        "input": "",  # No additional input
        "output": sample["output"]
    }


def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with SFT")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to sft_train.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank")
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
    print(f"LoRA rank:  {args.lora_r}")
    print()
    
    # =========================================================================
    # Step 1: Load and validate data
    # =========================================================================
    print("📂 Loading training data...")
    
    if not args.data.exists():
        print(f"❌ Data file not found: {args.data}")
        print("   Run: python convert_to_training.py first")
        return
    
    raw_data = load_dataset(args.data)
    print(f"   Loaded {len(raw_data)} samples")
    
    # Validate format
    valid = 0
    for sample in raw_data:
        if "instruction" in sample and "output" in sample:
            if "<think>" in sample["output"] and "</think>" in sample["output"]:
                valid += 1
    
    print(f"   Valid samples: {valid}/{len(raw_data)} ({100*valid/len(raw_data):.1f}%)")
    
    if args.dry_run:
        print("\n✅ Dry run complete. Data looks good!")
        return
    
    # =========================================================================
    # Step 2: Load model with unsloth
    # =========================================================================
    print("\n🔌 Loading model with unsloth...")
    
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ unsloth not installed!")
        print("   Run: pip install unsloth")
        return
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    print("   Model loaded!")
    
    # =========================================================================
    # Step 3: Configure LoRA
    # =========================================================================
    print("\n⚙️ Configuring LoRA...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print(f"   LoRA rank: {args.lora_r}")
    
    # =========================================================================
    # Step 4: Prepare dataset
    # =========================================================================
    print("\n📊 Preparing dataset...")
    
    from datasets import Dataset
    
    # Format for training
    formatted_data = [format_for_training(s) for s in raw_data]
    dataset = Dataset.from_list(formatted_data)
    
    # Create prompt template
    prompt_template = """### Instruction:
{instruction}

### Response:
{output}"""
    
    def formatting_func(examples):
        texts = []
        for inst, out in zip(examples["instruction"], examples["output"]):
            text = prompt_template.format(instruction=inst, output=out)
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    print(f"   Dataset size: {len(dataset)}")
    
    # =========================================================================
    # Step 5: Train
    # =========================================================================
    print("\n🚀 Starting training...")
    
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="adamw_8bit",
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )
    
    # Train!
    trainer.train()
    
    # =========================================================================
    # Step 6: Save
    # =========================================================================
    print("\n💾 Saving model...")
    
    model.save_pretrained(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")
    
    print(f"✅ Training complete!")
    print(f"   Model saved to: {args.output / 'final'}")


if __name__ == "__main__":
    main()
