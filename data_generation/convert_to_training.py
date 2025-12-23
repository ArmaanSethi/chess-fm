#!/usr/bin/env python3
"""
Convert sft_data.jsonl to HuggingFace training format.

This script converts the generated data into the exact format expected for SFT training:

INPUT FORMAT (sft_data.jsonl):
{
    "fen": "...",
    "reasoning": "<think>...</think>",
    "move": "e2e4",
    ...
}

OUTPUT FORMAT (sft_train.jsonl - HuggingFace compatible):
{
    "instruction": "Position (FEN): ... Side to move: White",
    "output": "<think>...</think>\ne2e4"
}

USAGE:
    python convert_to_training.py
    python convert_to_training.py --input data/sft_data.jsonl --output data/sft_train.jsonl
"""

import json
import argparse
from pathlib import Path


def convert_sample(sample: dict) -> dict:
    """Convert one sample to training format."""
    
    # Build instruction (matches what we prompted with)
    instruction = f"Position (FEN): {sample['fen']}\nSide to move: {sample['color']}\n\nAnalyze this position and choose the best move."
    
    # Build output (reasoning + move, matching roadmap format)
    output = f"{sample['reasoning']}\n{sample['move']}"
    
    return {
        "instruction": instruction,
        "output": output,
        # Keep metadata for filtering
        "fen": sample["fen"],
        "move": sample["move"],
        "is_reasonable": sample.get("is_reasonable", True),
        "model": sample.get("model", "unknown")
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to training format")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data" / "sft_data.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "sft_train.jsonl",
                        help="Output training file")
    parser.add_argument("--only-reasonable", action="store_true",
                        help="Only include samples where is_reasonable=True")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print("   Run generate_sft_data_proxy.py first!")
        return
    
    print(f"📂 Reading: {args.input}")
    
    # Convert
    converted = 0
    skipped = 0
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            sample = json.loads(line.strip())
            
            # Filter if requested
            if args.only_reasonable and not sample.get("is_reasonable", True):
                skipped += 1
                continue
            
            # Convert
            train_sample = convert_sample(sample)
            fout.write(json.dumps(train_sample) + "\n")
            converted += 1
    
    print(f"✅ Converted: {converted} samples")
    if skipped:
        print(f"⏭️  Skipped:   {skipped} weak moves")
    print(f"💾 Output:    {args.output}")
    
    # Show example
    print("\n📄 Example output:")
    with open(args.output, 'r') as f:
        example = json.loads(f.readline())
        print(json.dumps(example, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
