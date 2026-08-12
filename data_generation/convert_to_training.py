#!/usr/bin/env python3
"""
Convert all_sft_data.jsonl to HuggingFace training format.

INPUT FORMAT (all_sft_data.jsonl):
{
    "fen": "...",
    "color": "white",
    "reasoning": "<think>...</think>",
    "move": "e2e4",
    "model": "...",
    "is_reasonable": true,
    "centipawn_loss": 12.0,
    "stockfish_best": "e2e4",
    "raw_response": "...",
    "generated_at": "..."
}

OUTPUT FORMAT (sft_train.jsonl - TRL SFTTrainer compatible):
{
    "prompt": "Position (FEN): ... Side to move: White\n\nAnalyze ...",
    "completion": "<think>...</think>\ne2e4",
    "fen": "...",
    "move": "e2e4",
    "is_reasonable": true,
    "centipawn_loss": 12.0,
    "stockfish_best": "e2e4",
    "model": "..."
}

The output uses TRL's ``prompt``/``completion`` column layout so SFT loss is
computed on completion tokens only (not the prompt). Rows are deduplicated by
FEN, keeping the highest-quality sample per position. A ``data_manifest.json``
is written alongside the output with provenance/counts.

USAGE:
    python convert_to_training.py
    python convert_to_training.py --input data/all_sft_data.jsonl --output data/sft_train.jsonl
    python convert_to_training.py --only-reasonable
"""

import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path so `config` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import make_prompt


def fix_think_tags(reasoning: str) -> str:
    """Ensure balanced <think>...</think> tags in a reasoning trace."""
    text = reasoning.strip()
    opens = text.count("<think>")
    closes = text.count("</think>")
    if opens > closes:
        text += "</think>" * (opens - closes)
    elif closes > opens:
        text = "<think>" * (closes - opens) + text
    return text


def sample_quality(sample: dict) -> float:
    """Ranking key for keeping the best sample per FEN."""
    if not sample.get("is_reasonable", True):
        return 1e6  # De-prioritize unreasonable samples
    cp = sample.get("centipawn_loss")
    if cp is None:
        return 1e9  # Unknown quality: keep only if no better sample exists
    return float(cp)


def convert_sample(sample: dict) -> dict:
    """Convert one sample to training format."""

    # Build instruction from the shared prompt template so SFT and RL
    # see the exact same instruction at train and inference time.
    instruction = make_prompt(sample['fen'])

    # Build output (reasoning + move, matching roadmap format)
    reasoning = fix_think_tags(sample.get("reasoning", ""))
    output = f"{reasoning}\n{sample['move']}"

    return {
        "prompt": instruction,
        "completion": output,
        # Keep metadata for filtering / reproducibility
        "fen": sample["fen"],
        "move": sample["move"],
        "is_reasonable": sample.get("is_reasonable", True),
        "centipawn_loss": sample.get("centipawn_loss"),
        "stockfish_best": sample.get("stockfish_best"),
        "model": sample.get("model", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to training format")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "all_sft_data.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sft_train.jsonl",
                        help="Output training file")
    parser.add_argument("--only-reasonable", action="store_true",
                        help="Only include samples where is_reasonable=True")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print("   Generate or place all_sft_data.jsonl first!")
        return

    print(f"📂 Reading: {args.input}")

    # Read all samples
    samples = []
    with open(args.input, 'r') as fin:
        for line in fin:
            line_str = line.strip()
            if line_str:
                samples.append(json.loads(line_str))

    total_in = len(samples)

    # Deduplicate by FEN, keeping the highest-quality sample per position.
    best_by_fen = {}
    for s in samples:
        key = s["fen"]
        cur = best_by_fen.get(key)
        if cur is None or sample_quality(s) < sample_quality(cur):
            best_by_fen[key] = s
    samples = list(best_by_fen.values())

    # Apply reasonable filter if requested
    if args.only_reasonable:
        samples = [s for s in samples if s.get("is_reasonable", True)]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    with open(args.output, 'w') as fout:
        for sample in samples:
            train_sample = convert_sample(sample)
            fout.write(json.dumps(train_sample) + "\n")
            converted += 1

    # Provenance manifest
    manifest = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "raw_samples": total_in,
        "unique_fens": len(best_by_fen),
        "output_samples": converted,
        "only_reasonable": args.only_reasonable,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_template": "make_prompt() in config.py",
    }
    manifest_path = args.output.with_name("data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Converted: {converted} samples (from {total_in} raw, {len(best_by_fen)} unique FENs)")
    print(f"💾 Output:    {args.output}")
    print(f"📄 Manifest:  {manifest_path}")

    # Show example
    print("\n📄 Example output:")
    with open(args.output, 'r') as f:
        example = json.loads(f.readline())
        print(json.dumps(example, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
