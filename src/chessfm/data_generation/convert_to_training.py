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
    "generated_at": "..."
}

OUTPUT FORMAT (sft_train.jsonl - TRL SFTTrainer compatible):
{
    "prompt": "<|im_start|>user\\nPosition (FEN): ...<|im_end|>\\n<|im_start|>assistant\\n",
    "completion": "<think>...</think>\\ne2e4<|im_end|>",
    "fen": "...",
    "move": "e2e4",
    "is_reasonable": true,
    "centipawn_loss": 12.0,
    "stockfish_best": "e2e4",
    "model": "..."
}

The output uses TRL's ``prompt``/``completion`` column layout so SFT loss is
computed on completion tokens only. The prompt is ChatML-wrapped via
``build_chat_prompt`` so SFT and the GRPO/eval inference path see the exact
same instruction encoding. Rows are deduplicated by FEN, keeping the
highest-quality sample per position. By default only "reasonable" samples
(centipawn loss within ``MAX_CENTIPAWN_LOSS``) are kept. A portable
``data_manifest.json`` with provenance/counts is written alongside.

USAGE:
    python -m chessfm.data_generation.convert_to_training
    python -m chessfm.data_generation.convert_to_training --input data/all_sft_data.jsonl --output data/sft_train.jsonl
    python -m chessfm.data_generation.convert_to_training --include-unreasonable
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from chessfm.config import (
    CHATML_ASSISTANT_CLOSE,
    MAX_CENTIPAWN_LOSS,
    build_chat_prompt,
)
from chessfm.fens import read_jsonl

# Corpora at or below this size are flagged as reproducibility samples in the
# manifest; anything larger is treated as a full training run.
SAMPLE_THRESHOLD = 1000


def fix_think_tags(reasoning: str) -> str:
    """Ensure balanced <think>...</think> tags in a reasoning trace."""
    text = reasoning.strip()
    # Some teacher traces emit a duplicated closing tag
    # (``</think>\n</think>``). Collapse consecutive duplicates before
    # balancing, otherwise the balancer would prepend a spurious opener.
    text = re.sub(r"(</think>)\s+(?=</think>)", r"\1", text)
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
        return 1e6
    cp = sample.get("centipawn_loss")
    if cp is None:
        return 1e9
    return float(cp)


def convert_sample(sample: dict) -> dict:
    """Convert one sample to training format."""
    instruction = build_chat_prompt(sample["fen"])
    reasoning = fix_think_tags(sample.get("reasoning", ""))
    # The ChatML turn-end token terminates the completion so the model learns
    # *when to stop*; without it every generation runs to the token budget.
    output = f"{reasoning}\n{sample['move']}{CHATML_ASSISTANT_CLOSE}"

    return {
        "prompt": instruction,
        "completion": output,
        "fen": sample["fen"],
        "move": sample["move"],
        "is_reasonable": sample.get("is_reasonable", True),
        "centipawn_loss": clamp_cp(sample.get("centipawn_loss")),
        "stockfish_best": sample.get("stockfish_best"),
        "model": sample.get("model", "unknown"),
    }


def clamp_cp(cp):
    """Centipawn loss can never be negative; clamp to a sane floor."""
    if cp is None:
        return None
    return max(0, int(cp))


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to training format")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_generation/all_sft_data.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_generation/sft_train.jsonl"),
        help="Output training file",
    )
    parser.add_argument(
        "--include-unreasonable",
        action="store_true",
        help="Include samples beyond the centipawn-loss threshold",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print(
            "Run the synthetic data generator (chessfm.data_generation.generate_synthetic_data) first!"
        )
        return

    print(f"Reading: {args.input}")

    samples = read_jsonl(args.input)
    total_in = len(samples)
    raw_unreasonable = sum(1 for s in samples if not s.get("is_reasonable", True))

    # Rows that cannot be converted (missing FEN or move) are dropped loudly.
    malformed = [s for s in samples if not s.get("fen") or not s.get("move")]
    if malformed:
        print(f"Warning: dropping {len(malformed)} malformed row(s) (missing fen/move).")
    samples = [s for s in samples if s.get("fen") and s.get("move")]

    # Deduplicate by FEN, keeping the highest-quality sample per position.
    best_by_fen = {}
    for s in samples:
        key = s["fen"]
        cur = best_by_fen.get(key)
        if cur is None or sample_quality(s) < sample_quality(cur):
            best_by_fen[key] = s
    samples = list(best_by_fen.values())

    # Apply the quality filter (excluded by default).
    if not args.include_unreasonable:
        samples = [
            s
            for s in samples
            if s.get("is_reasonable", True)
            and _loss_within(s.get("centipawn_loss"), MAX_CENTIPAWN_LOSS)
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    for sample in samples:
        converted.append(convert_sample(sample))

    with open(args.output, "w") as fout:
        for train_sample in converted:
            fout.write(json.dumps(train_sample) + "\n")

    n_reasonable = sum(1 for s in converted if s.get("is_reasonable", True))
    manifest = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "raw_samples": total_in,
        "raw_unreasonable": raw_unreasonable,
        "malformed_rows_dropped": len(malformed),
        "unique_fens": len(best_by_fen),
        "output_samples": len(converted),
        "output_reasonable": n_reasonable,
        "filtered_reasonable_only": not args.include_unreasonable,
        "max_centipawn_loss": MAX_CENTIPAWN_LOSS,
        "sample": len(converted) <= SAMPLE_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_template": "build_chat_prompt() in chessfm.config",
    }
    manifest_path = args.output.with_name("data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Converted: {len(converted)} samples (from {total_in} raw, {len(best_by_fen)} unique FENs)"
    )
    print(f"Output:    {args.output}")
    print(f"Manifest:  {manifest_path}")

    if converted:
        print("\nExample output:")
        print(json.dumps(converted[0], indent=2)[:500] + "...")

    print("\nNote: `sample: true` in the manifest marks the output as a small")
    print(f"reproducibility slice (<= {SAMPLE_THRESHOLD} rows). Generate the full corpus with")
    print("python -m chessfm.data_generation.generate_synthetic_data.")


def _loss_within(cp, threshold: float) -> bool:
    if cp is None:
        return False
    return float(cp) <= threshold


if __name__ == "__main__":
    main()
