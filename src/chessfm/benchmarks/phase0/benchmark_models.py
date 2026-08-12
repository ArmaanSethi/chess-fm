#!/usr/bin/env python3
"""
Phase 0 model benchmark.

Compares candidate base models on legal move rate and format adherence using
the same prompt template and move parser as the production pipeline, so the
baseline is directly comparable to evaluation results.

USAGE:
    python benchmarks/phase0/benchmark_models.py
"""

import time

import chess

from chessfm.config import DEFAULT_BASE_MODEL, PROJECT_ROOT, build_chat_prompt
from chessfm.rl.move_utils import parse_move_from_output

# Models to test
MODELS = [
    DEFAULT_BASE_MODEL,
]

SAMPLE_LIMIT = 50


def load_fens(filename=None):
    if filename is None:
        filename = PROJECT_ROOT / "data_generation" / "positions.txt"
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


def benchmark(model_name, fens):
    print(f"Benchmarking {model_name}...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    legal_moves = 0
    format_adherence = 0
    start_time = time.time()
    total_tokens = 0

    for i, fen in enumerate(fens[:SAMPLE_LIMIT]):
        prompt = build_chat_prompt(fen)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][inputs.input_ids.shape[1] :]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        if "<think>" in completion and "</think>" in completion:
            format_adherence += 1

        move_uci = parse_move_from_output(completion)
        board = chess.Board(fen)
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    legal_moves += 1
            except ValueError:
                pass

        total_tokens += len(generated)

        if i % 10 == 0:
            print(f"Processed {i}/{SAMPLE_LIMIT}...")

    duration = time.time() - start_time
    print(f"\nResults for {model_name}:")
    print(f"  Legal Move Rate:   {legal_moves}/{SAMPLE_LIMIT} ({legal_moves / SAMPLE_LIMIT:.2%})")
    print(
        f"  Format Adherence:  {format_adherence}/{SAMPLE_LIMIT} ({format_adherence / SAMPLE_LIMIT:.2%})"
    )
    print(f"  Inference Speed:   {total_tokens / duration:.2f} tokens/sec")


if __name__ == "__main__":
    import sys

    fens = load_fens()
    if not fens:
        print("No FENs found. Run data_generation/fetch_elite_data.py first.")
        sys.exit(1)

    for model in MODELS:
        benchmark(model, fens)
