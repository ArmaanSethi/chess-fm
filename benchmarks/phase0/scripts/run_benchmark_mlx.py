import mlx.core as mx
from mlx_lm import load, generate
import chess
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def load_fens(filename="data_generation/positions.txt", limit=None):
    with open(filename, "r") as f:
        fens = [line.strip() for line in f.readlines() if line.strip()]
    if limit:
        fens = fens[:limit]
    return fens

def benchmark_model(model_path, fens, output_file):
    print(f"🚀 Loading {model_path}...")
    model, tokenizer = load(model_path)
    print("✅ Model loaded!")

    results = {
        "model": model_path,
        "total_positions": len(fens),
        "legal_moves": 0,
        "format_adherence": 0,
        "time_elapsed": 0,
        "speed_pos_sec": 0,
        "details": []
    }

    start_time = time.time()
    
    for i, fen in enumerate(tqdm(fens, desc="Benchmarking")):
        prompt = f"<|im_start|>user\\nYou are a chess expert. Output ONLY the best legal move in UCI format (e.g. e2e4) for this position.\\nFEN: {fen}<|im_end|>\\n<|im_start|>assistant\\n"
        
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=1000, verbose=False) # Increased token limit for thinking
            generated_text = response.strip()
            
            # Remove <think> blocks if present
            import re
            clean_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
            
            move_candidate = clean_text.split()[0] if clean_text else ""
        except Exception as e:
            print(f"Error generating for {fen}: {e}")
            move_candidate = ""
            generated_text = ""

        # Check legality
        board = chess.Board(fen)
        is_legal = False
        is_format_ok = False
        
        if len(move_candidate) >= 4:
            is_format_ok = True
            try:
                move = chess.Move.from_uci(move_candidate[:5] if len(move_candidate) >= 4 else "xxxx")
                if move in board.legal_moves:
                    is_legal = True
            except:
                pass

        if is_legal: results["legal_moves"] += 1
        if is_format_ok: results["format_adherence"] += 1

        results["details"].append({
            "fen": fen,
            "output": generated_text,
            "move_candidate": move_candidate,
            "is_legal": is_legal
        })

    end_time = time.time()
    results["time_elapsed"] = end_time - start_time
    results["speed_pos_sec"] = len(fens) / results["time_elapsed"]

    # Calculate percentages
    results["legal_rate"] = (results["legal_moves"] / len(fens)) * 100
    results["format_rate"] = (results["format_adherence"] / len(fens)) * 100

    print(f"\n📊 Results for {model_path}:")
    print(f"  Legal Moves: {results['legal_moves']}/{len(fens)} ({results['legal_rate']:.1f}%)")
    print(f"  Format OK:   {results['format_adherence']}/{len(fens)} ({results['format_rate']:.1f}%)")
    print(f"  Speed:       {results['speed_pos_sec']:.1f} pos/sec")
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    fens = load_fens(limit=args.limit)
    output_path = f"results/benchmark_{args.model.split('/')[-1]}_{args.limit}.json"
    
    benchmark_model(args.model, fens, output_path)
