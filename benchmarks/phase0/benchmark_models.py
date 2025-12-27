import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import chess

# Models to test
MODELS = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "deepseek-ai/deepseek-coder-1.3b-base",
    "google/gemma-2-2b-it",
]

import os

def load_fens(filename=None):
    if filename is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, "../data/fens.txt")
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def benchmark(model_name, fens):
    print(f"Benchmarking {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    legal_moves = 0
    format_adherence = 0
    start_time = time.time()
    total_tokens = 0

    for i, fen in enumerate(fens[:50]): # Limit to 50 for quick test first
        prompt = f"FEN: {fen}\nBest Move:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=10, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Check format (simple check: is it a short string?)
        if len(response) < 10 and len(response) > 1:
            format_adherence += 1
            
        # Check legality
        board = chess.Board(fen)
        try:
            # Clean up response (take first token/word)
            move_str = response.split()[0] if response else ""
            move = board.parse_san(move_str)
            legal_moves += 1
        except:
            pass
            
        # Count tokens (approx)
        total_tokens += len(output[0]) - inputs.input_ids.shape[1]

        if i % 10 == 0:
            print(f"Processed {i}/50...")

    duration = time.time() - start_time
    print(f"Results for {model_name}:")
    print(f"  Legal Move Rate: {legal_moves}/50 ({legal_moves/50:.2%})")
    print(f"  Format Adherence: {format_adherence}/50 ({format_adherence/50:.2%})")
    print(f"  Inference Speed: {total_tokens/duration:.2f} tokens/sec")

if __name__ == "__main__":
    fens = load_fens()
    if not fens:
        print("No FENs found. Run generate_fens.py first.")
        exit(1)
        
    for model in MODELS:
        benchmark(model, fens)
