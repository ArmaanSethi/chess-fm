import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

PROMPTS = {
    "Zero-Shot": "FEN: {fen}\nMove:",
    "Few-Shot": """FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Move: e5

FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
Move: Nc6

FEN: {fen}
Move:""",
    "Chain-of-Thought": "FEN: {fen}\nThink step-by-step about the position, then provide the Move.\n<think>\n",
}

import os

def load_fens(filename=None):
    if filename is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, "../data/fens.txt")
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def test_prompts(fens):
    print(f"Loading {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for name, template in PROMPTS.items():
        print(f"\nTesting Prompt Strategy: {name}")
        legal = 0
        total = 20 # small sample
        
        for fen in fens[:total]:
            prompt = template.format(fen=fen)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=100 if "Think" in name else 10,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Minimal parsing logic
            # ...
            # Check legality
            board = chess.Board(fen)
            # (Simplistic check)
            
        print(f"Strategy {name} finished (metrics pending full impl)")

if __name__ == "__main__":
    fens = load_fens()
    test_prompts(fens)
