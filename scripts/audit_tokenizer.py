import sys
from transformers import AutoTokenizer

MODELS = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    # "meta-llama/Llama-3.2-1B", # Access usually restricted or requires login
    # "deepseek-ai/deepseek-coder-1.3b-base" 
]

SAMPLE_PGN = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
SAMPLE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def audit_model(model_name):
    print(f"Auditing {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        return

    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Check simple PGN tokenization
    tokens = tokenizer.tokenize(SAMPLE_PGN)
    print(f"PGN Tokens: {tokens}")
    print(f"PGN Token IDs: {tokenizer.encode(SAMPLE_PGN)}")
    
    # Check FEN tokenization
    tokens_fen = tokenizer.tokenize(SAMPLE_FEN)
    print(f"FEN Tokens: {tokens_fen}")
    
    # specific check for pieces
    pieces = ["Nf3", "Bxc6", "O-O", "e4", "Qxd5"]
    print("Piece tokenization consistency:")
    for p in pieces:
        print(f"  {p}: {tokenizer.tokenize(p)}")

if __name__ == "__main__":
    for model in MODELS:
        audit_model(model)
