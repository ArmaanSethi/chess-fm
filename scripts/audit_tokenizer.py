import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_BASE_MODEL
from transformers import AutoTokenizer

MODELS = [
    DEFAULT_BASE_MODEL,
]

SAMPLE_UCI = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"
SAMPLE_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def audit_model(model_name):
    print(f"Auditing {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer for {model_name}: {e}")
        return

    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Check UCI tokenization
    tokens_uci = tokenizer.tokenize(SAMPLE_UCI)
    print(f"UCI Tokens: {tokens_uci}")
    print(f"UCI Token IDs: {tokenizer.encode(SAMPLE_UCI)}")
    
    # Check FEN tokenization
    tokens_fen = tokenizer.tokenize(SAMPLE_FEN)
    print(f"FEN Tokens: {tokens_fen}")
    
    # Check specific move tokenization consistency
    moves = ["e2e4", "g1f3", "e1g1", "a7a8q", "c7c5"]
    print("UCI Move Tokenization:")
    for m in moves:
        print(f"  {m}: {tokenizer.tokenize(m)}")

if __name__ == "__main__":
    for model in MODELS:
        audit_model(model)
