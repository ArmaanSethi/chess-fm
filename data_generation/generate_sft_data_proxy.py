"""
ChessFM SFT Data Generator (Proxy Version)

Uses AIClient-2-API proxy to access free AI credits from multiple sources.
Validates moves with Stockfish.

=== SETUP ===

1. Clone and run the proxy:
   git clone https://github.com/justlovemaki/AIClient-2-API
   cd AIClient-2-API
   chmod +x install-and-run.sh && ./install-and-run.sh

2. Open http://localhost:3000 and configure your credentials (see FREE CREDITS below)

3. Run this script:
   pip install openai python-chess
   python generate_sft_data_proxy.py

=== FREE CREDITS SOURCES ===

┌─────────────────────────────────────────────────────────────────────────────┐
│ PROVIDER        │ MODEL                  │ FREE LIMIT    │ HOW TO GET      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Gemini CLI      │ gemini-2.5-pro         │ 60 RPM        │ Install Gemini  │
│                 │ gemini-2.5-flash       │ 1000/day      │ CLI, login      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Antigravity     │ gemini-3-pro           │ Generous      │ This tool!      │
│ (Google intern) │ claude-sonnet-4.5      │               │ Auth via Google │
├─────────────────────────────────────────────────────────────────────────────┤
│ Kiro            │ claude-sonnet-4.5      │ 500 credits   │ Sign up at      │
│ (AWS IDE)       │ claude-opus-4.5        │ on signup     │ kiro.dev        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Qwen Code       │ qwen3-coder-plus       │ Generous      │ Alibaba Cloud   │
│                 │                        │               │ account         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Google AI       │ gemini-1.5-flash       │ 1500/day      │ aistudio.google │
│ Studio          │ gemini-1.5-pro         │ 50/day        │ .com/apikey     │
├─────────────────────────────────────────────────────────────────────────────┤
│ OpenRouter      │ Various                │ Some free     │ openrouter.ai   │
│                 │                        │ models        │                 │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMMENDED COMBO: 
  - Gemini CLI (best free tier)
  - + Kiro (free Claude)
  - + Google AI Studio (backup)
  = Thousands of requests per day!

=== USAGE ===

# Basic usage (uses localhost:3000 proxy)
python generate_sft_data_proxy.py

# Custom proxy URL
python generate_sft_data_proxy.py --proxy-url http://localhost:3000/v1

# Custom model
python generate_sft_data_proxy.py --model gemini-3-pro

# Generate specific number of samples
python generate_sft_data_proxy.py --samples 1000
"""

import os
import json
import time
import re
import random
import argparse
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Run: pip install openai")
    exit(1)

try:
    import chess
    import chess.engine
except ImportError:
    print("Run: pip install python-chess")
    exit(1)

# ============ CONFIG ============
DEFAULT_PROXY_URL = "http://localhost:3000/v1"
DEFAULT_MODEL = "gemini-2.5-flash"  # Fast and generous free tier
OUTPUT_FILE = "sft_data.jsonl"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS Homebrew path
# Alternative paths:
# STOCKFISH_PATH = "/usr/bin/stockfish"  # Linux
# STOCKFISH_PATH = "C:\\stockfish\\stockfish.exe"  # Windows

REQUESTS_PER_MINUTE = 50  # Stay safe
CENTIPAWN_THRESHOLD = 200  # Accept moves within 200cp of best

# ============ PROMPT ============
SYSTEM_PROMPT = """You are a chess grandmaster explaining your thought process to a student.

Given a chess position in FEN notation, analyze it and explain your reasoning, then choose the best move.

Your response MUST follow this EXACT format:
<think>
[Your analysis: threats, candidate moves, reasoning for your choice]
</think>
[Your move in UCI format, e.g., e2e4]

Rules:
1. The <think> section should be 50-200 words
2. Consider threats, tactics, and positional factors
3. Mention 2-3 candidate moves before choosing
4. Your final move must be on a new line after </think>
5. Use UCI format (e.g., e2e4, g1f3, e1g1 for castling)

Example:
<think>
White has just played e4, claiming central space. This is the most popular opening move.
I have several options: e5 (symmetrical, solid), c5 (Sicilian, fighting), e6 (French, defensive).
The Sicilian with c5 is the most aggressive response, leading to sharp play.
However, e5 is simpler and leads to more open positions which I'll choose.
</think>
e7e5"""

USER_PROMPT_TEMPLATE = """Position (FEN): {fen}
Side to move: {color}

Analyze this position and give your best move."""

# ============ SAMPLE POSITIONS ============
SAMPLE_POSITIONS = [
    # Common openings
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # After 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # After 2.Nf3
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After 2...Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian Game
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Two Knights
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Scandinavian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Sicilian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # Open Sicilian
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",  # Sicilian Najdorf
    # More middlegame positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",  # Italian castled
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 6 5",  # Giuoco Piano
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",  # After 1.d4
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",  # After 1.d4 Nf6
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",  # After 2.c4
]

def load_positions_from_file(filepath: str) -> list[str]:
    """Load FEN positions from a file (one per line)."""
    path = Path(filepath)
    if not path.exists():
        print(f"📂 No positions.txt found, using {len(SAMPLE_POSITIONS)} sample positions")
        return SAMPLE_POSITIONS
    
    with open(path, 'r') as f:
        positions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"📂 Loaded {len(positions)} positions from {filepath}")
    return positions

def setup_client(proxy_url: str) -> OpenAI:
    """Initialize OpenAI client pointing to proxy."""
    print(f"🔌 Connecting to proxy: {proxy_url}")
    return OpenAI(
        base_url=proxy_url,
        api_key="not-needed-for-proxy"  # Proxy handles auth
    )

def setup_stockfish() -> chess.engine.SimpleEngine:
    """Initialize Stockfish engine."""
    if not Path(STOCKFISH_PATH).exists():
        print(f"⚠️ Stockfish not found at {STOCKFISH_PATH}")
        print("   Install: brew install stockfish (macOS)")
        print("   Or update STOCKFISH_PATH in this script")
        print("   Continuing without move validation...")
        return None
    
    print(f"♟️ Starting Stockfish: {STOCKFISH_PATH}")
    return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def parse_response(response_text: str) -> tuple[str, str]:
    """Extract thinking and move from response."""
    # Extract <think>...</think> block
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # Extract move (UCI format)
    move_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)'
    
    # Get the move after </think>
    after_think = response_text.split('</think>')[-1] if '</think>' in response_text else response_text
    after_moves = re.findall(move_pattern, after_think.lower())
    
    if after_moves:
        move = after_moves[0]
    else:
        # Fallback: get last UCI-looking move in entire response
        all_moves = re.findall(move_pattern, response_text.lower())
        move = all_moves[-1] if all_moves else ""
    
    return thinking, move

def validate_move(board: chess.Board, move_uci: str, engine) -> dict:
    """Check if move is legal and reasonable."""
    result = {
        "is_legal": False,
        "is_reasonable": False,
        "centipawn_loss": None,
        "best_move": None
    }
    
    # Check legality
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return result
        result["is_legal"] = True
    except:
        return result
    
    # If no engine, skip evaluation
    if engine is None:
        result["is_reasonable"] = True
        return result
    
    # Get Stockfish evaluation
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        best_move = info["pv"][0]
        best_score = info["score"].white().score(mate_score=10000)
        result["best_move"] = best_move.uci()
        
        # Evaluate the proposed move
        board.push(move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=15))
        move_score = -info_after["score"].white().score(mate_score=10000)
        board.pop()
        
        # Calculate centipawn loss
        if best_score is not None and move_score is not None:
            # Flip sign based on side to move
            if board.turn == chess.BLACK:
                best_score = -best_score
                move_score = -move_score
            cp_loss = best_score - move_score
            result["centipawn_loss"] = cp_loss
            result["is_reasonable"] = cp_loss <= CENTIPAWN_THRESHOLD
    except Exception as e:
        print(f"    ⚠️ Engine error: {e}")
        result["is_reasonable"] = True  # Assume OK if engine fails
    
    return result

def generate_sample(client: OpenAI, model: str, engine, fen: str) -> dict | None:
    """Generate one training sample."""
    try:
        board = chess.Board(fen)
        color = "White" if board.turn == chess.WHITE else "Black"
        
        # Call API via proxy
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(fen=fen, color=color)}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        if not response_text:
            return None
        
        # Parse response
        thinking, move = parse_response(response_text)
        
        if not thinking or not move:
            print(f"    ⚠️ Failed to parse response")
            return None
        
        # Validate with Stockfish
        validation = validate_move(board, move, engine)
        
        if not validation["is_legal"]:
            print(f"    ❌ Illegal move: {move}")
            return None
        
        cp_info = f", loss: {validation['centipawn_loss']}cp" if validation['centipawn_loss'] else ""
        if not validation["is_reasonable"]:
            print(f"    ⚠️ Weak move: {move}{cp_info}")
        
        return {
            "fen": fen,
            "color": color,
            "reasoning": f"<think>\n{thinking}\n</think>",
            "move": move,
            "is_reasonable": validation["is_reasonable"],
            "centipawn_loss": validation["centipawn_loss"],
            "best_move": validation["best_move"],
            "model": model,
            "raw_response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"    ❌ API Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate chess SFT data using AI proxy")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL, 
                        help=f"Proxy URL (default: {DEFAULT_PROXY_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--samples", type=int, default=0,
                        help="Max samples to generate (0 = all positions)")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output file (default: {OUTPUT_FILE})")
    args = parser.parse_args()
    
    print("🧠 ChessFM SFT Data Generator (Proxy Version)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Proxy: {args.proxy_url}")
    print(f"Output: {args.output}")
    print()
    
    # Load positions
    positions = load_positions_from_file("positions.txt")
    random.shuffle(positions)
    
    if args.samples > 0:
        positions = positions[:args.samples]
    
    # Setup
    client = setup_client(args.proxy_url)
    engine = setup_stockfish()
    
    # Track progress
    samples_generated = 0
    samples_failed = 0
    start_time = time.time()
    
    # Check existing data
    output_path = Path(args.output)
    existing_samples = 0
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_samples = sum(1 for _ in f)
        print(f"📄 Appending to {existing_samples} existing samples")
    
    print(f"\n🚀 Generating {len(positions)} samples...")
    print(f"   Rate: ~{REQUESTS_PER_MINUTE}/min")
    print()
    
    try:
        with open(args.output, 'a') as f:
            for i, fen in enumerate(positions):
                print(f"[{i + 1}/{len(positions)}] {fen[:40]}...")
                
                sample = generate_sample(client, args.model, engine, fen)
                
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                    samples_generated += 1
                    
                    status = "✅" if sample["is_reasonable"] else "⚠️"
                    print(f"    {status} {sample['move']}")
                else:
                    samples_failed += 1
                
                # Rate limiting
                time.sleep(60 / REQUESTS_PER_MINUTE)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    
    finally:
        if engine:
            engine.quit()
        
        elapsed = time.time() - start_time
        total = existing_samples + samples_generated
        
        print(f"\n{'=' * 60}")
        print(f"📊 Session Summary")
        print(f"   ✅ Generated: {samples_generated}")
        print(f"   ❌ Failed: {samples_failed}")
        print(f"   📄 Total in file: {total}")
        print(f"   ⏱️ Time: {elapsed/60:.1f} minutes")
        print(f"   📁 Output: {args.output}")

if __name__ == "__main__":
    main()
