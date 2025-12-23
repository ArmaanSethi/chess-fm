#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ChessFM SFT Data Generator                                  ║
║                    ─────────────────────────────                              ║
║                    Generate reasoning traces for chess using FREE AI credits   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

SUPPORTED PROVIDERS (All FREE):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Provider     │ Model                │ Quality │ How to Set Up                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Antigravity  │ gemini-3-pro         │ ⭐⭐⭐⭐⭐  │ You already have it! (this IDE)│
│              │ claude-sonnet-4.5    │ ⭐⭐⭐⭐⭐  │                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Kiro         │ claude-sonnet-4.5    │ ⭐⭐⭐⭐⭐  │ Sign up at kiro.dev            │
│              │ claude-opus-4.5      │ ⭐⭐⭐⭐⭐  │ 500 free credits on signup     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Qwen Code    │ qwen3-coder-plus     │ ⭐⭐⭐⭐   │ Alibaba Cloud account          │
└─────────────────────────────────────────────────────────────────────────────────┘

SETUP:
    1. Install the AIClient-2-API proxy:
       git clone https://github.com/justlovemaki/AIClient-2-API
       cd AIClient-2-API
       chmod +x install-and-run.sh && ./install-and-run.sh

    2. Open http://localhost:3000 in browser

    3. Configure your providers:
       - Antigravity: Already authorized if you're using this IDE
       - Kiro: Download from kiro.dev, login, credentials auto-saved
       - Qwen: Login via Alibaba Cloud

    4. Run this script:
       pip install openai python-chess
       python generate_sft_data_proxy.py

USAGE EXAMPLES:
    # Use Antigravity's Gemini 3 Pro (RECOMMENDED - best reasoning)
    python generate_sft_data_proxy.py --model gemini-3-pro

    # Use Kiro's Claude (excellent reasoning traces)
    python generate_sft_data_proxy.py --model claude-sonnet-4.5

    # Use Qwen Code (good for chess logic)
    python generate_sft_data_proxy.py --model qwen3-coder-plus

    # Generate specific number of samples
    python generate_sft_data_proxy.py --samples 500 --model gemini-3-pro

OUTPUT:
    Data is saved to: data/sft_data.jsonl
    Each line contains:
    {
        "fen": "position",
        "reasoning": "<think>...</think>",
        "move": "e2e4",
        "model": "gemini-3-pro",
        ...
    }
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
    print("❌ Missing dependency: openai")
    print("   Run: pip install openai")
    exit(1)

try:
    import chess
    import chess.engine
except ImportError:
    print("❌ Missing dependency: python-chess")
    print("   Run: pip install python-chess")
    exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Proxy URL (AIClient-2-API default)
DEFAULT_PROXY_URL = "http://localhost:3000/v1"

# Best models for chess reasoning (in order of preference)
RECOMMENDED_MODELS = {
    "gemini-3-pro": "Antigravity - Best overall reasoning",
    "claude-sonnet-4.5": "Kiro/Antigravity - Excellent chain-of-thought",
    "claude-opus-4.5": "Kiro - Most powerful (if you have credits)",
    "qwen3-coder-plus": "Qwen Code - Good logic, fast",
}

DEFAULT_MODEL = "gemini-3-pro"  # Best for SFT data

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "sft_data.jsonl"

# Stockfish path (for move validation)
# macOS (Homebrew): /opt/homebrew/bin/stockfish
# Linux: /usr/bin/stockfish
# Windows: C:\stockfish\stockfish.exe
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# Rate limiting (requests per minute)
# Adjust based on your provider's limits
# Antigravity/Kiro: ~30 RPM is safe
# Qwen: ~20 RPM is safe
RPM = 30

# Move quality threshold (centipawns)
# Moves losing more than this are flagged as weak
CENTIPAWN_THRESHOLD = 200


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a chess grandmaster explaining your thought process to a student.

Given a chess position in FEN notation, analyze it and explain your reasoning, then choose the best move.

Your response MUST follow this EXACT format:
<think>
[Your analysis here - 50 to 200 words]
- First, assess the position (material, king safety, piece activity)
- Identify any immediate threats or tactics
- Consider 2-3 candidate moves
- Explain why you chose your move
</think>
[Your move in UCI format]

IMPORTANT:
- UCI format examples: e2e4, g1f3, e1g1 (kingside castle), e1c1 (queenside castle)
- Your move must be on a new line AFTER the </think> tag
- Do NOT include "Best move:" or any other text before the move

Example response:
<think>
The position after 1.e4 is the King's Pawn opening. White claims central space and opens lines for the bishop and queen.

As Black, I have several good responses:
- e7e5: Symmetric, fights for the center directly
- c7c5: Sicilian Defense, asymmetric and fighting
- e7e6: French Defense, solid but cramped

I'll play e5, the most principled response, directly contesting the center.
</think>
e7e5"""

USER_TEMPLATE = """Position (FEN): {fen}
Side to move: {color}

Analyze this position and choose the best move."""


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLE POSITIONS (used if positions.txt doesn't exist)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_POSITIONS = [
    # Opening phase
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    # Middlegame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6",
    "r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R b KQkq - 0 6",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b - - 0 7",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_positions(filepath: Path) -> list[str]:
    """Load FEN positions from file, or use samples."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            positions = [line.strip() for line in f 
                        if line.strip() and not line.startswith('#')]
        print(f"📂 Loaded {len(positions)} positions from {filepath}")
        return positions
    else:
        print(f"📂 No {filepath.name} found, using {len(SAMPLE_POSITIONS)} sample positions")
        print(f"   Tip: Create {filepath} with FEN positions (one per line)")
        return SAMPLE_POSITIONS


def setup_client(proxy_url: str) -> OpenAI:
    """Initialize OpenAI-compatible client for proxy."""
    print(f"🔌 Connecting to proxy: {proxy_url}")
    return OpenAI(
        base_url=proxy_url,
        api_key="not-needed"  # Proxy handles authentication
    )


def setup_stockfish() -> chess.engine.SimpleEngine | None:
    """Initialize Stockfish for move validation."""
    path = Path(STOCKFISH_PATH)
    if not path.exists():
        print(f"⚠️  Stockfish not found at {STOCKFISH_PATH}")
        print("   Install: brew install stockfish (macOS)")
        print("   Or: sudo apt install stockfish (Linux)")
        print("   Continuing without move validation...")
        return None
    
    print(f"♟️  Stockfish ready: {STOCKFISH_PATH}")
    return chess.engine.SimpleEngine.popen_uci(str(path))


def parse_response(text: str) -> tuple[str, str]:
    """Extract <think> reasoning and UCI move from response."""
    
    # Extract thinking
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # Extract UCI move (after </think>)
    uci_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)'
    
    # Look for move after </think>
    after_think = text.split('</think>')[-1] if '</think>' in text.lower() else text
    moves = re.findall(uci_pattern, after_think.lower())
    
    if moves:
        move = moves[0]
    else:
        # Fallback: last UCI-like string in entire text
        all_moves = re.findall(uci_pattern, text.lower())
        move = all_moves[-1] if all_moves else ""
    
    return thinking, move


def validate_move(board: chess.Board, uci: str, engine) -> dict:
    """Check if move is legal and evaluate quality with Stockfish."""
    result = {
        "is_legal": False,
        "is_reasonable": True,
        "centipawn_loss": None,
        "best_move": None
    }
    
    # Check legality
    try:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return result
        result["is_legal"] = True
    except:
        return result
    
    # Skip engine analysis if no Stockfish
    if engine is None:
        return result
    
    try:
        # Get best move evaluation
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        best_move = info["pv"][0]
        best_score = info["score"].white().score(mate_score=10000)
        result["best_move"] = best_move.uci()
        
        # Evaluate proposed move
        board.push(move)
        after_info = engine.analyse(board, chess.engine.Limit(depth=15))
        move_score = -after_info["score"].white().score(mate_score=10000)
        board.pop()
        
        # Calculate centipawn loss
        if best_score is not None and move_score is not None:
            if board.turn == chess.BLACK:
                best_score, move_score = -best_score, -move_score
            cp_loss = best_score - move_score
            result["centipawn_loss"] = cp_loss
            result["is_reasonable"] = cp_loss <= CENTIPAWN_THRESHOLD
    except Exception as e:
        pass  # Engine error, assume move is OK
    
    return result


def generate_sample(client: OpenAI, model: str, engine, fen: str) -> dict | None:
    """Generate one training sample."""
    try:
        board = chess.Board(fen)
        color = "White" if board.turn == chess.WHITE else "Black"
        
        # Call API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(fen=fen, color=color)}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        text = response.choices[0].message.content
        if not text:
            return None
        
        # Parse
        thinking, move = parse_response(text)
        if not thinking or not move:
            print("    ⚠️  Could not parse response")
            return None
        
        # Validate
        validation = validate_move(board, move, engine)
        
        if not validation["is_legal"]:
            print(f"    ❌ Illegal: {move}")
            return None
        
        return {
            "fen": fen,
            "color": color,
            "reasoning": f"<think>\n{thinking}\n</think>",
            "move": move,
            "model": model,
            "is_reasonable": validation["is_reasonable"],
            "centipawn_loss": validation["centipawn_loss"],
            "stockfish_best": validation["best_move"],
            "raw_response": text,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate chess SFT data using free AI credits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (all FREE via AIClient-2-API proxy):
  gemini-3-pro       Best overall (Antigravity) ⭐ RECOMMENDED
  claude-sonnet-4.5  Excellent reasoning (Kiro/Antigravity)
  claude-opus-4.5    Most powerful if you have credits (Kiro)
  qwen3-coder-plus   Good and fast (Qwen Code)

Examples:
  python generate_sft_data_proxy.py --model gemini-3-pro --samples 100
  python generate_sft_data_proxy.py --model claude-sonnet-4.5
        """
    )
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL,
                        help=f"Proxy URL (default: {DEFAULT_PROXY_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--samples", type=int, default=0,
                        help="Max samples to generate (0 = all positions)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help=f"Output file (default: {OUTPUT_FILE})")
    parser.add_argument("--rpm", type=int, default=RPM,
                        help=f"Requests per minute (default: {RPM})")
    args = parser.parse_args()
    
    # Banner
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║           ChessFM SFT Data Generator                          ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Model:  {args.model}")
    print(f"  Proxy:  {args.proxy_url}")
    print(f"  Output: {args.output}")
    print(f"  Rate:   {args.rpm} requests/min")
    print()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load positions
    positions_file = Path(__file__).parent / "positions.txt"
    positions = load_positions(positions_file)
    random.shuffle(positions)
    
    if args.samples > 0:
        positions = positions[:args.samples]
    
    # Setup
    client = setup_client(args.proxy_url)
    engine = setup_stockfish()
    
    # Progress tracking
    generated = 0
    failed = 0
    start = time.time()
    
    # Count existing
    existing = 0
    if args.output.exists():
        with open(args.output, 'r') as f:
            existing = sum(1 for _ in f)
        print(f"📄 Appending to {existing} existing samples")
    
    print()
    print(f"🚀 Generating from {len(positions)} positions...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        with open(args.output, 'a') as f:
            for i, fen in enumerate(positions):
                print(f"[{i+1}/{len(positions)}] {fen[:35]}...")
                
                sample = generate_sample(client, args.model, engine, fen)
                
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                    generated += 1
                    
                    status = "✅" if sample["is_reasonable"] else "⚠️ "
                    cp = sample["centipawn_loss"]
                    cp_str = f" ({cp:+d}cp)" if cp is not None else ""
                    print(f"    {status} {sample['move']}{cp_str}")
                else:
                    failed += 1
                
                # Rate limit
                time.sleep(60 / args.rpm)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    finally:
        if engine:
            engine.quit()
        
        elapsed = time.time() - start
        total = existing + generated
        
        print()
        print("═" * 50)
        print("📊 Summary")
        print(f"   ✅ Generated: {generated}")
        print(f"   ❌ Failed:    {failed}")
        print(f"   � Total:     {total} samples")
        print(f"   ⏱️  Time:      {elapsed/60:.1f} min")
        print(f"   � Saved to:  {args.output}")
        print()


if __name__ == "__main__":
    main()
