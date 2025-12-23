"""
ChessFM SFT Data Generator

Uses Gemini API (free tier) to generate reasoning traces for chess positions.
Validates moves with Stockfish.

Usage:
    pip install google-generativeai python-chess
    export GOOGLE_API_KEY="your-api-key"
    python generate_sft_data.py

Free Tier Limits (Gemini CLI):
    - 60 requests per minute
    - 1,000 requests per day
    - ~15k samples in 2 weeks
"""

import os
import json
import time
import re
import random
from datetime import datetime
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Run: pip install google-generativeai")
    exit(1)

try:
    import chess
    import chess.engine
except ImportError:
    print("Run: pip install python-chess")
    exit(1)

# ============ CONFIG ============
OUTPUT_FILE = "sft_data.jsonl"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Update for your system
REQUESTS_PER_MINUTE = 55  # Stay under 60 limit
DAILY_LIMIT = 950  # Stay under 1000 limit
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
5. Use UCI format (e.g., e2e4, g1f3, e1g1 for castling)"""

USER_PROMPT_TEMPLATE = """Position (FEN): {fen}
Side to move: {color}

Analyze this position and give your best move."""

# ============ SAMPLE POSITIONS ============
# Starting with some common opening/middlegame positions
# In production, load from Lichess database
SAMPLE_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # After 1.e4
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # After 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # After 2.Nf3
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # After 2...Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian Game
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Two Knights
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Scandinavian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # Sicilian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # Open Sicilian
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",  # Sicilian Najdorf setup
]

def load_positions_from_file(filepath: str) -> list[str]:
    """Load FEN positions from a file (one per line)."""
    if not Path(filepath).exists():
        return SAMPLE_POSITIONS
    
    with open(filepath, 'r') as f:
        positions = [line.strip() for line in f if line.strip()]
    return positions

def setup_gemini() -> genai.GenerativeModel:
    """Initialize Gemini API."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        print("Get your key at: https://aistudio.google.com/app/apikey")
        exit(1)
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 1.5 Flash for best free tier limits
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    return model

def setup_stockfish() -> chess.engine.SimpleEngine:
    """Initialize Stockfish engine."""
    if not Path(STOCKFISH_PATH).exists():
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        print("Install: brew install stockfish")
        exit(1)
    
    return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def parse_response(response_text: str) -> tuple[str, str]:
    """Extract thinking and move from Gemini response."""
    # Extract <think>...</think> block
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # Extract move (last word that looks like UCI)
    move_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)'
    moves = re.findall(move_pattern, response_text.lower())
    
    # Get the move after </think>
    after_think = response_text.split('</think>')[-1] if '</think>' in response_text else response_text
    after_moves = re.findall(move_pattern, after_think.lower())
    
    move = after_moves[0] if after_moves else (moves[-1] if moves else "")
    
    return thinking, move

def validate_move(board: chess.Board, move_uci: str, engine: chess.engine.SimpleEngine) -> dict:
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
    
    # Get Stockfish evaluation
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
        cp_loss = best_score - move_score
        result["centipawn_loss"] = cp_loss
        result["is_reasonable"] = cp_loss <= CENTIPAWN_THRESHOLD
    
    return result

def generate_sample(model, engine, fen: str) -> dict | None:
    """Generate one training sample."""
    try:
        board = chess.Board(fen)
        color = "White" if board.turn == chess.WHITE else "Black"
        
        # Call Gemini
        prompt = USER_PROMPT_TEMPLATE.format(fen=fen, color=color)
        response = model.generate_content(prompt)
        
        if not response.text:
            return None
        
        # Parse response
        thinking, move = parse_response(response.text)
        
        if not thinking or not move:
            print(f"  ⚠️ Failed to parse response")
            return None
        
        # Validate with Stockfish
        validation = validate_move(board, move, engine)
        
        if not validation["is_legal"]:
            print(f"  ❌ Illegal move: {move}")
            return None
        
        if not validation["is_reasonable"]:
            print(f"  ⚠️ Move {move} loses {validation['centipawn_loss']}cp (best: {validation['best_move']})")
            # Still save it, but flag it
        
        return {
            "fen": fen,
            "color": color,
            "reasoning": f"<think>\n{thinking}\n</think>",
            "move": move,
            "is_reasonable": validation["is_reasonable"],
            "centipawn_loss": validation["centipawn_loss"],
            "best_move": validation["best_move"],
            "raw_response": response.text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("🧠 ChessFM SFT Data Generator")
    print("=" * 50)
    
    # Load positions
    positions = load_positions_from_file("positions.txt")
    print(f"📂 Loaded {len(positions)} positions")
    
    # Shuffle for variety
    random.shuffle(positions)
    
    # Setup
    print("🔌 Connecting to Gemini API...")
    model = setup_gemini()
    
    print("♟️ Starting Stockfish...")
    engine = setup_stockfish()
    
    # Track progress
    samples_today = 0
    samples_total = 0
    start_time = time.time()
    
    # Output file
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        with open(output_path, 'r') as f:
            samples_total = sum(1 for _ in f)
        print(f"📄 Resuming from {samples_total} existing samples")
    
    print(f"\n🚀 Starting generation...")
    print(f"   Rate limit: {REQUESTS_PER_MINUTE}/min, {DAILY_LIMIT}/day")
    print()
    
    try:
        with open(OUTPUT_FILE, 'a') as f:
            for i, fen in enumerate(positions):
                if samples_today >= DAILY_LIMIT:
                    print(f"\n⏸️ Daily limit reached ({DAILY_LIMIT}). Resume tomorrow!")
                    break
                
                print(f"[{samples_today + 1}/{len(positions)}] Processing: {fen[:30]}...")
                
                sample = generate_sample(model, engine, fen)
                
                if sample:
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                    samples_today += 1
                    samples_total += 1
                    
                    status = "✅" if sample["is_reasonable"] else "⚠️"
                    print(f"  {status} Saved: {sample['move']} (loss: {sample['centipawn_loss']}cp)")
                
                # Rate limiting
                time.sleep(60 / REQUESTS_PER_MINUTE)
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    
    finally:
        engine.quit()
        
        elapsed = time.time() - start_time
        print(f"\n📊 Session Summary")
        print(f"   Generated: {samples_today} samples")
        print(f"   Total: {samples_total} samples")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
