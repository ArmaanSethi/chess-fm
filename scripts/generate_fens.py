import chess
import chess.pgn
import random
import sys

def generate_random_fens(n=1000):
    fens = []
    # simple random play to get diverse positions
    # Or just load from a file if we had one.
    # Let's play some random games and grab positions.
    
    while len(fens) < n:
        board = chess.Board()
        game_len = random.randint(10, 60)
        for _ in range(game_len):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        fens.append(board.fen())
        if len(fens) % 100 == 0:
            print(f"Generated {len(fens)} FENs...")
            
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "../data/fens.txt")
    with open(output_path, "w") as f:
        for fen in fens:
            f.write(fen + "\n")
    print(f"Saved {len(fens)} FENs to {output_path}")

if __name__ == "__main__":
    generate_random_fens()
