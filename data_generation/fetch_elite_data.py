#!/usr/bin/env python3
import io
import time
import argparse
from pathlib import Path

import chess


def _get_requests():
    import requests
    return requests


def _get_chess_pgn():
    import chess.pgn
    return chess.pgn


def fetch_lila_fens(limit=25000):
    """
    Fetch FENs from real high-quality games on Lichess.
    """
    requests = _get_requests()
    chess_pgn = _get_chess_pgn()
    print(f"🌐 Sourcing {limit} elite FEN positions...")
    fens = set()
    
    try:
        # Top engine + Top GM accounts on Lichess for diverse elite data
        accounts = ["leela-chess-zero", "Stockfish", "Maverick", "Berserk", "DrNykterstein", "nihalsarin", "RebeccaHarris"]
        
        for account in accounts:
            print(f"📥 Fetching games from {account}...")
            try:
                response = requests.get(f"https://lichess.org/api/games/user/{account}?max=200",
                                      headers={"Accept": "application/x-chess-pgn"},
                                      timeout=30)
                if response.status_code == 200:
                    pgn = io.StringIO(response.text)
                    while len(fens) < limit:
                        game = chess_pgn.read_game(pgn)
                        if not game:
                            break
                        board = game.board()
                        if len(list(game.mainline_moves())) < 20:
                            continue

                        for move in game.mainline_moves():
                            board.push(move)
                            if 10 < board.fullmove_number < 45 and not board.is_game_over():
                                try:
                                    # Validate FEN
                                    b_check = chess.Board(board.fen())
                                    fens.add(b_check.fen())
                                except Exception:
                                    pass
                                if len(fens) >= limit:
                                    break
                    print(f"   Now at {len(fens)} positions...")
            except Exception as req_err:
                print(f"⚠️ Failed fetching for account {account}: {req_err}")

            time.sleep(1) # Gentle rate limiting for Lichess API
            if len(fens) >= limit:
                break
                            
    except Exception as e:
        print(f"⚠️ Error during fetch: {e}")
        
    return list(fens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=25000)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "positions.txt")
    args = parser.parse_args()

    fens = fetch_lila_fens(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w") as f:
        for fen in fens:
            f.write(fen + "\n")
            
    print(f"✅ Success! Saved {len(fens)} high-quality positions to {args.output}")

if __name__ == "__main__":
    main()
