#!/usr/bin/env python3
"""Fetch FENs from real high-quality games on Lichess."""

import argparse
import io
import logging
import time
from pathlib import Path

import chess
import chess.pgn
import requests

from chessfm.config import PROJECT_ROOT

logger = logging.getLogger("chessfm.fetch_elite_data")

# Elite *human* GM accounts on Lichess. Bot accounts (Stockfish, Leela, Berserk
# etc.) are deliberately excluded: bot-vs-bot games contain engine-typical
# openings and structures that make poor grounding for human-style reasoning
# traces. Extend this list with more verified usernames as needed.
ACCOUNTS = [
    "DrNykterstein",  # Magnus Carlsen
    "GMHikaru",  # Hikaru Nakamura
    "DanielNaroditsky",  # Daniel Naroditsky
    "GMWSO",  # Wesley So
    "Firouzja2003",  # Alireza Firouzja
    "penguingim1",  # Andrew Tang
    "nihalsarin",  # Nihal Sarin
]

# Games fetched per account per run. Lichess allows larger exports with an
# authenticated token; raise this (and add auth) when scaling the pool.
GAMES_PER_ACCOUNT = 300


def fetch_lila_fens(
    limit: int = 25000,
    min_fullmove: int = 3,
    max_fullmove: int = 55,
    games_per_account: int = GAMES_PER_ACCOUNT,
) -> list[str]:
    """Fetch up to ``limit`` unique FENs from elite Lichess games.

    Returns a deterministically sorted list so repeated runs are stable.
    Raises ``SystemExit`` if no positions could be fetched at all (network
    down, API changed, all accounts unavailable) — writing an empty positions
    file would silently break every downstream pipeline stage.
    """
    logger.info(
        "Sourcing %d elite FEN positions (fullmove %d..%d, %d games/account)...",
        limit,
        min_fullmove,
        max_fullmove,
        games_per_account,
    )
    fens: set[str] = set()

    for account in ACCOUNTS:
        logger.info("Fetching games from %s...", account)
        try:
            response = requests.get(
                f"https://lichess.org/api/games/user/{account}?max={games_per_account}",
                headers={
                    "Accept": "application/x-chess-pgn",
                    # Lichess asks automated clients to identify themselves.
                    "User-Agent": "chess-fm data pipeline (github.com/ArmaanSethi/chess-fm)",
                },
                timeout=30,
            )
            if response.status_code != 200:
                logger.warning("Non-200 response for %s: %s", account, response.status_code)
                continue

            pgn = io.StringIO(response.text)
            while len(fens) < limit:
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                board = game.board()
                moves = list(game.mainline_moves())
                if len(moves) < 20:
                    continue

                for move in moves:
                    board.push(move)
                    if (
                        min_fullmove <= board.fullmove_number <= max_fullmove
                        and not board.is_game_over()
                    ):
                        fens.add(board.fen())
                        if len(fens) >= limit:
                            break
            logger.info("Now at %d positions...", len(fens))
        except Exception as exc:
            logger.warning("Failed fetching for account %s: %s", account, exc)

        time.sleep(1)  # Gentle rate limiting for Lichess API
        if len(fens) >= limit:
            break

    if not fens:
        raise SystemExit(
            "Fetched 0 positions from Lichess. Check network access / API availability; "
            "refusing to overwrite the positions file with nothing."
        )
    if len(fens) < limit:
        logger.warning(
            "Only fetched %d/%d requested positions. The pool is smaller than requested; "
            "downstream sampling will see duplicates less often but coverage is reduced.",
            len(fens),
            limit,
        )
    return sorted(fens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=25000)
    parser.add_argument(
        "--min-fullmove",
        type=int,
        default=3,
        help="Minimum fullmove number to include (default: 3, includes openings)",
    )
    parser.add_argument(
        "--max-fullmove",
        type=int,
        default=55,
        help="Maximum fullmove number to include (default: 55, includes endgames)",
    )
    parser.add_argument(
        "--games-per-account",
        type=int,
        default=GAMES_PER_ACCOUNT,
        help="Maximum games to request per GM account (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_generation" / "positions.txt",
        help="Output positions file",
    )
    args = parser.parse_args()

    fens = fetch_lila_fens(
        limit=args.count,
        min_fullmove=args.min_fullmove,
        max_fullmove=args.max_fullmove,
        games_per_account=args.games_per_account,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for fen in fens:
            f.write(fen + "\n")

    print(f"Success! Saved {len(fens)} high-quality positions to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
