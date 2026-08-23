"""
Generate seed/curated FEN positions for initial testing and small training runs.

USAGE:
    python -m chessfm.data_generation.download_positions
    python -m chessfm.data_generation.download_positions --count 1000
"""

import argparse
import random
from pathlib import Path

from chessfm.config import PROJECT_ROOT

# Curated list of common chess positions for training
# These are real positions from standard openings and middlegames

OPENING_POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # After 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    # After 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # After 1.e4 e5 2.Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # After 1.e4 e5 2.Nf3 Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Italian Game: 3...Bc5
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Italian Game: 3...Nf6 (Two Knights)
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Ruy Lopez: 3.Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Ruy Lopez: 3...a6
    "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    # Ruy Lopez: 4.Ba4
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",
    # Sicilian Defense: 1.e4 c5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # Sicilian: 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    # Sicilian: 2...d6
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    # Sicilian: 2...Nc6
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    # Sicilian Najdorf: 5...a6
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    # French Defense: 1.e4 e6
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # French: 2.d4 d5
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    # Caro-Kann: 1.e4 c6
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # Scandinavian: 1.e4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    # Queen's Gambit: 1.d4 d5 2.c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    # QGD: 2...e6
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    # QGA: 2...dxc4
    "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    # Slav: 2...c6
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    # King's Indian: 1.d4 Nf6 2.c4 g6
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    # King's Indian: 3.Nc3 Bg7
    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # Nimzo-Indian: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # English: 1.c4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    # London System: 1.d4 d5 2.Bf4
    "rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2",
]

MIDDLEGAME_POSITIONS = [
    # Italian Game mainline
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6",
    # Ruy Lopez closed
    "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 b - - 0 8",
    # Sicilian Dragon
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ - 0 9",
    # French Advance
    "rnbqk2r/ppp2ppp/4pn2/3pP3/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 0 5",
    # Queen's Gambit Declined
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5",
    # King's Indian Classical
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 6",
    # Caro-Kann Advance
    "rnbqkb1r/pp3ppp/4pn2/3pP3/3P4/8/PPP1NPPP/R1BQKBNR b KQkq - 0 5",
    # Semi-Slav
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    # Catalan
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 2 5",
    # Grunfeld
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    # Various tactical positions
    "r2qkb1r/ppp2ppp/2n1bn2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 5",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 b - - 0 7",
    "r2q1rk1/ppp2ppp/2n1bn2/3pp3/2B1P3/3P1N2/PPPN1PPP/R1BQR1K1 b - - 0 8",
    "r1bqr1k1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQ1RK1 b - - 0 8",
]

ENDGAME_POSITIONS = [
    # King and pawn endings
    "8/8/8/4k3/8/8/4PK2/8 w - - 0 1",
    "8/5k2/8/8/8/8/4PK2/8 w - - 0 1",
    "8/8/4k3/8/3K4/8/4P3/8 w - - 0 1",
    # Rook endings
    "8/8/8/4k3/8/8/4R3/4K3 w - - 0 1",
    "8/8/4k3/8/8/8/R7/4K3 w - - 0 1",
    "4r3/8/4k3/8/8/8/4R3/4K3 w - - 0 1",
    # Queen vs pieces
    "8/8/4k3/8/8/8/4Q3/4K3 w - - 0 1",
    "3r4/8/4k3/8/8/8/4Q3/4K3 w - - 0 1",
]


def generate_positions(count: int, seed: int = 42) -> list[str]:
    """Generate a list of FEN positions.

    The curated lists are small (a few dozen). If ``count`` exceeds them, we
    cycle with shuffling, which necessarily produces duplicates — so we warn
    loudly here rather than silently shipping thousands of repeated FENs. For
    large, diverse position sets use ``fetch_elite_data``.
    """
    all_positions = OPENING_POSITIONS + MIDDLEGAME_POSITIONS + ENDGAME_POSITIONS
    # W293-friendly dedupe: keep source order, drop any accidental duplicates.
    all_positions = list(dict.fromkeys(all_positions))

    if count > len(all_positions):
        print(
            f"Requested {count} positions but only {len(all_positions)} unique curated "
            "positions exist. Cycling with shuffling will DUPLICATE FENs."
        )
        print("For large-scale data, use `python -m chessfm.data_generation.fetch_elite_data`.")

    rng = random.Random(seed)
    result = []
    while len(result) < count:
        batch = all_positions.copy()
        rng.shuffle(batch)
        result.extend(batch)

    return result[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate FEN positions for training")
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of positions to generate (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        # Deliberately NOT positions.txt: that file is the elite-FEN pool
        # produced by fetch_elite_data, and this toy curated generator must
        # never overwrite it by default.
        default=PROJECT_ROOT / "data_generation" / "curated_positions.txt",
        help="Output file",
    )
    args = parser.parse_args()

    print(f"Generating {args.count} FEN positions...")

    positions = generate_positions(args.count)

    with open(args.output, "w") as f:
        for fen in positions:
            f.write(fen + "\n")

    unique = len(set(positions))
    if unique < len(positions):
        print(f"Saved {len(positions)} positions to {args.output} ({unique} unique)")
    else:
        print(f"Saved {len(positions)} unique positions to {args.output}")
    print("\nBreakdown:")
    print(f"   Opening positions:   {len(OPENING_POSITIONS)}")
    print(f"   Middlegame positions: {len(MIDDLEGAME_POSITIONS)}")
    print(f"   Endgame positions:   {len(ENDGAME_POSITIONS)}")
    print("\nFor large-scale data generation, use fetch_elite_data.py")


if __name__ == "__main__":
    main()
