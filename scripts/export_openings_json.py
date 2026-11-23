"""Export ChessFM's ECO opening book to static JSON for the standalone web app.

Reads ``OPENING_BOOK`` from ``chessfm.coach.openings`` and replays each line
with ``python-chess`` to emit canonical 4-field board FENs (``normalize_board_fen``)
so the browser Opening Coach supports 100% transposition-aware lookup without
duplicating the book in JavaScript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess

from chessfm.coach.openings import OPENING_BOOK
from chessfm.data_generation.phase_sampler import normalize_board_fen


def build_openings_payload() -> dict[str, Any]:
    """Build the JSON-serializable opening book payload with precomputed FENs."""
    entries: list[dict[str, Any]] = []
    for entry in OPENING_BOOK:
        board = chess.Board()
        fens: list[str] = []
        moves_uci: list[str] = []
        for san in entry.moves:
            move = board.parse_san(san)
            moves_uci.append(move.uci())
            board.push(move)
            fens.append(normalize_board_fen(board.fen()))
        entries.append(
            {
                "eco": entry.eco,
                "name": entry.name,
                "moves": list(entry.moves),
                "moves_uci": moves_uci,
                "white_plan": entry.white_plan,
                "black_plan": entry.black_plan,
                "common_trap": entry.common_trap,
                "fens": fens,
                "final_fen": fens[-1] if fens else normalize_board_fen(chess.STARTING_FEN),
            }
        )
    return {
        "schema_version": "1.0",
        "count": len(entries),
        "entries": entries,
    }


def export_openings_json(output_path: Path) -> Path:
    """Write the opening book JSON payload to ``output_path``."""
    payload = build_openings_payload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for exporting ``web/data/openings.json``."""
    parser = argparse.ArgumentParser(
        description="Export ChessFM ECO opening book to JSON for the static web app."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/data/openings.json"),
        help="Output JSON path (default: web/data/openings.json).",
    )
    args = parser.parse_args(argv)
    written = export_openings_json(args.output)
    payload = json.loads(written.read_text(encoding="utf-8"))
    print(f"Exported {payload['count']} openings to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
