#!/usr/bin/env python3
"""
Generate synthetic reasoning traces for ChessFM SFT.

This is the clean, reproducible version of the earlier "proxy distillation"
experiment. It queries a teacher LLM (OpenAI / Anthropic / Gemini) for
reasoning over a FEN, then hard-validates the proposed move with Stockfish,
recording a ``centipawn_loss`` so downstream conversion can filter to
reasonable moves.

Design goals
------------
- No secrets in the repo: API keys come from environment variables only.
- Deterministic, resumable: (optionally) a Hugging Face-style cache file lets
  an interrupted run pick up where it left off.
- Stockfish-validated output, mirroring exactly what ``convert_to_training.py``
  consumes.

Keys (set in your shell, never commit them):
    OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY

USAGE:
    python -m chessfm.data_generation.generate_synthetic_data \
        --provider openai --model <model> --positions data_generation/positions.txt \
        --limit 500 --output data_generation/full/all_sft_data.jsonl
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import chess

from chessfm.config import MAX_CENTIPAWN_LOSS, PROJECT_ROOT, make_prompt
from chessfm.fens import read_fens
from chessfm.rl.engine import best_cp, best_move, close_engine, eval_after_move, open_engine
from chessfm.rl.move_utils import parse_move_from_output


class ProviderAPI:
    """Minimal adapter around teacher-LLM chat endpoints."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._openai(prompt)
        if self.provider == "anthropic":
            return self._anthropic(prompt)
        if self.provider == "gemini":
            return self._gemini(prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _openai(self, prompt: str) -> str:
        import requests

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        import requests

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 512,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _gemini(self, prompt: str) -> str:
        import requests

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )
        resp = requests.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


SYSTEM_PROMPT = (
    "You are a strong chess player. Given a position in FEN notation, reason "
    "briefly about the position inside <think>...</think> tags, then output "
    "the single best move in UCI format on the final line (e.g. e2e4). Keep the "
    "reasoning concise and correct."
)


def validate_with_stockfish(engine, fen: str, move_uci: str, depth: int = 12):
    """Validate a proposed move; returns (is_legal, centipawn_loss, best_move).

    ``centipawn_loss`` is clamped to >= 0 (a move cannot beat Stockfish's best).
    ``best_move`` is Stockfish's top move in UCI, or None if the engine did not
    report a principal variation.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return False, None, None

        best = best_cp(engine, board, depth)
        chosen = eval_after_move(engine, board, move_uci, depth)
        if best is None or chosen is None:
            return False, None, None
        loss = max(0.0, best - chosen)
        return True, round(loss, 1), best_move(engine, board, depth)
    except ValueError:
        return False, None, None


def load_positions(path: Path, limit: int | None) -> list:
    return read_fens(path, limit)


def resume_from(output_path: Path):
    """Return the set of FENs already processed, for resumable runs."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path) as f:
        for line in f:
            if line.strip():
                try:
                    done.add(json.loads(line)["fen"])
                except Exception:
                    pass
    return done


def gen_batch(
    api: ProviderAPI, engine, fens, output_path: Path, depth: int = 12, delay: float = 1.0
):
    done = resume_from(output_path)
    total, kept, legal, skipped = len(fens), 0, 0, 0
    out = open(output_path, "a")
    try:
        for i, fen in enumerate(fens):
            if fen in done:
                continue
            try:
                response = api.complete(make_prompt(fen))
            except Exception as e:
                print(f"  API error on {i}: {e}")
                time.sleep(delay * 3)
                continue
            move = parse_move_from_output(response)
            if not move:
                skipped += 1
                continue
            is_legal, loss, best = validate_with_stockfish(engine, fen, move, depth)
            if not is_legal or loss is None:
                # Unvalidated or illegal moves are never written: a row without
                # a Stockfish evaluation is unusable for SFT and only pollutes
                # the corpus (see the data-quality note in the README).
                skipped += 1
                continue
            legal += 1
            row = {
                "fen": fen,
                "color": "White" if fen.split()[1] == "w" else "Black",
                "reasoning": response,
                "move": move,
                "model": f"{api.provider}:{api.model}",
                "is_reasonable": loss <= MAX_CENTIPAWN_LOSS,
                "centipawn_loss": loss,
                "stockfish_best": best,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            kept += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{total} processed, {kept} kept, {legal} legal, {skipped} skipped")
            time.sleep(delay)
    finally:
        out.close()
    print(f"Done: {kept} samples kept ({legal} legal, {skipped} skipped).")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT traces")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], required=True)
    parser.add_argument("--model", required=True, help="Teacher model id (e.g. gpt-4o)")
    parser.add_argument("--positions", type=Path, required=True, help="FEN list (one per line)")
    parser.add_argument("--limit", type=int, default=None, help="Cap on positions to process")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_generation" / "full" / "all_sft_data.jsonl",
    )
    parser.add_argument("--depth", type=int, default=12, help="Stockfish search depth")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    args = parser.parse_args()

    key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }[args.provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"Missing {key_env}. Set it in your environment (never commit it).")

    fens = load_positions(args.positions, args.limit)
    print(f"Loaded {len(fens)} positions from {args.positions}")

    engine = open_engine()
    if engine is None:
        raise SystemExit("Stockfish not found; it is required to validate moves.")

    try:
        api = ProviderAPI(args.provider, args.model, api_key)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gen_batch(api, engine, fens, args.output, depth=args.depth, delay=args.delay)
    finally:
        close_engine(engine)
    print(f"Output written to {args.output} (gitignored).")


if __name__ == "__main__":
    main()
