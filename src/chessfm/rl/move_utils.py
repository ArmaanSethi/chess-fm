#!/usr/bin/env python3
"""
Pure-python move parsing helpers (no heavy dependencies).

Used by both GRPO training and evaluation so move extraction is identical
everywhere.
"""


def parse_move_from_output(output: str) -> str | None:
    """
    Extract the move from model output.

    Expects output like ``<think>...</think> e2e4``. Returns the last
    UCI-looking token after the closing ``</think>`` tag, or None.
    """
    text = output.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    parts = text.split()
    if not parts:
        return None

    for part in reversed(parts):
        part = part.strip().lower().rstrip(".,!?")
        if 4 <= len(part) <= 5:
            if part[0] in "abcdefgh" and part[1] in "12345678":
                if part[2] in "abcdefgh" and part[3] in "12345678":
                    if len(part) == 4 or part[4] in "qrbn":
                        return part

    return None


def extract_fen_from_prompt(prompt: str) -> str:
    """Extract the FEN string from a ChessFM prompt."""
    for line in prompt.split("\n"):
        if line.startswith("Position (FEN):"):
            return line.replace("Position (FEN):", "").strip()
    raise ValueError(f"Could not find FEN in prompt: {prompt[:200]}")
