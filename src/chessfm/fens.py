"""Shared file I/O helpers for FEN position lists and JSONL records.

Every entry point (data generation, training, evaluation, benchmarks) reads
position lists from plain-text files and records from JSONL. Centralizing the
two operations keeps parsing behavior identical across the pipeline.
"""

import json
from pathlib import Path


def read_fens(path: Path, limit: int | None = None) -> list[str]:
    """Read one FEN per line from ``path``.

    Blank lines are skipped. ``limit`` truncates the result. Raises
    ``FileNotFoundError`` if ``path`` does not exist.
    """
    with open(path) as f:
        fens = [line.strip() for line in f if line.strip()]
    return fens[:limit] if limit else fens


def read_jsonl(path: Path) -> list[dict]:
    """Read JSON records from a JSONL file, skipping blank lines.

    Raises ``JSONDecodeError`` on malformed rows: silently dropping corrupt
    records would hide data-integrity problems.
    """
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                out.append(json.loads(line_str))
    return out
