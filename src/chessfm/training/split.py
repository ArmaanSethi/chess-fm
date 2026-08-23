"""FEN-level dataset splitting (pure, no heavy dependencies)."""

import random


def fen_level_split(rows: list[dict], val_fraction: float = 0.15, seed: int = 42):
    """Split rows by unique FEN so no position appears in both train and val.

    Returns ``(train_indices, val_indices, val_fens)``.
    """
    fens = sorted({r["fen"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(fens)
    val_fens = set(fens[: max(1, round(val_fraction * len(fens)))])
    val_indices = [i for i, r in enumerate(rows) if r["fen"] in val_fens]
    val_set = set(val_indices)
    train_indices = [i for i in range(len(rows)) if i not in val_set]
    return train_indices, val_indices, val_fens
