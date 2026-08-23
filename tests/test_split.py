#!/usr/bin/env python3
"""Unit tests for the FEN-level train/validation split."""

from chessfm.training.split import fen_level_split


def _rows(n: int, fens=None):
    return [
        {"fen": fens[i] if fens else f"fen-{i}", "prompt": f"p{i}", "completion": f"c{i}"}
        for i in range(n)
    ]


def test_separates_fens():
    rows = _rows(10)
    train_idx, val_idx, val_fens = fen_level_split(rows, val_fraction=0.3, seed=42)
    train_fens = {rows[i]["fen"] for i in train_idx}
    val_fens_set = {rows[i]["fen"] for i in val_idx}
    assert not (train_fens & val_fens_set)
    assert len(val_fens) == len(val_idx) == 3
    assert len(train_idx) == 7


def test_deterministic_given_seed():
    rows = _rows(100)
    assert fen_level_split(rows, seed=7) == fen_level_split(rows, seed=7)


def test_multi_row_per_fen_stays_together():
    # 10 rows all sharing one FEN -> the whole group lands in one split.
    rows = _rows(10, fens=["same-fen"] * 10)
    train_idx, val_idx, _ = fen_level_split(rows, val_fraction=0.5, seed=1)
    assert len(val_idx) in (0, 10)
    assert len(train_idx) == 10 - len(val_idx)


def test_empty_rows():
    train_idx, val_idx, val_fens = fen_level_split([], val_fraction=0.5, seed=42)
    assert train_idx == [] and val_idx == [] and val_fens == set()
