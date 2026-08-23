#!/usr/bin/env python3
"""Unit tests for the shared FEN/JSONL readers (chessfm.fens)."""

import json

import pytest

from chessfm.fens import read_fens, read_jsonl


def test_read_fens_skips_blank_lines_and_strips(tmp_path):
    p = tmp_path / "fens.txt"
    p.write_text("  fen-a  \n\nfen-b\n")
    assert read_fens(p) == ["fen-a", "fen-b"]


def test_read_fens_limit(tmp_path):
    p = tmp_path / "fens.txt"
    p.write_text("a\nb\nc\n")
    assert read_fens(p, limit=2) == ["a", "b"]


def test_read_fens_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fens(tmp_path / "nope.txt")


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_raises_on_malformed_row(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a": 1}\nnot-json\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(p)
