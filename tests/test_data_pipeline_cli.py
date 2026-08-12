#!/usr/bin/env python3
"""End-to-end tests for the data-pipeline CLI mains (no network/hardware)."""

import json
import sys

from chessfm.data_generation.convert_to_training import main as convert_main
from chessfm.data_generation.convert_to_training import read_jsonl
from chessfm.data_generation.download_positions import main as download_main

SAMPLE = {
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "color": "black",
    "reasoning": "<think>solid and standard.</think>",
    "move": "e7e5",
    "model": "gpt-test",
    "is_reasonable": True,
    "centipawn_loss": 12,
}

UNREASONABLE = dict(SAMPLE, fen="8/8/8/8/4k3/8/4K3/8 w - - 0 1", centipawn_loss=4000)


def _run(main, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"] + argv)
    main()


class TestConvertCLI:
    def test_converts_and_writes_manifest(self, tmp_path, monkeypatch):
        raw = tmp_path / "all.jsonl"
        out = tmp_path / "out" / "sft_train.jsonl"
        raw.write_text(json.dumps(SAMPLE) + "\n" + json.dumps(UNREASONABLE) + "\n")
        _run(convert_main, ["--input", str(raw), "--output", str(out)], monkeypatch)
        assert out.exists()
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 1  # unreasonable sample filtered by default
        assert rows[0]["prompt"].startswith("<|im_start|>user\nPosition (FEN):")
        assert rows[0]["completion"].endswith("\ne7e5")
        manifest = json.loads(out.with_name("data_manifest.json").read_text())
        assert manifest["raw_samples"] == 2
        assert manifest["output_samples"] == 1

    def test_include_unreasonable_flag(self, tmp_path, monkeypatch):
        raw = tmp_path / "all.jsonl"
        out = tmp_path / "sft_train.jsonl"
        raw.write_text(json.dumps(SAMPLE) + "\n" + json.dumps(UNREASONABLE) + "\n")
        _run(
            convert_main,
            ["--input", str(raw), "--output", str(out), "--include-unreasonable"],
            monkeypatch,
        )
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 2

    def test_missing_input_prints_not_raises(self, tmp_path, monkeypatch, capsys):
        _run(
            convert_main,
            ["--input", str(tmp_path / "nope.jsonl"), "--output", str(tmp_path / "x.jsonl")],
            monkeypatch,
        )
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_read_jsonl_skips_blank_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text(json.dumps(SAMPLE) + "\n\n" + json.dumps(SAMPLE) + "\n")
        assert len(read_jsonl(p)) == 2


class TestDownloadCLI:
    def test_generates_positions_file(self, tmp_path, monkeypatch):
        out = tmp_path / "positions.txt"
        _run(download_main, ["--count", "5", "--output", str(out)], monkeypatch)
        assert out.exists()
        lines = [line.strip() for line in out.open() if line.strip()]
        assert len(lines) == 5
        assert all(len(fen.split()) == 6 for fen in lines)
