#!/usr/bin/env python3
"""
Unit tests for ChessFM data generation.

Run with: python -m pytest tests/ -v
Or just: python tests/test_data_generation.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

POSITIONS = PROJECT_ROOT / "data_generation" / "positions.txt"
SFT_TRAIN = PROJECT_ROOT / "data_generation" / "sft_train.jsonl"
MANIFEST = PROJECT_ROOT / "data_generation" / "data_manifest.json"


class TestFENPositions:
    def test_positions_file_exists(self):
        assert POSITIONS.exists(), "positions.txt not found"
        lines = [line.strip() for line in POSITIONS.open() if line.strip()]
        assert len(lines) > 0, "positions.txt is empty"

    def test_fen_format(self):
        if not POSITIONS.exists():
            return
        positions = [line.strip() for line in POSITIONS.open() if line.strip()][:50]
        errors = []
        for i, fen in enumerate(positions):
            if len(fen.split()) != 6:
                errors.append(f"Line {i + 1}: expected 6 parts, got {len(fen.split())}")
        assert not errors, "FEN format errors:\n" + "\n".join(errors)


class TestOutputFormat:
    def test_think_tags_regex(self):
        test_cases = [
            ("<think>\nThis is reasoning\n</think>\ne2e4", "This is reasoning", "e2e4"),
            ("<think>Short</think>\ng1f3", "Short", "g1f3"),
            ("<THINK>Case insensitive</THINK>\na2a4", "Case insensitive", "a2a4"),
        ]
        for text, expected_think, expected_move in test_cases:
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
            assert think_match, f"Failed to match: {text}"
            assert think_match.group(1).strip() == expected_think

            uci_pattern = r"([a-h][1-8][a-h][1-8][qrbn]?)"
            after_think = text.split("</think>")[-1]
            moves = re.findall(uci_pattern, after_think.lower())
            assert moves, f"No move found in: {text}"
            assert moves[0] == expected_move

    def test_uci_move_format(self):
        valid = ["e2e4", "g1f3", "e1g1", "e7e8q", "a7a8n"]
        invalid = ["Nf3", "e4", "O-O", "exd5", "1.e4"]
        uci_pattern = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
        for move in valid:
            assert uci_pattern.match(move), f"Should be valid: {move}"
        for move in invalid:
            assert not uci_pattern.match(move), f"Should be invalid: {move}"


class TestTrainingFormat:
    def test_jsonl_format(self):
        sample = {
            "prompt": "<|im_start|>user\nPosition (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1<|im_end|>\n<|im_start|>assistant\n",
            "completion": "<think>\nWhite played e4...\n</think>\ne7e5",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "move": "e7e5",
        }
        parsed = json.loads(json.dumps(sample))
        assert "prompt" in parsed
        assert "completion" in parsed
        assert "<think>" in parsed["completion"]
        assert "</think>" in parsed["completion"]

    def test_sft_train_deduped_by_fen(self):
        if not SFT_TRAIN.exists():
            return
        rows = [json.loads(line) for line in SFT_TRAIN.open() if line.strip()]
        fens = {r["fen"] for r in rows}
        assert len(rows) == len(fens), "Duplicate FENs found in sft_train.jsonl"
        for r in rows:
            assert r["completion"].count("<think>") == r["completion"].count("</think>"), (
                f"Unbalanced think tags in {r['fen']}"
            )

    def test_sft_train_prompt_is_chatml(self):
        """SFT prompts must be ChatML-wrapped to match the inference path."""
        if not SFT_TRAIN.exists():
            return
        rows = [json.loads(line) for line in SFT_TRAIN.open() if line.strip()]
        assert len(rows) > 0
        for r in rows:
            assert r["prompt"].startswith("<|im_start|>user\nPosition (FEN):"), (
                "Prompt must be ChatML-wrapped"
            )
            assert r["prompt"].endswith("<|im_start|>assistant\n"), (
                "Prompt must end with assistant open tag"
            )

    def test_data_manifest_exists(self):
        if not MANIFEST.exists():
            return
        data = json.loads(MANIFEST.read_text())
        assert "raw_samples" in data
        assert "unique_fens" in data
        assert "output_samples" in data


class TestRoadmapFormat:
    def test_matches_roadmap_format(self):
        valid_outputs = [
            "<think>\nThe position shows White with a strong center.\n</think>\ne2e4",
            "<think>\nI should develop my knight.\n</think>\ng1f3",
        ]
        for output in valid_outputs:
            normalized = output.strip()
            assert "<think>" in normalized
            assert "</think>" in normalized
            after_think = normalized.split("</think>")[-1].strip()
            assert re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", after_think), (
                f"Move format wrong: {after_think}"
            )
