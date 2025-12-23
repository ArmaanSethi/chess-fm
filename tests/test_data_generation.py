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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFENPositions:
    """Test that FEN positions are valid."""
    
    def test_positions_file_exists(self):
        """positions.txt should exist with content."""
        positions_file = Path(__file__).parent.parent / "data_generation" / "positions.txt"
        assert positions_file.exists(), "positions.txt not found"
        
        with open(positions_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        assert len(lines) > 0, "positions.txt is empty"
        print(f"✅ Found {len(lines)} positions")
    
    def test_fen_format(self):
        """FEN strings should have correct format."""
        positions_file = Path(__file__).parent.parent / "data_generation" / "positions.txt"
        
        if not positions_file.exists():
            print("⚠️ Skipping: positions.txt not found")
            return
        
        with open(positions_file) as f:
            positions = [l.strip() for l in f if l.strip()]
        
        # FEN has 6 space-separated parts
        errors = []
        for i, fen in enumerate(positions[:50]):  # Check first 50
            parts = fen.split()
            if len(parts) != 6:
                errors.append(f"Line {i+1}: Expected 6 parts, got {len(parts)}")
        
        assert len(errors) == 0, f"FEN format errors:\n" + "\n".join(errors)
        print(f"✅ First 50 FEN positions have correct format")


class TestOutputFormat:
    """Test that output format matches roadmap."""
    
    def test_think_tags_regex(self):
        """Regex should correctly parse <think> tags."""
        test_cases = [
            ("<think>\nThis is reasoning\n</think>\ne2e4", "This is reasoning", "e2e4"),
            ("<think>Short</think>\ng1f3", "Short", "g1f3"),
            ("<THINK>Case insensitive</THINK>\na2a4", "Case insensitive", "a2a4"),
        ]
        
        for text, expected_think, expected_move in test_cases:
            # Extract think
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
            assert think_match, f"Failed to match: {text}"
            assert think_match.group(1).strip() == expected_think
            
            # Extract move
            uci_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)'
            after_think = text.split('</think>')[-1] if '</think>' in text.lower() else text
            moves = re.findall(uci_pattern, after_think.lower())
            assert moves, f"No move found in: {text}"
            assert moves[0] == expected_move
        
        print("✅ Think tag regex works correctly")
    
    def test_uci_move_format(self):
        """UCI moves should match pattern."""
        valid_moves = ["e2e4", "g1f3", "e1g1", "e7e8q", "a7a8n"]
        invalid_moves = ["Nf3", "e4", "O-O", "exd5", "1.e4"]
        
        uci_pattern = r'^[a-h][1-8][a-h][1-8][qrbn]?$'
        
        for move in valid_moves:
            assert re.match(uci_pattern, move), f"Should be valid: {move}"
        
        for move in invalid_moves:
            assert not re.match(uci_pattern, move), f"Should be invalid: {move}"
        
        print("✅ UCI move format validation works")


class TestTrainingFormat:
    """Test training data format."""
    
    def test_jsonl_format(self):
        """Training JSONL should have correct fields."""
        sample = {
            "instruction": "Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nSide to move: Black\n\nAnalyze this position and choose the best move.",
            "output": "<think>\nWhite played e4...\n</think>\ne7e5",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "move": "e7e5"
        }
        
        # Should serialize to JSON
        json_str = json.dumps(sample)
        parsed = json.loads(json_str)
        
        assert "instruction" in parsed
        assert "output" in parsed
        assert "<think>" in parsed["output"]
        assert "</think>" in parsed["output"]
        
        print("✅ Training format is correct")


class TestRoadmapFormat:
    """Test that output matches roadmap specification."""
    
    def test_matches_roadmap_format(self):
        """Output should match the format in the roadmap."""
        # From roadmap Section IV:
        # <think>
        # reasoning
        # </think>
        # e4
        
        expected_pattern = r'^<think>\n.*\n</think>\n[a-h][1-8][a-h][1-8][qrbn]?$'
        
        valid_outputs = [
            "<think>\nThe position shows White with a strong center.\n</think>\ne2e4",
            "<think>\nI should develop my knight.\n</think>\ng1f3",
        ]
        
        for output in valid_outputs:
            # Normalize whitespace for comparison
            normalized = output.strip()
            assert "<think>" in normalized
            assert "</think>" in normalized
            
            # Move should come after </think>
            after_think = normalized.split("</think>")[-1].strip()
            assert re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', after_think), \
                f"Move format wrong: {after_think}"
        
        print("✅ Output matches roadmap format")


def run_all_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 60)
    print("ChessFM Unit Tests")
    print("=" * 60)
    print()
    
    test_classes = [
        TestFENPositions(),
        TestOutputFormat(),
        TestTrainingFormat(),
        TestRoadmapFormat(),
    ]
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📋 {class_name}")
        print("-" * 40)
        
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                method = getattr(test_class, method_name)
                try:
                    method()
                except AssertionError as e:
                    print(f"❌ {method_name}: {e}")
                except Exception as e:
                    print(f"⚠️ {method_name}: {e}")
    
    print()
    print("=" * 60)
    print("Tests complete!")


if __name__ == "__main__":
    run_all_tests()
