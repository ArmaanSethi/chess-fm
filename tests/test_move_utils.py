#!/usr/bin/env python3
"""Unit tests for the move parsing helpers (no heavy deps required)."""

import unittest

from chessfm.rl.move_utils import extract_fen_from_prompt, parse_move_from_output


class TestParseMove(unittest.TestCase):
    def test_basic_think_format(self):
        self.assertEqual(
            parse_move_from_output("<think>develop the knight</think>\ne2e4"),
            "e2e4",
        )

    def test_newline_separated(self):
        self.assertEqual(
            parse_move_from_output("<think>abc</think>\ne7e8q"),
            "e7e8q",
        )

    def test_no_think_tag(self):
        self.assertEqual(parse_move_from_output("g1f3"), "g1f3")

    def test_promotion_move(self):
        self.assertEqual(parse_move_from_output("a7a8n"), "a7a8n")

    def test_invalid_five_char_token_rejected(self):
        # "e2e4x" is 5 chars but the 5th is not a promotion piece -> rejected.
        self.assertIsNone(parse_move_from_output("<think>x</think> e2e4x"))
        self.assertEqual(
            parse_move_from_output("<think>x</think> a7a8q"),
            "a7a8q",
        )

    def test_no_move_returns_none(self):
        self.assertIsNone(parse_move_from_output("just some text"))
        self.assertIsNone(parse_move_from_output(""))

    def test_trailing_punctuation_stripped(self):
        self.assertEqual(
            parse_move_from_output("<think>x</think> e2e4."),
            "e2e4",
        )

    def test_uppercase_uci(self):
        self.assertEqual(
            parse_move_from_output("<think>x</think> E2E4"),
            "e2e4",
        )


class TestExtractFen(unittest.TestCase):
    def test_extracts_fen(self):
        prompt = "Position (FEN): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nMove:"
        self.assertEqual(
            extract_fen_from_prompt(prompt),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        )

    def test_missing_fen_raises(self):
        with self.assertRaises(ValueError):
            extract_fen_from_prompt("no fen present here")


if __name__ == "__main__":
    unittest.main()
