#!/usr/bin/env python3
"""Unit tests for ChessFM central configuration."""

from chessfm.config import (
    CHATML_ASSISTANT_OPEN,
    CHATML_USER_OPEN,
    PROMPT_TEMPLATE,
    build_chat_prompt,
    make_prompt,
)


class TestPrompts:
    def test_make_prompt_white_to_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        prompt = make_prompt(fen)
        assert "Side to move: Black" in prompt
        assert fen in prompt

    def test_make_prompt_black_to_move(self):
        fen = "8/8/8/4k3/8/8/4PK2/8 w - - 0 1"
        prompt = make_prompt(fen)
        assert "Side to move: White" in prompt

    def test_make_prompt_contains_template_fields(self):
        prompt = make_prompt("8/8/8/8/8/8/8/8 w - - 0 1")
        assert "Analyze this position and choose the best move." in prompt
        assert "<think>..." in prompt

    def test_build_chat_prompt_wraps_in_chatml(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        chat = build_chat_prompt(fen)
        assert chat.startswith(CHATML_USER_OPEN)
        assert chat.endswith(CHATML_ASSISTANT_OPEN)
        assert "Position (FEN):" in chat
        assert "Side to move: White" in chat

    def test_prompt_template_is_shared_source(self):
        assert "Reason inside <think>...</think> tags" in PROMPT_TEMPLATE
        assert "in UCI format (e.g. e2e4)" in PROMPT_TEMPLATE
