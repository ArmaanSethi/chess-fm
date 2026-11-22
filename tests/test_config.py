#!/usr/bin/env python3
"""Unit tests for ChessFM central configuration."""

from chessfm.config import (
    ALT_TEACHER_MODEL,
    CHATML_ASSISTANT_OPEN,
    CHATML_USER_OPEN,
    DEFAULT_BASE_MODEL,
    DEFAULT_BROWSER_STUDENT_MODEL,
    DEFAULT_COACH_MODEL,
    DEFAULT_LOCAL_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_V1_5_MODEL,
    DEFAULT_V1_MODEL,
    OFFICIAL_GEMMA_COACH_MODEL,
    PROMPT_TEMPLATE,
    STOCKFISH_PATHS,
    build_chat_prompt,
    get_assistant_role,
    is_gemma4_model,
    make_prompt,
    render_ascii_board,
    requires_modern_transformers,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestConstants:
    def test_model_constants(self):
        assert DEFAULT_BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
        assert DEFAULT_V1_MODEL == "Qwen/Qwen3-4B"
        assert DEFAULT_V1_5_MODEL == "Qwen/Qwen3.5-4B"
        assert DEFAULT_COACH_MODEL == "unsloth/gemma-4-E4B-it"
        assert OFFICIAL_GEMMA_COACH_MODEL == "google/gemma-4-E4B-it"
        assert DEFAULT_BROWSER_STUDENT_MODEL == "Qwen/Qwen3.5-0.8B"
        assert DEFAULT_LOCAL_STUDENT_MODEL == "google/gemma-4-E2B-it"
        assert DEFAULT_TEACHER_MODEL == "google/gemma-4-26B-A4B-it"
        assert ALT_TEACHER_MODEL == "Qwen/Qwen3.6-35B-A3B"

    def test_stockfish_paths_include_usr_games(self):
        assert "/usr/games/stockfish" in STOCKFISH_PATHS

    def test_is_gemma4_model(self):
        assert is_gemma4_model("unsloth/gemma-4-E4B-it") is True
        assert is_gemma4_model("google/gemma-4-E4B-it") is True
        assert is_gemma4_model("local/gemma4-finetuned") is True
        assert is_gemma4_model("Qwen/Qwen2.5-3B-Instruct") is False
        assert is_gemma4_model("Qwen/Qwen3-4B") is False

    def test_requires_modern_transformers(self):
        assert requires_modern_transformers("Qwen/Qwen3.5-4B") is True
        assert requires_modern_transformers("Qwen/Qwen3.5-0.8B") is True
        assert requires_modern_transformers("Qwen/Qwen3.6-35B-A3B") is True
        assert requires_modern_transformers("google/gemma-4-26B-A4B-it") is True
        assert requires_modern_transformers("google/gemma-4-E2B-it") is True
        assert requires_modern_transformers("Qwen/Qwen3-4B") is False
        assert requires_modern_transformers("Qwen/Qwen2.5-3B-Instruct") is False

    def test_get_assistant_role(self):
        assert get_assistant_role("unsloth/gemma-4-E4B-it") == "model"
        assert get_assistant_role("google/gemma-2-2b-it") == "model"
        assert get_assistant_role("Qwen/Qwen2.5-3B-Instruct") == "assistant"
        assert get_assistant_role("Qwen/Qwen3-4B") == "assistant"


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

    def test_render_ascii_board(self):
        board_str = render_ascii_board(START_FEN)
        lines = board_str.splitlines()
        assert len(lines) == 9
        assert lines[0] == "8 r n b q k b n r"
        assert lines[7] == "1 R N B Q K B N R"
        assert lines[8] == "  a b c d e f g h"

    def test_make_prompt_include_ascii(self):
        prompt = make_prompt(START_FEN, include_ascii=True)
        assert "Board:\n8 r n b q k b n r" in prompt
        assert "  a b c d e f g h" in prompt
        assert f"Position (FEN): {START_FEN}" in prompt

    def test_build_chat_prompt_wraps_in_chatml(self):
        chat = build_chat_prompt(START_FEN)
        assert chat.startswith(CHATML_USER_OPEN)
        assert chat.endswith(CHATML_ASSISTANT_OPEN)
        assert "Position (FEN):" in chat
        assert "Side to move: White" in chat

    def test_build_chat_prompt_uses_tokenizer_template_with_enable_thinking(self):
        captured = {}

        class _TokWithThinking:
            def apply_chat_template(
                self,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ):
                captured["messages"] = messages
                captured["tokenize"] = tokenize
                captured["add_generation_prompt"] = add_generation_prompt
                captured["enable_thinking"] = enable_thinking
                return f"<custom>{messages[0]['content']}</custom>"

        out = build_chat_prompt(START_FEN, tokenizer=_TokWithThinking(), enable_thinking=True)
        assert out.startswith("<custom>")
        assert captured["tokenize"] is False
        assert captured["add_generation_prompt"] is True
        assert captured["enable_thinking"] is True
        assert captured["messages"][0]["role"] == "user"

    def test_build_chat_prompt_handles_tokenizer_without_enable_thinking_kwarg(self):
        class _TokNoThinking:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return f"<no-think>{messages[0]['content']}</no-think>"

        out = build_chat_prompt(START_FEN, tokenizer=_TokNoThinking(), enable_thinking=True)
        assert out.startswith("<no-think>")

    def test_build_chat_prompt_falls_back_to_chatml_when_template_missing(self):
        class _TokMissingTemplate:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                raise ValueError("tokenizer.chat_template is not set")

        out = build_chat_prompt(START_FEN, tokenizer=_TokMissingTemplate())
        assert out == build_chat_prompt(START_FEN)

    def test_prompt_template_is_shared_source(self):
        assert "Reason inside <think>...</think> tags" in PROMPT_TEMPLATE
        assert "in UCI format (e.g. e2e4)" in PROMPT_TEMPLATE
