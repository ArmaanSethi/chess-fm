#!/usr/bin/env python3
"""Unit tests for ChessFM central configuration."""

import pytest

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
    render_legal_moves,
    render_piece_list,
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

    def test_render_piece_list_starting_position(self):
        listing = render_piece_list(START_FEN)
        assert listing == (
            "White: Ke1, Qd1, Ra1, Rh1, Bc1, Bf1, Nb1, Ng1, pawns a2 b2 c2 d2 e2 f2 g2 h2"
            " | Black: Ke8, Qd8, Ra8, Rh8, Bc8, Bf8, Nb8, Ng8, pawns a7 b7 c7 d7 e7 f7 g7 h7"
            " | Castling: KQkq"
        )

    def test_render_piece_list_empty_and_en_passant_and_promoted(self):
        empty = render_piece_list("8/8/8/8/8/8/8/8 w - - 0 1")
        assert empty == "White: none | Black: none"

        ep_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ep_listing = render_piece_list(ep_fen)
        assert "Castling: KQkq" in ep_listing
        assert "En passant: e3" in ep_listing

        two_queens = render_piece_list("4k3/8/8/8/8/8/Q7/Q3K3 w - - 0 1")
        assert two_queens == "White: Ke1, Qa1, Qa2 | Black: Ke8"

        with pytest.raises(ValueError):
            render_piece_list("not a valid fen")

    def test_render_legal_moves_sorted_and_checkmate(self):
        moves = render_legal_moves(START_FEN).split(", ")
        assert len(moves) == 20
        assert moves == sorted(moves)
        assert "e2e4" in moves

        # Fool's mate checkmate position: 0 legal moves
        mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        assert render_legal_moves(mate_fen) == "none"

        with pytest.raises(ValueError):
            render_legal_moves("invalid fen")

    def test_make_prompt_and_chat_prompt_with_piece_list_and_legal_moves(self):
        base = make_prompt(START_FEN)
        assert "Pieces:" not in base
        assert "Legal moves:" not in base

        with_pieces = make_prompt(START_FEN, include_piece_list=True)
        assert with_pieces.startswith(f"Position (FEN): {START_FEN}\nSide to move: White\n\n")
        assert "Pieces: White: Ke1," in with_pieces
        assert "Legal moves:" not in with_pieces

        with_all = make_prompt(
            START_FEN,
            include_ascii=True,
            include_piece_list=True,
            include_legal_moves=True,
        )
        assert (
            with_all.index("Board:\n")
            < with_all.index("Pieces: ")
            < with_all.index("Legal moves: ")
        )

        chat = build_chat_prompt(
            START_FEN,
            include_piece_list=True,
            include_legal_moves=True,
        )
        assert "Pieces: White: Ke1," in chat
        assert "Legal moves: a2a3," in chat
