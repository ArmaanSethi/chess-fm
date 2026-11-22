#!/usr/bin/env python3
"""Unit tests for data conversion pure functions."""

from chessfm.data_generation.convert_to_training import (
    _loss_within,
    clamp_cp,
    convert_sample,
    fix_think_tags,
    sample_quality,
)


def test_fix_think_tags_balances_extra_open():
    assert fix_think_tags("<think>hi") == "<think>hi</think>"


def test_fix_think_tags_balances_extra_close():
    assert fix_think_tags("hi</think>") == "<think>hi</think>"


def test_fix_think_tags_balanced_unchanged():
    assert fix_think_tags("<think>hi</think>") == "<think>hi</think>"


def test_clamp_cp_never_negative():
    assert clamp_cp(-1243) == 0
    assert clamp_cp(12.6) == 12
    assert clamp_cp(None) is None


def test_convert_sample_chatml_and_completion():
    sample = {
        "fen": "8/8/8/8/4k3/8/4K3/8 w - - 0 1",
        "color": "white",
        "reasoning": "<think>keep the king active</think>",
        "move": "e2e3",
        "model": "gpt-test",
        "is_reasonable": True,
        "centipawn_loss": -5,
    }
    row = convert_sample(sample)
    assert row["prompt"].startswith("<|im_start|>user\nPosition (FEN):")
    assert row["prompt"].endswith("<|im_start|>assistant\n")
    assert row["completion"] == "<think>keep the king active</think>\ne2e3<|im_end|>"
    assert row["centipawn_loss"] == 0  # clamped


def test_sample_quality_penalizes_unreasonable():
    assert sample_quality({"is_reasonable": False, "centipawn_loss": 1}) == 1e6
    assert sample_quality({"is_reasonable": True, "centipawn_loss": 50}) == 50.0
    assert sample_quality({"is_reasonable": True}) == 1e9


def test_loss_within_threshold():
    assert _loss_within(100, 200) is True
    assert _loss_within(300, 200) is False
    assert _loss_within(None, 200) is False


def test_fix_think_tags_collapses_duplicate_and_nested_tags():
    assert fix_think_tags("<think>hi\n</think>\n</think>") == "<think>hi\n</think>"
    assert fix_think_tags("<think><think>hi</think></think>") == "<think>hi</think>"
    assert fix_think_tags("<think>\n<think>hi</think>\n</think>") == "<think>hi</think>"
    assert fix_think_tags("<think>outer <think>inner</think> end</think>") == (
        "<think>outer inner end</think>"
    )
    assert fix_think_tags("") == ""


def test_convert_sample_prevents_duplicate_trailing_move_and_nested_tags():
    base = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "color": "white",
        "move": "e2e4",
        "model": "test-model",
        "is_reasonable": True,
        "centipawn_loss": 0,
    }

    # Reasoning already ends with the UCI move after </think>.
    row1 = convert_sample({**base, "reasoning": "<think>control the center</think>\ne2e4"})
    assert row1["completion"] == "<think>control the center</think>\ne2e4<|im_end|>"

    # Reasoning has nested tags, duplicate trailing move, and ChatML close token.
    row2 = convert_sample(
        {**base, "reasoning": "<think><think>control the center</think></think>\ne2e4<|im_end|>"}
    )
    assert row2["completion"] == "<think>control the center</think>\ne2e4<|im_end|>"

    # Move attached directly to </think> without whitespace.
    row3 = convert_sample({**base, "reasoning": "<think>control the center</think>e2e4"})
    assert row3["completion"] == "<think>control the center</think>\ne2e4<|im_end|>"

    # Move mentioned inside <think>...</think> is preserved.
    row4 = convert_sample({**base, "reasoning": "<think>I should play e2e4</think>"})
    assert row4["completion"] == "<think>I should play e2e4</think>\ne2e4<|im_end|>"
