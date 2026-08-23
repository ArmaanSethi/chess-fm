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
