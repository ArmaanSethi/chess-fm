#!/usr/bin/env python3
"""Unit tests for GRPO wiring helpers that don't require a GPU or a model."""

from chessfm.rl.train_grpo import load_positions, make_reward_func

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_reads_fens_into_chat_prompts(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text(START_FEN + "\n\n" + "8/8/8/4k3/8/8/4K3/8 w - - 0 1\n")
    positions = load_positions(p)
    assert len(positions) == 2
    assert positions[0]["prompt"].startswith("<|im_start|>user\nPosition (FEN):")
    assert positions[0]["prompt"].endswith("<|im_start|>assistant\n")


def test_respects_max_positions(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text((START_FEN + "\n") * 5)
    positions = load_positions(p, max_positions=3)
    assert len(positions) == 3


def test_missing_file_exits(tmp_path):
    try:
        load_positions(tmp_path / "missing.txt")
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for missing file")


def test_empty_file_exits(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text("\n\n")
    try:
        load_positions(p)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for empty file")


def test_stage0_returns_floats():
    fn = make_reward_func(stage=0)
    prompt = f"Position (FEN): {START_FEN}\nSide to move: White"
    rewards = fn(
        prompts=[prompt, prompt],
        completions=["<think>center.</think>\ne2e4", "gibberish"],
    )
    assert len(rewards) == 2
    assert rewards[0] > 0.0  # legal
    assert rewards[1] < 0.0  # no parseable move -> -1.0


def test_bad_prompt_warns_not_crashes():
    fn = make_reward_func(stage=0)
    rewards = fn(prompts=["no fen here"], completions=["e2e4"])
    assert rewards == [-1.0]
