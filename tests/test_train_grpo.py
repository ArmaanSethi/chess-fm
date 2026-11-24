#!/usr/bin/env python3
"""Unit tests for GRPO wiring helpers that don't require a GPU or a model."""

import unittest

import chess
import pytest

from chessfm.rl.train_grpo import (
    load_positions,
    make_reward_func,
    parse_args,
    save_grpo_checkpoint,
    validate_grpo_batch_divisibility,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class _BrokenEngine:
    def analyse(self, board, limit):
        raise chess.engine.EngineError("engine died")


def test_validate_grpo_batch_divisibility_valid():
    validate_grpo_batch_divisibility(8, 8, world_size=1)
    validate_grpo_batch_divisibility(4, 8, world_size=2)
    validate_grpo_batch_divisibility(16, 8, world_size=1)


def test_validate_grpo_batch_divisibility_invalid():
    with pytest.raises(ValueError, match="divisible by num_generations"):
        validate_grpo_batch_divisibility(4, 8, world_size=1)
    with pytest.raises(ValueError, match="positive"):
        validate_grpo_batch_divisibility(0, 8, world_size=1)
    with pytest.raises(ValueError, match="positive"):
        validate_grpo_batch_divisibility(8, 0, world_size=1)
    with pytest.raises(ValueError, match="positive"):
        validate_grpo_batch_divisibility(8, 8, world_size=0)


def test_reads_fens_into_chat_prompts(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text(START_FEN + "\n\n" + "8/8/8/4k3/8/8/4K3/8 w - - 0 1\n")
    positions = load_positions(p)
    assert len(positions) == 2
    assert positions[0]["prompt"].startswith("<|im_start|>user\nPosition (FEN):")
    assert positions[0]["prompt"].endswith("<|im_start|>assistant\n")


def test_load_positions_supports_include_ascii(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text(START_FEN + "\n")
    positions = load_positions(p, include_ascii=True)
    assert "Board:\n8 r n b q k b n r" in positions[0]["prompt"]


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
    # A reward computation failure is loud (logged) and neutral, not punished
    # as if it were an illegal move.
    assert rewards == [0.0]


class TestEngineFailureGuard(unittest.TestCase):
    """Engine failures are neutral per-completion but abort the run at scale."""

    PROMPT = f"Position (FEN): {START_FEN}\nSide to move: White"
    COMPLETION = "<think>center control and development.</think>\ne2e4"

    def test_engine_failure_is_neutral_not_negative(self):
        fn = make_reward_func(stage=1, engine=_BrokenEngine())
        with self.assertLogs("chessfm.train_grpo", level="ERROR"):
            rewards = fn(prompts=[self.PROMPT], completions=[self.COMPLETION])
        self.assertEqual(rewards, [0.0])

    def test_few_early_failures_do_not_abort(self):
        fn = make_reward_func(stage=1, engine=_BrokenEngine())
        for _ in range(5):
            fn(prompts=[self.PROMPT], completions=[self.COMPLETION])

    def test_sustained_engine_failure_aborts_run(self):
        fn = make_reward_func(stage=1, engine=_BrokenEngine())
        with self.assertRaises(SystemExit):
            for _ in range(30):
                fn(prompts=[self.PROMPT] * 4, completions=[self.COMPLETION] * 4)


class _DummyTokenizer:
    def __init__(self):
        self.saved_paths: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_paths.append(path)


class _DummyStandaloneModel:
    def __init__(self):
        self.saved_paths: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_paths.append(path)


class _DummyPeftModel(_DummyStandaloneModel):
    def __init__(self):
        super().__init__()
        self.merged_calls = 0
        self.merged_model = _DummyStandaloneModel()

    def merge_and_unload(self):
        self.merged_calls += 1
        return self.merged_model


def test_parse_args_merge_and_save_defaults_and_toggle():
    default_args = parse_args([])
    assert default_args.merge_and_save is True

    enabled_args = parse_args(["--merge-and-save"])
    assert enabled_args.merge_and_save is True

    disabled_args = parse_args(["--no-merge-and-save"])
    assert disabled_args.merge_and_save is False


def test_save_grpo_checkpoint_merges_peft_model(tmp_path):
    model = _DummyPeftModel()
    tok = _DummyTokenizer()

    res = save_grpo_checkpoint(model, tok, tmp_path, stage=0, merge_and_save=True)

    adapter_dir = tmp_path / "stage0_final"
    merged_dir = tmp_path / "stage0_final_merged"
    assert res == {"adapter_path": adapter_dir, "merged_path": merged_dir}
    assert model.saved_paths == [str(adapter_dir)]
    assert model.merged_calls == 1
    assert model.merged_model.saved_paths == [str(merged_dir)]
    assert tok.saved_paths == [str(adapter_dir), str(merged_dir)]


def test_save_grpo_checkpoint_handles_non_peft_model(tmp_path):
    model = _DummyStandaloneModel()
    tok = _DummyTokenizer()

    res = save_grpo_checkpoint(model, tok, tmp_path, stage=1, merge_and_save=True)

    adapter_dir = tmp_path / "stage1_final"
    merged_dir = tmp_path / "stage1_final_merged"
    assert res == {"adapter_path": adapter_dir, "merged_path": merged_dir}
    assert model.saved_paths == [str(adapter_dir), str(merged_dir)]
    assert tok.saved_paths == [str(adapter_dir), str(merged_dir)]


def test_save_grpo_checkpoint_no_merge_skips_merged_dir(tmp_path):
    model = _DummyPeftModel()
    tok = _DummyTokenizer()

    res = save_grpo_checkpoint(model, tok, tmp_path, stage=2, merge_and_save=False)

    adapter_dir = tmp_path / "stage2_final"
    assert res == {"adapter_path": adapter_dir, "merged_path": None}
    assert model.saved_paths == [str(adapter_dir)]
    assert model.merged_calls == 0
    assert model.merged_model.saved_paths == []
    assert tok.saved_paths == [str(adapter_dir)]


def test_parse_args_dapo_and_lora_defaults_and_overrides():
    defaults = parse_args([])
    assert defaults.num_generations == 8
    assert defaults.per_device_batch_size == 8
    assert defaults.beta == pytest.approx(0.04)
    assert defaults.epsilon == pytest.approx(0.20)
    assert defaults.epsilon_high == pytest.approx(0.28)
    assert defaults.reward_mode == "legacy"
    assert defaults.lora_r == 16
    assert defaults.all_linear is True
    assert defaults.filter_zero_variance is False
    assert defaults.include_piece_list is False
    assert defaults.include_legal_moves is False

    custom = parse_args(
        [
            "--reward-mode",
            "wdl_loss",
            "--num-generations",
            "16",
            "--per-device-batch-size",
            "16",
            "--lora-r",
            "32",
            "--no-all-linear",
            "--filter-zero-variance",
            "--loss-type",
            "dr_grpo",
            "--include-piece-list",
            "--include-legal-moves",
        ]
    )
    assert custom.reward_mode == "wdl_loss"
    assert custom.num_generations == 16
    assert custom.lora_r == 32
    assert custom.all_linear is False
    assert custom.filter_zero_variance is True
    assert custom.loss_type == "dr_grpo"
    assert custom.include_piece_list is True
    assert custom.include_legal_moves is True


def test_load_positions_supports_piece_list_and_legal_moves(tmp_path):
    p = tmp_path / "positions.txt"
    p.write_text(START_FEN + "\n")
    positions = load_positions(p, include_piece_list=True, include_legal_moves=True)
    assert "Pieces: White: Ke1," in positions[0]["prompt"]
    assert "Legal moves: a2a3," in positions[0]["prompt"]


def test_make_reward_func_wdl_uses_cache_across_rollouts():
    class _FakeScore:
        def __init__(self, cp):
            self._cp = cp

        def is_mate(self):
            return False

        def score(self):
            return self._cp

        def white(self):
            return self

    class _CountingEngine:
        def __init__(self):
            self.calls = 0

        def analyse(self, board, limit):
            self.calls += 1
            return {"score": _FakeScore(25), "pv": []}

    engine = _CountingEngine()
    fn = make_reward_func(stage=0, engine=engine, reward_mode="wdl_loss", use_cache=True)
    prompt = f"Position (FEN): {START_FEN}\nSide to move: White"
    comp = "<think>control the center with e2e4 cleanly.</think>\ne2e4"
    rewards = fn(prompts=[prompt] * 8, completions=[comp] * 8)
    assert len(rewards) == 8
    assert all(r == pytest.approx(3.1) for r in rewards)
    assert engine.calls == 2  # 1 before + 1 after, reused across all 8 rollouts
    assert fn.cache is not None
    assert fn.cache.stats()["fen_hits"] >= 7


def test_filter_zero_variance_groups_guards_empty_active_mask():
    import torch

    from chessfm.rl.train_grpo import filter_zero_variance_groups

    # Group 0 has zero reward variance and non-zero completion tokens;
    # Group 1 has non-zero reward variance but all-zero completion tokens.
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.5, -0.5])
    mask = torch.zeros((8, 5), dtype=torch.long)
    mask[:4, :] = 1
    filtered_mask, frac = filter_zero_variance_groups(rewards, mask, num_generations=4)
    assert filtered_mask.sum().item() == mask.sum().item()
    assert torch.equal(filtered_mask, mask)
    assert frac == 0.0


def test_parse_args_and_build_grpo_config_num_iterations(tmp_path):
    from chessfm.rl.train_grpo import build_grpo_config

    defaults = parse_args([])
    assert defaults.num_iterations == 1

    custom = parse_args(
        [
            "--num-iterations",
            "2",
            "--per-device-batch-size",
            "16",
            "--num-generations",
            "8",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert custom.num_iterations == 2
    cfg = build_grpo_config(custom)
    assert cfg.num_iterations == 2

    invalid = parse_args(["--num-iterations", "0", "--output", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="num_iterations must be >= 1"):
        build_grpo_config(invalid)

    sharp = parse_args(["--reward-mode", "sharp_wdl"])
    assert sharp.reward_mode == "sharp_wdl"


if __name__ == "__main__":
    unittest.main()
