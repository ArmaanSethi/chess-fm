#!/usr/bin/env python3
"""
GRPO training script for ChessFM.

Trains a language model to play chess with Group Relative Policy Optimization
(TRL GRPOTrainer). Supports staged training:

    Stage 0: Legality only      (+1 legal, -1 illegal) + format/budget bonuses
    Stage 1: + Stockfish move quality (how close to Stockfish's best move)
    Stage 2: + Opponent-reply delta (outcome after Stockfish's best counterplay)

USAGE:
    python -m chessfm.rl.train_grpo --stage 0 --steps 500
    python -m chessfm.rl.train_grpo --stage 1 --steps 2000 --checkpoint-every 500
    python -m chessfm.rl.train_grpo --stage 1 --resume  # continue last checkpoint
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trl import GRPOConfig

from chessfm.config import (
    DEFAULT_BASE_MODEL,
    GRPO_BETA,
    GRPO_NUM_GENERATIONS,
    GRPO_TEMPERATURE,
    LORA_R,
    MAX_COMPLETION_LENGTH,
    MAX_PROMPT_LENGTH,
    build_chat_prompt,
)
from chessfm.fens import read_fens
from chessfm.rl.engine import close_engine, open_engine
from chessfm.rl.move_utils import extract_fen_from_prompt, parse_move_from_output
from chessfm.rl.rewards import EngineAnalysisError, StockfishEvalCache, combined_reward
from chessfm.training.manifest import write_run_manifest

logger = logging.getLogger("chessfm.train_grpo")

# Abort the run if more than this fraction of reward computations fail due to
# engine/infrastructure errors. A broken engine silently poisoning rewards is
# worse than stopping training.
_MAX_ENGINE_FAILURE_RATE = 0.05
_MIN_FAILURE_SAMPLE = 20

DEFAULT_EPSILON_LOW = 0.20
DEFAULT_EPSILON_HIGH = 0.28


def validate_grpo_batch_divisibility(
    per_device_batch_size: int,
    num_generations: int,
    world_size: int = 1,
) -> None:
    """Validate that ``(per_device_batch_size * world_size)`` is divisible by ``num_generations``.

    TRL's ``GRPOTrainer`` requires the global generation batch size to be a
    multiple of ``num_generations``; failing fast here prevents crashing after
    model loading on the GPU.
    """
    if per_device_batch_size <= 0:
        raise ValueError(f"per_device_batch_size must be positive, got {per_device_batch_size}")
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    global_batch = per_device_batch_size * world_size
    if global_batch % num_generations != 0:
        raise ValueError(
            f"GRPO requires (per_device_batch_size * world_size) to be divisible by "
            f"num_generations, got ({per_device_batch_size} * {world_size}) = {global_batch} "
            f"and num_generations = {num_generations}. "
            f"Set --per-device-batch-size to a multiple of {num_generations} "
            f"(e.g. {num_generations})."
        )


def make_reward_func(
    stage: int | str,
    engine=None,
    *,
    reward_mode: str = "legacy",
    cache: StockfishEvalCache | None = None,
    use_cache: bool = True,
):
    """Build a TRL-compatible reward function.

    TRL calls reward funcs as ``reward_func(prompts=..., completions=..., **kwargs)``
    and expects a list of floats back, one per completion.

    When ``engine`` is provided and ``use_cache=True`` (the default), a
    :class:`~chessfm.rl.rewards.StockfishEvalCache` is attached so repeated
    per-FEN and per-(FEN, move) evaluations across GRPO rollouts reuse cached
    Stockfish analyses.

    Engine failures (``EngineAnalysisError``) are treated as neutral (0.0) and
    counted; once the failure rate exceeds ``_MAX_ENGINE_FAILURE_RATE`` over at
    least ``_MIN_FAILURE_SAMPLE`` engine-backed calls, the run aborts — a dying
    Stockfish process must not quietly degrade into a garbage training signal.
    """
    stats = {"calls": 0, "engine_failures": 0}
    active_cache = (
        cache
        if cache is not None
        else (StockfishEvalCache() if engine is not None and use_cache else None)
    )

    def _abort_if_engine_dying():
        if stats["engine_failures"] == 0 or stats["calls"] < _MIN_FAILURE_SAMPLE:
            return
        rate = stats["engine_failures"] / max(1, stats["calls"])
        if rate > _MAX_ENGINE_FAILURE_RATE:
            raise SystemExit(
                f"Aborting: {stats['engine_failures']}/{stats['calls']} reward "
                f"computations failed ({rate:.1%} > {_MAX_ENGINE_FAILURE_RATE:.0%}). "
                "The Stockfish engine appears to be unhealthy."
            )

    def chess_reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions, strict=True):
            # Count every attempt, successful or not: a *fully* broken engine
            # produces zero successes, so successes alone can never trip the
            # failure-rate guard.
            stats["calls"] += 1
            try:
                fen = extract_fen_from_prompt(prompt)
                move = parse_move_from_output(completion) or "xxxx"
                r = combined_reward(
                    fen,
                    move,
                    completion,
                    stage=stage,
                    engine=engine,
                    reward_mode=reward_mode,
                    cache=active_cache,
                )
                rewards.append(r["total"])
            except EngineAnalysisError as e:
                # An infrastructure failure must be loud and *neutral*, not
                # treated as an illegal move or a blunder: penalizing it would
                # corrupt the training signal at scale.
                logger.error("Reward computation failed, using neutral 0.0: %s", e)
                rewards.append(0.0)
                stats["engine_failures"] += 1
            except ValueError as e:
                # Malformed prompt/output (e.g. no FEN in the prompt) is a data
                # bug, not an engine problem: neutral, logged, not counted
                # against the engine-health budget.
                logger.error("Malformed sample, using neutral 0.0: %s", e)
                rewards.append(0.0)
        _abort_if_engine_dying()
        return rewards

    chess_reward_fn.cache = active_cache  # type: ignore[attr-defined]
    return chess_reward_fn


def load_positions(
    filepath: Path,
    max_positions: int = 1000,
    seed: int = 42,
    tokenizer=None,
    include_ascii: bool = False,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
) -> list[dict]:
    """Load positions into chat-formatted prompt dataset.

    When the pool has more FENs than ``max_positions``, a seeded random sample
    is taken instead of the file head, so the training set is not biased toward
    whatever order positions were fetched in.

    Raises:
        SystemExit: if the positions file is missing or contains no positions.
    """
    if not filepath.exists():
        raise SystemExit(
            f"Positions file not found: {filepath}\n"
            "Generate it first: python -m chessfm.data_generation.fetch_elite_data "
            "(or pass --positions)."
        )
    fens = read_fens(filepath)
    if len(fens) > max_positions:
        fens = random.Random(seed).sample(fens, max_positions)
    positions = [
        {
            "prompt": build_chat_prompt(
                fen,
                tokenizer=tokenizer,
                include_ascii=include_ascii,
                include_piece_list=include_piece_list,
                include_legal_moves=include_legal_moves,
                enable_thinking=True,
            )
        }
        for fen in fens
    ]
    if not positions:
        raise SystemExit(f"Positions file contains no positions: {filepath}")
    return positions


def filter_zero_variance_groups(
    rewards,
    completion_mask,
    num_generations: int,
    eps: float = 1e-8,
):
    """Mask out completion tokens belonging to zero-variance reward groups.

    In DAPO/RLVR, groups where all ``num_generations`` completions receive the
    same reward contribute zero advantage but still dilute the token-level loss
    denominator (and KL term when ``beta > 0``). This helper zeroes their
    ``completion_mask`` rows while guarding against the all-zero-variance batch
    edge case: if every group in the batch has zero variance, the mask is
    returned unchanged so ``completion_mask.sum()`` never becomes ``0`` (NaN).

    Returns ``(filtered_completion_mask, filtered_group_fraction)``.
    """
    if num_generations <= 1:
        return completion_mask, 0.0
    total_rows = int(rewards.shape[0])
    if total_rows == 0 or total_rows % num_generations != 0:
        return completion_mask, 0.0

    grouped = rewards.reshape(-1, num_generations)
    spread = grouped.max(dim=1).values - grouped.min(dim=1).values
    active_groups = spread > eps
    if not bool(active_groups.any()):
        # Guard against 0/0 NaN in compute_loss when every group has zero variance.
        return completion_mask, 0.0

    row_mask = active_groups.repeat_interleave(num_generations).unsqueeze(1)
    filtered_mask = completion_mask * row_mask.to(dtype=completion_mask.dtype)
    if not bool(filtered_mask.any()):
        # Guard against 0/0 NaN in compute_loss if active groups have empty masks.
        return completion_mask, 0.0
    filtered_frac = float((~active_groups).float().mean().item())
    return filtered_mask, filtered_frac


def build_grpo_config(args) -> GRPOConfig:
    """Construct the TRL GRPOConfig from parsed CLI args.

    Imported lazily so pure helpers above stay importable without the full
    torch/trl stack. Kept separate from ``main`` so the exact kwargs the
    training script passes are exercised by tests (guards against TRL API
    drift).
    """
    world_size = getattr(args, "world_size", int(os.environ.get("WORLD_SIZE", "1")))
    validate_grpo_batch_divisibility(
        args.per_device_batch_size,
        args.num_generations,
        world_size=world_size,
    )

    import torch
    from trl import GRPOConfig

    use_gpu = torch.cuda.is_available()
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    grpo_params = inspect.signature(GRPOConfig.__init__).parameters
    extra_kwargs: dict = {}
    if use_gpu:
        extra_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    epsilon = getattr(args, "epsilon", None)
    if epsilon is not None and "epsilon" in grpo_params:
        extra_kwargs["epsilon"] = epsilon

    epsilon_high = getattr(args, "epsilon_high", None)
    if epsilon_high is not None and "epsilon_high" in grpo_params:
        extra_kwargs["epsilon_high"] = epsilon_high

    scale_rewards = getattr(args, "scale_rewards", None)
    if scale_rewards is not None and "scale_rewards" in grpo_params:
        extra_kwargs["scale_rewards"] = scale_rewards

    num_iterations = getattr(args, "num_iterations", None)
    if num_iterations is not None:
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        if "num_iterations" in grpo_params:
            extra_kwargs["num_iterations"] = int(num_iterations)

    loss_type = getattr(args, "loss_type", None)
    if loss_type:
        if "loss_type" in grpo_params:
            extra_kwargs["loss_type"] = loss_type
        else:
            logger.warning(
                "Installed TRL GRPOConfig does not accept loss_type=%r; "
                "falling back to TRL default token-level loss.",
                loss_type,
            )

    return GRPOConfig(
        output_dir=str(args.output),
        learning_rate=args.lr,
        max_steps=args.steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        temperature=args.temperature,
        beta=args.beta,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=10,
        save_steps=args.checkpoint_every,
        save_total_limit=3,
        report_to=report_to,
        seed=args.seed,
        bf16=use_gpu and torch.cuda.is_bf16_supported(),
        fp16=use_gpu and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=use_gpu,
        **extra_kwargs,
    )


class _KVCacheGenerationMixin:
    """Temporarily switch to eval + use_cache=True during rollout generation.

    TRL 0.16 calls ``model.generate()`` while ``model.training`` is True,
    causing Qwen3's forward pass to force ``use_cache=False`` whenever
    ``gradient_checkpointing=True`` (~10-20x slower rollouts). Switching to
    eval mode strictly inside ``_generate_and_score_completions`` restores
    KV-cached generation while preserving gradient checkpointing and
    ``use_cache=False`` during ``compute_loss()`` backward passes.
    """

    def _generate_and_score_completions(self, inputs):
        accelerator = getattr(self, "accelerator", None)
        unwrapped = (
            accelerator.unwrap_model(self.model)
            if accelerator is not None and hasattr(accelerator, "unwrap_model")
            else self.model
        )
        cfg = getattr(unwrapped, "config", None)
        was_training = self.model.training
        prev_use_cache = getattr(cfg, "use_cache", False) if cfg is not None else False
        self.model.eval()
        if cfg is not None:
            cfg.use_cache = True
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            if cfg is not None:
                cfg.use_cache = prev_use_cache
            if was_training:
                self.model.train()


class _ZeroVarianceFilterMixin:
    """DAPO-style dynamic zero-variance group filter for GRPOTrainer.

    Wraps ``_generate_and_score_completions`` and masks out ``completion_mask``
    for any group whose rewards/advantages have zero variance, while leaving
    all-zero-variance batches intact to prevent ``0 / 0`` NaN loss.
    """

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not isinstance(output, dict) or "completion_mask" not in output:
            return output
        signal = output.get("rewards")
        if signal is None:
            signal = output.get("advantages")
        if signal is None:
            return output
        num_gen = getattr(self, "num_generations", None) or getattr(
            getattr(self, "args", None), "num_generations", 1
        )
        filtered_mask, frac = filter_zero_variance_groups(
            signal, output["completion_mask"], int(num_gen)
        )
        output["completion_mask"] = filtered_mask
        self._last_zero_variance_filtered_frac = frac
        return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for GRPO training."""
    parser = argparse.ArgumentParser(description="Train ChessFM with GRPO")
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Training stage (0=legality, 1=+move quality, 2=+opponent-reply delta)",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="legacy",
        choices=["legacy", "wdl_loss"],
        help="Reward mode ('legacy' for stages 0/1/2, 'wdl_loss' for expected-score delta)",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of max training steps")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Base model to train")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=GRPO_NUM_GENERATIONS,
        help="Completions sampled per prompt (GRPO group size)",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=GRPO_NUM_GENERATIONS,
        help="Per-device training batch size (must satisfy batch*world_size %% num_generations == 0)",
    )
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="GRPO policy updates per rollout generation (>=2 activates PPO/DAPO Clip-Higher)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--temperature",
        type=float,
        default=GRPO_TEMPERATURE,
        help="Sampling temperature for generation",
    )
    parser.add_argument("--beta", type=float, default=GRPO_BETA, help="KL penalty coefficient")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON_LOW,
        help="Lower PPO/GRPO clipping bound (epsilon_low)",
    )
    parser.add_argument(
        "--epsilon-high",
        type=float,
        default=DEFAULT_EPSILON_HIGH,
        help="Upper DAPO Clip-Higher bound (epsilon_high)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=["grpo", "bnpo", "dr_grpo", "dapo"],
        help="Optional TRL loss type (forwarded when supported by installed TRL version)",
    )
    parser.add_argument(
        "--scale-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide advantages by group reward std (pass --no-scale-rewards for Dr. GRPO)",
    )
    parser.add_argument(
        "--filter-zero-variance",
        action="store_true",
        help="Mask out completion tokens from zero-variance reward groups (DAPO dynamic filtering)",
    )
    parser.add_argument("--lora-r", type=int, default=LORA_R, help="LoRA adapter rank")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA scaling alpha (defaults to 2 * --lora-r)",
    )
    parser.add_argument(
        "--all-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Target all linear layers (q/k/v/o/gate/up/down_proj) vs attention-only",
    )
    parser.add_argument(
        "--full-ft",
        action="store_true",
        help="Train all parameters without LoRA adapters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path.cwd() / "data_generation" / "positions.txt",
        help="Path to positions file",
    )
    parser.add_argument(
        "--max-positions", type=int, default=1000, help="Number of positions to sample"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100, help="Save a checkpoint every N steps"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output if one exists",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include an ASCII board rendering in position prompts",
    )
    parser.add_argument(
        "--include-piece-list",
        action="store_true",
        help="Include a lossless piece-by-square listing in position prompts",
    )
    parser.add_argument(
        "--include-legal-moves",
        action="store_true",
        help="Include sorted UCI legal moves in position prompts (isolated comparison mode only)",
    )
    parser.add_argument(
        "--merge-and-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights into standalone checkpoint at <output>/stage{N}_final_merged",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def save_grpo_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    stage: int,
    *,
    merge_and_save: bool = True,
) -> dict[str, Path | None]:
    """Save GRPO adapter checkpoint and optionally a standalone merged checkpoint.

    When chaining multiple GRPO stages (Stage 0 -> Stage 1 -> Stage 2), each
    stage loads the previous checkpoint with ``merge_peft=True`` and attaches a
    fresh LoRA adapter. Saving only the adapter would point its
    ``base_model_name_or_path`` at the original pre-GRPO weights, silently
    discarding earlier GRPO stages. Merging into ``stage{stage}_final_merged``
    preserves all accumulated weights for subsequent stages and HF uploads.
    """
    final_path = output_dir / f"stage{stage}_final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nTraining complete! Adapter saved to {final_path}")

    merged_path: Path | None = None
    if merge_and_save:
        merged_path = output_dir / f"stage{stage}_final_merged"
        if hasattr(model, "merge_and_unload"):
            print(f"Merging LoRA adapter into standalone weights -> {merged_path}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged_path))
        else:
            print(f"Model has no PEFT adapter; saving standalone weights -> {merged_path}")
            model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        print(f"Merged model saved to {merged_path}")

    return {"adapter_path": final_path, "merged_path": merged_path}


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_grpo_batch_divisibility(
        args.per_device_batch_size,
        args.num_generations,
        world_size=world_size,
    )

    if args.num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {args.num_iterations}")
    if args.filter_zero_variance and args.per_device_batch_size <= args.num_generations:
        logger.warning(
            "filter_zero_variance=True with per_device_batch_size (%d) <= num_generations (%d) "
            "places only 1 prompt group per micro-batch, so the all-zero-variance NaN guard "
            "prevents masking degenerate groups; set per_device_batch_size >= 2 * num_generations "
            "(e.g. %d) to activate zero-variance group filtering.",
            args.per_device_batch_size,
            args.num_generations,
            2 * args.num_generations,
        )

    effective_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    print("=" * 60)
    print("ChessFM RL Training (GRPO via TRL)")
    print("=" * 60)
    print(f"Stage:            {args.stage} (reward_mode={args.reward_mode})")
    print(f"Model:            {args.model}")
    print(f"Max Steps:        {args.steps}")
    print(f"Generations/grp:  {args.num_generations}")
    print(f"Batch size/dev:   {args.per_device_batch_size}")
    print(f"Num iterations:   {args.num_iterations}")
    print(f"Temperature:      {args.temperature}")
    print(f"Beta (KL):        {args.beta}")
    print(f"Clip bounds:      [{args.epsilon}, {args.epsilon_high}]")
    print(f"Zero-var filter:  {args.filter_zero_variance}")
    print(
        f"LoRA / Full-FT:   {'Full FT' if args.full_ft else f'r={args.lora_r}, alpha={effective_alpha}'}"
    )
    print(f"Checkpoint every: {args.checkpoint_every}")
    print(f"Resume:           {args.resume}")
    print(f"Merge & save:     {args.merge_and_save}")
    print()

    import torch  # noqa: F401  (kept so an early CUDA error surfaces here)
    from datasets import Dataset
    from trl import GRPOTrainer

    from chessfm.load_model import load_model

    torch.manual_seed(args.seed)

    model, tokenizer = load_model(
        args.model,
        merge_peft=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        all_linear=args.all_linear,
        full_ft=args.full_ft,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_positions = load_positions(
        args.positions,
        args.max_positions,
        seed=args.seed,
        tokenizer=tokenizer,
        include_ascii=args.include_ascii,
        include_piece_list=args.include_piece_list,
        include_legal_moves=args.include_legal_moves,
    )
    dataset = Dataset.from_list(raw_positions)
    print(f"Loaded {len(dataset)} prompts/positions for training")

    engine = None
    needs_engine = args.stage >= 1 or args.reward_mode == "wdl_loss"
    if needs_engine:
        engine = open_engine()
        if engine is None:
            raise SystemExit(
                "Stockfish engine not found. Stages 1/2 and wdl_loss reward mode require "
                "Stockfish for move evaluation. Install it (brew install stockfish / "
                "apt install stockfish) or run with --stage 0 --reward-mode legacy."
            )

    if hasattr(model, "config"):
        model.config.use_cache = False

    training_args = build_grpo_config(args)

    reward_fn = make_reward_func(
        args.stage,
        engine=engine,
        reward_mode=args.reward_mode,
        use_cache=True,
    )

    if args.filter_zero_variance:

        class _ChessGRPOTrainer(_ZeroVarianceFilterMixin, _KVCacheGenerationMixin, GRPOTrainer):
            pass

    else:

        class _ChessGRPOTrainer(_KVCacheGenerationMixin, GRPOTrainer):
            pass

    trainer = _ChessGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    try:
        print("\nStarting GRPO Training Loop...")
        train_start = time.monotonic()
        if args.resume and any(args.output.glob("checkpoint-*")):
            print("Resuming from latest checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        wall_clock_seconds = round(time.monotonic() - train_start, 2)

        save_grpo_checkpoint(
            model,
            tokenizer,
            args.output,
            args.stage,
            merge_and_save=args.merge_and_save and not args.full_ft,
        )

        gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        manifest_path = write_run_manifest(
            args.output,
            model_id=args.model,
            stage=f"grpo_stage{args.stage}",
            seed=args.seed,
            hyperparameters={
                "steps": args.steps,
                "reward_mode": args.reward_mode,
                "num_generations": args.num_generations,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "num_iterations": args.num_iterations,
                "lr": args.lr,
                "temperature": args.temperature,
                "beta": args.beta,
                "epsilon": args.epsilon,
                "epsilon_high": args.epsilon_high,
                "loss_type": args.loss_type,
                "scale_rewards": args.scale_rewards,
                "filter_zero_variance": args.filter_zero_variance,
                "lora_r": None if args.full_ft else args.lora_r,
                "lora_alpha": None if args.full_ft else effective_alpha,
                "all_linear": None if args.full_ft else args.all_linear,
                "full_ft": args.full_ft,
                "include_ascii": args.include_ascii,
                "include_piece_list": args.include_piece_list,
                "include_legal_moves": args.include_legal_moves,
                "merge_and_save": args.merge_and_save and not args.full_ft,
            },
            dataset_path=args.positions,
            gpu_type=gpu_type,
            wall_clock_seconds=wall_clock_seconds,
        )
        print(f"Run manifest saved to: {manifest_path}")
    finally:
        close_engine(engine)


if __name__ == "__main__":
    main()
