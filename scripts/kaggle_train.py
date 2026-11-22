#!/usr/bin/env python3
"""Resumable Kaggle Notebook runner (2x T4 GPU) for ChessFM training and evaluation.

Designed for Kaggle's 12-hour session limits and Debian/Ubuntu environment where
Stockfish is installed under ``/usr/games``. All command construction is exposed
as pure helpers so CI can verify exact CLI flags without a GPU.

Usage inside a Kaggle Notebook cell:
    !apt-get -qq update && apt-get -qq install -y stockfish
    !python scripts/kaggle_train.py --task pipeline \\
        --model unsloth/gemma-4-E4B-it \\
        --stage 0 --steps 500 \\
        --output /kaggle/working/checkpoints \\
        --include-ascii --resume

    # Individual stages:
    !python scripts/kaggle_train.py --task sft --model Qwen/Qwen3-4B --epochs 3
    !python scripts/kaggle_train.py --task grpo --stage 1 --steps 500 --resume
    !python scripts/kaggle_train.py --task eval --games 20 --stockfish-level 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT = "/kaggle/working/checkpoints"
DEFAULT_DATA = "data_generation/sft_train.jsonl"

STOCKFISH_CANDIDATES = (
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
)


def ensure_usr_games_on_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment mapping with ``/usr/games`` prepended to ``PATH``."""
    target = dict(os.environ if env is None else env)
    current_path = target.get("PATH", "")
    parts = [p for p in current_path.split(os.pathsep) if p]
    if "/usr/games" not in parts:
        target["PATH"] = os.pathsep.join(["/usr/games", *parts]) if parts else "/usr/games"
    return target


def find_stockfish(env: dict[str, str] | None = None) -> str | None:
    """Locate the Stockfish binary after ensuring ``/usr/games`` is on ``PATH``."""
    checked_env = ensure_usr_games_on_path(env)
    found = shutil.which("stockfish", path=checked_env.get("PATH"))
    if found:
        return found
    for candidate in STOCKFISH_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def require_stockfish(env: dict[str, str] | None = None) -> str:
    """Return the Stockfish binary path or raise ``RuntimeError`` with install steps."""
    binary = find_stockfish(env)
    if binary is None:
        raise RuntimeError(
            "Stockfish binary not found (required for GRPO stages >= 1 and Stockfish eval). "
            "Install it in your Kaggle notebook with: "
            "!apt-get -qq update && apt-get -qq install -y stockfish"
        )
    return binary


def build_sft_cmd(
    *,
    model: str = DEFAULT_MODEL,
    data: str = DEFAULT_DATA,
    output: str = DEFAULT_OUTPUT,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-5,
    include_ascii: bool = False,
    merge_and_save: bool = True,
    python_bin: str = sys.executable,
) -> list[str]:
    """Build the ``chessfm.training.train_sft`` CLI command for Kaggle T4 GPUs."""
    cmd = [
        python_bin,
        "-m",
        "chessfm.training.train_sft",
        "--model",
        model,
        "--data",
        data,
        "--output",
        output,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--merge-and-save" if merge_and_save else "--no-merge-and-save",
    ]
    if include_ascii:
        cmd.append("--include-ascii")
    return cmd


def build_grpo_cmd(
    *,
    model: str = DEFAULT_MODEL,
    stage: int = 0,
    steps: int = 500,
    output: str = DEFAULT_OUTPUT,
    num_generations: int = 8,
    per_device_batch_size: int = 8,
    grad_accum: int = 2,
    max_positions: int = 1000,
    checkpoint_every: int = 100,
    resume: bool = False,
    include_ascii: bool = False,
    python_bin: str = sys.executable,
) -> list[str]:
    """Build the ``chessfm.rl.train_grpo`` CLI command with optional resume & ASCII."""
    cmd = [
        python_bin,
        "-m",
        "chessfm.rl.train_grpo",
        "--model",
        model,
        "--stage",
        str(stage),
        "--steps",
        str(steps),
        "--output",
        output,
        "--num-generations",
        str(num_generations),
        "--per-device-batch-size",
        str(per_device_batch_size),
        "--grad-accum",
        str(grad_accum),
        "--max-positions",
        str(max_positions),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
    if resume:
        cmd.append("--resume")
    if include_ascii:
        cmd.append("--include-ascii")
    return cmd


def build_eval_cmd(
    *,
    model: str = DEFAULT_MODEL,
    games: int = 20,
    stockfish_level: int = 1,
    random_opponent: bool = False,
    seed: int = 42,
    python_bin: str = sys.executable,
) -> list[str]:
    """Build the ``chessfm.rl.evaluate`` CLI command."""
    cmd = [
        python_bin,
        "-m",
        "chessfm.rl.evaluate",
        "--model",
        model,
        "--games",
        str(games),
        "--stockfish-level",
        str(stockfish_level),
        "--seed",
        str(seed),
    ]
    if random_opponent:
        cmd.append("--random-opponent")
    return cmd


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool = False) -> None:
    print(f">>> {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable 2x T4 Kaggle runner for ChessFM SFT, GRPO, and evaluation."
    )
    parser.add_argument(
        "--task",
        choices=["sft", "grpo", "eval", "pipeline"],
        default="pipeline",
        help="Workflow to run (default: pipeline)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID or checkpoint")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to SFT JSONL file")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=0, help="GRPO stage")
    parser.add_argument("--steps", type=int, default=500, help="GRPO max training steps")
    parser.add_argument("--epochs", type=int, default=3, help="SFT training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="SFT per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="SFT learning rate")
    parser.add_argument(
        "--num-generations", type=int, default=8, help="GRPO completions per prompt"
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=8, help="GRPO per-device batch size"
    )
    parser.add_argument("--grad-accum", type=int, default=2, help="GRPO gradient accumulation")
    parser.add_argument("--max-positions", type=int, default=1000, help="Max GRPO positions")
    parser.add_argument(
        "--checkpoint-every", type=int, default=100, help="Save GRPO checkpoint every N steps"
    )
    parser.add_argument("--games", type=int, default=20, help="Number of evaluation games")
    parser.add_argument("--stockfish-level", type=int, default=1, help="Stockfish skill level")
    parser.add_argument(
        "--random-opponent",
        action="store_true",
        help="Evaluate against a random mover instead of Stockfish",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Checkpoint root directory (default: /kaggle/working/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume GRPO from latest checkpoint in output directory",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include ASCII board rendering in SFT and GRPO prompts",
    )
    parser.add_argument(
        "--merge-and-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights after SFT into <output>/final_merged",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing training",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env = ensure_usr_games_on_path()
    os.environ["PATH"] = env["PATH"]

    needs_stockfish = (args.task in {"grpo", "pipeline"} and args.stage >= 1) or (
        args.task in {"eval", "pipeline"} and not args.random_opponent
    )
    if needs_stockfish and not args.dry_run:
        stockfish_bin = require_stockfish(env)
        print(f"Using Stockfish binary: {stockfish_bin}", flush=True)

    out_root = Path(args.output)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    if args.task == "sft":
        cmd = build_sft_cmd(
            model=args.model,
            data=args.data,
            output=str(out_root),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            include_ascii=args.include_ascii,
            merge_and_save=args.merge_and_save,
        )
        _run_cmd(cmd, env, dry_run=args.dry_run)
    elif args.task == "grpo":
        cmd = build_grpo_cmd(
            model=args.model,
            stage=args.stage,
            steps=args.steps,
            output=str(out_root),
            num_generations=args.num_generations,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            max_positions=args.max_positions,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            include_ascii=args.include_ascii,
        )
        _run_cmd(cmd, env, dry_run=args.dry_run)
    elif args.task == "eval":
        cmd = build_eval_cmd(
            model=args.model,
            games=args.games,
            stockfish_level=args.stockfish_level,
            random_opponent=args.random_opponent,
            seed=args.seed,
        )
        _run_cmd(cmd, env, dry_run=args.dry_run)
    elif args.task == "pipeline":
        sft_dir = out_root / "sft"
        grpo_dir = out_root / f"grpo_stage{args.stage}"
        sft_merged = sft_dir / ("final_merged" if args.merge_and_save else "final")
        grpo_final = grpo_dir / f"stage{args.stage}_final"

        print("=== Step 1/3: SFT ===", flush=True)
        _run_cmd(
            build_sft_cmd(
                model=args.model,
                data=args.data,
                output=str(sft_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                include_ascii=args.include_ascii,
                merge_and_save=args.merge_and_save,
            ),
            env,
            dry_run=args.dry_run,
        )

        print(f"=== Step 2/3: GRPO Stage {args.stage} ===", flush=True)
        _run_cmd(
            build_grpo_cmd(
                model=str(sft_merged),
                stage=args.stage,
                steps=args.steps,
                output=str(grpo_dir),
                num_generations=args.num_generations,
                per_device_batch_size=args.per_device_batch_size,
                grad_accum=args.grad_accum,
                max_positions=args.max_positions,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                include_ascii=args.include_ascii,
            ),
            env,
            dry_run=args.dry_run,
        )

        print("=== Step 3/3: Evaluation ===", flush=True)
        _run_cmd(
            build_eval_cmd(
                model=str(grpo_final),
                games=args.games,
                stockfish_level=args.stockfish_level,
                random_opponent=args.random_opponent,
                seed=args.seed,
            ),
            env,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
