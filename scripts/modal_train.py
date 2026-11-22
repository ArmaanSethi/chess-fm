#!/usr/bin/env python3
"""Modal runner for $0-budget ChessFM training and evaluation on an L4 GPU.

Supports all three ChessFM model tracks:
    - ``v0``: ``Qwen/Qwen2.5-3B-Instruct`` (3B baseline)
    - ``v1``: ``Qwen/Qwen3-4B`` (4B reasoning upgrade)
    - ``v2`` / ``coach``: ``unsloth/gemma-4-E4B-it`` (ungated default; set
      ``HF_TOKEN`` in your environment to use ``google/gemma-4-E4B-it``)

Usage:
    # One-time setup:
    pip install modal && modal setup

    # Run SFT (merged) -> GRPO Stage 0 -> Stockfish Eval end-to-end:
    modal run scripts/modal_train.py --task smoke-pipeline --include-ascii

    # Individual stages (select model via --track or explicit --model):
    modal run scripts/modal_train.py --task sft --track v2 --epochs 3 --include-ascii
    modal run scripts/modal_train.py --task grpo --stage 0 --steps 500 --include-ascii
    modal run scripts/modal_train.py --task eval --games 20 --stockfish-level 1
    modal run scripts/modal_train.py --task coach-eval
    modal run scripts/modal_train.py --task export-gguf --model /vol/checkpoints/sft_smoke/final_merged

    # Download checkpoints and eval reports locally:
    modal volume get chessfm-artifacts results ./results
    modal volume get chessfm-artifacts checkpoints ./checkpoints
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "chessfm-trainer"
VOLUME_NAME = "chessfm-artifacts"
REMOTE_REPO = "/root/chess-fm"
VOL_MOUNT = "/vol"

REPO_ROOT = Path(__file__).resolve().parents[1]

# Supported model tracks: v0 baseline, v1 Qwen3 upgrade, v2 Gemma 4 Coach.
DEFAULT_V0_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_V1_MODEL = "Qwen/Qwen3-4B"
DEFAULT_V2_MODEL = "unsloth/gemma-4-E4B-it"
OFFICIAL_V2_MODEL = "google/gemma-4-E4B-it"

TRACK_MODELS: dict[str, str] = {
    "v0": DEFAULT_V0_MODEL,
    "v1": DEFAULT_V1_MODEL,
    "v2": DEFAULT_V2_MODEL,
    "coach": DEFAULT_V2_MODEL,
}

VALID_TASKS = (
    "smoke-pipeline",
    "sft",
    "grpo",
    "eval",
    "coach-eval",
    "export-gguf",
)


def resolve_track_model(model: str = "", track: str = "", use_official_gemma: bool = False) -> str:
    """Resolve the Hugging Face model ID from ``--model`` / ``--track`` flags.

    An explicit ``model`` takes precedence unless ``track`` is provided without
    overriding the default model. For the ``v2``/``coach`` track, setting
    ``use_official_gemma=True`` selects ``google/gemma-4-E4B-it`` (requires
    ``HF_TOKEN`` in the environment).
    """
    if track:
        key = track.strip().lower()
        if key not in TRACK_MODELS:
            raise ValueError(
                f"Unknown track {track!r}. Choose from: {', '.join(sorted(TRACK_MODELS))}"
            )
        if key in {"v2", "coach"} and use_official_gemma:
            return OFFICIAL_V2_MODEL
        return TRACK_MODELS[key]
    return model or DEFAULT_V0_MODEL


# Forward HF_TOKEN when present locally (required only for gated checkpoints
# such as google/gemma-4-E4B-it; ungated unsloth/gemma-4-E4B-it needs no token).
_HF_SECRETS = (
    [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})]
    if os.environ.get("HF_TOKEN")
    else []
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "stockfish")
    .env(
        {
            "PATH": "/usr/games:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HOME": f"{VOL_MOUNT}/hf_cache",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.51.0,<4.52",
        "accelerate>=0.34.0",
        "peft>=0.11.0,<0.25",
        "trl>=0.16.0,<0.17",
        "bitsandbytes>=0.45.0",
        "python-chess>=1.10.0",
        "datasets>=3.0.0,<6",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REMOTE_REPO,
        ignore=[
            "venv",
            ".venv",
            ".git",
            ".agents",
            "__pycache__",
            ".pytest_cache",
            "checkpoints",
            "results",
            "*.pyc",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)


def _ensure_installed() -> None:
    os.makedirs(f"{VOL_MOUNT}/checkpoints", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/results/pgns", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/exports", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/hf_cache", exist_ok=True)
    _run(["pip", "install", "--no-deps", "-e", "."])


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_HF_SECRETS,
    timeout=60 * 60 * 4,  # 4 hours max per invocation
)
def run_sft(
    model: str = DEFAULT_V0_MODEL,
    data: str = "data_generation/sft_train.jsonl",
    output_subdir: str = "sft_qwen25_3b",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-5,
    include_ascii: bool = False,
    merge_and_save: bool = True,
) -> str:
    _ensure_installed()
    out_dir = f"{VOL_MOUNT}/checkpoints/{output_subdir}"
    cmd = [
        "python",
        "-m",
        "chessfm.training.train_sft",
        "--model",
        model,
        "--data",
        data,
        "--output",
        out_dir,
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
    _run(cmd)
    volume.commit()
    return f"{out_dir}/final_merged" if merge_and_save else f"{out_dir}/final"


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_HF_SECRETS,
    timeout=60 * 60 * 6,  # 6 hours max per invocation
)
def run_grpo(
    model: str = DEFAULT_V0_MODEL,
    stage: int = 0,
    steps: int = 500,
    output_subdir: str = "grpo_stage0",
    num_generations: int = 8,
    per_device_batch_size: int = 8,
    grad_accum: int = 2,
    max_positions: int = 1000,
    checkpoint_every: int = 100,
    resume: bool = True,
    include_ascii: bool = False,
) -> str:
    _ensure_installed()
    out_dir = f"{VOL_MOUNT}/checkpoints/{output_subdir}"
    cmd = [
        "python",
        "-m",
        "chessfm.rl.train_grpo",
        "--model",
        model,
        "--stage",
        str(stage),
        "--steps",
        str(steps),
        "--output",
        out_dir,
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
    _run(cmd)
    volume.commit()
    return f"{out_dir}/stage{stage}_final"


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_HF_SECRETS,
    timeout=60 * 60 * 2,
)
def run_eval(
    model: str = DEFAULT_V0_MODEL,
    games: int = 20,
    stockfish_level: int = 1,
    random_opponent: bool = False,
) -> None:
    _ensure_installed()
    cmd = [
        "python",
        "-m",
        "chessfm.rl.evaluate",
        "--model",
        model,
        "--games",
        str(games),
        "--stockfish-level",
        str(stockfish_level),
    ]
    if random_opponent:
        cmd.append("--random-opponent")
    _run(cmd)
    # Copy generated JSON + PGN reports into the persistent Modal Volume
    _run(["cp", "-r", "results/.", f"{VOL_MOUNT}/results/"])
    volume.commit()


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_HF_SECRETS,
    timeout=60 * 60,
)
def run_coach_eval() -> None:
    _ensure_installed()
    _run(["python", "-m", "chessfm.eval.suite", "--demo"])
    if Path(f"{REMOTE_REPO}/results").exists():
        _run(["cp", "-r", "results/.", f"{VOL_MOUNT}/results/"])
    volume.commit()


@app.function(
    volumes={VOL_MOUNT: volume},
    secrets=_HF_SECRETS,
    timeout=60 * 60,
)
def run_export_gguf(
    model: str = f"{VOL_MOUNT}/checkpoints/sft_smoke/final_merged",
    output_subdir: str = "gguf_export",
    quant: str = "q4_k_m",
    family: str = "gemma4",
) -> str:
    _ensure_installed()
    out_dir = f"{VOL_MOUNT}/exports/{output_subdir}"
    _run(
        [
            "python",
            "scripts/export_gguf.py",
            "--model",
            model,
            "--output",
            out_dir,
            "--quant",
            quant,
            "--family",
            family,
        ]
    )
    volume.commit()
    return out_dir


@app.local_entrypoint()
def main(
    task: str = "smoke-pipeline",
    model: str = DEFAULT_V0_MODEL,
    track: str = "",
    use_official_gemma: bool = False,
    data: str = "data_generation/sft_train.jsonl",
    stage: int = 0,
    steps: int = 200,
    epochs: int = 3,
    games: int = 20,
    stockfish_level: int = 1,
    random_opponent: bool = False,
    include_ascii: bool = False,
    quant: str = "q4_k_m",
    family: str = "gemma4",
    output_subdir: str = "",
) -> None:
    resolved_model = resolve_track_model(
        model=model, track=track, use_official_gemma=use_official_gemma
    )

    if task == "sft":
        ckpt = run_sft.remote(
            model=resolved_model,
            data=data,
            output_subdir=output_subdir or "sft_baseline",
            epochs=epochs,
            include_ascii=include_ascii,
            merge_and_save=True,
        )
        print(f"SFT merged checkpoint saved to Modal volume: {ckpt}")
    elif task == "grpo":
        ckpt = run_grpo.remote(
            model=resolved_model,
            stage=stage,
            steps=steps,
            output_subdir=output_subdir or f"grpo_stage{stage}",
            include_ascii=include_ascii,
        )
        print(f"GRPO Stage {stage} checkpoint saved to Modal volume: {ckpt}")
    elif task == "eval":
        run_eval.remote(
            model=resolved_model,
            games=games,
            stockfish_level=stockfish_level,
            random_opponent=random_opponent,
        )
        print(
            "Evaluation complete. Download reports with: modal volume get chessfm-artifacts results ./results"
        )
    elif task == "coach-eval":
        run_coach_eval.remote()
        print("Coach evaluation suite complete.")
    elif task == "export-gguf":
        export_dir = run_export_gguf.remote(
            model=resolved_model,
            output_subdir=output_subdir or "gguf_export",
            quant=quant,
            family=family,
        )
        print(
            f"GGUF export artifacts saved to Modal volume: {export_dir}. "
            "Download with: modal volume get chessfm-artifacts exports ./exports"
        )
    elif task == "smoke-pipeline":
        sft_subdir = output_subdir or "sft_smoke"
        print("=== Step 1/3: SFT (merge-and-save) ===")
        sft_merged_ckpt = run_sft.remote(
            model=resolved_model,
            data=data,
            output_subdir=sft_subdir,
            epochs=epochs,
            include_ascii=include_ascii,
            merge_and_save=True,
        )
        # Ensure we hand off the merged checkpoint ({sft_dir}/final_merged) to GRPO Stage 0.
        if not sft_merged_ckpt.endswith("/final_merged"):
            sft_merged_ckpt = f"{VOL_MOUNT}/checkpoints/{sft_subdir}/final_merged"
        print(f"SFT complete (merged): {sft_merged_ckpt}")

        print("=== Step 2/3: GRPO Stage 0 ===")
        grpo_ckpt = run_grpo.remote(
            model=sft_merged_ckpt,
            stage=0,
            steps=steps,
            output_subdir=(output_subdir or "smoke") + "_grpo_stage0",
            include_ascii=include_ascii,
        )
        print(f"GRPO Stage 0 complete: {grpo_ckpt}")

        print("=== Step 3/3: Stockfish Eval ===")
        run_eval.remote(
            model=grpo_ckpt,
            games=games,
            stockfish_level=stockfish_level,
            random_opponent=random_opponent,
        )
        print(
            "Pipeline complete! Fetch reports: modal volume get chessfm-artifacts results ./results"
        )
    else:
        raise ValueError(f"Unknown task: {task}. Choose from: {', '.join(VALID_TASKS)}")
