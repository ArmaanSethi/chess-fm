#!/usr/bin/env python3
"""Modal runner for ChessFM GPU training, open-weight distillation, and evaluation.

Supports both the verified TRL 0.16 container (``v0`` / ``v1``) and a modern
``transformers>=5.2`` / ``trl>=1.18`` container (``v1.5``, ``v2``/``coach``,
``student``, ``student-e2b``), plus a vLLM container for zero-API open-weight
coaching distillation (``distill-teacher``).

Model tracks:
    - ``v0``: ``Qwen/Qwen2.5-3B-Instruct`` (3B control baseline)
    - ``v1``: ``Qwen/Qwen3-4B`` (4B reasoning backbone, fast TRL/vLLM GRPO)
    - ``v1.5``: ``Qwen/Qwen3.5-4B`` (4B hybrid Gated DeltaNet reasoning backbone)
    - ``v2`` / ``coach``: ``unsloth/gemma-4-E4B-it`` (or ``google/gemma-4-E4B-it``;
      all Gemma 4 checkpoints are Apache-2.0 and ungated)
    - ``student``: ``Qwen/Qwen3.5-0.8B`` (browser WebGPU distilled coach target)
    - ``student-e2b``: ``google/gemma-4-E2B-it`` (local/Ollama distilled coach target)
    - ``teacher``: ``google/gemma-4-26B-A4B-it`` (open-weight MoE distillation writer)
    - ``teacher-qwen``: ``Qwen/Qwen3.6-35B-A3B`` (alternative MoE distillation writer)

Usage:
    # One-time setup:
    pip install modal && modal setup

    # Run SFT (merged) -> GRPO Stage 0 -> Stockfish Eval end-to-end:
    modal run scripts/modal_train.py --task smoke-pipeline --track v1 --include-ascii

    # Individual stages (select model via --track or explicit --model):
    modal run scripts/modal_train.py --task sft --track v1 --epochs 3 --include-ascii
    modal run scripts/modal_train.py --task sft --track v1.5 --epochs 3 --include-ascii
    modal run scripts/modal_train.py --task grpo --track v1 --stage 0 --steps 500 --include-ascii
    modal run scripts/modal_train.py --task eval --model /vol/checkpoints/grpo_stage0/stage0_final --games 50
    modal run scripts/modal_train.py --task distill-teacher --track teacher --limit 1000
    modal run scripts/modal_train.py --task coach-eval
    modal run scripts/modal_train.py --task export-gguf --model /vol/checkpoints/sft_smoke/final_merged

    # Download checkpoints, datasets, and eval reports locally:
    modal volume get chessfm-artifacts results ./results
    modal volume get chessfm-artifacts checkpoints ./checkpoints
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "chessfm-trainer"
VOLUME_NAME = "chessfm-artifacts"
REMOTE_REPO = "/root/chess-fm"
VOL_MOUNT = "/vol"

REPO_ROOT = Path(__file__).resolve().parents[1]

# Supported model tracks.
DEFAULT_V0_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_V1_MODEL = "Qwen/Qwen3-4B"
DEFAULT_V1_5_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_V2_MODEL = "unsloth/gemma-4-E4B-it"
OFFICIAL_V2_MODEL = "google/gemma-4-E4B-it"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-0.8B"
FALLBACK_STUDENT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LOCAL_STUDENT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_TEACHER_MODEL = "google/gemma-4-26B-A4B-it"
ALT_TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B"

TRACK_MODELS: dict[str, str] = {
    "v0": DEFAULT_V0_MODEL,
    "v1": DEFAULT_V1_MODEL,
    "v1.5": DEFAULT_V1_5_MODEL,
    "v2": DEFAULT_V2_MODEL,
    "coach": DEFAULT_V2_MODEL,
    "student": DEFAULT_STUDENT_MODEL,
    "student-qwen3": FALLBACK_STUDENT_MODEL,
    "student-e2b": DEFAULT_LOCAL_STUDENT_MODEL,
    "teacher": DEFAULT_TEACHER_MODEL,
    "teacher-qwen": ALT_TEACHER_MODEL,
}

VALID_TASKS = (
    "smoke-pipeline",
    "sft",
    "sft-student",
    "grpo",
    "eval",
    "static-eval",
    "pass-at-k",
    "coach-eval",
    "coach-benchmark",
    "distill-teacher",
    "p4a-vllm-batch",
    "export-gguf",
    "export-onnx",
    "upload-hf",
)


def requires_modern_stack(model_name: str) -> bool:
    """Return True if ``model_name`` needs the transformers>=5.2 / TRL>=1.18 image."""
    lower = str(model_name).lower()
    return any(
        tag in lower for tag in ("gemma-4", "gemma4", "qwen3.5", "qwen3-5", "qwen3.6", "qwen3.8")
    )


def resolve_track_model(model: str = "", track: str = "", use_official_gemma: bool = False) -> str:
    """Resolve the model ID or checkpoint path from ``--model`` / ``--track`` flags.

    An explicit ``--model`` (e.g. a checkpoint path on the Modal volume) always
    takes precedence over ``--track`` so combining ``--track v1 --model /vol/...``
    never silently evaluates the base model instead of the checkpoint.
    """
    if model:
        return model
    if track:
        key = track.strip().lower()
        if key not in TRACK_MODELS:
            raise ValueError(
                f"Unknown track {track!r}. Choose from: {', '.join(sorted(TRACK_MODELS))}"
            )
        if key in {"v2", "coach"} and use_official_gemma:
            return OFFICIAL_V2_MODEL
        return TRACK_MODELS[key]
    return DEFAULT_V1_MODEL


# Forward optional runtime secrets (HF_TOKEN for Hub uploads / private repos,
# WANDB_API_KEY for experiment tracking) strictly from the local environment.
# When WANDB_API_KEY is absent, explicitly disable W&B so Hugging Face Trainer
# does not block on interactive login prompts inside headless containers.
_RUNTIME_SECRET_DICT: dict[str, str] = {
    key: os.environ[key] for key in ("HF_TOKEN", "WANDB_API_KEY") if os.environ.get(key)
}
if "WANDB_API_KEY" in _RUNTIME_SECRET_DICT:
    _RUNTIME_SECRET_DICT.setdefault("WANDB_PROJECT", "chess-fm")
else:
    _RUNTIME_SECRET_DICT["WANDB_DISABLED"] = "true"
    _RUNTIME_SECRET_DICT["WANDB_MODE"] = "disabled"
_SECRETS = [modal.Secret.from_dict(_RUNTIME_SECRET_DICT)]

_REPO_IGNORE = [
    "venv",
    ".venv",
    ".git",
    ".agents",
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "results",
    "*.pyc",
]

_BASE_ENV = {
    "PATH": "/usr/games:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HF_HOME": f"{VOL_MOUNT}/hf_cache",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# Stable image for v0 (Qwen2.5-3B) and v1 (Qwen3-4B) verified against TRL 0.16.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "stockfish")
    .env(_BASE_ENV)
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
        "wandb>=0.19.0",
    )
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO, ignore=_REPO_IGNORE)
)

# Modern base image for Qwen3.5 and Gemma 4 fine-tuning / RL (local repo mounted last).
_modern_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "stockfish")
    .env(_BASE_ENV)
    .pip_install(
        "torch>=2.5.0",
        "transformers>=5.2.0",
        "accelerate>=1.0.0",
        "peft>=0.15.0",
        "trl>=1.18.0",
        "bitsandbytes>=0.45.0",
        "python-chess>=1.10.0",
        "datasets>=3.0.0,<6",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
        "wandb>=0.19.0",
        "huggingface-hub>=0.28.0",
    )
)

modern_image = _modern_base_image.add_local_dir(
    str(REPO_ROOT), remote_path=REMOTE_REPO, ignore=_REPO_IGNORE
)

# vLLM image for high-throughput open-weight teacher distillation (zero external APIs).
vllm_image = _modern_base_image.pip_install("vllm").add_local_dir(
    str(REPO_ROOT), remote_path=REMOTE_REPO, ignore=_REPO_IGNORE
)

# Dedicated CPU image for Transformers.js v3 WebGPU ONNX export & 4-bit quantization.
onnx_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env(_BASE_ENV)
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.51.0,<5.0",
        "optimum[onnxruntime]",
        "onnx>=1.17.0",
        "onnxruntime>=1.20.0",
        "onnxscript>=0.1.0",
        "huggingface-hub>=0.28.0",
        "numpy<2.3",
    )
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO, ignore=_REPO_IGNORE)
)

# Dedicated CPU-only image for llama.cpp GGUF conversion & Q4_K_M quantization.
gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF",
        "cmake --build /opt/llama.cpp/build --target llama-quantize -j4",
        "cp /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize",
    )
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.51.0,<4.52",
        "safetensors",
        "numpy",
        "sentencepiece",
        "protobuf",
        "gguf",
    )
    .env({**_BASE_ENV, "LLAMA_CPP_DIR": "/opt/llama.cpp"})
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO, ignore=_REPO_IGNORE)
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)


def _ensure_installed() -> None:
    os.makedirs(f"{VOL_MOUNT}/checkpoints", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/data", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/results/pgns", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/exports", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/hf_cache", exist_ok=True)
    _run(["pip", "install", "--no-deps", "-e", "."])
    src_path = str(Path(REMOTE_REPO) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _exec_sft(
    model: str,
    data: str,
    output_subdir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    include_ascii: bool,
    merge_and_save: bool,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    optim: str = "",
    packing: bool = False,
    resume: bool = False,
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
        "--grad-accum",
        str(grad_accum),
        "--lora-r",
        str(lora_r),
        "--all-linear" if all_linear else "--no-all-linear",
        "--merge-and-save" if merge_and_save else "--no-merge-and-save",
    ]
    if lr > 0:
        cmd.extend(["--lr", str(lr)])
    if lora_alpha > 0:
        cmd.extend(["--lora-alpha", str(lora_alpha)])
    if full_ft:
        cmd.append("--full-ft")
    if optim:
        cmd.extend(["--optim", optim])
    if packing:
        cmd.append("--packing")
    if resume:
        cmd.append("--resume")
    if include_ascii:
        cmd.append("--include-ascii")
    if include_piece_list:
        cmd.append("--include-piece-list")
    if include_legal_moves:
        cmd.append("--include-legal-moves")
    _run(cmd)
    volume.commit()
    merged_dir = f"{out_dir}/final_merged"
    if merge_and_save and not full_ft and Path(merged_dir).exists():
        return merged_dir
    return f"{out_dir}/final"


def _exec_grpo(
    model: str,
    stage: int,
    steps: int,
    output_subdir: str,
    num_generations: int,
    per_device_batch_size: int,
    grad_accum: int,
    max_positions: int,
    checkpoint_every: int,
    resume: bool,
    include_ascii: bool,
    positions: str = "",
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    reward_mode: str = "legacy",
    lr: float = 0.0,
    beta: float = -1.0,
    epsilon: float = -1.0,
    epsilon_high: float = -1.0,
    loss_type: str = "",
    scale_rewards: bool = True,
    filter_zero_variance: bool = False,
    num_iterations: int = 0,
    temperature: float = -1.0,
    seed: int = -1,
    max_prompt_length: int = 0,
    max_completion_length: int = 0,
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
        "--reward-mode",
        reward_mode,
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
        "--checkpoint-every",
        str(checkpoint_every),
        "--lora-r",
        str(lora_r),
        "--all-linear" if all_linear else "--no-all-linear",
        "--scale-rewards" if scale_rewards else "--no-scale-rewards",
        "--merge-and-save" if merge_and_save else "--no-merge-and-save",
    ]
    if lora_alpha > 0:
        cmd.extend(["--lora-alpha", str(lora_alpha)])
    if full_ft:
        cmd.append("--full-ft")
    if lr > 0:
        cmd.extend(["--lr", str(lr)])
    if beta >= 0:
        cmd.extend(["--beta", str(beta)])
    if epsilon >= 0:
        cmd.extend(["--epsilon", str(epsilon)])
    if epsilon_high >= 0:
        cmd.extend(["--epsilon-high", str(epsilon_high)])
    if loss_type:
        cmd.extend(["--loss-type", loss_type])
    if filter_zero_variance:
        cmd.append("--filter-zero-variance")
    if num_iterations > 0:
        cmd.extend(["--num-iterations", str(num_iterations)])
    if temperature >= 0:
        cmd.extend(["--temperature", str(temperature)])
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    if max_prompt_length > 0:
        cmd.extend(["--max-prompt-length", str(max_prompt_length)])
    if max_completion_length > 0:
        cmd.extend(["--max-completion-length", str(max_completion_length)])
    if positions:
        cmd.extend(["--positions", positions])
    if max_positions > 0:
        cmd.extend(["--max-positions", str(max_positions)])
    if resume:
        cmd.append("--resume")
    if include_ascii:
        cmd.append("--include-ascii")
    if include_piece_list:
        cmd.append("--include-piece-list")
    if include_legal_moves:
        cmd.append("--include-legal-moves")
    _run(cmd)
    volume.commit()
    merged_dir = f"{out_dir}/stage{stage}_final_merged"
    if merge_and_save and not full_ft and Path(merged_dir).exists():
        return merged_dir
    return f"{out_dir}/stage{stage}_final"


def _exec_eval(
    model: str,
    games: int,
    stockfish_level: int,
    random_opponent: bool,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    openings: str = "",
    maia_elo: int = 0,
    opponent_depth: int = 0,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    stockfish_elo: int = 0,
    manifest_output: str = "",
    seed: int = -1,
    avoid_repetition: bool = False,
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
        "--decoding-mode",
        decoding_mode,
        "--enable-thinking" if enable_thinking else "--no-enable-thinking",
    ]
    if random_opponent:
        cmd.append("--random-opponent")
    if avoid_repetition:
        cmd.append("--avoid-repetition")
    if include_ascii:
        cmd.append("--include-ascii")
    if include_piece_list:
        cmd.append("--include-piece-list")
    if include_legal_moves:
        cmd.append("--include-legal-moves")
    if openings:
        cmd.extend(["--openings", openings])
    if maia_elo > 0:
        cmd.extend(["--maia-elo", str(maia_elo)])
    if opponent_depth > 0:
        cmd.extend(["--opponent-depth", str(opponent_depth)])
    if batch_size > 0:
        cmd.extend(["--batch-size", str(batch_size)])
    if stockfish_elo > 0:
        cmd.extend(["--stockfish-elo", str(stockfish_elo)])
    if manifest_output:
        cmd.extend(["--manifest-output", manifest_output])
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    _run(cmd)
    _run(["cp", "-r", "results/.", f"{VOL_MOUNT}/results/"])
    volume.commit()


def _suite_supports_flag(flag: str) -> bool:
    """Return True if ``chessfm.eval.suite`` accepts ``flag`` (e.g. ``--include-piece-list``)."""
    param_name = flag.lstrip("-").replace("-", "_")
    src_path = str(Path(REMOTE_REPO) / "src")
    try:
        if Path(src_path).is_dir() and src_path not in sys.path:
            sys.path.insert(0, src_path)
    except OSError:
        pass
    try:
        from chessfm.eval.suite import evaluate_model_suite
    except ImportError:
        return False
    return param_name in inspect.signature(evaluate_model_suite).parameters


def _exec_static_eval(
    model: str,
    benchmark: str,
    decoding_mode: str,
    include_ascii: bool,
    enable_thinking: bool,
    max_positions: int,
    skip_probes: bool,
    output_name: str,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    puzzles: str = "",
    chessqa: str = "",
    english_forgetting: str = "",
    depth: int = 0,
    coaching_jsonl: str = "",
) -> str:
    _ensure_installed()
    out_path = output_name if output_name.startswith("/") else f"{VOL_MOUNT}/results/{output_name}"
    cmd = [
        "python",
        "-m",
        "chessfm.eval.suite",
        "--model",
        model,
        "--decoding-mode",
        decoding_mode,
        "--enable-thinking" if enable_thinking else "--no-enable-thinking",
        "--output",
        out_path,
    ]
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    if include_ascii:
        cmd.append("--include-ascii")
    if include_piece_list:
        if not _suite_supports_flag("--include-piece-list"):
            raise ValueError(
                "chessfm.eval.suite does not yet support --include-piece-list. "
                "Update chessfm.eval.suite or omit --include-piece-list for static-eval."
            )
        cmd.append("--include-piece-list")
    if include_legal_moves:
        if not _suite_supports_flag("--include-legal-moves"):
            raise ValueError(
                "chessfm.eval.suite does not yet support --include-legal-moves. "
                "Update chessfm.eval.suite or omit --include-legal-moves for static-eval."
            )
        cmd.append("--include-legal-moves")
    if max_positions > 0:
        cmd.extend(["--max-positions", str(max_positions)])
    if skip_probes:
        cmd.append("--skip-probes")
    if batch_size > 0:
        cmd.extend(["--batch-size", str(batch_size)])
    if puzzles:
        cmd.extend(["--puzzles", puzzles])
    if chessqa:
        cmd.extend(["--chessqa", chessqa])
    if english_forgetting:
        cmd.extend(["--english-forgetting", english_forgetting])
    if depth > 0:
        cmd.extend(["--depth", str(depth)])
    if coaching_jsonl:
        cmd.extend(["--coaching-jsonl", coaching_jsonl])
    _run(cmd)
    volume.commit()
    return out_path


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 4,
)
def run_sft(
    model: str = DEFAULT_V1_MODEL,
    data: str = "data_generation/sft_train.jsonl",
    output_subdir: str = "sft_qwen3_4b",
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 0.0,
    include_ascii: bool = False,
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    optim: str = "",
    packing: bool = False,
    resume: bool = False,
) -> str:
    return _exec_sft(
        model,
        data,
        output_subdir,
        epochs,
        batch_size,
        grad_accum,
        lr,
        include_ascii,
        merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        optim=optim,
        packing=packing,
        resume=resume,
    )


@app.function(
    gpu="A100-80GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 6,
)
def run_sft_a100(
    model: str = DEFAULT_V1_MODEL,
    data: str = "data_generation/sft_train.jsonl",
    output_subdir: str = "sft_a100",
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 0.0,
    include_ascii: bool = False,
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    optim: str = "",
    packing: bool = False,
    resume: bool = False,
) -> str:
    return _exec_sft(
        model,
        data,
        output_subdir,
        epochs,
        batch_size,
        grad_accum,
        lr,
        include_ascii,
        merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        optim=optim,
        packing=packing,
        resume=resume,
    )


@app.function(
    gpu="H100",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 6,
)
def run_sft_h100(
    model: str = DEFAULT_V1_MODEL,
    data: str = "data_generation/sft_train.jsonl",
    output_subdir: str = "sft_h100",
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 0.0,
    include_ascii: bool = False,
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = True,
    optim: str = "paged_adamw_8bit",
    packing: bool = False,
    resume: bool = False,
) -> str:
    return _exec_sft(
        model,
        data,
        output_subdir,
        epochs,
        batch_size,
        grad_accum,
        lr,
        include_ascii,
        merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        optim=optim,
        packing=packing,
        resume=resume,
    )


@app.function(
    image=modern_image,
    gpu="A100-40GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 4,
)
def run_sft_modern(
    model: str = DEFAULT_V1_5_MODEL,
    data: str = "data_generation/sft_train.jsonl",
    output_subdir: str = "sft_modern",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 0.0,
    include_ascii: bool = False,
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    optim: str = "",
    packing: bool = False,
    resume: bool = False,
) -> str:
    return _exec_sft(
        model,
        data,
        output_subdir,
        epochs,
        batch_size,
        grad_accum,
        lr,
        include_ascii,
        merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        optim=optim,
        packing=packing,
        resume=resume,
    )


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 6,
)
def run_grpo(
    model: str = DEFAULT_V1_MODEL,
    stage: int = 0,
    steps: int = 500,
    output_subdir: str = "grpo_stage0",
    num_generations: int = 4,
    per_device_batch_size: int = 4,
    grad_accum: int = 2,
    max_positions: int = 1000,
    checkpoint_every: int = 100,
    resume: bool = True,
    include_ascii: bool = False,
    positions: str = "",
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    reward_mode: str = "legacy",
    lr: float = 0.0,
    beta: float = -1.0,
    epsilon: float = -1.0,
    epsilon_high: float = -1.0,
    loss_type: str = "",
    scale_rewards: bool = True,
    filter_zero_variance: bool = False,
    num_iterations: int = 0,
    temperature: float = -1.0,
    seed: int = -1,
    max_prompt_length: int = 0,
    max_completion_length: int = 0,
) -> str:
    return _exec_grpo(
        model,
        stage,
        steps,
        output_subdir,
        num_generations,
        per_device_batch_size,
        grad_accum,
        max_positions,
        checkpoint_every,
        resume,
        include_ascii,
        positions=positions,
        merge_and_save=merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        reward_mode=reward_mode,
        lr=lr,
        beta=beta,
        epsilon=epsilon,
        epsilon_high=epsilon_high,
        loss_type=loss_type,
        scale_rewards=scale_rewards,
        filter_zero_variance=filter_zero_variance,
        num_iterations=num_iterations,
        temperature=temperature,
        seed=seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )


@app.function(
    gpu="A100-80GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 8,
)
def run_grpo_a100(
    model: str = DEFAULT_V1_MODEL,
    stage: int = 1,
    steps: int = 500,
    output_subdir: str = "grpo_a100",
    num_generations: int = 8,
    per_device_batch_size: int = 8,
    grad_accum: int = 2,
    max_positions: int = 1000,
    checkpoint_every: int = 100,
    resume: bool = True,
    include_ascii: bool = False,
    positions: str = "",
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    reward_mode: str = "wdl_loss",
    lr: float = 0.0,
    beta: float = 0.04,
    epsilon: float = 0.20,
    epsilon_high: float = 0.28,
    loss_type: str = "",
    scale_rewards: bool = True,
    filter_zero_variance: bool = True,
    num_iterations: int = 0,
    temperature: float = -1.0,
    seed: int = -1,
    max_prompt_length: int = 0,
    max_completion_length: int = 0,
) -> str:
    return _exec_grpo(
        model,
        stage,
        steps,
        output_subdir,
        num_generations,
        per_device_batch_size,
        grad_accum,
        max_positions,
        checkpoint_every,
        resume,
        include_ascii,
        positions=positions,
        merge_and_save=merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        reward_mode=reward_mode,
        lr=lr,
        beta=beta,
        epsilon=epsilon,
        epsilon_high=epsilon_high,
        loss_type=loss_type,
        scale_rewards=scale_rewards,
        filter_zero_variance=filter_zero_variance,
        num_iterations=num_iterations,
        temperature=temperature,
        seed=seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )


@app.function(
    image=modern_image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 6,
)
def run_grpo_modern(
    model: str = DEFAULT_V1_5_MODEL,
    stage: int = 0,
    steps: int = 500,
    output_subdir: str = "grpo_modern_stage0",
    num_generations: int = 8,
    per_device_batch_size: int = 8,
    grad_accum: int = 2,
    max_positions: int = 1000,
    checkpoint_every: int = 100,
    resume: bool = True,
    include_ascii: bool = False,
    positions: str = "",
    merge_and_save: bool = True,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    reward_mode: str = "legacy",
    lr: float = 0.0,
    beta: float = -1.0,
    epsilon: float = -1.0,
    epsilon_high: float = -1.0,
    loss_type: str = "",
    scale_rewards: bool = True,
    filter_zero_variance: bool = False,
    num_iterations: int = 0,
    temperature: float = -1.0,
    seed: int = -1,
    max_prompt_length: int = 0,
    max_completion_length: int = 0,
) -> str:
    return _exec_grpo(
        model,
        stage,
        steps,
        output_subdir,
        num_generations,
        per_device_batch_size,
        grad_accum,
        max_positions,
        checkpoint_every,
        resume,
        include_ascii,
        positions=positions,
        merge_and_save=merge_and_save,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        all_linear=all_linear,
        full_ft=full_ft,
        reward_mode=reward_mode,
        lr=lr,
        beta=beta,
        epsilon=epsilon,
        epsilon_high=epsilon_high,
        loss_type=loss_type,
        scale_rewards=scale_rewards,
        filter_zero_variance=filter_zero_variance,
        num_iterations=num_iterations,
        temperature=temperature,
        seed=seed,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_eval(
    model: str = DEFAULT_V1_MODEL,
    games: int = 20,
    stockfish_level: int = 1,
    random_opponent: bool = False,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    openings: str = "",
    maia_elo: int = 0,
    opponent_depth: int = 0,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    stockfish_elo: int = 0,
    manifest_output: str = "",
    seed: int = -1,
    avoid_repetition: bool = False,
) -> None:
    _exec_eval(
        model,
        games,
        stockfish_level,
        random_opponent,
        decoding_mode=decoding_mode,
        include_ascii=include_ascii,
        enable_thinking=enable_thinking,
        openings=openings,
        maia_elo=maia_elo,
        opponent_depth=opponent_depth,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        batch_size=batch_size,
        stockfish_elo=stockfish_elo,
        manifest_output=manifest_output,
        seed=seed,
        avoid_repetition=avoid_repetition,
    )


@app.function(
    image=modern_image,
    gpu="A100-40GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_eval_modern(
    model: str = DEFAULT_V1_5_MODEL,
    games: int = 20,
    stockfish_level: int = 1,
    random_opponent: bool = False,
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    openings: str = "",
    maia_elo: int = 0,
    opponent_depth: int = 0,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    stockfish_elo: int = 0,
    manifest_output: str = "",
    seed: int = -1,
    avoid_repetition: bool = False,
) -> None:
    _exec_eval(
        model,
        games,
        stockfish_level,
        random_opponent,
        decoding_mode=decoding_mode,
        include_ascii=include_ascii,
        enable_thinking=enable_thinking,
        openings=openings,
        maia_elo=maia_elo,
        opponent_depth=opponent_depth,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        batch_size=batch_size,
        stockfish_elo=stockfish_elo,
        manifest_output=manifest_output,
        seed=seed,
        avoid_repetition=avoid_repetition,
    )


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_static_eval(
    model: str = DEFAULT_V1_MODEL,
    benchmark: str = "data_generation/eval_benchmark.jsonl",
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    max_positions: int = 0,
    skip_probes: bool = False,
    output_name: str = "eval_manifest.json",
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    puzzles: str = "",
    chessqa: str = "",
    english_forgetting: str = "",
    depth: int = 0,
    coaching_jsonl: str = "",
) -> str:
    return _exec_static_eval(
        model,
        benchmark,
        decoding_mode,
        include_ascii,
        enable_thinking,
        max_positions,
        skip_probes,
        output_name,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        batch_size=batch_size,
        puzzles=puzzles,
        chessqa=chessqa,
        english_forgetting=english_forgetting,
        depth=depth,
        coaching_jsonl=coaching_jsonl,
    )


@app.function(
    image=modern_image,
    gpu="A100-40GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_static_eval_modern(
    model: str = DEFAULT_V1_5_MODEL,
    benchmark: str = "data_generation/eval_benchmark.jsonl",
    decoding_mode: str = "strict",
    include_ascii: bool = False,
    enable_thinking: bool = True,
    max_positions: int = 0,
    skip_probes: bool = False,
    output_name: str = "eval_manifest.json",
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    batch_size: int = 0,
    puzzles: str = "",
    chessqa: str = "",
    english_forgetting: str = "",
    depth: int = 0,
    coaching_jsonl: str = "",
) -> str:
    return _exec_static_eval(
        model,
        benchmark,
        decoding_mode,
        include_ascii,
        enable_thinking,
        max_positions,
        skip_probes,
        output_name,
        include_piece_list=include_piece_list,
        include_legal_moves=include_legal_moves,
        batch_size=batch_size,
        puzzles=puzzles,
        chessqa=chessqa,
        english_forgetting=english_forgetting,
        depth=depth,
        coaching_jsonl=coaching_jsonl,
    )


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_pass_at_k(
    model: str = DEFAULT_V1_MODEL,
    puzzles: str = "/vol/data/phase2/test_puzzles_900.jsonl",
    output: str = "pass_at_k_summary.json",
    raw_output: str = "",
    num_puzzles: int = 200,
    k: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
    seed: int = 42,
    include_ascii: bool = False,
    include_piece_list: bool = True,
    include_legal_moves: bool = False,
    enable_thinking: bool = True,
) -> str:
    _ensure_installed()
    out_path = output if output.startswith("/") else f"{VOL_MOUNT}/results/{output}"
    cmd = [
        "python",
        "-m",
        "chessfm.eval.pass_at_k",
        "--model",
        model,
        "--puzzles",
        puzzles,
        "--output",
        out_path,
        "--num-puzzles",
        str(num_puzzles),
        "--k",
        str(k),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--enable-thinking" if enable_thinking else "--no-enable-thinking",
    ]
    if raw_output:
        raw_path = raw_output if raw_output.startswith("/") else f"{VOL_MOUNT}/results/{raw_output}"
        cmd.extend(["--raw-output", raw_path])
    if include_ascii:
        cmd.append("--include-ascii")
    if include_piece_list:
        cmd.append("--include-piece-list")
    if include_legal_moves:
        cmd.append("--include-legal-moves")
    _run(cmd)
    volume.commit()
    return out_path


@app.function(
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 30,
)
def run_upload_hf(
    local_dir: str,
    repo_id: str,
    repo_type: str = "model",
    path_in_repo: str = "",
    commit_message: str = "Upload ChessFM artifacts",
    dry_run: bool = False,
    private: bool = True,
) -> str:
    _ensure_installed()
    cmd = [
        "python",
        "scripts/upload_hf.py",
        "--local-dir",
        local_dir,
        "--repo-id",
        repo_id,
        "--repo-type",
        repo_type,
        "--commit-message",
        commit_message,
    ]
    if private:
        cmd.append("--private")
    if path_in_repo:
        cmd.extend(["--path-in-repo", path_in_repo])
    if dry_run:
        cmd.append("--dry-run")
    _run(cmd)
    return repo_id


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 4,
)
def run_distill_teacher(
    model: str = DEFAULT_TEACHER_MODEL,
    input_data: str = "data_generation/coaching_sample.jsonl",
    output_name: str = "distilled_coach.jsonl",
    max_attempts: int = 3,
    limit: int = 0,
    workers: int = 24,
) -> str:
    """Serve an open-weight teacher locally via vLLM and generate ACT-Eval-verified traces."""
    import requests

    _ensure_installed()
    out_path = f"{VOL_MOUNT}/data/{output_name}"
    port = 8001
    server_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
            "--dtype",
            "bfloat16",
        ],
        cwd=REMOTE_REPO,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        for _ in range(450):
            if server_proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {server_proc.returncode}")
            try:
                if requests.get(health_url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Timed out waiting for local vLLM teacher server to become ready.")

        cmd = [
            "python",
            "-m",
            "chessfm.data_generation.distill_open_teacher",
            "--input",
            input_data,
            "--output",
            out_path,
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--model",
            model,
            "--max-attempts",
            str(max_attempts),
            "--workers",
            str(workers),
        ]
        if limit > 0:
            cmd.extend(["--limit", str(limit)])
        _run(cmd)
        volume.commit()
        return out_path
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 45,
)
def run_p4a_vllm_batch(
    model: str = DEFAULT_TEACHER_MODEL,
    coach_test_path: str = f"{VOL_MOUNT}/data/phase2/test_coach_300.jsonl",
    fresh_seed_path: str = f"{VOL_MOUNT}/data/phase2/coach_fresh_1k_seed.jsonl",
    puzzles_path: str = f"{VOL_MOUNT}/data/phase2/test_puzzles_900.jsonl",
    chessqa_path: str = f"{VOL_MOUNT}/data/phase2/test_chessqa_1000.jsonl",
    out_dir: str = f"{VOL_MOUNT}/results/phase2/e5",
    workers: int = 24,
    eval_workers: int = 32,
    skip_coach: bool = False,
) -> str:
    """Execute all 3 Phase 4A vLLM workloads in ONE warm A100-80GB container boot.

    Workloads:
    1. Row 6 Teacher (single-shot, temperature=0.2, tactical facts) on test_coach_300.jsonl.
    2. Distill 1,000 fresh tactical-fact-grounded traces (max_attempts=3, Strict V2 filtered).
    3. Piece #3 Frontier Baseline (zero-shot Piece-by-Square, temperature=0.0, no legal moves list)
       on test_puzzles_900.jsonl (max_tokens=512) and test_chessqa_1000.jsonl (max_tokens=128).
    """
    import concurrent.futures
    import json
    import threading
    from datetime import datetime, timezone

    import requests

    container_t0 = time.monotonic()
    _ensure_installed()

    from chessfm.config import make_prompt, render_piece_list
    from chessfm.fens import read_jsonl
    from chessfm.training.manifest import _get_git_sha

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    port = 8001
    base_url = f"http://127.0.0.1:{port}/v1"

    server_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            "4096",
            "--gpu-memory-utilization",
            "0.90",
            "--dtype",
            "bfloat16",
        ],
        cwd=REMOTE_REPO,
    )

    stage_timings: dict[str, float] = {}
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        for _ in range(450):
            if server_proc.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {server_proc.returncode}")
            try:
                if requests.get(health_url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Timed out waiting for local vLLM server to become ready.")

        vllm_ready_t = time.monotonic()
        vllm_startup_s = round(vllm_ready_t - container_t0, 2)
        print(f"[p4a-vllm-batch] vLLM ready in {vllm_startup_s}s", flush=True)

        row6_out = str(out_root / "coach_eval_teacher_26b.jsonl")
        fresh_out = str(out_root / "coach_fresh_1k_distilled.jsonl")
        if not skip_coach:
            # Stage 1: Row 6 Teacher single-shot on test_coach_300.jsonl
            t_s1 = time.monotonic()
            _run(
                [
                    "python",
                    "-m",
                    "chessfm.data_generation.distill_open_teacher",
                    "--input",
                    coach_test_path,
                    "--output",
                    row6_out,
                    "--base-url",
                    base_url,
                    "--model",
                    model,
                    "--include-tactical-facts",
                    "--single-shot",
                    "--temperature",
                    "0.2",
                    "--workers",
                    str(workers),
                ]
            )
            stage_timings["row6_teacher_300_s"] = round(time.monotonic() - t_s1, 2)
            volume.commit()
            print(
                f"[p4a-vllm-batch] Stage 1 (Row 6 Teacher 300) done in {stage_timings['row6_teacher_300_s']}s",
                flush=True,
            )

            # Stage 2: Distill 1,000 fresh tactical-fact-grounded traces
            t_s2 = time.monotonic()
            _run(
                [
                    "python",
                    "-m",
                    "chessfm.data_generation.distill_open_teacher",
                    "--input",
                    fresh_seed_path,
                    "--output",
                    fresh_out,
                    "--base-url",
                    base_url,
                    "--model",
                    model,
                    "--include-tactical-facts",
                    "--max-attempts",
                    "3",
                    "--workers",
                    str(workers),
                ]
            )
            stage_timings["fresh_1k_distill_s"] = round(time.monotonic() - t_s2, 2)
            volume.commit()
            print(
                f"[p4a-vllm-batch] Stage 2 (Fresh 1K distill) done in {stage_timings['fresh_1k_distill_s']}s",
                flush=True,
            )
        else:
            print("[p4a-vllm-batch] Skipping Stages 1 & 2 (--skip-coach)", flush=True)

        # Shared threaded HTTP helper for Piece #3 (Puzzles 900 + ChessQA 1000)
        tls = threading.local()

        def _post_chat(user_content: str, max_tokens: int) -> dict:
            sess = getattr(tls, "session", None)
            if sess is None:
                sess = requests.Session()
                tls.session = sess
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            resp = sess.post(f"{base_url}/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            content = str(msg.get("content") or "")
            reasoning = str(msg.get("reasoning_content") or "")
            if reasoning and "<think>" not in content.lower():
                raw_output = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                raw_output = content
            usage = data.get("usage") or {}
            return {
                "raw_output": raw_output,
                "content": content,
                "reasoning_content": reasoning,
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "finish_reason": choice.get("finish_reason"),
            }

        # Stage 3A: Piece #3 Puzzles 900 (exact Piece-by-Square prompt, NO legal moves list)
        t_s3a = time.monotonic()
        puzzle_rows = read_jsonl(Path(puzzles_path))

        def _eval_puzzle(item: dict) -> dict:
            fen = item["fen"]
            prompt = make_prompt(
                fen,
                include_ascii=False,
                include_piece_list=True,
                include_legal_moves=False,
            )
            try:
                res = _post_chat(prompt, max_tokens=512)
            except Exception as exc:
                res = {
                    "raw_output": "",
                    "content": "",
                    "reasoning_content": "",
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "finish_reason": f"error:{exc}",
                }
            return {
                "fen": fen,
                "best_move": item.get("best_move"),
                "line": item.get("line"),
                "rating": item.get("rating"),
                "split_group": item.get("split_group"),
                "primary_theme": item.get("primary_theme"),
                "difficulty_band": item.get("difficulty_band"),
                **res,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as ex:
            puzzle_outputs = list(ex.map(_eval_puzzle, puzzle_rows))

        puzzles_out = out_root / "gemma26b_puzzles_900_raw.jsonl"
        with puzzles_out.open("w", encoding="utf-8") as fh:
            for rec in puzzle_outputs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stage_timings["piece3_puzzles_900_s"] = round(time.monotonic() - t_s3a, 2)
        volume.commit()
        print(
            f"[p4a-vllm-batch] Stage 3A (Piece #3 Puzzles 900) done in {stage_timings['piece3_puzzles_900_s']}s",
            flush=True,
        )

        # Stage 3B: Piece #3 ChessQA 1000 (exact _build_chessqa_prompt user content)
        t_s3b = time.monotonic()
        chessqa_rows = read_jsonl(Path(chessqa_path))

        def _build_chessqa_user_content(item: dict) -> str:
            question = str(item.get("question", ""))
            fen = item.get("fen")
            if isinstance(fen, str) and fen.strip():
                extra_str = f"Pieces: {render_piece_list(fen)}"
                lines = question.splitlines()
                if lines and lines[0].lstrip().upper().startswith("FEN:"):
                    rest = "\n".join(lines[1:])
                    return f"{lines[0]}\n{extra_str}" + (f"\n{rest}" if rest else "")
                return f"FEN: {fen}\n{extra_str}\n{question}"
            return question

        def _eval_chessqa(item: dict) -> dict:
            user_content = _build_chessqa_user_content(item)
            try:
                res = _post_chat(user_content, max_tokens=128)
            except Exception as exc:
                res = {
                    "raw_output": "",
                    "content": "",
                    "reasoning_content": "",
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "finish_reason": f"error:{exc}",
                }
            return {
                "fen": item.get("fen"),
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "answer": item.get("answer", item.get("expected", "")),
                **res,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=eval_workers) as ex:
            chessqa_outputs = list(ex.map(_eval_chessqa, chessqa_rows))

        chessqa_out = out_root / "gemma26b_chessqa_1000_raw.jsonl"
        with chessqa_out.open("w", encoding="utf-8") as fh:
            for rec in chessqa_outputs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stage_timings["piece3_chessqa_1000_s"] = round(time.monotonic() - t_s3b, 2)

        total_container_s = round(time.monotonic() - container_t0, 2)
        active_gpu_s = round(time.monotonic() - vllm_ready_t, 2)
        rate_per_sec = 2.50 / 3600.0
        manifest = {
            "model": model,
            "gpu": "A100-80GB",
            "rate_usd_per_hr": 2.50,
            "git_sha": _get_git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vllm_startup_seconds": vllm_startup_s,
            "active_gpu_seconds": active_gpu_s,
            "container_wall_clock_seconds": total_container_s,
            "estimated_active_cost_usd": round(active_gpu_s * rate_per_sec, 4),
            "estimated_conservative_cost_usd": round(total_container_s * rate_per_sec, 4),
            "stage_timings_seconds": stage_timings,
            "outputs": {
                "row6_teacher": row6_out,
                "fresh_1k_distilled": fresh_out,
                "piece3_puzzles_900_raw": str(puzzles_out),
                "piece3_chessqa_1000_raw": str(chessqa_out),
            },
            "counts": {
                "puzzles_rows": len(puzzle_outputs),
                "chessqa_rows": len(chessqa_outputs),
            },
        }
        manifest_path = out_root / "p4a_vllm_batch_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        volume.commit()
        print(json.dumps(manifest, indent=2), flush=True)
        return str(manifest_path)
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60,
)
def run_coach_eval() -> None:
    _ensure_installed()
    _run(["python", "-m", "chessfm.eval.suite", "--demo"])
    if Path(f"{REMOTE_REPO}/results").exists():
        _run(["cp", "-r", "results/.", f"{VOL_MOUNT}/results/"])
    volume.commit()


@app.function(
    image=gguf_image,
    cpu=4.0,
    memory=16384,
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60,
)
def run_export_gguf(
    model: str = f"{VOL_MOUNT}/checkpoints/sft_smoke/final_merged",
    output_subdir: str = "gguf_export",
    quant: str = "q4_k_m",
    family: str = "gemma4",
    coach_v2: bool = False,
    keep_f16: bool = False,
) -> str:
    os.makedirs(f"{VOL_MOUNT}/exports", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/hf_cache", exist_ok=True)
    out_dir = f"{VOL_MOUNT}/exports/{output_subdir}"
    cmd = [
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
    if coach_v2:
        cmd.append("--coach-v2")
    if keep_f16:
        cmd.append("--keep-f16")
    _run(cmd)
    volume.commit()
    return out_dir


@app.function(
    image=modern_image,
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60 * 2,
)
def run_sft_student_l4(
    model: str = DEFAULT_STUDENT_MODEL,
    data: str = f"{VOL_MOUNT}/data/distilled_coach.jsonl",
    output_subdir: str = "sft_student_coach",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    include_ascii: bool = False,
    merge_and_save: bool = True,
) -> str:
    """Fine-tune a small browser-target student model (0.6B-0.8B) on L4."""
    return _exec_sft(
        model,
        data,
        output_subdir,
        epochs,
        batch_size,
        grad_accum,
        lr,
        include_ascii,
        merge_and_save,
    )


@app.function(
    image=onnx_image,
    cpu=4.0,
    memory=16384,
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60,
)
def run_export_onnx(
    model: str = FALLBACK_STUDENT_MODEL,
    output_subdir: str = "onnx_webgpu_export",
    quant: str = "q4f16",
    opset: int = 17,
    upload_hf: str = "",
    dry_run: bool = False,
) -> str:
    """Export a merged student model to quantized Transformers.js v3 WebGPU ONNX."""
    os.makedirs(f"{VOL_MOUNT}/exports", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/hf_cache", exist_ok=True)
    out_dir = f"{VOL_MOUNT}/exports/{output_subdir}"
    cmd = [
        "python",
        "scripts/export_onnx_webgpu.py",
        "--model",
        model,
        "--output",
        out_dir,
        "--quant",
        quant,
        "--opset",
        str(opset),
    ]
    if upload_hf:
        cmd.extend(["--upload-hf", upload_hf])
    if dry_run:
        cmd.append("--dry-run")
    _run(cmd)
    if not dry_run:
        volume.commit()
    return out_dir


def _exec_coach_benchmark(
    model: str,
    benchmark: str,
    output_subdir: str,
    limit: int = 0,
    use_tactical_facts: bool = True,
    batch_size: int = 16,
    max_rows: int = 0,
    base_model: str = "",
) -> str:
    """Shared batched Coach benchmark runner supporting merged models & PEFT LoRA adapters."""
    import json
    from datetime import datetime, timezone

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_installed()

    from chessfm.coach.coach_rewards import summarize_coach_eval
    from chessfm.data_generation.distill_open_teacher import (
        SYSTEM_PROMPT,
        _is_non_coaching_row,
        _resolve_tactical_facts,
        build_teacher_prompt,
        infer_task_type,
        strip_reasoning_blocks,
    )
    from chessfm.fens import read_jsonl

    bench_path = Path(benchmark)
    if not bench_path.is_absolute():
        bench_path = Path(REMOTE_REPO) / bench_path
    rows = [r for r in read_jsonl(bench_path) if r.get("fen") and not _is_non_coaching_row(r)]
    effective_limit = max_rows if max_rows > 0 else limit
    if effective_limit > 0:
        rows = rows[:effective_limit]

    model_path = Path(model)
    adapter_cfg_path = model_path / "adapter_config.json"
    if adapter_cfg_path.is_file():
        from peft import PeftModel

        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        resolved_base = base_model or str(adapter_cfg.get("base_model_name_or_path", ""))
        if not resolved_base:
            raise ValueError(
                f"PEFT adapter at {model} has no base_model_name_or_path; pass --base-model"
            )
        tok_src = model if (model_path / "tokenizer_config.json").is_file() else resolved_base
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        base_lm = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        lm = PeftModel.from_pretrained(base_lm, model)
        if hasattr(lm, "merge_and_unload"):
            lm = lm.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        lm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm.eval()

    start_t = time.monotonic()
    prepared_prompts: list[str] = []
    prepared_meta: list[dict] = []

    for row in rows:
        fen = row["fen"]
        task_type = infer_task_type(row)
        best_move = str(row.get("best_move_uci") or row.get("best_move") or "")
        tf = _resolve_tactical_facts(row) if use_tactical_facts else None

        if use_tactical_facts:
            # Always build grounded tactical-fact prompt; never reuse raw ungrounded row["messages"].
            prompt_row = {**row, "tactical_facts": tf} if tf is not None else row
            user_prompt = build_teacher_prompt(
                prompt_row,
                task_type=task_type,
                include_tactical_facts=True,
            )
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = row.get("messages")
            if isinstance(messages, list) and len(messages) >= 2:
                prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            else:
                user_prompt = build_teacher_prompt(row, task_type=task_type)
                prompt_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

        try:
            prompt_str = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_str = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prepared_prompts.append(prompt_str)
        prepared_meta.append(
            {
                "fen": fen,
                "task_type": task_type,
                "best_move": best_move,
                "student_move": str(row.get("student_move") or row.get("played_move_uci") or ""),
                "pv": row.get("pv") or row.get("principal_variation") or [],
                "cp_loss": row.get("cp_loss"),
                "tactical_facts": tf,
            }
        )

    eff_bs = max(1, int(batch_size))
    raw_responses: list[str] = []
    for start_idx in range(0, len(prepared_prompts), eff_bs):
        chunk_prompts = prepared_prompts[start_idx : start_idx + eff_bs]
        inputs = tokenizer(
            chunk_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(lm.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen_ids = lm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq in gen_ids:
            new_tokens = seq[input_len:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            raw_responses.append(strip_reasoning_blocks(raw_text))

    eval_rows = [
        {**meta, "response": resp} for meta, resp in zip(prepared_meta, raw_responses, strict=True)
    ]
    outputs_jsonl, summary = summarize_coach_eval(eval_rows)
    wall_s = round(time.monotonic() - start_t, 2)
    summary.update(
        {
            "model": model,
            "base_model": base_model or None,
            "benchmark": str(benchmark),
            "use_tactical_facts": bool(use_tactical_facts),
            "batch_size": eff_bs,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_clock_seconds": wall_s,
        }
    )

    if output_subdir.startswith("/"):
        base_out = Path(output_subdir)
        out_jsonl_path = base_out.with_suffix(".jsonl")
        out_summary_path = base_out.parent / f"{base_out.stem}_summary.json"
    else:
        out_jsonl_path = Path(f"{VOL_MOUNT}/results/{output_subdir}.jsonl")
        out_summary_path = Path(f"{VOL_MOUNT}/results/{output_subdir}_summary.json")

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl_path.open("w", encoding="utf-8") as fh:
        for item in outputs_jsonl:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    out_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    print(json.dumps(summary, indent=2))
    return str(out_summary_path)


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60,
)
def run_coach_benchmark_l4(
    model: str = FALLBACK_STUDENT_MODEL,
    benchmark: str = "data_generation/coaching_sample.jsonl",
    output_subdir: str = "coach_benchmark",
    limit: int = 0,
    use_tactical_facts: bool = True,
    batch_size: int = 16,
    max_rows: int = 0,
    base_model: str = "",
) -> str:
    """Generate batched coaching outputs on L4 (stable image) and score with V1/V2 + Anti-Slop."""
    return _exec_coach_benchmark(
        model=model,
        benchmark=benchmark,
        output_subdir=output_subdir,
        limit=limit,
        use_tactical_facts=use_tactical_facts,
        batch_size=batch_size,
        max_rows=max_rows,
        base_model=base_model,
    )


@app.function(
    image=modern_image,
    gpu="L4",
    volumes={VOL_MOUNT: volume},
    secrets=_SECRETS,
    timeout=60 * 60,
)
def run_coach_benchmark_modern_l4(
    model: str = DEFAULT_STUDENT_MODEL,
    benchmark: str = "data_generation/coaching_sample.jsonl",
    output_subdir: str = "coach_benchmark",
    limit: int = 0,
    use_tactical_facts: bool = True,
    batch_size: int = 16,
    max_rows: int = 0,
    base_model: str = "",
) -> str:
    """Generate batched coaching outputs on L4 (modern image) and score with V1/V2 + Anti-Slop."""
    return _exec_coach_benchmark(
        model=model,
        benchmark=benchmark,
        output_subdir=output_subdir,
        limit=limit,
        use_tactical_facts=use_tactical_facts,
        batch_size=batch_size,
        max_rows=max_rows,
        base_model=base_model,
    )


@app.local_entrypoint()
def main(
    task: str = "smoke-pipeline",
    model: str = "",
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
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    enable_thinking: bool = True,
    decoding_mode: str = "strict",
    openings: str = "",
    maia_elo: int = 0,
    opponent_depth: int = 0,
    benchmark: str = "data_generation/eval_benchmark.jsonl",
    positions: str = "",
    max_positions: int = 0,
    skip_probes: bool = False,
    quant: str = "q4_k_m",
    family: str = "gemma4",
    output_subdir: str = "",
    limit: int = 0,
    max_attempts: int = 3,
    local_dir: str = "",
    repo_id: str = "",
    repo_type: str = "model",
    path_in_repo: str = "",
    commit_message: str = "Upload ChessFM artifacts",
    dry_run: bool = False,
    workers: int = 24,
    opset: int = 17,
    gpu: str = "l4",
    lora_r: int = 16,
    lora_alpha: int = 0,
    all_linear: bool = True,
    full_ft: bool = False,
    optim: str = "",
    packing: bool = False,
    lr: float = 0.0,
    batch_size: int = 0,
    grad_accum: int = 0,
    num_generations: int = 0,
    reward_mode: str = "legacy",
    beta: float = -1.0,
    epsilon: float = -1.0,
    epsilon_high: float = -1.0,
    loss_type: str = "",
    scale_rewards: bool = True,
    filter_zero_variance: bool = False,
    puzzles: str = "",
    chessqa: str = "",
    english_forgetting: str = "",
    depth: int = 0,
    coaching_jsonl: str = "",
    output: str = "",
    stockfish_elo: int = 0,
    manifest_output: str = "",
    seed: int = -1,
    num_iterations: int = 0,
    checkpoint_every: int = 0,
    resume: bool = True,
    temperature: float = -1.0,
    avoid_repetition: bool = False,
    private: bool = True,
    raw_output: str = "",
    num_puzzles: int = 200,
    k: int = 8,
    top_p: float = 0.95,
    max_prompt_length: int = 0,
    max_completion_length: int = 0,
    use_tactical_facts: bool = True,
    max_rows: int = 0,
    base_model: str = "",
    coach_v2: bool = False,
    keep_f16: bool = False,
    out_dir: str = "",
    eval_workers: int = 32,
    skip_coach: bool = False,
    sft_resume: bool = False,
) -> None:
    gpu_tier = gpu.strip().lower()
    if gpu_tier not in {"l4", "a100", "h100"}:
        raise ValueError(f"Unknown --gpu {gpu!r}. Choose from: l4, a100, h100")

    resolved_model = resolve_track_model(
        model=model, track=track, use_official_gemma=use_official_gemma
    )
    use_modern = requires_modern_stack(resolved_model)
    if gpu_tier == "h100":
        if task != "sft":
            raise ValueError(f"--gpu h100 is only supported for --task sft (got --task {task!r}).")
        if use_modern:
            raise ValueError(
                f"--gpu h100 uses the legacy TRL image and does not support modern-stack "
                f"model {resolved_model!r}. Use --gpu l4 for modern tracks."
            )
    elif gpu_tier == "a100":
        a100_tasks = {"sft", "grpo", "smoke-pipeline"}
        if task not in a100_tasks:
            raise ValueError(
                f"--gpu a100 is only supported for tasks {sorted(a100_tasks)} "
                f"(got --task {task!r}). Use --gpu l4 for this task."
            )
        if use_modern:
            raise ValueError(
                f"--gpu a100 uses the legacy TRL image and does not support modern-stack "
                f"model {resolved_model!r}. Use --gpu l4 for modern tracks."
            )

    if gpu_tier == "h100":
        sft_fn = run_sft_h100
    elif gpu_tier == "a100":
        sft_fn = run_sft_a100
    else:
        sft_fn = run_sft_modern if use_modern else run_sft

    if gpu_tier == "a100":
        grpo_fn = run_grpo_a100
    else:
        grpo_fn = run_grpo_modern if use_modern else run_grpo

    eval_fn = run_eval_modern if use_modern else run_eval
    static_eval_fn = run_static_eval_modern if use_modern else run_static_eval

    if task == "sft":
        sft_kwargs: dict = {
            "model": resolved_model,
            "data": data,
            "output_subdir": output_subdir or "sft_baseline",
            "epochs": epochs,
            "include_ascii": include_ascii,
            "include_piece_list": include_piece_list,
            "include_legal_moves": include_legal_moves,
            "merge_and_save": not full_ft,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "all_linear": all_linear,
            "full_ft": full_ft,
            "packing": packing,
        }
        if lr > 0:
            sft_kwargs["lr"] = lr
        if batch_size > 0:
            sft_kwargs["batch_size"] = batch_size
        if grad_accum > 0:
            sft_kwargs["grad_accum"] = grad_accum
        if optim:
            sft_kwargs["optim"] = optim
        if sft_resume:
            sft_kwargs["resume"] = True
        ckpt = sft_fn.remote(**sft_kwargs)
        print(f"SFT checkpoint saved to Modal volume: {ckpt}")
    elif task == "sft-student":
        ckpt = run_sft_student_l4.remote(
            model=resolved_model,
            data=data
            if data != "data_generation/sft_train.jsonl"
            else f"{VOL_MOUNT}/data/distilled_coach.jsonl",
            output_subdir=output_subdir or "sft_student_coach",
            epochs=epochs,
            include_ascii=include_ascii,
            merge_and_save=True,
        )
        print(f"Student SFT merged checkpoint saved to Modal volume: {ckpt}")
    elif task == "grpo":
        grpo_kwargs: dict = {
            "model": resolved_model,
            "stage": stage,
            "steps": steps,
            "output_subdir": output_subdir or f"grpo_stage{stage}",
            "include_ascii": include_ascii,
            "include_piece_list": include_piece_list,
            "include_legal_moves": include_legal_moves,
            "positions": positions,
            "merge_and_save": not full_ft,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "all_linear": all_linear,
            "full_ft": full_ft,
            "reward_mode": reward_mode,
            "scale_rewards": scale_rewards,
            "filter_zero_variance": filter_zero_variance,
        }
        if max_positions > 0:
            grpo_kwargs["max_positions"] = max_positions
        if num_generations > 0:
            grpo_kwargs["num_generations"] = num_generations
            if batch_size <= 0:
                grpo_kwargs["per_device_batch_size"] = num_generations
        if batch_size > 0:
            grpo_kwargs["per_device_batch_size"] = batch_size
        if grad_accum > 0:
            grpo_kwargs["grad_accum"] = grad_accum
        if lr > 0:
            grpo_kwargs["lr"] = lr
        if beta >= 0:
            grpo_kwargs["beta"] = beta
        if epsilon >= 0:
            grpo_kwargs["epsilon"] = epsilon
        if epsilon_high >= 0:
            grpo_kwargs["epsilon_high"] = epsilon_high
        if loss_type:
            grpo_kwargs["loss_type"] = loss_type
        if num_iterations > 0:
            grpo_kwargs["num_iterations"] = num_iterations
        if checkpoint_every > 0:
            grpo_kwargs["checkpoint_every"] = checkpoint_every
        grpo_kwargs["resume"] = resume
        if temperature >= 0:
            grpo_kwargs["temperature"] = temperature
        if seed >= 0:
            grpo_kwargs["seed"] = seed
        if max_prompt_length > 0:
            grpo_kwargs["max_prompt_length"] = max_prompt_length
        if max_completion_length > 0:
            grpo_kwargs["max_completion_length"] = max_completion_length
        ckpt = grpo_fn.remote(**grpo_kwargs)
        print(f"GRPO Stage {stage} checkpoint saved to Modal volume: {ckpt}")
    elif task == "eval":
        eval_fn.remote(
            model=resolved_model,
            games=games,
            stockfish_level=stockfish_level,
            random_opponent=random_opponent,
            decoding_mode=decoding_mode,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=enable_thinking,
            openings=openings,
            maia_elo=maia_elo,
            opponent_depth=opponent_depth,
            batch_size=batch_size,
            stockfish_elo=stockfish_elo,
            manifest_output=manifest_output,
            seed=seed,
            avoid_repetition=avoid_repetition,
        )
        print(
            "Evaluation complete. Download reports with: modal volume get chessfm-artifacts results ./results"
        )
    elif task == "static-eval":
        out_path = static_eval_fn.remote(
            model=resolved_model,
            benchmark=benchmark,
            decoding_mode=decoding_mode,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=enable_thinking,
            max_positions=max_positions,
            skip_probes=skip_probes,
            output_name=output or output_subdir or "eval_manifest.json",
            batch_size=batch_size,
            puzzles=puzzles,
            chessqa=chessqa,
            english_forgetting=english_forgetting,
            depth=depth,
            coaching_jsonl=coaching_jsonl,
        )
        print(f"Static 4-tier eval manifest saved to Modal volume: {out_path}")
    elif task == "pass-at-k":
        out_path = run_pass_at_k.remote(
            model=resolved_model,
            puzzles=puzzles or f"{VOL_MOUNT}/data/phase2/test_puzzles_900.jsonl",
            output=output or output_subdir or "pass_at_k_summary.json",
            raw_output=raw_output,
            num_puzzles=num_puzzles,
            k=k,
            temperature=temperature if temperature > 0 else 0.7,
            top_p=top_p,
            batch_size=batch_size if batch_size > 0 else 16,
            seed=seed if seed >= 0 else 42,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=enable_thinking,
        )
        print(f"pass@k evaluation summary saved to Modal volume: {out_path}")
    elif task == "upload-hf":
        if not local_dir or not repo_id:
            raise ValueError("upload-hf requires --local-dir and --repo-id")
        uploaded = run_upload_hf.remote(
            local_dir=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            dry_run=dry_run,
            private=private,
        )
        print(f"Hugging Face Hub upload complete: {uploaded}")
    elif task == "distill-teacher":
        out_path = run_distill_teacher.remote(
            model=resolved_model,
            input_data=data
            if data != "data_generation/sft_train.jsonl"
            else "data_generation/coaching_sample.jsonl",
            output_name=output_subdir or "distilled_coach.jsonl",
            max_attempts=max_attempts,
            limit=limit,
            workers=workers,
        )
        print(f"Verified distillation dataset saved to Modal volume: {out_path}")
    elif task == "p4a-vllm-batch":
        teacher_model = model or DEFAULT_TEACHER_MODEL
        manifest_path = run_p4a_vllm_batch.remote(
            model=teacher_model,
            coach_test_path=benchmark
            if benchmark != "data_generation/eval_benchmark.jsonl"
            else f"{VOL_MOUNT}/data/phase2/test_coach_300.jsonl",
            fresh_seed_path=data
            if data != "data_generation/sft_train.jsonl"
            else f"{VOL_MOUNT}/data/phase2/coach_fresh_1k_seed.jsonl",
            puzzles_path=puzzles or f"{VOL_MOUNT}/data/phase2/test_puzzles_900.jsonl",
            chessqa_path=chessqa or f"{VOL_MOUNT}/data/phase2/test_chessqa_1000.jsonl",
            out_dir=out_dir or f"{VOL_MOUNT}/results/phase2/e5",
            workers=workers,
            eval_workers=eval_workers,
            skip_coach=skip_coach,
        )
        print(f"Phase 4A warm vLLM batch manifest saved to Modal volume: {manifest_path}")
    elif task == "coach-eval":
        run_coach_eval.remote()
        print("Coach evaluation suite complete.")
    elif task == "coach-benchmark":
        bench_fn = run_coach_benchmark_modern_l4 if use_modern else run_coach_benchmark_l4
        summary_path = bench_fn.remote(
            model=resolved_model,
            benchmark=benchmark
            if benchmark != "data_generation/eval_benchmark.jsonl"
            else "data_generation/coaching_sample.jsonl",
            output_subdir=output_subdir or "coach_benchmark",
            limit=limit,
            use_tactical_facts=use_tactical_facts,
            batch_size=batch_size if batch_size > 0 else 16,
            max_rows=max_rows,
            base_model=base_model,
        )
        print(f"Coach ACT-Eval benchmark summary saved to Modal volume: {summary_path}")
    elif task == "export-gguf":
        export_dir = run_export_gguf.remote(
            model=resolved_model,
            output_subdir=output_subdir or "gguf_export",
            quant=quant,
            family=family,
            coach_v2=coach_v2,
            keep_f16=keep_f16,
        )
        print(
            f"GGUF export artifacts saved to Modal volume: {export_dir}. "
            "Download with: modal volume get chessfm-artifacts exports ./exports"
        )
    elif task == "export-onnx":
        onnx_quant = "q4f16" if quant == "q4_k_m" else quant
        export_dir = run_export_onnx.remote(
            model=resolved_model,
            output_subdir=output_subdir or "onnx_webgpu_export",
            quant=onnx_quant,
            opset=opset,
            upload_hf=repo_id,
            dry_run=dry_run,
        )
        print(
            f"ONNX WebGPU export artifacts saved to Modal volume: {export_dir}. "
            "Download with: modal volume get chessfm-artifacts exports ./exports"
        )
    elif task == "smoke-pipeline":
        sft_subdir = output_subdir or "sft_smoke"
        print(f"=== Step 1/3: SFT (merge-and-save, model={resolved_model}) ===")
        sft_merged_ckpt = sft_fn.remote(
            model=resolved_model,
            data=data,
            output_subdir=sft_subdir,
            epochs=epochs,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            merge_and_save=True,
            full_ft=False,
        )
        print(f"SFT complete (merged): {sft_merged_ckpt}")

        print("=== Step 2/3: GRPO Stage 0 ===")
        grpo_ckpt = grpo_fn.remote(
            model=sft_merged_ckpt,
            stage=0,
            steps=steps,
            output_subdir=(output_subdir or "smoke") + "_grpo_stage0",
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
        )
        print(f"GRPO Stage 0 complete: {grpo_ckpt}")

        print("=== Step 3/3: Stockfish Eval ===")
        eval_fn.remote(
            model=grpo_ckpt,
            games=games,
            stockfish_level=stockfish_level,
            random_opponent=random_opponent,
            decoding_mode=decoding_mode,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=enable_thinking,
            openings=openings,
            maia_elo=maia_elo,
            opponent_depth=opponent_depth,
        )
        print(
            "Pipeline complete! Fetch reports: modal volume get chessfm-artifacts results ./results"
        )
    else:
        raise ValueError(f"Unknown task: {task}. Choose from: {', '.join(VALID_TASKS)}")
