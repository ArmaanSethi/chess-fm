#!/usr/bin/env python3
"""
ChessFM SFT Training Script.

Train the base model on chess reasoning data with TRL's SFTTrainer. Model
loading is handled by ``chessfm.load_model`` (unsloth-accelerated when the
``unsloth`` extra is installed, with a plain PEFT fallback otherwise).

USAGE:
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl
    python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import inspect
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trl import SFTConfig

from chessfm.config import (
    CHATML_ASSISTANT_OPEN,
    DEFAULT_BASE_MODEL,
    LORA_R,
    MAX_SEQ_LENGTH,
    build_chat_prompt,
    get_assistant_role,
)
from chessfm.fens import read_jsonl
from chessfm.training.manifest import write_run_manifest
from chessfm.training.split import fen_level_split

logger = logging.getLogger("chessfm.train_sft")

GEMMA_ASSISTANT_OPEN = "<|turn>model\n"
DEFAULT_LORA_LR = 2e-4
DEFAULT_FULL_FT_LR = 2e-5


def get_response_template(model_name: str = DEFAULT_BASE_MODEL) -> str:
    """Return the assistant turn-open marker for completion-only loss masking."""
    if get_assistant_role(model_name) == "model":
        return GEMMA_ASSISTANT_OPEN
    return CHATML_ASSISTANT_OPEN


def _format_messages_chatml(
    messages: list[dict[str, Any]], *, add_generation_prompt: bool = False
) -> str:
    """Fallback ChatML renderer when a tokenizer lacks a ``chat_template``."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    if add_generation_prompt:
        parts.append(CHATML_ASSISTANT_OPEN)
    return "".join(parts)


def _apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Render ``messages`` via ``tokenizer.apply_chat_template`` with graceful fallbacks."""
    if (
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
    return _format_messages_chatml(messages, add_generation_prompt=add_generation_prompt)


def _is_valid_sft_sample(sample: dict[str, Any]) -> bool:
    """Return True if ``sample`` has valid ``messages`` or ``prompt``+``completion``."""
    if not sample.get("is_reasonable", True):
        return False
    messages = sample.get("messages")
    if (
        isinstance(messages, list)
        and len(messages) >= 2
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages)
    ):
        return True
    return "prompt" in sample and "completion" in sample


def format_sft_example(
    sample: dict[str, Any],
    tokenizer: Any,
    *,
    include_ascii: bool = False,
    include_piece_list: bool = False,
    include_legal_moves: bool = False,
    use_prompt_completion: bool = False,
) -> dict[str, str]:
    """Format one SFT example for TRL 0.16 (``text``) or TRL >=1.18 (``prompt``/``completion``).

    Checks ``messages`` FIRST so coaching rows containing ``fen`` are never
    clobbered by P1 FEN prompts when board representation flags are enabled.
    """
    messages = sample.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        full_str = _apply_chat_template_safe(
            tokenizer,
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if use_prompt_completion:
            prompt_str = _apply_chat_template_safe(
                tokenizer,
                messages[:-1],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if full_str.startswith(prompt_str):
                completion_str = full_str[len(prompt_str) :]
            else:
                eos_tok = getattr(tokenizer, "eos_token", None) or "<|im_end|>"
                completion_str = f"{messages[-1]['content']}{eos_tok}"
            return {"prompt": prompt_str, "completion": completion_str}
        return {"text": full_str}

    if (include_ascii or include_piece_list or include_legal_moves) and "fen" in sample:
        prompt = build_chat_prompt(
            sample["fen"],
            tokenizer=tokenizer,
            include_ascii=include_ascii,
            include_piece_list=include_piece_list,
            include_legal_moves=include_legal_moves,
            enable_thinking=True,
        )
    else:
        prompt = sample["prompt"]
    completion = sample["completion"]
    if use_prompt_completion:
        return {"prompt": prompt, "completion": completion}
    return {"text": f"{prompt}{completion}"}


def _sft_config_supports_completion_only_loss() -> bool:
    """Return True if the installed TRL ``SFTConfig`` accepts ``completion_only_loss``."""
    try:
        from trl import SFTConfig
    except ImportError:
        return False
    return "completion_only_loss" in inspect.signature(SFTConfig.__init__).parameters


def build_completion_collator(tokenizer, model_name: str = DEFAULT_BASE_MODEL):
    """Build a TRL collator that masks prompt tokens to -100 during SFT.

    Returns ``None`` on TRL 1.x+ where ``DataCollatorForCompletionOnlyLM`` is
    superseded by trainer-level completion masking.
    """
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError:
        return None

    response_template = get_response_template(model_name)
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )


def resolve_sft_lr(lr: float | None, *, full_ft: bool = False) -> float:
    """Resolve SFT learning rate (defaults to 2e-5 for Full FT, 2e-4 for LoRA)."""
    if lr is not None:
        return float(lr)
    return DEFAULT_FULL_FT_LR if full_ft else DEFAULT_LORA_LR


def build_sft_config(
    args,
    *,
    completion_only_loss: bool = False,
    has_completion_collator: bool = False,
) -> SFTConfig:
    """Construct the TRL SFTConfig from parsed CLI args.

    Imported lazily so ``read_jsonl`` stays usable without the full torch/trl
    stack. Kept separate from ``main`` so the exact kwargs the training script
    passes are exercised by tests (guards against TRL API drift).
    """
    import os

    import torch
    from trl import SFTConfig

    use_gpu = torch.cuda.is_available()
    full_ft = bool(getattr(args, "full_ft", False))
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    grad_accum = getattr(args, "grad_accum", 2)
    lr = resolve_sft_lr(getattr(args, "lr", None), full_ft=full_ft)
    explicit_optim = getattr(args, "optim", None)
    if explicit_optim:
        optim = explicit_optim
    elif use_gpu:
        optim = "adamw_bnb_8bit" if full_ft else "adamw_8bit"
    else:
        optim = "adamw_torch"

    sft_params = inspect.signature(SFTConfig.__init__).parameters
    extra_kwargs: dict[str, Any] = {}
    if use_gpu:
        extra_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "warmup_ratio" in sft_params:
        extra_kwargs["warmup_ratio"] = 0.1
    if completion_only_loss and "completion_only_loss" in sft_params:
        extra_kwargs["completion_only_loss"] = True

    if getattr(args, "packing", False):
        if has_completion_collator:
            logger.warning(
                "packing=True is incompatible with DataCollatorForCompletionOnlyLM "
                "(packed sequences would leave subsequent prompts unmasked); "
                "falling back to packing=False with group_by_length=True."
            )
            if "packing" in sft_params:
                extra_kwargs["packing"] = False
            if "group_by_length" in sft_params:
                extra_kwargs["group_by_length"] = True
        elif "packing" in sft_params:
            extra_kwargs["packing"] = True

    return SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="epoch",
        report_to=report_to,
        # bf16 is preferred on modern GPUs (matches Qwen2.5/Qwen3 native dtype);
        # fp16 is only a fallback for pre-Ampere hardware.
        bf16=use_gpu and torch.cuda.is_bf16_supported(),
        fp16=use_gpu and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=use_gpu,
        optim=optim,
        seed=args.seed,
        max_length=MAX_SEQ_LENGTH,
        **extra_kwargs,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for SFT training."""
    parser = argparse.ArgumentParser(description="Train ChessFM with SFT")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.cwd() / "data_generation" / "sft_train.jsonl",
        help="Path to sft_train.jsonl",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL, help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults to 2e-4 for LoRA, 2e-5 for --full-ft)",
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
        help="Run full-parameter fine-tuning (bf16 + gradient checkpointing) instead of LoRA",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optimizer name (e.g. adamw_bnb_8bit, paged_adamw_8bit, adamw_8bit, adamw_torch)",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help=(
            "Enable sequence packing when compatible with completion-only masking "
            "(falls back to group_by_length=True with DataCollatorForCompletionOnlyLM)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Just validate data, don't train")
    parser.add_argument(
        "--merge-and-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge LoRA weights and save full checkpoint to <output>/final_merged",
    )
    parser.add_argument(
        "--include-ascii",
        action="store_true",
        help="Include an ASCII board rendering in prompts when FEN is available",
    )
    parser.add_argument(
        "--include-piece-list",
        action="store_true",
        help="Include a lossless piece-by-square listing in prompts when FEN is available",
    )
    parser.add_argument(
        "--include-legal-moves",
        action="store_true",
        help="Include sorted UCI legal moves in prompts (isolated comparison mode only)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from the latest checkpoint-<step> subdirectory inside --output if present",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)
    args.lr = resolve_sft_lr(args.lr, full_ft=args.full_ft)
    return args


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the highest-step ``checkpoint-<N>`` directory under ``output_dir``, if any."""
    if not output_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for entry in output_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            suffix = entry.name.removeprefix("checkpoint-")
            if suffix.isdigit():
                candidates.append((int(suffix), entry))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return str(candidates[-1][1])


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    effective_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
    mode_desc = (
        "Full FT"
        if args.full_ft
        else f"LoRA r={args.lora_r}, alpha={effective_alpha}, all_linear={args.all_linear}"
    )

    print("=" * 60)
    print("ChessFM SFT Training")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Mode:       {mode_desc}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum}")
    print(f"LR:         {args.lr}")
    print(f"Packing:    {args.packing}")
    print(f"Seed:       {args.seed}")
    print()

    print("Loading training data...")
    if not args.data.exists():
        print(f"Data file not found: {args.data}")
        print("Run: python -m chessfm.data_generation.convert_to_training first")
        return

    raw_data = read_jsonl(args.data)
    print(f"   Loaded {len(raw_data)} samples")

    # Filter to valid, reasonable samples (supports both messages and prompt/completion rows).
    valid_data = [s for s in raw_data if _is_valid_sft_sample(s)]
    print(f"   Valid & reasonable samples: {len(valid_data)}/{len(raw_data)}")

    if not valid_data:
        print("No valid samples after filtering. Check your data.")
        return

    if args.dry_run:
        bad_tags = 0
        for s in valid_data:
            comp = s.get("completion")
            if comp is None and isinstance(s.get("messages"), list) and s["messages"]:
                comp = str(s["messages"][-1].get("content", ""))
            comp_str = str(comp or "")
            if comp_str.count("<think>") != comp_str.count("</think>"):
                bad_tags += 1
        if bad_tags:
            print(f"   ERROR: {bad_tags} sample(s) with unbalanced <think> tags.")
            print("   Fix the data before training: re-run")
            print("   python -m chessfm.data_generation.convert_to_training")
            raise SystemExit(1)
        print("   All completions have balanced <think> tags.")
        print("\nDry run complete. Data looks good!")
        return

    import torch
    from datasets import Dataset
    from trl import SFTTrainer

    from chessfm.load_model import load_model

    print("\nLoading model...")
    model, tokenizer = load_model(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        all_linear=args.all_linear,
        full_ft=args.full_ft,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = build_completion_collator(tokenizer, model_name=args.model)
    use_prompt_completion = data_collator is None and _sft_config_supports_completion_only_loss()

    print("\nPreparing formatted dataset with completion-only loss masking...")
    formatted_rows = [
        format_sft_example(
            sample,
            tokenizer,
            include_ascii=args.include_ascii,
            include_piece_list=args.include_piece_list,
            include_legal_moves=args.include_legal_moves,
            use_prompt_completion=use_prompt_completion,
        )
        for sample in valid_data
    ]
    dataset = Dataset.from_list(formatted_rows)

    train_indices, val_indices, val_fens = fen_level_split(valid_data)
    train_ds = dataset.select(train_indices)
    eval_ds = dataset.select(val_indices)
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(eval_ds)} (unique FENs: {len(val_fens)})")
    print("   Split is by FEN: no position appears in both sets.")

    print("\nStarting training...")
    args.output.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "config"):
        model.config.use_cache = False

    training_args = build_sft_config(
        args,
        completion_only_loss=use_prompt_completion,
        has_completion_collator=data_collator is not None,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        args=training_args,
    )

    train_start = time.monotonic()
    resume_ckpt = find_latest_checkpoint(args.output) if args.resume else None
    if resume_ckpt is not None:
        print(f"Resuming SFT from latest checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    wall_clock_seconds = round(time.monotonic() - train_start, 2)

    print("\nSaving model...")
    final_dir = args.output / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    label = "Full-FT model" if args.full_ft else "Adapter"
    print(f"Training complete! {label} saved to: {final_dir}")

    if not args.full_ft and args.merge_and_save and hasattr(model, "merge_and_unload"):
        merged_dir = args.output / "final_merged"
        print(f"Merging LoRA adapter into base weights -> {merged_dir}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    manifest_path = write_run_manifest(
        args.output,
        model_id=args.model,
        stage="sft",
        seed=args.seed,
        hyperparameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "lora_r": None if args.full_ft else args.lora_r,
            "lora_alpha": None if args.full_ft else effective_alpha,
            "all_linear": None if args.full_ft else args.all_linear,
            "full_ft": args.full_ft,
            "optim": training_args.optim,
            "packing": bool(getattr(training_args, "packing", False)),
            "group_by_length": bool(getattr(training_args, "group_by_length", False)),
            "include_ascii": args.include_ascii,
            "include_piece_list": args.include_piece_list,
            "include_legal_moves": args.include_legal_moves,
            "merge_and_save": args.merge_and_save and not args.full_ft,
        },
        dataset_path=args.data,
        gpu_type=gpu_type,
        wall_clock_seconds=wall_clock_seconds,
    )
    print(f"Run manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
