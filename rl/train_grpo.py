#!/usr/bin/env python3
"""
GRPO Training Script for ChessFM

Train a language model to play chess using Group Relative Policy Optimization (TRL GRPOTrainer).
Supports staged training:

    Stage 0: Legality only      (+1 legal, -1 illegal) + format/budget bonuses
    Stage 1: + Stockfish move quality (how close to Stockfish's best move)
    Stage 2: + Stockfish eval delta (dense per-move signal)

USAGE:
    python rl/train_grpo.py --stage 0 --steps 500
    python rl/train_grpo.py --stage 1 --steps 2000 --checkpoint-every 500
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path so `config`, `rl.*` are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_BASE_MODEL, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, make_prompt
from rl.move_utils import parse_move_from_output, extract_fen_from_prompt
from rl.rewards import combined_reward

# torch/datasets are imported lazily so `--help` works without a GPU stack.
_torch = None
_dataset = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_datasets():
    global _dataset
    if _dataset is None:
        from datasets import Dataset
        _dataset = Dataset
    return _dataset


# ---------------------------------------------------------------------------
# Reward function for TRL GRPOTrainer
# ---------------------------------------------------------------------------

def make_reward_func(stage: int, engine=None):
    """
    Build a TRL-compatible reward function.

    TRL calls reward funcs as ``reward_func(prompts=..., completions=..., **kwargs)``
    and expects a list of floats back, one per completion.
    """
    def chess_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                fen = extract_fen_from_prompt(prompt)
                move = parse_move_from_output(completion)
                if move is None:
                    move = "xxxx"  # Guaranteed illegal
                r = combined_reward(fen, move, completion, stage=stage, engine=engine)
                rewards.append(r["total"])
            except Exception as e:
                # Never let a single bad reward crash a training run.
                rewards.append(-1.0)
        return rewards
    return chess_reward_fn


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_positions(filepath: Path, max_positions: int = 1000) -> List[dict]:
    """Load positions into a ChatML-formatted prompt dataset.

    Raises:
        SystemExit: if the positions file is missing or contains no positions.
    """
    if not filepath.exists():
        raise SystemExit(
            f"❌ Positions file not found: {filepath}\n"
            "   Generate it first: python data_generation/fetch_elite_data.py "
            "(or pass --positions)."
        )
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            fen = line.strip()
            if fen:
                prompt = (
                    f"<|im_start|>user\n{make_prompt(fen)}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                positions.append({"prompt": prompt})
            if len(positions) >= max_positions:
                break
    if not positions:
        raise SystemExit(
            f"❌ Positions file contains no positions: {filepath}"
        )
    return positions


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """Load the base model with unsloth (4-bit + LoRA) or a PEFT fallback."""
    torch = _get_torch()
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    except ImportError:
        print("⚠️ unsloth not available, falling back to standard transformers/peft")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer


def get_stockfish_engine():
    """
    Return a Stockfish engine instance.

    Stages 1 and 2 depend on Stockfish for their reward signal, so a missing
    engine is a hard error rather than a silent downgrade to Stage 0.
    """
    import chess.engine
    from config import STOCKFISH_PATHS
    for path in STOCKFISH_PATHS:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Threads": 1, "Hash": 16})
            return engine
        except Exception:
            continue
    raise SystemExit(
        "❌ Stockfish engine not found (checked: {}).\n"
        "   Stages 1 and 2 require Stockfish for the move-quality reward. "
        "Install it (e.g. `brew install stockfish` / `apt install stockfish`) "
        "or run with --stage 0.".format(", ".join(STOCKFISH_PATHS))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ChessFM with GRPO")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Training stage (0=legality, 1=+move quality, 2=+eval delta)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of max training steps")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Base model to train")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--per-device-batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--positions", type=Path,
                        default=Path(__file__).parent.parent / "data_generation" / "positions.txt",
                        help="Path to positions file")
    parser.add_argument("--max-positions", type=int, default=1000,
                        help="Maximum number of positions to sample")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessFM RL Training (GRPO via TRL)")
    print("=" * 60)
    print(f"Stage:            {args.stage}")
    print(f"Model:            {args.model}")
    print(f"Max Steps:        {args.steps}")
    print(f"Generations/grp:  {args.batch_size}")
    print(f"Checkpoint every: {args.checkpoint_every}")
    print()

    torch = _get_torch()
    torch.manual_seed(args.seed)

    # Load Dataset
    Dataset = _get_datasets()
    raw_positions = load_positions(args.positions, args.max_positions)
    dataset = Dataset.from_list(raw_positions)
    print(f"📂 Loaded {len(dataset)} prompts/positions for training")

    # Load Model
    model, tokenizer = load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Stockfish engine only needed for stages 1/2
    engine = None
    if args.stage >= 1:
        engine = get_stockfish_engine()

    # Configure GRPO Trainer
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("❌ 'trl' package is required for GRPO. Install via `pip install 'trl>=0.12'`.")
        return

    training_args = GRPOConfig(
        output_dir=str(args.output),
        learning_rate=args.lr,
        max_steps=args.steps,
        num_generations=args.batch_size,
        per_device_train_batch_size=args.per_device_batch_size,
        max_prompt_length=256,
        max_completion_length=256,
        logging_steps=10,
        save_steps=args.checkpoint_every,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    reward_fn = make_reward_func(args.stage, engine=engine)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n🚀 Starting GRPO Training Loop...")
    trainer.train()

    final_path = args.output / f"stage{args.stage}_final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n✅ Training complete! Model saved to {final_path}")

    if engine is not None:
        engine.quit()


if __name__ == "__main__":
    main()
