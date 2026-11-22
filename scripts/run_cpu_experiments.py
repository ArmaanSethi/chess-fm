#!/usr/bin/env python3
"""Run reproducible CPU + Stockfish-16 micro-experiments for ChessFM.

Executes three fast, deterministic engineering experiments without a GPU:

1. ``exp_constrained_vs_strict``:
   Compares unconstrained Strict Pass@1 decoding against
   :class:`chessfm.rl.constrained_decoding.LegalMoveLogitsProcessor` using a
   tiny causal LM (``sshleifer/tiny-gpt2``, with an offline mock fallback)
   across positions from ``data_generation/eval_benchmark.jsonl``.
2. ``exp_grpo_reward_variance``:
   Builds an 8-candidate GRPO group per position (top-3 engine moves, 3 random
   legal moves, 2 illegal/malformed outputs) and evaluates Stage 0, Stage 1,
   and Stage 2 rewards with :func:`chessfm.rl.rewards.combined_reward` and
   live Stockfish to quantify group variance (GRPO gradient signal) and
   best-vs-blunder reward separation.
3. ``exp_sft_completion_masking``:
   Runs :func:`chessfm.training.train_sft.build_completion_collator`
   (``DataCollatorForCompletionOnlyLM``) on ``data_generation/sft_train.jsonl``
   and measures exact prompt tokens masked to ``-100`` vs. active completion
   tokens trained (both standard FEN prompts and ``--include-ascii`` prompts).

Usage:
    venv/bin/python scripts/run_cpu_experiments.py --positions 20 --depth 6
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

from chessfm.config import DATA_DIR, build_chat_prompt
from chessfm.fens import read_jsonl
from chessfm.rl.constrained_decoding import (
    DualModeGenerator,
    LegalMovePrefixTrie,
    legal_uci_moves,
)
from chessfm.rl.engine import close_engine, open_engine
from chessfm.rl.rewards import combined_reward
from chessfm.training.train_sft import build_completion_collator, get_response_template

DEFAULT_TINY_MODEL = "sshleifer/tiny-gpt2"


# ---------------------------------------------------------------------------
# Offline / unit-test deterministic mock LM & tokenizer
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal byte-level tokenizer compatible with LegalMoveLogitsProcessor and TRL."""

    def __init__(self) -> None:
        # Reserve ASCII printable + newline + special tokens.
        self._chars = [chr(i) for i in range(32, 127)] + ["\n"]
        self._stoi: dict[str, int] = {ch: idx for idx, ch in enumerate(self._chars)}
        self._itos: dict[int, str] = {idx: ch for ch, idx in self._stoi.items()}
        self.eos_token_id: int = len(self._chars)
        self.pad_token_id: int = self.eos_token_id + 1
        self.eos_token: str = "<|endoftext|>"
        self.pad_token: str = "<|pad|>"
        self.padding_side: str = "right"
        self.vocab_size: int = self.pad_token_id + 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._stoi.get(ch, 0) for ch in text]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out: list[str] = []
        for tid in token_ids:
            tid_int = int(tid)
            if skip_special_tokens and tid_int in (self.eos_token_id, self.pad_token_id):
                continue
            out.append(self._itos.get(tid_int, ""))
        return "".join(out)

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(
        self,
        encoded_inputs: list[dict[str, Any]],
        padding: bool = True,
        return_tensors: str = "pt",
        **_: Any,
    ) -> dict[str, Any]:
        del padding, return_tensors
        import torch

        max_len = max(len(row["input_ids"]) for row in encoded_inputs)
        input_ids = []
        attention_mask = []
        for row in encoded_inputs:
            ids = list(row["input_ids"])
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_n)
            attention_mask.append([1] * len(ids) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _MockCausalLM:
    """Deterministic greedy decoder that emits an illegal filler token unless constrained."""

    def __init__(self, tokenizer: _MockTokenizer) -> None:
        self.tokenizer = tokenizer
        self.device = None
        self._filler_id = tokenizer.encode("z")[0]

    def generate(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        max_new_tokens: int = 12,
        logits_processor: Sequence[Any] | None = None,
        **_: Any,
    ) -> Any:
        del attention_mask
        import torch

        seq = input_ids.clone()
        batch_size = seq.shape[0]
        vocab_size = self.tokenizer.vocab_size

        for _ in range(max_new_tokens):
            scores = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
            # Strongly prefer the illegal filler token 'z' so strict mode fails
            # while constrained decoding must steer into the legal-move trie.
            scores[:, self._filler_id] = 5.0
            if logits_processor:
                for proc in logits_processor:
                    scores = proc(seq, scores)
            next_tok = torch.argmax(scores, dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=-1)
            if int(next_tok[0].item()) == self.tokenizer.eos_token_id:
                break
        return seq


def _load_tiny_lm(model_id: str, use_mock: bool = False) -> tuple[Any, Any, str, bool]:
    """Load ``model_id`` on CPU, falling back to the deterministic mock if offline."""
    if use_mock:
        tok = _MockTokenizer()
        return _MockCausalLM(tok), tok, "mock_causal_lm", True

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer, model_id, False
    except Exception:
        tok = _MockTokenizer()
        return _MockCausalLM(tok), tok, "mock_causal_lm", True


def _load_tokenizer(model_id: str, use_mock: bool = False) -> tuple[Any, str, bool]:
    """Load tokenizer for ``model_id``, falling back to ``_MockTokenizer`` if offline."""
    if use_mock:
        return _MockTokenizer(), "mock_tokenizer", True
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "model_max_length", 1024) < 4096:
            tokenizer.model_max_length = 4096
        return tokenizer, model_id, False
    except Exception:
        return _MockTokenizer(), "mock_tokenizer", True


# ---------------------------------------------------------------------------
# Experiment 1: Constrained Decoding vs. Strict Pass@1
# ---------------------------------------------------------------------------


def exp_constrained_vs_strict(
    benchmark_rows: Sequence[dict[str, Any]],
    model_id: str = DEFAULT_TINY_MODEL,
    max_positions: int = 20,
    max_new_tokens: int = 12,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Compare unconstrained Strict Pass@1 vs. LegalMoveLogitsProcessor decoding.

    Prompts end immediately after ``</think>\\n`` and ``require_think_close=False``
    so both decoding modes are compared directly on the post-reasoning move
    emission boundary.
    """
    rows = list(benchmark_rows[:max_positions])
    model, tokenizer, resolved_model_id, used_mock = _load_tiny_lm(model_id, use_mock=use_mock)

    def _post_think_prompt(fen: str) -> str:
        return f"{build_chat_prompt(fen)}<think>Evaluating candidate moves.</think>\n"

    generator = DualModeGenerator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        prompt_builder=_post_think_prompt,
        require_think_close=False,
    )

    strict_legal = 0
    constrained_legal = 0
    strict_latencies_ms: list[float] = []
    constrained_latencies_ms: list[float] = []
    trie_build_ms: list[float] = []
    sample_outputs: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        fen = row["fen"]

        t0 = time.perf_counter()
        strict_res = generator.generate_move(fen, mode="strict")
        strict_ms = (time.perf_counter() - t0) * 1000.0
        strict_latencies_ms.append(strict_ms)
        if strict_res["is_legal"]:
            strict_legal += 1

        t_trie = time.perf_counter()
        _ = LegalMovePrefixTrie.from_fen(fen, tokenizer)
        trie_build_ms.append((time.perf_counter() - t_trie) * 1000.0)

        t1 = time.perf_counter()
        constrained_res = generator.generate_move(fen, mode="constrained")
        constrained_ms = (time.perf_counter() - t1) * 1000.0
        constrained_latencies_ms.append(constrained_ms)
        if constrained_res["is_legal"]:
            constrained_legal += 1

        if idx < 3:
            sample_outputs.append(
                {
                    "fen": fen,
                    "strict_move": strict_res["move"],
                    "strict_legal": bool(strict_res["is_legal"]),
                    "constrained_move": constrained_res["move"],
                    "constrained_legal": bool(constrained_res["is_legal"]),
                }
            )

    n = len(rows)
    strict_mean_ms = round(statistics.mean(strict_latencies_ms), 2) if n else 0.0
    constrained_mean_ms = round(statistics.mean(constrained_latencies_ms), 2) if n else 0.0
    trie_mean_ms = round(statistics.mean(trie_build_ms), 2) if n else 0.0

    return {
        "model_id": resolved_model_id,
        "used_mock_fallback": used_mock,
        "positions_evaluated": n,
        "max_new_tokens": max_new_tokens,
        "strict_legal_count": strict_legal,
        "strict_legal_rate": round(strict_legal / n, 4) if n else 0.0,
        "constrained_legal_count": constrained_legal,
        "constrained_legal_rate": round(constrained_legal / n, 4) if n else 0.0,
        "strict_mean_latency_ms": strict_mean_ms,
        "constrained_mean_latency_ms": constrained_mean_ms,
        "latency_overhead_ms": round(constrained_mean_ms - strict_mean_ms, 2),
        "trie_build_mean_ms": trie_mean_ms,
        "sample_outputs": sample_outputs,
    }


# ---------------------------------------------------------------------------
# Experiment 2: GRPO Staged Reward Variance & Separation
# ---------------------------------------------------------------------------


def _build_grpo_candidate_group(
    row: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Build an 8-candidate GRPO group for a single benchmark position.

    Composition:
    - 3 top engine moves (best, 2nd best, 3rd best)
    - 3 random legal moves drawn from remaining legal moves
    - 2 illegal / malformed outputs (1 illegal UCI move, 1 malformed string)
    """
    fen = row["fen"]
    all_legal = legal_uci_moves(fen)
    legal_set = set(all_legal)

    top_moves: list[str] = []
    for entry in row.get("top_moves", []):
        mv = entry.get("move") if isinstance(entry, dict) else None
        if mv and mv in legal_set and mv not in top_moves:
            top_moves.append(mv)
    best_mv = row.get("best_move")
    if best_mv and best_mv in legal_set and best_mv not in top_moves:
        top_moves.insert(0, best_mv)

    # Ensure exactly 3 top-move slots even in forced/few-move endgames.
    for mv in all_legal:
        if len(top_moves) >= 3:
            break
        if mv not in top_moves:
            top_moves.append(mv)
    while len(top_moves) < 3 and all_legal:
        top_moves.append(all_legal[0])
    top_moves = top_moves[:3]

    remaining_legal = [mv for mv in all_legal if mv not in top_moves]
    if len(remaining_legal) >= 3:
        random_moves = rng.sample(remaining_legal, 3)
    elif remaining_legal:
        random_moves = [rng.choice(remaining_legal) for _ in range(3)]
    else:
        random_moves = [rng.choice(all_legal) for _ in range(3)]

    def _fmt_legal(mv: str) -> str:
        return (
            f"<think>Evaluating candidate move {mv} for piece activity, "
            f"tactical safety, and pawn structure in this position.</think>\n{mv}"
        )

    # Pick a syntactically valid UCI string guaranteed illegal in this position.
    illegal_uci = "a1a1"
    for cand in ("a1h8", "h1a8", "e1e8", "a1a1"):
        try:
            if chess.Move.from_uci(cand) not in chess.Board(fen).legal_moves:
                illegal_uci = cand
                break
        except ValueError:
            continue

    candidates: list[dict[str, str]] = [
        {"role": "best", "move": top_moves[0], "output": _fmt_legal(top_moves[0])},
        {"role": "top_2", "move": top_moves[1], "output": _fmt_legal(top_moves[1])},
        {"role": "top_3", "move": top_moves[2], "output": _fmt_legal(top_moves[2])},
    ]
    for idx, mv in enumerate(random_moves, start=1):
        candidates.append({"role": f"random_{idx}", "move": mv, "output": _fmt_legal(mv)})

    candidates.append(
        {
            "role": "illegal_uci",
            "move": illegal_uci,
            "output": (
                f"<think>Considering {illegal_uci} without verifying board geometry "
                f"or legal piece trajectory.</think>\n{illegal_uci}"
            ),
        }
    )
    candidates.append(
        {
            "role": "malformed",
            "move": "",
            "output": "I think White should attack the king soon.",
        }
    )
    return candidates


def exp_grpo_reward_variance(
    benchmark_rows: Sequence[dict[str, Any]],
    engine: Any = None,
    max_positions: int = 20,
    depth: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure GRPO group reward variance and best-vs-blunder separation across Stages 0-2."""
    rows = list(benchmark_rows[:max_positions])
    owns_engine = engine is None
    if owns_engine:
        engine = open_engine()
        if engine is None:
            raise RuntimeError("Stockfish binary is required for exp_grpo_reward_variance.")

    t0 = time.perf_counter()
    rng = random.Random(seed)
    groups = [(row["fen"], _build_grpo_candidate_group(row, rng)) for row in rows]

    stage_results: dict[str, dict[str, Any]] = {}
    try:
        for stage in (0, 1, 2):
            all_rewards: list[float] = []
            group_stds: list[float] = []
            legal_group_stds: list[float] = []
            nonzero_groups = 0
            legal_nonzero_groups = 0
            best_rewards: list[float] = []
            random_legal_rewards: list[float] = []
            worst_legal_rewards: list[float] = []
            illegal_rewards: list[float] = []

            for fen, candidates in groups:
                group_scores: list[float] = []
                legal_scores: list[float] = []

                for cand in candidates:
                    res = combined_reward(
                        fen=fen,
                        move_uci=cand["move"],
                        output=cand["output"],
                        stage=stage,
                        engine=engine if stage >= 1 else None,
                        move_quality_depth=depth,
                    )
                    score = float(res["total"])
                    group_scores.append(score)
                    all_rewards.append(score)
                    if res["legal"]:
                        legal_scores.append(score)
                    if cand["role"] == "best":
                        best_rewards.append(score)
                    elif cand["role"].startswith("random_"):
                        random_legal_rewards.append(score)
                    elif cand["role"] in ("illegal_uci", "malformed"):
                        illegal_rewards.append(score)

                g_std = statistics.pstdev(group_scores) if len(group_scores) > 1 else 0.0
                l_std = statistics.pstdev(legal_scores) if len(legal_scores) > 1 else 0.0
                group_stds.append(g_std)
                legal_group_stds.append(l_std)
                if g_std > 1e-9:
                    nonzero_groups += 1
                if l_std > 1e-9:
                    legal_nonzero_groups += 1
                if legal_scores:
                    worst_legal_rewards.append(min(legal_scores))

            n = len(groups)
            best_mean = statistics.mean(best_rewards) if best_rewards else 0.0
            random_mean = statistics.mean(random_legal_rewards) if random_legal_rewards else 0.0
            worst_legal_mean = statistics.mean(worst_legal_rewards) if worst_legal_rewards else 0.0
            illegal_mean = statistics.mean(illegal_rewards) if illegal_rewards else 0.0

            stage_results[f"stage_{stage}"] = {
                "mean_reward": round(statistics.mean(all_rewards), 4) if all_rewards else 0.0,
                "mean_group_std": round(statistics.mean(group_stds), 4) if group_stds else 0.0,
                "legal_only_mean_group_std": (
                    round(statistics.mean(legal_group_stds), 4) if legal_group_stds else 0.0
                ),
                "nonzero_variance_group_rate": round(nonzero_groups / n, 4) if n else 0.0,
                "legal_nonzero_variance_group_rate": (
                    round(legal_nonzero_groups / n, 4) if n else 0.0
                ),
                "best_move_mean_reward": round(best_mean, 4),
                "random_legal_mean_reward": round(random_mean, 4),
                "worst_legal_mean_reward": round(worst_legal_mean, 4),
                "illegal_mean_reward": round(illegal_mean, 4),
                "best_vs_worst_legal_separation": round(best_mean - worst_legal_mean, 4),
                "best_vs_random_legal_separation": round(best_mean - random_mean, 4),
            }
    finally:
        if owns_engine:
            close_engine(engine)

    return {
        "positions_evaluated": len(groups),
        "group_size": 8,
        "stockfish_depth": depth,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "stages": stage_results,
    }


# ---------------------------------------------------------------------------
# Experiment 3: SFT Completion-Only Loss Masking (BUG-01 Analysis)
# ---------------------------------------------------------------------------


def _measure_masking_for_prompts(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    collator: Any,
    include_ascii: bool = False,
) -> dict[str, Any]:
    total_counts: list[int] = []
    masked_counts: list[int] = []
    active_counts: list[int] = []
    masked_fractions: list[float] = []
    zero_active_rows = 0

    for row in rows:
        if include_ascii and "fen" in row:
            prompt = build_chat_prompt(row["fen"], include_ascii=True)
        else:
            prompt = row["prompt"]
        text = f"{prompt}{row['completion']}"
        encoded = tokenizer(text, add_special_tokens=False)
        batch = collator([encoded])
        labels = batch["labels"][0].tolist()
        total = len(labels)
        masked = sum(1 for tok_id in labels if tok_id == -100)
        active = total - masked
        if active == 0:
            zero_active_rows += 1
        total_counts.append(total)
        masked_counts.append(masked)
        active_counts.append(active)
        masked_fractions.append(masked / total if total else 0.0)

    n = len(rows)
    total_sum = sum(total_counts)
    masked_sum = sum(masked_counts)
    return {
        "rows_evaluated": n,
        "mean_total_tokens": round(statistics.mean(total_counts), 2) if n else 0.0,
        "mean_masked_prompt_tokens": round(statistics.mean(masked_counts), 2) if n else 0.0,
        "mean_active_completion_tokens": round(statistics.mean(active_counts), 2) if n else 0.0,
        "mean_masked_fraction": round(statistics.mean(masked_fractions), 4) if n else 0.0,
        "corpus_masked_fraction": round(masked_sum / total_sum, 4) if total_sum else 0.0,
        "min_masked_fraction": round(min(masked_fractions), 4) if n else 0.0,
        "max_masked_fraction": round(max(masked_fractions), 4) if n else 0.0,
        "zero_active_rows": zero_active_rows,
    }


def exp_sft_completion_masking(
    sft_rows: Sequence[dict[str, Any]],
    model_id: str = DEFAULT_TINY_MODEL,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Measure prompt tokens masked to -100 vs. active completion tokens via TRL collator."""
    valid_rows = [
        r for r in sft_rows if "prompt" in r and "completion" in r and r.get("is_reasonable", True)
    ]
    tokenizer, resolved_tok_id, used_mock = _load_tokenizer(model_id, use_mock=use_mock)
    collator = build_completion_collator(tokenizer)

    standard_stats = _measure_masking_for_prompts(
        valid_rows, tokenizer, collator, include_ascii=False
    )
    ascii_rows = [r for r in valid_rows if "fen" in r]
    ascii_stats = (
        _measure_masking_for_prompts(ascii_rows, tokenizer, collator, include_ascii=True)
        if ascii_rows
        else None
    )

    return {
        "tokenizer_id": resolved_tok_id,
        "used_mock_fallback": used_mock,
        "response_template": get_response_template(),
        "rows_evaluated": len(valid_rows),
        "standard_prompt": standard_stats,
        "ascii_board_prompt": ascii_stats,
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_all_experiments(
    benchmark_path: Path = DATA_DIR / "eval_benchmark.jsonl",
    sft_path: Path = DATA_DIR / "sft_train.jsonl",
    positions: int = 20,
    depth: int = 6,
    model_id: str = DEFAULT_TINY_MODEL,
    seed: int = 42,
    use_mock: bool = False,
) -> dict[str, Any]:
    """Run all three CPU micro-experiments and return a structured JSON-serializable report."""
    t0 = time.perf_counter()
    benchmark_rows = read_jsonl(benchmark_path)
    sft_rows = read_jsonl(sft_path)

    exp1 = exp_constrained_vs_strict(
        benchmark_rows,
        model_id=model_id,
        max_positions=positions,
        use_mock=use_mock,
    )
    exp2 = exp_grpo_reward_variance(
        benchmark_rows,
        max_positions=positions,
        depth=depth,
        seed=seed,
    )
    exp3 = exp_sft_completion_masking(
        sft_rows,
        model_id=model_id,
        use_mock=use_mock,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(benchmark_path),
        "sft_file": str(sft_path),
        "positions": positions,
        "stockfish_depth": depth,
        "seed": seed,
        "total_elapsed_seconds": round(time.perf_counter() - t0, 2),
        "exp1_constrained_vs_strict": exp1,
        "exp2_grpo_reward_variance": exp2,
        "exp3_sft_completion_masking": exp3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible CPU + Stockfish-16 micro-experiments for ChessFM."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DATA_DIR / "eval_benchmark.jsonl",
        help="Path to Stockfish-verified eval benchmark JSONL.",
    )
    parser.add_argument(
        "--sft-data",
        type=Path,
        default=DATA_DIR / "sft_train.jsonl",
        help="Path to cleaned SFT training JSONL.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=20,
        help="Number of benchmark positions for Experiments 1 and 2.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Stockfish search depth for Experiment 2.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TINY_MODEL,
        help="Tiny causal LM / tokenizer ID for Experiments 1 and 3.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Experiment 2.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON experiment report.",
    )
    args = parser.parse_args()

    report = run_all_experiments(
        benchmark_path=args.benchmark,
        sft_path=args.sft_data,
        positions=args.positions,
        depth=args.depth,
        model_id=args.model,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
