# ChessFM — Reproducible CPU & Stockfish-16 Micro-Experiments

This document records verified engineering micro-experiments executed directly on CPU + Stockfish 16 without GPU hardware or external API keys. Each experiment isolates a core training or inference design choice in ChessFM and is reproducible from a single CLI command.

## Reproduction

```bash
venv/bin/python scripts/run_cpu_experiments.py --positions 20 --depth 6
```

- **Wall-clock time (all 3 experiments):** `12.16 s`
- **Engine:** Stockfish 16 (`depth=6` for Experiment 2)
- **Causal LM / Tokenizer:** `sshleifer/tiny-gpt2` (CPU, `used_mock_fallback=false`)
- **Datasets:** `data_generation/eval_benchmark.jsonl` (first 20 positions), `data_generation/sft_train.jsonl` (all 40 cleaned rows)
- **Seed:** `42`

---

## Experiment 1 — Constrained Decoding vs. Strict Pass@1

**Hypothesis:** Unconstrained autoregressive decoding (`Strict Pass@1`) from an unaligned or under-trained LM frequently emits illegal or unparseable UCI moves, whereas attaching `LegalMoveLogitsProcessor` (prefix-trie token masking after `</think>`) guarantees 100% move legality with minimal latency overhead.

**Setup:** Both `strict` and `constrained` modes are invoked via `DualModeGenerator` on the exact same post-`</think>` prompt boundary (`max_new_tokens=12`, greedy decoding, `require_think_close=False`) across 20 positions from `data_generation/eval_benchmark.jsonl` using `sshleifer/tiny-gpt2`.

| Decoding Mode | Legal Moves | Legal Rate ↑ | Mean Latency (ms/pos) ↓ | Mean Trie Build (ms/pos) |
|---|---:|---:|---:|---:|
| `strict` (unconstrained Pass@1) | 0 / 20 | 0.0% | 34.79 | — |
| `constrained` (`LegalMoveLogitsProcessor`) | 20 / 20 | **100.0%** | **26.25** (`-8.54 ms`) | 7.44 |

**Engineering Takeaways:**
1. **Zero illegal moves:** `LegalMoveLogitsProcessor` raises move legality from `0.0%` to `100.0%` (`20 / 20`) even on a tiny untrained causal LM.
2. **Net negative latency overhead on short move targets:** Constrained decoding was `8.54 ms/position` *faster* than strict decoding (`26.25 ms` vs. `34.79 ms`). Because the trie permits EOS immediately after a complete legal UCI move (4–5 tokens), generation terminates early instead of running to `max_new_tokens=12`, more than offsetting the `7.44 ms` per-position prefix-trie build cost.
3. **Evaluation discipline:** Because constrained decoding masks illegal moves at inference time, ChessFM reports both `strict` Pass@1 (measuring true model legality) and `constrained` decoding (production play) side by side.

---

## Experiment 2 — GRPO Staged Reward Variance & Blunder Separation

**Hypothesis:** GRPO normalizes rewards within each candidate group (`G = 8`); if a reward stage assigns identical scores to all legal moves, policy gradients vanish once SFT achieves high legality. Adding Stockfish move quality (Stage 1) and opponent-reply-aware delta (Stage 2) should create strong within-group variance and clear separation between best moves and blunders.

**Setup:** For each of 20 benchmark positions, an 8-candidate GRPO group is constructed (`top-1`, `top-2`, `top-3` engine moves + 3 deterministic random legal moves + 1 illegal UCI move + 1 malformed output) and scored with `combined_reward` across Stages 0, 1, and 2 (`Stockfish 16, depth=6`, `2.01 s` total).

| GRPO Stage | Mean Reward | Full-Group Std (8 cands) | Legal-Only Std (6 cands) ↑ | Non-Zero Legal Variance Groups ↑ | Best Move Mean | Random Legal Mean | Worst Legal Mean | Best vs. Worst Legal Gap ↑ | Best vs. Random Legal Gap ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stage 0** (Legality + Format) | 0.5750 | 0.9093 | **0.0000** | 0.0% (0/20) | 1.1000 | 1.1000 | 1.1000 | **0.0000** | **0.0000** |
| **Stage 1** (+ Move Quality) | 1.0201 | 1.1930 | **0.2626** | 100.0% (20/20) | 1.9260 | 1.5120 | 1.3203 | **+0.6058** | **+0.4140** |
| **Stage 2** (+ Reply Delta) | 0.7526 | 1.1751 | **0.6376** | 100.0% (20/20) | 1.9682 | 0.8992 | 0.4826 | **+1.4856** | **+1.0690** |

*Illegal/malformed candidates score `-1.0000` across all three stages (`100.0%` full-group non-zero variance across all stages).*

**Engineering Takeaways:**
1. **Why Stage 0 alone plateaus:** Stage 0 yields `0.0000` std and `0.0000` reward separation among legal moves (`0/20` groups have non-zero legal variance). It is strictly a fast, Stockfish-free bootstrapping stage for legality and `<think>...</think>` formatting.
2. **Stage 1 unlocks legal-move gradients:** Adding Stockfish centipawn-loss quality gives `100.0%` (`20/20`) of groups non-zero legal-move variance (`std = 0.2626`) and a `+0.6058` reward gap between best and worst legal moves.
3. **Stage 2 amplifies blunder discrimination by >2.4×:** Incorporating Stockfish's best counterplay reply increases legal-only group std to `0.6376` (`2.43×` Stage 1) and widens the best-vs-worst legal separation to `+1.4856` (`2.45×` Stage 1) and best-vs-random separation to `+1.0690` (`2.58×` Stage 1), concentrating GRPO advantage magnitude on punishing tactical blunders.

---

## Experiment 3 — SFT Completion-Only Loss Masking (`BUG-01` Verification)

**Hypothesis:** Standard causal LM SFT computes cross-entropy loss over the entire `prompt + completion` sequence, wasting gradient updates memorizing static ChatML headers, FEN strings, and ASCII board grids. Using TRL's `DataCollatorForCompletionOnlyLM` (`build_completion_collator`, response template `<|im_start|>assistant\n`) masks all prompt tokens to `-100` so 100% of gradient signal targets `<think>...</think>` reasoning and UCI moves.

**Setup:** Runs `build_completion_collator` with `sshleifer/tiny-gpt2` across all 40 cleaned rows in `data_generation/sft_train.jsonl`, comparing the standard FEN prompt against `--include-ascii` (ASCII board injected into the user prompt).

| Prompt Configuration | Rows | Mean Total Tokens | Mean Masked Prompt Tokens (`-100`) | Mean Active Completion Tokens | Mean Per-Row Masked % | Corpus Masked % | Per-Row Masked Range | Zero-Active Rows |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Standard FEN prompt | 40 | 377.02 | **127.50** | 249.53 | **35.85%** | 33.82% | `10.58% – 50.20%` | **0 / 40** |
| `--include-ascii` board prompt | 40 | 471.02 | **221.50** | 249.53 | **48.94%** | 47.03% | `17.76% – 63.69%` | **0 / 40** |

**Engineering Takeaways:**
1. **Substantial wasted gradient prevented:** Without completion-only masking, `35.85%` of per-example loss tokens on standard prompts — and `48.94%` (up to `63.69%` on concise reasoning rows) when `--include-ascii` is enabled — would be spent memorizing fixed instruction/board prompt tokens instead of chess reasoning.
2. **100% template alignment:** `0 / 40` rows suffered template-matching failures (`zero_active_rows = 0`), confirming that `CHATML_ASSISTANT_OPEN = "<|im_start|>assistant\n"` aligns cleanly with `sft_train.jsonl` and `build_chat_prompt(..., include_ascii=True)`.
3. **Invariant active target:** Injecting the ASCII board adds exactly `94.0` prompt tokens per position (`221.50 - 127.50`), all masked to `-100`, while keeping the active completion target identical (`249.53` tokens).
