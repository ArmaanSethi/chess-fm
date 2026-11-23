# ChessFM — Reproducible Local Benchmarks

This document records verified baseline measurements produced directly by the ChessFM evaluation harness on CPU + Stockfish 16. Every figure below is reproducible from a single CLI command without GPU access or external API keys.

> **Note:** Trained LLM checkpoints (`v0` Qwen2.5-3B, `v1` Qwen3-4B, `v2` Gemma 4 E4B) are generated on GPU via `scripts/modal_train.py` or `scripts/kaggle_train.py` and evaluated with the exact same 4-tier harness. No unverified or extrapolated LLM Elo numbers are listed here.

---

## 1. Benchmark Suite & Corpus Integrity

Generated and audited with live Stockfish 16 (depth 10):

```bash
venv/bin/python -m chessfm.data_generation.build_corpus \
  --output-dir data_generation --depth 10 --clean-sft
```

| Artifact | Rows | Phase Balance | Audit Result |
|---|---:|---|---|
| `data_generation/eval_benchmark.jsonl` | 40 | 25% Opening / 50% Middlegame / 25% Endgame | 100% valid, Stockfish depth-10 ground truth |
| `data_generation/coaching_sample.jsonl` | 240 | 25% Opening / 50% Middlegame / 25% Endgame | **0 claim violations**, **0 Socratic L1/L2 spoilers**, 100% legal |
| `data_generation/sft_train.jsonl` | 40 | Cleaned from 44 committed rows | **4 mislabeled blunders (>200 cp) removed**, `passed=True` |

---

## 2. 4-Tier Local Baseline Results (Stockfish 16, Depth 6)

Reproduce:

```bash
venv/bin/python scripts/run_local_benchmarks.py \
  --benchmark data_generation/eval_benchmark.jsonl \
  --depth 6 --match-games 10
```

*Total wall-clock time on CPU: **14.24 s** (`positions_evaluated = 40`).*

### Tier 1 — Static Engine Metrics

| Baseline | Legal Rate | Format Adherence | ACPL ↓ | MAR (≤50 cp) ↑ | Mean WDL Loss ↓ | Exact Best-Move ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `random_legal` (seed 42) | 100.0% | 100.0% | 179.35 | 45.0% | 11.68% | 7.5% |
| `maia_fallback_1200` | 100.0% | 100.0% | 55.12 | 77.5% | 4.04% | 37.5% |
| `maia_fallback_2000` | 100.0% | 100.0% | 66.90 | 62.5% | 4.89% | 25.0% |

### Tier 2 — Board-State Probing (ChessQA Ground-Truth Validation)

| Category | Probes | Oracle Verified |
|---|---:|---:|
| `side_to_move` | 40 | 40 / 40 (100.0%) |
| `piece_on_square` | 40 | 40 / 40 (100.0%) |
| `move_legality` (legal + illegal pairs) | 80 | 80 / 80 (100.0%) |
| `legal_move_count` | 40 | 40 / 40 (100.0%) |
| **Total** | **200** | **200 / 200 (100.0%)** |

### Tier 3 — Coaching Faithfulness (ACT-Eval Deterministic Claim Verification)

| System Under Test | Cases | Mean Factuality ↑ | Full-Pass Rate ↑ | Spoiler Rate ↓ |
|---|---:|---:|---:|---:|
| `CoachOrchestrator` (engine-grounded explanations) | 40 | **89.58%** (gate: ≥85%) | 80.0% | **0.0%** |
| `generate_socratic_hint` (Levels 1 & 2) | 80 | **100.0%** | **100.0%** | **0.0%** |

### Tier 4 — Paired Color-Balanced Match (10 Games vs. Stockfish Depth 1)

| Model / Sidecar | Games (W/B) | W-D-L | Score | 95% Wilson CI | Decoding Mode |
|---|---:|---|---:|---|---|
| `maia-sidecar-1800-calibrated_fallback` | 10 (5W / 5B) | `0-4-6` | 0.20 | `[0.0567, 0.5098]` | `constrained` |

### PGN Game Review Smoke Check

| Sample Game | Student Accuracy | Classification Counts | Critical Moments |
|---|---:|---|---:|
| Benchmark Game 1 (tactical blunders) | 82.6% | 4 Best, 2 Good, 2 Inaccuracy, 2 Blunder | 5 |
| Benchmark Game 2 (Scholar's Mate) | 90.3% | 3 Best, 1 Mistake | 1 |

---

## 3. Part 1 GPU Benchmarks — `Qwen/Qwen3-4B` (`v1`, NVIDIA L4)

All Part 1 GPU measurements were collected on Modal NVIDIA L4 GPUs (`$0.80/hr`) over the zero-leakage, phase-balanced **200-FEN held-out benchmark** (`25%` opening / `55%` middlegame / `20%` endgame, Stockfish 16 depth-10 ground truth, `verify_zero_leakage=True`) and **25 held-out paired openings** (`eval_openings_25.txt`). See `docs/EXPERIMENT_LOG.md` (`EXP-003`–`EXP-006`) for exact Modal App IDs, W&B run links, and GPU cost ledgers.

### Reproducible CLI Commands

```bash
# Static 4-Tier Dual-Mode Evaluation (200 FENs + 1,000 Tier 2 ChessQA Probes)
modal run --detach scripts/modal_train.py --task static-eval \
  --model /vol/checkpoints/grpo_stage2/stage2_final_merged \
  --benchmark /vol/data/eval_benchmark_200.jsonl \
  --decoding-mode strict --enable-thinking --include-ascii \
  --output-subdir grpo_s2_think_on_strict.json

# N=50 Paired Color-Balanced Match vs Stockfish Level 1 (Depth 6)
modal run --detach scripts/modal_train.py --task eval \
  --model /vol/checkpoints/grpo_stage2/stage2_final_merged \
  --games 50 --stockfish-level 1 \
  --decoding-mode constrained --enable-thinking --include-ascii \
  --openings /vol/data/eval_openings_25.txt
```

### 3.1 Training Progression: Base -> SFT -> Staged GRPO (200 Held-Out FENs, ASCII ON)

| Checkpoint | Strict Legal Rate ↑ | Format Adherence ↑ | Strict ACPL ↓ | Strict MAR (≤50cp) ↑ | Constr. Legal ↑ | Constr. ACPL ↓ | Constr. MAR ↑ | Constr. Exact ↑ | Tier 2 ChessQA (`piece_on_sq`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base (Think OFF)** | `0.5%` (1/200) | `0.0%` | `80.0` *(n=1)* | `0.0%` | `100.0%` | `465.36` | `20.00%` | `2.0%` | `41.3%` (`12.5%`) |
| **Base (Think ON)** | `0.0%` (0/200) | `0.0%` | `—` *(n=0)* | `0.0%` | `0.0%` *(512-tok)* | `—` | `—` | `0.0%` | `41.3%` (`12.5%`) |
| **Completion-Masked SFT** | `21.0%` (42/200) | `100.0%` | `345.07` | `11.90%` | `100.0%` | `380.10` | `19.50%` | `6.5%` | `41.4%` (`33.5%`) |
| **GRPO Stage 0 (100 steps)** | `28.5%` (57/200) | `100.0%` | `267.30` | `17.54%` | `100.0%` | `328.11` | `19.50%` | `7.0%` | *(--skip-probes)* |
| **GRPO Stage 1 (150 steps)** | `28.5%` (57/200) | `100.0%` | `269.21` | `21.05%` | `100.0%` | `327.12` | `22.00%` | `7.0%` | *(--skip-probes)* |
| **GRPO Stage 2 (150 steps)** | **`32.0%`** (64/200) | **`100.0%`** | **`277.16`** | **`25.00%`** | **`100.0%`** | **`325.88`** | **`22.00%`** | **`7.0%`** | **`41.7%` (`34.0%`)** |

### 3.2 Dual-Mode (`Strict` vs `Constrained`) & `Think OFF` vs `Think ON` Ablation (`GRPO Stage 2`)

| Configuration | Legal Rate ↑ | Format Adherence ↑ | ACPL ↓ | MAR (≤50cp) ↑ | Mean WDL Loss ↓ | Exact Match ↑ | Wall-Clock (L4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Think OFF · `strict`** | `12.0%` (24/200) | `0.0%` | `594.71` | `16.67%` | `15.43%` | `0.0%` | `1,757.01s` |
| **Think OFF · `constrained`** | `100.0%` (200/200) | `0.0%` | `451.79` | `21.00%` | `18.88%` | `5.5%` | `77.12s` |
| **Think ON · `strict`** | **`32.0%`** (64/200) | **`100.0%`** | **`277.16`** | **`25.00%`** | `19.70%` | `1.5%` | `1,062.44s` |
| **Think ON · `constrained`** | **`100.0%`** (200/200) | **`100.0%`** | **`325.88`** | **`22.00%`** | `19.61%` | **`7.0%`** | `520.28s` |

### 3.3 N=50 Paired Color-Balanced Matches (`GRPO Stage 2`, Think ON, ASCII ON, `seed=42`)

| Opponent (`backend`) | Decoding Mode | Games (W/B) | Record (W-L-D) | Score | 95% Wilson CI | White Split | Black Split | Legal Rate | Est. Elo |
|---|---|---:|---|---:|---|---|---|---:|---:|
| **Stockfish L1 (`stockfish`, d6)** | `strict` | 50 (25W / 25B) | `0-50-0` | `0.00%` | `[0.0000, 0.0713]` | `0-25-0` (`0.00`) | `0-25-0` (`0.00`) | `33.33%` | `-299.8` |
| **Stockfish L1 (`stockfish`, d6)** | `constrained` | 50 (25W / 25B) | **`0-38-12`** | **`12.00%`** | **`[0.0562, 0.2380]`** | `0-17-8` (`0.16`) | `0-21-4` (`0.08`) | **`100.0%`** | **`553.9`** |
| **Maia-1100 (`calibrated_fallback`)** | `constrained` | 50 *(120m timeout)* | *Deferred to batched eval* | — | — | — | — | — | — |

