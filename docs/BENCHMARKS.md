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

---

## 4. Phase II Controlled Ablation Ladder — `Qwen/Qwen3-4B` (`E0 -> E1a -> E1b -> E2a -> E2b`)

Phase II isolates four axes on frozen, **0.00% normalized-FEN-overlap** benchmarks (`test_fens_1000.jsonl`, `test_puzzles_900.jsonl`, `test_chessqa_1000.jsonl`, `test_english_100.jsonl`, and 5-anchor 200-game paired matches over `test_openings_20.txt`):
1. **Board Representation & LR (`E0 -> E1a`):** Lossless Piece-by-Square coordinate list (`--include-piece-list`) + All-Linear LoRA `lr=2e-4` (10× `E0`'s `2e-5`).
2. **Corpus Scaling & Mixing (`E1a -> E1b`):** Scaling `2.5K` game-only SFT -> `20,000` mixed rows (elite games + theme-balanced Lichess puzzles + 5-category ChessQA, 2 epochs, `7,030,121` tokens).
3. **LoRA Rank Scaling (`E1b -> E2a`):** All-Linear LoRA `r=16` (`alpha=32`) -> `r=64` (`alpha=128`) on identical data (`seed=42`).
4. **Full Fine-Tuning Control (`E2a -> E2b`):** 100%-parameter Full Fine-Tuning (`--full-ft`, `lr=2e-5`, `paged_adamw_8bit`, H100-80GB) vs All-Linear LoRA `r=64` (`lr=2e-4`, A100-80GB).

### 4.1 Reproducible Phase II CLI Commands

```bash
# Train E2a (All-Linear LoRA r=64, A100-80GB) or E2b (Full-FT, H100-80GB)
modal run --detach scripts/modal_train.py --task sft --track v1 --gpu a100 \
  --data /vol/data/phase2/sft_phase2_20k.jsonl --include-piece-list \
  --lora-r 64 --all-linear --lr 2e-4 --epochs 2 --batch-size 8 --grad-accum 4 \
  --output-subdir sft_e2a_20k_r64

modal run --detach scripts/modal_train.py --task sft --track v1 --gpu h100 --full-ft \
  --data /vol/data/phase2/sft_phase2_20k.jsonl --include-piece-list \
  --lr 2e-5 --epochs 2 --batch-size 8 --grad-accum 4 \
  --output-subdir sft_e2b_20k_fullft

# Frozen Static 1,000-FEN + CoT Verifier V2 (Strict & Constrained, --batch-size 16)
modal run --detach scripts/modal_train.py --task static-eval \
  --model /vol/checkpoints/sft_e2a_20k_r64/final_merged \
  --benchmark /vol/data/phase2/test_fens_1000.jsonl \
  --decoding-mode strict --enable-thinking --include-piece-list \
  --skip-probes --batch-size 16 \
  --output /vol/results/phase2/e2a_test_fens_1000_strict.json

# Frozen 900 Puzzles + 1,000 ChessQA + 100 General-English Forgetting Probe
modal run --detach scripts/modal_train.py --task static-eval \
  --model /vol/checkpoints/sft_e2a_20k_r64/final_merged --benchmark "" \
  --puzzles /vol/data/phase2/test_puzzles_900.jsonl \
  --chessqa /vol/data/phase2/test_chessqa_1000.jsonl \
  --english-forgetting /vol/data/phase2/test_english_100.jsonl \
  --decoding-mode strict --enable-thinking --include-piece-list \
  --skip-probes --batch-size 16 \
  --output /vol/results/phase2/e2a_puzzles_chessqa_english.json

# Batched 40-Game Paired Anchor Match (repeat for random, sf0, sf1, maia1100, sf1320)
modal run --detach scripts/modal_train.py --task eval \
  --model /vol/checkpoints/sft_e2a_20k_r64/final_merged \
  --games 40 --stockfish-level 1 --decoding-mode constrained \
  --enable-thinking --include-piece-list \
  --openings /vol/data/phase2/test_openings_20.txt --batch-size 20 \
  --manifest-output /vol/results/phase2/e2a_bt_sf1.json
```

### 4.2 Complete Phase II Ablation Ladder (`Think ON`, Deterministic Greedy)

| Stage (Exp ID) | Setup | Val Loss | Strict Legal ↑ | Strict ACPL ↓ | Strict MAR ↑ | Constr. ACPL ↓ | CoT V2 Acc ↑ | 900 Puzzles Solved ↑ | 1,000 ChessQA ↑ | English (`Leak`) ↑ | 5-Anchor BT Elo (`95% CI`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`E0` SFT** (`EXP-004`) | 2.5K ASCII, `r=16`, `2e-5` | `0.5181` | `21.0%` *(200)* | `345.07` | `11.90%` | `380.10` | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |
| **`E0` GRPO S2** (`EXP-006`) | +400 GRPO steps, ASCII | — | `32.0%` *(200)* | `277.16` | `25.00%` | `325.88` | *N/A* | *N/A* | *N/A* | *N/A* | `553.9` *(SF L1 only)* |
| **`E1a` SFT** (`EXP-007`) | 2.5K Piece List, `r=16`, `2e-4` | `0.4302` | `63.5%` *(200)* | `421.31` | `29.13%` | `330.18` | `64.02%` | `4.67%` (`42`) | `11.9%` | *N/A* | *N/A* |
| **`E1b` SFT** (`EXP-008`) | 20K Mix, Piece List, `r=16`, `2e-4` | `0.2775` | `64.4%` *(1,000)* | `336.51` | `31.52%` | `393.00` | `60.02%` | `18.33%` (`165`) | `38.8%` | `89.0%` (`0%`) | *N/A* |
| **`E2a` SFT** (`EXP-009`) | 20K Mix, Piece List, **`r=64`, `2e-4`** | **`0.2660`** | **`66.2%`** *(1,000)* | **`297.56`** | `32.02%` | **`387.90`** | **`60.66%`** | **`21.44%`** (`193`) | **`45.2%`** | **`90.0%`** (`0%`) | **`542.35`** `[470.8, 610.5]` |
| **`E2b` SFT** (`EXP-010`) | 20K Mix, Piece List, **Full-FT, `2e-5`** | `0.2842` | `62.4%` *(1,000)* | `322.13` | **`32.53%`** | `410.67` | `57.74%` | `17.44%` (`157`) | `31.2%` | `89.0%` (`0%`) | **`530.98`** `[458.2, 595.6]` |

### 4.3 5-Anchor 200-Game Paired Bradley-Terry Elo (`E2a` vs `E2b`, Constrained Decoding)

| Anchor Opponent (Calibrated Elo) | `E2a` (`r=64`) W-L-D | `E2a` Score | `E2b` (Full-FT) W-L-D | `E2b` Score |
|---|---:|---:|---:|---:|
| Random Bot (`200`) | `1-2-37` | `48.8%` | `3-0-37` | **`53.8%`** |
| Stockfish Skill 0, d6 (`800`) | `0-30-10` | `12.5%` | `0-27-13` | **`16.2%`** |
| Stockfish Skill 1, d6 (`950`) | `0-29-11` | **`13.8%`** | `0-30-10` | `12.5%` |
| Maia-1100 fallback (`1100`) | `0-22-18` | `22.5%` | `0-22-18` | `22.5%` |
| Stockfish `UCI_Elo=1320`, d6 (`1320`) | `0-22-18` | **`22.5%`** | `0-30-10` | `12.5%` |
| **Bradley-Terry Elo ± 95% Bootstrap CI** | **`542.35`** | **`[470.81, 610.53]`** | **`530.98`** | **`[458.17, 595.64]`** |

### 4.4 Key Phase II Takeaways & Sample Verified Traces
- **Piece-by-Square Coordinate List + Proper LoRA LR (`E0 -> E1a`)** tripled strict legality (`21.0% -> 63.5%`) and raised `piece_on_square` probing from `33.5% -> 80.5%` on the exact same `2.5K` rows.
- **20K Mixed-Corpus Scaling (`E1a -> E1b`)** increased 900-puzzle solve rate `3.93×` (`4.67% -> 18.33%`) and 1,000-ChessQA `3.26×` (`11.9% -> 38.8%`) with `0.0%` English leakage.
- **All-Linear LoRA `r=64` (`lr=2e-4`) vs Full-FT (`lr=2e-5`) (*LoRA Without Regret*):** In the 20K-sample / 7M-token regime, `E2a` (`r=64`) beat `E2b` (Full-FT) on held-out loss (`0.2660` vs `0.2842`), strict ACPL (`297.56` vs `322.13 cp`), 900 puzzles (`21.44%` vs `17.44%`), and 1,000 ChessQA (`45.2%` vs `31.2%`), while both preserved `89–90%` general English with `0.0%` chess leakage.
- **Stateless Greedy Repetition Trap:** Across 400 combined `E2a` + `E2b` match games, `89.0%` of draws (`84/94` for `E2a`, `78/88` for `E2b`) ended by threefold repetition under single-FEN greedy decoding, motivating history-aware prompting and RLVR post-training.
- **Sample Exact-Match + Zero-Violation CoT Verifier V2 Traces:**
  - `E2a`: `1r3r1k/1R2R2p/1p6/p2B1bp1/8/8/PP4PP/6K1 b - - 1 31` -> `b8b7` (`cp_loss=0.0`, `6/6` V2 CoT claims verified).
  - `E2b`: `rn1qkbnr/ppp2pp1/8/3p3p/3P3P/2Nb1P2/PPP3P1/R1BQK1NR w KQkq - 0 8` -> `d1d3` (`cp_loss=0.0`, `7/7` V2 CoT claims verified).
  - `E2b`: `2r5/1R2pk2/3pp3/6p1/3NP2p/1RPn1P1P/4K1P1/r7 b - - 3 33` -> `d3c1` (`cp_loss=0.0`, `6/6` V2 CoT claims verified).

---

## 5. Phase III RLVR — Reward Hacking Diagnosis (`E3a`), Anti-Hack `sharp_wdl` (`E3b`), `pass@k`, and Repetition-Aware BT Elo

Phase III applies DAPO-style GRPO (`G=8`, `ε_low=0.20`, `ε_high=0.28`, `β=0.04`, `num_iterations=2`, All-Linear LoRA `r=32`, `lr=1.5e-5`, 200 steps on A100-80GB) on top of `E2a` (`r=64` merged) using `20,000` zero-leakage sharp tactical positions (`rl_sharp_tactical.txt`, `0.00%` overlap with all frozen test sets).

### 5.1 Reproducible Phase III CLI Commands

```bash
# Train E3b (Anti-Hack sharp_wdl DAPO, A100-80GB)
modal run --detach scripts/modal_train.py --task grpo --track v1 --gpu a100 \
  --model /vol/checkpoints/sft_e2a_20k_r64/final_merged \
  --positions /vol/data/phase2/rl_sharp_tactical.txt \
  --stage 1 --reward-mode sharp_wdl --steps 200 \
  --num-generations 8 --batch-size 16 --grad-accum 2 --num-iterations 2 \
  --lr 1.5e-5 --beta 0.04 --epsilon 0.20 --epsilon-high 0.28 \
  --lora-r 32 --all-linear --include-piece-list --merge-and-save \
  --output-subdir grpo_e3b_sharp_wdl

# Stratified pass@1..pass@8 (N=200 puzzles, k=8, T=0.7, top_p=0.95)
modal run --detach scripts/modal_train.py --task pass-at-k \
  --model /vol/checkpoints/grpo_e3b_sharp_wdl/stage1_final_merged \
  --puzzles /vol/data/phase2/test_puzzles_900.jsonl --limit 200 \
  --k 8 --temperature 0.7 --top-p 0.95 --include-piece-list

# Repetition-Aware 5-Anchor 200-Game Paired Match (--avoid-repetition)
modal run --detach scripts/modal_train.py --task eval \
  --model /vol/checkpoints/grpo_e3b_sharp_wdl/stage1_final_merged \
  --games 40 --stockfish-level 1 --decoding-mode constrained --avoid-repetition \
  --enable-thinking --include-piece-list --openings /vol/data/phase2/test_openings_20.txt \
  --batch-size 20 --manifest-output /vol/results/phase2/e3b/e3b_norep_bt_sf1.json
```

### 5.2 `E2a` (SFT) vs `E3a` (Naive WDL-DAPO: A-Pawn Hack) vs `E3b` (Anti-Hack `sharp_wdl`)

| Metric (`test_fens_1000` / `test_puzzles_900` / `test_chessqa_1000`) | `E2a` (SFT `r=64`) | `E3a` (Naive `wdl_loss` — Hacked) | **`E3b` (`sharp_wdl` — Clean)** | Delta (`E3b` vs `E2a`) |
|---|---:|---:|---:|---:|
| **Strict Legal Rate** (`test_fens_1000`) ↑ | `66.2%` | `86.4%` *(hacked)* | **`76.5%`** | **`+10.3 pp`** |
| **Strict MAR (`≤50 cp`, all 1,000)** ↑ | `20.5%` | `23.1%` | **`23.5%`** | **`+3.0 pp`** |
| **Strict Exact Best Move** ↑ | `8.1%` | `4.0%` *(collapsed)* | **`8.0%`** | `-0.1 pp` |
| **Constrained ACPL** ↓ | `387.90 cp` | `356.87 cp` | **`361.94 cp`** | **`-25.96 cp`** |
| **Constrained MAR (`≤50 cp`) / Exact Match** ↑ | `30.2%` / `11.8%` | `28.7%` / `6.5%` | **`31.5%` / `11.8%`** | **`+1.3 pp` / `0.0 pp`** |
| **Emergent CoT V2 Claim Accuracy** *(0 CoT reward!)* ↑ | `60.66%` (`19.8%` 0-viol) | `82.43%` *(a-pawn templates)* | **`70.68%` (`33.8%` 0-viol)** | **`+10.02 pp`** |
| **Unique Predicted Moves** (`1,000` FENs) ↑ | `528` | `206` *(collapsed)* | **`443`** | `-85` *(passes gate)* |
| **Strict `a`-File Pawn Pushes** (`^a[2-7]a[2-7]$`) ↓ | `1.2%` (`12`) | **`60.3%` (`603`)** *(FAIL)* | **`1.5%` (`15`)** *(PASS)* | `+0.3 pp` |
| **Top-1 Move Frequency** ↓ | `3.9%` (`e1g1`) | `18.7%` (`a2a3`) | **`3.7%` (`h2h3`)** | **`-0.2 pp`** |
| **900 Puzzles Solved / Strict Legal** ↑ | **`21.44%`** / `72.67%` | `11.56%` / `81.89%` | `16.56%` / **`78.33%`** | `-4.88 pp` / `+5.66 pp` |
| **1,000 ChessQA / 100 English (`0%` Leak)** ↑ | **`45.2%`** / `90.0%` | `43.7%` / `90.0%` | `44.6%` / `89.0%` | `-0.6 pp` / `-1.0 pp` |

### 5.3 Exploration vs Sharpening: Stratified `pass@1`–`pass@8` (`N=200` Puzzles, `k=8`, `T=0.7`)

| Model | `pass@1` (`95% CI`) | `pass@2` | `pass@4` | `pass@8` (`95% CI`) | Strict Legal (`1,600` rollouts) | CoT V2 Acc | Legal Move Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`E2a` (SFT `r=64`)** | **`14.56%`** `[11.4, 17.9]` | **`22.75%`** | **`32.69%`** | **`43.50%`** `[36.5, 50.5]` | `68.56%` | `52.25%` | `1.35 bits` |
| **`E3a` (A-Pawn Hack)** | `8.06%` `[6.3, 10.1]` | `14.34%` | `23.76%` | `36.00%` `[29.5, 43.0]` | `74.19%` | `64.00%` | `1.88 bits` |
| **`E3b` (`sharp_wdl`)** | `11.13%` `[8.4, 14.1]` | `17.62%` | `25.76%` | `34.00%` `[27.5, 41.0]` | **`75.62%`** | **`59.32%`** | `1.61 bits` |

### 5.4 Exposing the Stateless Repetition Mirage: 4-Suite 200-Game Paired Bradley-Terry Elo

| Suite (`200` Paired Games across 5 Anchors) | W-L-D | 3-Fold Rep Draws | Repetition-Filtered Plies | Wins vs Maia-1100 | BT Elo (`95% Bootstrap CI`) |
|---|---|---:|---:|---:|---|
| **`E2a` Standard (`constrained`)** | `1-105-94` | `84 / 94` (`89.4%`) | `0` | `0` | `542.35` `[470.81, 610.53]` *(inflated)* |
| **`E2a` Repetition-Aware (`--avoid-repetition`)** | `3-143-54` | `17 / 54` (`31.5%`) | `317` | `0` | **`311.43`** `[263.14, 365.87]` *(`-230.9` mirage)* |
| **`E3b` Standard (`constrained`)** | `5-131-64` | `55 / 64` (`85.9%`) | `0` | **`2`** | `406.07` `[331.93, 483.26]` |
| **`E3b` Repetition-Aware (`--avoid-repetition`)** | `4-140-56` | `22 / 56` (`39.29%`) | `285` | **`1`** | **`338.74`** `[277.01, 406.07]` **(`+27.31` clean RLVR lift)** |

---

## 6. Phase IV — Real `4B` AI Chess Coach (`E5`), Goodhart's Law Showdown, and `26B` Anchor

### 6.1 Exposing Goodhart's Law on Grounded `0.6B` Coaching (`N=292`, Verifier V1 vs Strict V2)

| Model (`N=292` grounded coaching outputs) | Weak V1 Pass ↑ | Strict V2 Pass ↑ | V2 + Non-Slop ↑ | Slop-Flagged Rate ↓ | Distinct-4gram ↑ | Top Repeated 4-Gram |
|---|---:|---:|---:|---:|---:|---|
| **`Gemma-4-26B` Teacher** | `100.0%` | **`93.84%`** | **`93.84%`** | **`0.00%`** | **`0.7616`** | `"is there a way"` (`13.4%`) |
| **`Qwen3-0.6B` Base (grounded)** | `82.88%` | `81.16%` | `32.53%` | `64.38%` | `0.5060` | `"the best move is"` (`37.7%`) |
| **`Qwen3-0.6B` SFT (grounded)** | `86.64%` | `61.99%` | **`12.67%`** | **`79.79%`** | `0.4097` | `"tension in the center"` (`41.1%`) |

### 6.2 Complete 7-Row `test_coach_300` Benchmark (`4,518` Spotless V2 Rows, `0.00%` Leakage)

All 7 arms evaluated with identical deterministic tactical-fact grounding (`test_coach_300.jsonl`, 60 × 5 modes, Stockfish depth-10 truth, greedy decoding).

| Row & Model | Weak V1 ↑ | Strict V2 ↑ | V2 + Non-Slop ↑ | L1/L2 Spoiler-Free ↑ | **All-Clean** (`95% CI`) ↑ | Tactical Fact Cov. ↑ | Distinct-4gram ↑ |
|---|---:|---:|---:|---:|---|---:|---:|
| **1. Base `Qwen3-4B`** (0-shot) | `82.67%` | `77.67%` | `63.33%` | `86.67%` | `58.00%` `[52.4, 63.4]` | `89.67%` | `0.7416` |
| **2. `Arm A` SFT** (`Base 4B → SFT`) | `93.00%` | `88.67%` | `88.67%` | `100.0%` | `88.67%` `[84.6, 91.8]` | `79.56%` | `0.7255` |
| **3. `Arm A` Strict `V2-GRPO`** *(WINNER)* | `95.67%` | **`92.00%`** | **`92.00%`** | **`100.0%`** | **`92.00%`** `[88.4, 94.6]` | `77.14%` | `0.7175` |
| **4. `Arm B` SFT** (`E3b → SFT`) | `93.33%` | `89.00%` | `89.00%` | `98.33%` | `88.33%` `[84.2, 91.5]` | `79.25%` | `0.7226` |
| **5. `Arm B` Strict `V2-GRPO`** *(Play→Coach Transfer)* | `95.00%` | `91.33%` | `91.33%` | `99.17%` | **`91.00%`** `[87.2, 93.7]` | **`81.67%`** *(`+4.53 pp`)* | `0.7123` |
| **6. `Gemma-4-26B` Teacher** (single-shot) | `90.00%` | `88.67%` | `88.67%` | `100.0%` | `88.67%` `[84.6, 91.8]` | **`89.78%`** | **`0.7862`** |
| **7. `Arm C` Weak `V1-GRPO`** *(Goodhart Control)* | **`96.00%`** | `89.33%` | `89.33%` | `100.0%` | `89.33%` `[85.3, 92.3]` | `76.44%` | `0.7237` |

- **David-vs-Goliath `4B` vs `26B` (`gemma26b_puzzles_chessqa.json`):** Zero-shot `google/gemma-4-26B-A4B-it` scored **`0.22%` strict / `10.78%` lenient** (`97/900`) on `test_puzzles_900` and **`26.30%`** on `test_chessqa_1000` (exhausting the 512-token cap on `666/900` puzzles), whereas our `6.5×` smaller `4B` models achieved **`21.44%` strict puzzles (`E2a`, `97×` strict / `2.0×` lenient)**, **`45.20%` ChessQA (`+18.9 pp`)**, and **`92.00%` All-Clean coaching (`Arm A`, `+3.33 pp` above the `26B` Teacher itself!)**.
- **Local Deployment Artifact:** `Arm A` Strict `V2-GRPO` passed the `>=90% V2+Non-Slop` and `100% L1/L2 Spoiler-Free` export gate and was quantized to `Q4_K_M` GGUF (`2.50 GB`) with an Ollama `Modelfile` (`winsonly/chessfm-qwen3-4b-coach-v2`, `private=True`).

---

## 7. Phase IV.5 — Mechanistic Interpretability Suite (`INTERP #1`–`#4`, `Qwen/Qwen3-4B`)

Reproduce all four mechanistic interpretability experiments on one A100-80GB GPU:

```bash
modal run --detach scripts/modal_train.py --task interp-suite
modal volume get chessfm-artifacts results/interp ./results/interp
```

### 7.1 INTERP #1 — 64-Square Linear Board-State Probes & Causal Board Surgery (`800` Train / `200` Held-Out FENs)

| Checkpoint (Prompt Mode) | L0 Occ. ↑ | L6 Occ. ↑ | L12 Occ. ↑ | L18 Occ. ↑ | L24 Occ. ↑ | **L30 Occ. ↑** | **L35 Occ. ↑** | L6 → L35 Trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Base_ASCII` (`Qwen3-4B`) | `70.71%` | `71.34%` | `68.88%` | `65.53%` | `63.33%` | `61.76%` | **`60.91%`** | `-10.43 pp` monotonic decay |
| `E0_ASCII` (2.5K ASCII) | `70.49%` | `70.92%` | `68.46%` | `66.28%` | `64.78%` | `64.60%` | **`61.76%`** | `-9.16 pp` monotonic decay |
| `E1a_PieceList` (2.5K PL) | `70.03%` | **`72.97%`** | `69.73%` | `68.19%` | `66.80%` | **`70.13%`** | **`69.00%`** | **U-shaped L30–35 rebound (`+8.09 pp` vs Base)** |
| `E2a_r64` (20K PL, `r=64`) | `70.81%` | `72.09%` | `69.28%` | `68.23%` | `63.76%` | `67.59%` | `66.52%` | **U-shaped L30–35 rebound (`+5.61 pp` vs Base)** |
| `E3b_SharpWDL` (RLVR) | `70.15%` | `71.12%` | `69.34%` | `68.19%` | `64.20%` | `67.53%` | **`66.82%`** | **U-shaped L30–35 rebound (`+5.91 pp` vs Base)** |

- **Causal Control (Layer-6 Board Surgery on `E3b`, `N=50`):** Erasing the moving piece along the Layer-6 probe axis flipped `18.0%` (`9/50`) of moves vs **`24.0%`** (`12/50`, excess `-6.0 pp`) for a norm-matched orthogonal random control (`steer_norm=10.46`). This null Layer-6 bottleneck result shows the final prompt token is *not* the sole causal board bottleneck—later attention heads re-read piece coordinates directly from the Piece-List KV cache (confirmed causally in INTERP #4).

### 7.2 INTERP #2 — Exact QR-SVD RL Weight-Diffing & Causal Goodhart "Slop Vector" Steering

| RL Adapter (`r=32`, 252 Modules) | Total Frobenius | MLP Energy Share | Attn Energy Share | **Peak Update Layer** | **Mean Eff. Rank** | Top-1 SV Energy | Top-2 Diff Layers vs Paired Arm |
|---|---:|---:|---:|---:|---:|---:|---|
| **`E3a` (`wdl_loss` — A-Pawn Hack)** | `1.3404` | `71.24%` | `28.76%` | **`Layer 35`** *(final layer!)* | **`11.41`** *(collapsed)* | **`40.66%`** | **`L34` (`+0.0207`), `L35` (`+0.0207`)** |
| **`E3b` (`sharp_wdl` — Clean RLVR)** | `1.3291` | `71.15%` | `28.85%` | **`Layer 17`** *(mid-network)* | **`12.83`** | `36.77%` | Larger update at **`L29`** (`+0.0078`) |
| **`Arm A` (Strict `V2-GRPO`, `92.0%`)** | `0.8757` | `71.66%` | `28.34%` | `Layer 27` | `15.64` | `32.01%` | Larger updates at **`L16–L20`** |
| **`Arm C` (Weak `V1-GRPO`, `89.3%`)** | `0.8559` | `72.02%` | `27.98%` | **`Layer 31`** *(later)* | `15.67` | `31.40%` | Larger updates at **`L2`** & late layers |

- **Causal Goodhart "Slop Vector" Sweep (`mean(h_ArmC) - mean(h_ArmA)` injected into `Arm A`, `N=50` held-out coach prompts):**

| Steering `α` | Weak V1 Pass | Strict V2 Pass | `V1 − V2` Gap | **Slop-Flagged Rate** | Distinct-4gram | L1/L2 Spoiler-Free | Mean Words |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **`-2.0`** *(Anti-Goodhart)* | `88.0%` | `84.0%` | `4.0 pp` | **`0.0%`** | `0.8706` | `100.0%` | `44.38` |
| **`-1.0`** | `90.0%` | `88.0%` | **`2.0 pp`** | `20.0%` | `0.8632` | `100.0%` | `44.00` |
| **`0.0`** *(Unsteered `Arm A`)* | `96.0%` | **`92.0%`** | `4.0 pp` | `26.0%` | **`0.8730`** | `100.0%` | `43.80` |
| **`+1.0`** | `90.0%` | `86.0%` | `4.0 pp` | `22.0%` | `0.8714` | `100.0%` | `43.56` |
| **`+2.0`** *(Pro-Goodhart)* | `92.0%` | `86.0%` | **`6.0 pp`** | **`36.0%`** | **`0.8569`** | `100.0%` | `42.80` |

### 7.3 INTERP #3 & #4 — Causal `<think>` Sabotage & Piece-Tracker Attention Circuit (`E3b`, `N=100` Solved Puzzles)

| Experiment | Measurement | Result | Control / Reference |
|---|---|---|---|
| **INTERP #3A** Logit Lens | `P(target_move)` across `<think>` `0% → 50% → 100% → post_</think>` | `0.0000 → 0.0200 → 0.0000 →` **`0.99997`** (`100%` top-1) | Move emission commits cleanly at `</think>` boundary |
| **INTERP #3B** `<think>` Sabotage | Corrupted `<think>` prefill steering to legal distractor (`N=100`) | **`100.0%` flip (`100/100`), `100.0%` adopted distractor** | `0.0%` resisted — `<think>` is 100% causally load-bearing |
| **INTERP #4A** Head Hunt (`60` puzzles) | Top-5 Piece-Tracker heads (`1,152` total heads) | **`L27.H17` (`+0.0186`), `L29.H9` (`+0.0097`), `L26.H20` (`+0.0076`), `L26.H24` (`+0.0075`), `L28.H2` (`+0.0066`)** | All 5 tightly clustered in **Layers 26–29** right before L30–35 board rebound! |
| **INTERP #4B** Causal Zero-Ablation | Zero-ablate Top-5 (`0.43%` of heads) on `100` solved puzzles | **`-12.00 pp` solve drop** (`88%` retained, `98%` legal) | Layer-matched Random-5: **`-4.00 pp`** (`+8.00 pp` / **`3.0×`** causal excess drop!) |


