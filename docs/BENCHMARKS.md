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
