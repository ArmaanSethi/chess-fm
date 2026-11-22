# ChessFM

> Engine-grounded chess reasoning models and an offline AI Chess Coach built on SFT, staged GRPO, constrained decoding, and deterministic verification.

[![CI](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArmaanSethi/chess-fm)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Models](https://img.shields.io/badge/Models-Qwen2.5--3B%20%7C%20Qwen3--4B%20%7C%20Gemma--4--E4B-blue)]()

---

## Overview

Most chess engines play by brute-force search without explaining their plans. ChessFM trains compact language models to **reason in `<think>...</think>` tags before committing to a legal UCI move**, and pairs that reasoning foundation with a deterministic, engine-grounded **AI Chess Coach**:

```xml
<think>
Black's queen attacks f7. Castling immediately drops material, so developing
with Nf6 defends the square while contesting the center.
</think>
g8f6
```

All training harnesses, reward math, constrained decoding, coaching engines, 4-tier evaluation suites, and cloud/local deployment scripts are implemented and verified end-to-end in CI.

---

## Two-Track Architecture

### Track 1 — Research & RL (`v0` / `v1`)

| Track | Default Model | Role |
|-------|---------------|------|
| **v0 (Control)** | `Qwen/Qwen2.5-3B-Instruct` | 3B baseline selected in [Phase 0 benchmarks](docs/phase0/BASELINE_REPORT.md) |
| **v1 (Reasoning Upgrade)** | `Qwen/Qwen3-4B` | 4B reasoning backbone with native `<think>` alignment |

Key capabilities:
- **Model-Agnostic Prompting & Board Representation**: Single source of truth in `src/chessfm/config.py` (`build_chat_prompt`) using `tokenizer.apply_chat_template` with ChatML fallback and optional ASCII+FEN board rendering (`--include-ascii`).
- **Best-Line SFT & Leakage-Free Splits**: Completion-only loss (`train_sft.py`), FEN-level splitting (`training/split.py`), phase-stratified sampling across opening/middlegame/endgame (`data_generation/phase_sampler.py`), and deterministic `$0`-API best-line trace construction (`data_generation/coaching_dataset.py`).
- **Merged SFT → Staged GRPO**: LoRA adapters merge cleanly (`--merge-adapter`) into a standalone checkpoint before staged GRPO (`rl/train_grpo.py`):
  - **Stage 0** — Legality (`+1` legal / `-1` illegal) + `<think>` format and length-budget shaping (Stockfish-free).
  - **Stage 1** — Adds Stockfish move quality relative to the engine's best move.
  - **Stage 2** — Adds opponent-reply delta (evaluation after Stockfish's best counterplay).
  - Engine errors are isolated as neutral infrastructure failures and abort training if the failure rate exceeds 5%.
- **100%-Legal Constrained Decoding**: Trie-based UCI `LogitsProcessor` (`rl/constrained_decoding.py`) masks illegal move continuations after `</think>` at inference time without altering pre-move reasoning.

### Track 2 — AI Chess Coach (`v2`)

| Track | Default Model | Gated Official Variant |
|-------|---------------|------------------------|
| **v2 (Coach)** | `unsloth/gemma-4-E4B-it` | `google/gemma-4-E4B-it` (via `HF_TOKEN`) |

Key capabilities:
- **Stockfish MultiPV Partial Strategy Trees & WDL Classification** (`coach/classifier.py`): Evaluates candidate lines, computes win-probability deltas, and classifies moves (`brilliant`, `best`, `good`, `inaccuracy`, `mistake`, `blunder`).
- **Maia-2 Human-Elo Calibration (600–2600)** (`coach/maia_sidecar.py`): Estimates human move probabilities at the student's target Elo and flags practical traps where engine-best lines diverge from likely human replies.
- **3-Level Socratic Hints with 0% Spoilers** (`coach/socratic.py`): Progressive coaching hints (Level 1 strategic theme → Level 2 target piece/square → Level 3 tactical mechanism) verified by `contains_move_spoiler` so answers are never leaked.
- **ACT-Eval Deterministic Claim Verifier** (`coach/verifier.py`): Audits natural-language coaching claims (checks, captures, hanging pieces, legal moves, tactical motifs) directly against `python-chess` board ground truth.
- **Full-Game PGN Review & Weakness Profiler** (`coach/game_review.py`): Analyzes complete PGNs, identifies critical turning points, aggregates recurring weaknesses, and produces clickable tactical drills.
- **Offline Interactive Coach Studio UI** (`coach/server.py`, `coach/static/index.html`): Zero-dependency Python stdlib HTTP server (`python -m chessfm.coach.server` or `chessfm-coach`) supporting deterministic Stockfish+Maia coaching out of the box and optional local Ollama LLM narration (`OLLAMA_HOST`, `CHESSFM_OLLAMA_MODEL`).

---

## 4-Tier Evaluation Suite

Run the full structured harness locally with:

```bash
python -m chessfm.eval.suite --demo
# or via installed console script:
chessfm-eval-suite --demo
```

| Tier | Scope | Metrics |
|------|-------|---------|
| **Tier 1 — Static Engine Metrics** | Move quality & formatting | Legal-move rate, `<think>...</think>` format adherence, ACPL, Move Accuracy Rate (MAR), mean WDL win-probability loss, exact puzzle accuracy |
| **Tier 2 — Board-State Probing (ChessQA)** | Spatial comprehension | Deterministic accuracy on side to move, piece on square, move legality, and legal move count |
| **Tier 3 — Coaching Faithfulness (ACT-Eval)** | Groundedness & pedagogy | Deterministic claim verification rate (`verify_claims`) and Socratic spoiler rate (`contains_move_spoiler`) |
| **Tier 4 — Paired Color-Balanced Match** | Head-to-head play | Overall and White/Black split W-D-L, 95% Wilson CI, rough Elo indicator, and reproducibility metadata |

> Note: `legal rate`, `format adherence`, ACPL, and paired W-D-L are the primary reported metrics. Any Elo figure printed by the evaluator is an informal ballpark indicator derived from Stockfish skill levels, not a validated rating.

---

## Getting Started

```bash
# 1. Create venv, install Stockfish, and install package + dev tools
./setup_env.sh
source venv/bin/activate

# 2. Run the 4-tier evaluation suite in demo mode
python -m chessfm.eval.suite --demo

# 3. Launch the offline Interactive Coach Studio UI
python -m chessfm.coach.server --host 127.0.0.1 --port 8000

# 4. Verify the SFT data pipeline (dedupes by FEN, writes data_manifest.json)
python -m chessfm.data_generation.convert_to_training
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run
```

Installed console scripts after `pip install -e .`:
`chessfm-convert`, `chessfm-generate-data`, `chessfm-fetch-fens`, `chessfm-train-sft`, `chessfm-train-grpo`, `chessfm-evaluate`, `chessfm-coach`, `chessfm-eval-suite`.

---

## `$0` Cloud & Local Workflows

### Modal (L4 GPU) — `scripts/modal_train.py`

```bash
pip install modal && modal setup

# End-to-end smoke pipeline: merged SFT -> GRPO Stage 0 -> Stockfish eval
modal run scripts/modal_train.py --task smoke-pipeline --include-ascii

# Individual stages (v0, v1, or v2/coach; set HF_TOKEN for google/gemma-4-E4B-it)
modal run scripts/modal_train.py --task sft --track v2 --epochs 3 --include-ascii
modal run scripts/modal_train.py --task grpo --stage 0 --steps 500 --include-ascii
modal run scripts/modal_train.py --task eval --games 20 --stockfish-level 1
modal run scripts/modal_train.py --task coach-eval
modal run scripts/modal_train.py --task export-gguf --model /vol/checkpoints/sft_smoke/final_merged

# Pull artifacts back locally
modal volume get chessfm-artifacts results ./results
modal volume get chessfm-artifacts checkpoints ./checkpoints
```

### Kaggle Notebook (2x T4 GPU, Resumable 12h Sessions) — `scripts/kaggle_train.py`

```bash
# Inside a Kaggle Notebook cell
!apt-get -qq update && apt-get -qq install -y stockfish
!python scripts/kaggle_train.py --task pipeline \
    --model unsloth/gemma-4-E4B-it \
    --stage 0 --steps 500 \
    --output /kaggle/working/checkpoints \
    --include-ascii --resume

# Or run individual stages
!python scripts/kaggle_train.py --task sft --model Qwen/Qwen3-4B --epochs 3
!python scripts/kaggle_train.py --task grpo --stage 1 --steps 500 --resume
!python scripts/kaggle_train.py --task eval --games 20 --stockfish-level 1
```

### Local Ollama GGUF Export — `scripts/export_gguf.py`

```bash
# Preview the generated Modelfile and conversion command without llama.cpp
python scripts/export_gguf.py \
    --model checkpoints/sft_coach/final_merged \
    --output exports/chesscoach-gemma4 \
    --quant q4_k_m \
    --family gemma4 \
    --dry-run

# Export GGUF and register with Ollama
python scripts/export_gguf.py \
    --model checkpoints/sft_coach/final_merged \
    --output exports/chesscoach-gemma4 \
    --quant q4_k_m \
    --family gemma4
ollama create chesscoach -f exports/chesscoach-gemma4/Modelfile
```

---

## Reproducibility & Testing

No fabricated Elo ratings or uncommitted GPU metrics are claimed in this repository. Model checkpoints (`checkpoints/`), raw evaluation outputs (`results/`), and large training corpora are gitignored and regenerated via the commands below.

| Verification / Artifact | Command | Verified Result / Output |
|-------------------------|---------|--------------------------|
| Lint & formatting | `venv/bin/ruff check . && venv/bin/ruff format --check .` | Clean pass in CI |
| Unit & integration suite | `venv/bin/python -m pytest --cov=chessfm --cov-report=term-missing` | 432 passing tests, 93%+ coverage (CI gate ≥80%) |
| TRL CPU smoke test | `venv/bin/python -m pytest tests/test_training_smoke.py` | 1 real SFT step + 1 real GRPO step (`sshleifer/tiny-gpt2`) |
| 4-Tier Eval Suite | `python -m chessfm.eval.suite --demo` | Structured JSON report across Tiers 1–4 |
| Data conversion & manifest | `python -m chessfm.data_generation.convert_to_training` | `data_generation/sft_train.jsonl` + `data_manifest.json` |
| SFT dry-run validation | `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run` | Schema & split validation (no GPU) |
| SFT training (GPU) | `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --merge-adapter` | LoRA + merged checkpoint under `checkpoints/` |
| GRPO training (GPU) | `python -m chessfm.rl.train_grpo --stage 0 --steps 2000 --checkpoint-every 500` | RL checkpoint under `checkpoints/` |
| Stockfish match eval (GPU) | `python -m chessfm.rl.evaluate --model <checkpoint> --games 50` | `results/eval_<timestamp>.json` + `results/pgns/eval_<timestamp>.pgn` |

Each evaluation run records reproducibility metadata (git revision, seed, opponent level, fixed Stockfish search depth, and opening set) and alternates White/Black colors by default (`--no-alternate-colors` to pin a side).

---

## Committed Sample Data

The repository includes a small sample dataset for deterministic pipeline verification; full corpora are generated on demand:

| File | Contents |
|------|----------|
| `data_generation/all_sft_data.jsonl` | Raw engine-validated samples (85 rows) |
| `data_generation/sft_train.jsonl` | Converted `prompt`/`completion` dataset (44 rows, deduped by FEN) |
| `data_generation/positions.txt` | 25,000 elite-game FENs (position pool) |
| `data_generation/data_manifest.json` | Provenance manifest (counts, sources, timestamp) |

---

## Project Structure

```
chess-fm/
├── README.md                     # Portfolio & engineering documentation
├── AGENTS.md                     # Developer and agent operational guide
├── pyproject.toml                # Packaging, console scripts, ruff, pytest, coverage
├── setup_env.sh                  # Local venv + Stockfish setup
├── scripts/                      # $0 cloud & local deployment runners
│   ├── modal_train.py            # Modal L4 runner (v0/v1/v2, eval, export)
│   ├── kaggle_train.py           # Resumable Kaggle 2x T4 runner
│   └── export_gguf.py            # GGUF conversion + Ollama Modelfile generator
├── src/chessfm/
│   ├── config.py                 # Central config, model tracks, chat/ASCII prompts
│   ├── fens.py                   # Shared FEN/JSONL readers
│   ├── load_model.py             # Quantized & PEFT model loading
│   ├── coach/                    # AI Chess Coach (v2)
│   │   ├── classifier.py         # MultiPV Partial Strategy Trees & WDL classification
│   │   ├── maia_sidecar.py       # Maia-2 human-Elo (600-2600) calibration & traps
│   │   ├── socratic.py           # 3-level Socratic hints + 0%-spoiler verifier
│   │   ├── verifier.py           # ACT-Eval deterministic claim verifier
│   │   ├── game_review.py        # Full-game PGN review & weakness profiler
│   │   ├── multimodal.py         # CoachOrchestrator Q&A pipeline
│   │   ├── server.py             # Offline Coach Studio HTTP server & REST API
│   │   └── static/index.html     # Interactive Coach Studio UI
│   ├── data_generation/
│   │   ├── coaching_dataset.py   # Motif detection & $0-API best-line/coaching builder
│   │   ├── phase_sampler.py      # Opening/middlegame/endgame stratified sampling
│   │   ├── convert_to_training.py
│   │   ├── generate_synthetic_data.py
│   │   ├── download_positions.py
│   │   └── fetch_elite_data.py
│   ├── eval/
│   │   └── suite.py              # 4-Tier Evaluation Suite (Static, ChessQA, ACT-Eval, Match)
│   ├── rl/
│   │   ├── chess_env.py          # Chess environment
│   │   ├── constrained_decoding.py # 100%-legal UCI LogitsProcessor
│   │   ├── engine.py             # Stockfish helpers & failure accounting
│   │   ├── move_utils.py         # Move/FEN parsing utilities
│   │   ├── rewards.py            # Staged GRPO reward functions (Stages 0-2)
│   │   ├── train_grpo.py         # TRL GRPO training entry point
│   │   └── evaluate.py           # Paired color-balanced Stockfish evaluator
│   ├── training/
│   │   ├── split.py              # Leakage-free FEN-level train/val split
│   │   └── train_sft.py          # SFT training + adapter merging
│   ├── benchmarks/phase0/        # Phase 0 model-selection benchmarks
│   └── scripts/                  # Tokenizer audit & model download helpers
├── docs/phase0/                  # Phase 0 baseline reports
├── data_generation/              # Committed sample data & manifest
└── tests/                        # Unit, integration, and TRL smoke tests
```

---

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [Qwen2.5](https://arxiv.org/abs/2409.12122) — Base model architecture (v0)
- [ChessLLM](https://arxiv.org/abs/2501.17186) — LLM chess via SFT
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — `<think>` reasoning paradigm
- [Phase 0 Baseline Report](docs/phase0/BASELINE_REPORT.md) — Model selection
- [ChessFM Roadmap](chess_fm_roadmap.md) — Implementation plan

---

## License

MIT
