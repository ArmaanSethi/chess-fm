# ChessFM

> Engine-grounded chess reasoning models and an offline AI Chess Coach built on SFT, staged GRPO, constrained decoding, and deterministic verification.

[![CI](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArmaanSethi/chess-fm)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Models](https://img.shields.io/badge/Models-Qwen2.5--3B%20%7C%20Qwen3--4B%20%7C%20Gemma--4--E4B-blue)]()

---

## Overview

> **Read the full technical write-up:** [Inside a 4B Chess Reasoning & Coaching Model: Post-Training, Reward Hacking, and Mechanistic Interpretability Under Verified Oracles (`docs/BLOG_POST.md`)](docs/BLOG_POST.md) · **Try the live browser coach:** [armaansethi.github.io/chess-fm](https://armaansethi.github.io/chess-fm/) · **Live Dashboard:** [dashboard.html](https://armaansethi.github.io/chess-fm/dashboard.html)

Most chess engines play by brute-force search without explaining their plans. ChessFM trains compact language models (`Qwen/Qwen3-4B`) to **reason in `<think>...</think>` tags before committing to a legal UCI move**, pairs that foundation with a zero-slop **AI Chess Coach (`92.0%` All-Clean, `100%` spoiler-free, `2.50 GB` Q4_K_M GGUF)** that outperforms its `6.5×` larger `Gemma-4-26B` teacher (`88.7%`), and opens the weights with a **4-part Mechanistic Interpretability suite (`chessfm.interp`)**:

```xml
<think>
Black's queen attacks f7. Castling immediately drops material, so developing
with Nf6 defends the square while contesting the center.
</think>
g8f6
```

All training harnesses, reward math (`wdl_loss`, anti-hack `sharp_wdl`, Stockfish-free `coach_v2`), constrained/repetition-aware decoding, deterministic coaching engines, 4-tier evaluation suites, mechanistic interpretability tools, and cloud/local deployment scripts are verified end-to-end in CI.

---

## Verified Highlights (`0.00%` Board-Overlap Test Suites)

- **Board Representation & *LoRA Without Regret* (`E0 → E2a vs E2b`):** Piece-by-Square coordinates + All-Linear LoRA `lr=2e-4` tripled strict move legality (`21.0% → 63.5%`); at `20K` mixed rows (`7M` tokens), All-Linear `r=64` (`E2a`: `66.2%` legal, `21.4%` 900-Puzzles, `45.2%` 1K-ChessQA, `90%` English / `0%` leak) beat 100%-parameter Full-FT (`E2b`) on all 7 suites.
- **Reward Hacking Diagnosed & Fixed (`E3a → E3b`):** Caught naive WDL-DAPO collapsing to `60.3%` `a`-file pawn pushes (`unique_moves: 528 → 206`); quadratic `sharp_wdl` restored diversity (`443` moves, `1.5%` a-pawns), lifted strict legality **`+10.3 pp` (`76.5%`)**, MAR **`+3.0 pp`**, and **emergent CoT V2 faithfulness `+10.0 pp` (`70.7%`) with zero CoT reward**, while `--avoid-repetition` exposed a `+231` Elo stateless draw mirage and revealed `+27.3` Elo clean RLVR gain (plus wins vs. `Maia-1100`).
- **Zero-Slop `4B` AI Chess Coach (`E5`):** Strict Verifier V2 + deterministic tactical facts + Stockfish-free Coach-GRPO took `Qwen3-4B` from `58.0% → 92.0%` All-Clean (`100%` L1/L2 spoiler-free, `0%` slop), beating both `Gemma-4-26B` Teacher (`88.7%`) and Weak `V1-GRPO` (`89.3%`); Play→Coach transfer (`Arm B`: `E3b → Coach`) reached `91.0%` All-Clean with `+4.5 pp` deeper tactical fact coverage (`81.7%`).
- **Mechanistic Interpretability (`INTERP #1–#4`):** Pinpointed a **`0.43%`-of-heads Piece-Tracker circuit in Layers 26–29** (`L27.H17`, `L29.H9`, `L26.H20/24`, `L28.H2`, **`3.0×` causal ablation drop** `-12.0 pp` vs. `-4.0 pp` layer-matched random) driving a **Layer 30–35 board rebound**, proved the A-Pawn Hack is a **low-rank `L34–L35` output shortcut** (`eff_rank: 12.8 → 11.4`, peak `L35`) vs. clean RLVR (`peak L17`), causally dialed boilerplate slop `0% → 36%` via the linear `Arm C − Arm A` Goodhart vector, and proved `<think>` is `100%` causally load-bearing (`100/100` sabotage flips).

---

## Two-Track Architecture

> See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for the full Architecture Decision Record (`ADR-001`–`ADR-008`) and [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for reproducible multi-machine setup and GPU workflows.

### Track 1 — Research & RL (`v0` / `v1` / `v1.5`)

| Track | Model | Role |
|-------|-------|------|
| **v0 (Legacy Control)** | `Qwen/Qwen2.5-3B-Instruct` | 3B baseline selected in [Phase 0 benchmarks](docs/phase0/BASELINE_REPORT.md) |
| **v1 (Fast RL Backbone)** | `Qwen/Qwen3-4B` | 4B text-only reasoning backbone with native `<think>` alignment |
| **v1.5 (Hybrid Reasoning)** | `Qwen/Qwen3.5-4B` | 4B hybrid Gated DeltaNet reasoning backbone (`transformers>=5.2`) |

Key capabilities:
- **Model-Agnostic Prompting & Board Representation**: Single source of truth in `src/chessfm/config.py` (`build_chat_prompt`) using `tokenizer.apply_chat_template` with ChatML fallback and optional ASCII+FEN board rendering (`--include-ascii`).
- **Best-Line SFT & Leakage-Free Splits**: Completion-only loss (`train_sft.py`), FEN-level splitting (`training/split.py`), phase-stratified sampling across opening/middlegame/endgame (`data_generation/phase_sampler.py`), and deterministic `$0`-API best-line trace construction (`data_generation/coaching_dataset.py`).
- **Merged SFT → Staged GRPO**: LoRA adapters merge cleanly (`--merge-and-save`) into a standalone checkpoint before staged GRPO (`rl/train_grpo.py`):
  - **Stage 0** — Legality (`+1` legal / `-1` illegal) + `<think>` format and length-budget shaping (Stockfish-free).
  - **Stage 1** — Adds Stockfish move quality relative to the engine's best move.
  - **Stage 2** — Adds opponent-reply delta (evaluation after Stockfish's best counterplay).
  - Engine errors are isolated as neutral infrastructure failures and abort training if the failure rate exceeds 5%.
- **100%-Legal Constrained Decoding**: Trie-based UCI `LogitsProcessor` (`rl/constrained_decoding.py`) masks illegal move continuations after `</think>` at inference time without altering pre-move reasoning.

### Track 2 — AI Chess Coach (`v2` & Distilled Compact Students)

| Role | Model (Apache-2.0, Ungated) | Purpose |
|------|-----------------------------|---------|
| **v2 (Workstation Coach)** | `unsloth/gemma-4-E4B-it` / `google/gemma-4-E4B-it` | Local / Ollama multimodal chess coaching |
| **Browser Student** | `Qwen/Qwen3.5-0.8B` | Distilled compact WebGPU narration model |
| **Local Student** | `google/gemma-4-E2B-it` | Compact local / Ollama narration model |
| **Offline Prose Writer** | `google/gemma-4-26B-A4B-it` / `Qwen/Qwen3.6-35B-A3B` | Zero-API vLLM data writer filtered 100% by `ACT-Eval` |

Key capabilities:
- **Stockfish MultiPV Partial Strategy Trees & WDL Classification** (`coach/classifier.py`): Evaluates candidate lines, computes win-probability deltas, and classifies moves (`brilliant`, `best`, `good`, `inaccuracy`, `mistake`, `blunder`).
- **Maia-2 Human-Elo Calibration (600–2600)** (`coach/maia_sidecar.py`): Estimates human move probabilities at the student's target Elo and flags practical traps where engine-best lines diverge from likely human replies.
- **3-Level Socratic Hints with 0% Spoilers** (`coach/socratic.py`): Progressive coaching hints (Level 1 strategic theme → Level 2 target piece/square → Level 3 tactical mechanism) verified by `contains_move_spoiler` so answers are never leaked.
- **ACT-Eval Deterministic Claim Verifier** (`coach/verifier.py`): Audits natural-language coaching claims (checks, captures, hanging pieces, legal moves, tactical motifs) directly against `python-chess` board ground truth.
- **Full-Game PGN Review, Openings & Spaced-Repetition Drills** (`coach/game_review.py`, `coach/openings.py`, `coach/drills.py`, `coach/importers.py`): Analyzes PGNs and public Lichess/Chess.com games, detects ECO openings/traps, extracts critical turning points, and schedules SM-2 practice drills.
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

## Cloud & Local Workflows

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for full multi-machine setup, environment variables, and end-to-end training commands.

### Modal GPU Runner — `scripts/modal_train.py`

```bash
pip install modal && modal setup

# End-to-end smoke pipeline: merged SFT -> GRPO Stage 0 -> Stockfish eval
modal run scripts/modal_train.py --task smoke-pipeline --track v1 --include-ascii

# Individual stages (v0, v1, v1.5, v2/coach, student, student-e2b, teacher)
modal run scripts/modal_train.py --task sft --track v1 --epochs 3 --include-ascii
modal run scripts/modal_train.py --task sft --track v1.5 --epochs 3 --include-ascii
modal run scripts/modal_train.py --task grpo --track v1 --stage 0 --steps 500 --include-ascii
modal run scripts/modal_train.py --task distill-teacher --track teacher --limit 1000
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

No fabricated Elo ratings or uncommitted GPU metrics are claimed in this repository. Verified local Stockfish-16 baselines and CPU micro-experiments are documented in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Model checkpoints (`checkpoints/`), raw evaluation outputs (`results/`), and large training corpora are gitignored and regenerated via the commands below.

| Verification / Artifact | Command | Verified Result / Output |
|-------------------------|---------|--------------------------|
| Lint & formatting | `venv/bin/ruff check . && venv/bin/ruff format --check .` | Clean pass in CI |
| Unit & integration suite | `venv/bin/python -m pytest --cov=chessfm --cov-report=term-missing` | 843 passing tests, 93.81% coverage (CI gate ≥80%) |
| TRL CPU smoke test | `venv/bin/python -m pytest tests/test_training_smoke.py` | 1 real SFT step + 1 real GRPO step (`sshleifer/tiny-gpt2`) |
| 4-Tier Eval Suite | `python -m chessfm.eval.suite --demo` | Structured JSON report across Tiers 1–4 |
| Open-weight distiller dry-run | `python -m chessfm.data_generation.distill_open_teacher --dry-run --limit 5` | Validates engine-fact schema and teacher prompts (no GPU) |
| Data conversion & manifest | `python -m chessfm.data_generation.convert_to_training` | `data_generation/sft_train.jsonl` + `data_manifest.json` |
| SFT dry-run validation | `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run` | Schema & split validation (no GPU) |
| SFT training (GPU) | `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --merge-and-save` | LoRA + merged checkpoint under `checkpoints/` |
| GRPO training (GPU) | `python -m chessfm.rl.train_grpo --stage 0 --steps 2000 --checkpoint-every 500` | RL checkpoint under `checkpoints/` |
| Stockfish match eval (GPU) | `python -m chessfm.rl.evaluate --model <checkpoint> --games 50` | `results/eval_<timestamp>.json` + `results/pgns/eval_<timestamp>.pgn` |

Each evaluation run records reproducibility metadata (git revision, seed, opponent level, fixed Stockfish search depth, and opening set) and alternates White/Black colors by default (`--no-alternate-colors` to pin a side).

---

## Committed Sample Data

The repository includes small Stockfish-16-verified sample datasets for deterministic pipeline verification; full corpora are generated on demand:

| File | Contents |
|------|----------|
| `data_generation/all_sft_data.jsonl` | Raw engine-validated samples (85 rows) |
| `data_generation/sft_train.jsonl` | Cleaned `prompt`/`completion` dataset (40 rows, deduped by FEN, >200 cp blunders removed) |
| `data_generation/coaching_sample.jsonl` | Phase-balanced coaching sample (240 rows, 0 claim violations, 0 Socratic L1/L2 spoilers) |
| `data_generation/eval_benchmark.jsonl` | Phase-balanced benchmark FENs (40 rows, Stockfish depth-10 ground truth) |
| `data_generation/positions.txt` | 25,000 elite-game FENs (position pool) |
| `data_generation/data_manifest.json` | Provenance manifest (counts, sources, timestamp) |

---

## Project Structure

```
chess-fm/
├── README.md                     # Project overview and documentation
├── AGENTS.md                     # Developer and agent operational guide
├── pyproject.toml                # Packaging, console scripts, ruff, pytest, coverage
├── setup_env.sh                  # Local venv + Stockfish setup
├── docs/
│   ├── BLOG_POST.md              # Full technical write-up (RLVR, Goodhart fixes, Mech-Interp)
│   ├── BENCHMARKS.md             # Verified local & GPU benchmarks (Phases I-IV.5)
│   ├── EXPERIMENT_LOG.md         # Complete run ledger, W&B links, and failure autopsies
│   ├── ARCHITECTURE.md           # Module and dataflow architecture
│   ├── DESIGN_DECISIONS.md       # Architecture Decision Record (ADR-001 to ADR-008)
│   ├── TRAINING_GUIDE.md         # Multi-machine setup and GPU training workflows
│   ├── EXPERIMENTS.md            # Reproducible CPU/Stockfish micro-experiments
│   └── phase0/                   # Phase 0 baseline reports
├── scripts/                      # Cloud & local deployment runners
│   ├── modal_train.py            # Dual-image Modal GPU runner (RL, distillation, eval, interp, export)
│   ├── run_interp_suite.py       # 4-part Mechanistic Interpretability A100 suite runner
│   ├── kaggle_train.py           # Resumable Kaggle 2x T4 runner
│   └── export_gguf.py            # GGUF conversion + Ollama Modelfile generator
├── src/chessfm/
│   ├── config.py                 # Central config, model tracks, chat/ASCII/Piece-List prompts
│   ├── fens.py                   # Shared FEN/JSONL readers
│   ├── load_model.py             # Quantized & PEFT model loading
│   ├── interp/                   # Mechanistic Interpretability library
│   │   ├── probes.py             # 64-square linear board probes & orthogonal random controls
│   │   ├── lora_svd.py           # Exact QR-SVD per-layer & Attn-vs-MLP weight diffing
│   │   └── circuits.py           # Logit lens, <think> sabotage, BPE Piece-Tracker head circuits
│   ├── coach/                    # AI Chess Coach stack
│   │   ├── classifier.py         # MultiPV Partial Strategy Trees & WDL classification
│   │   ├── maia_sidecar.py       # Maia-2 human-Elo (600-2600) calibration & traps
│   │   ├── socratic.py           # 3-level Socratic hints + 0%-spoiler verifier
│   │   ├── tactical_facts.py     # Deterministic 11-pattern python-chess tactical extractor
│   │   ├── coach_rewards.py      # Stockfish-free Strict V2 & V1 Coach-GRPO rewards
│   │   ├── verifier.py           # ACT-Eval Strict V2 claim & boilerplate-slop verifier
│   │   ├── game_review.py        # Full-game PGN review & weakness profiler
│   │   ├── openings.py           # ECO opening & trap classifier
│   │   ├── drills.py             # SM-2 spaced-repetition drill engine
│   │   ├── importers.py          # Lichess & Chess.com public game importers
│   │   ├── multimodal.py         # CoachOrchestrator Q&A pipeline
│   │   ├── server.py             # Offline Coach Studio HTTP server & REST API
│   │   └── static/index.html     # Interactive Coach Studio UI
│   ├── data_generation/
│   │   ├── coaching_dataset.py   # Motif detection & $0-API best-line/coaching builder
│   │   ├── distill_open_teacher.py # Zero-API open-weight rejection-sampled distiller
│   │   ├── phase_sampler.py      # Opening/middlegame/endgame stratified sampling
│   │   ├── build_corpus.py       # Stockfish-verified sample corpus builder
│   │   ├── convert_to_training.py
│   │   ├── generate_synthetic_data.py
│   │   ├── download_positions.py
│   │   └── fetch_elite_data.py
│   ├── eval/
│   │   ├── suite.py              # 4-Tier Evaluation Suite (Static, ChessQA, ACT-Eval, Match)
│   │   ├── bradley_terry.py      # Paired 5-anchor Bradley-Terry Elo + bootstrap CI
│   │   ├── pass_at_k.py          # Stratified pass@1..pass@k + move entropy evaluator
│   │   └── forgetting.py         # 100-item general-English catastrophic-forgetting probe
│   ├── rl/
│   │   ├── chess_env.py          # Chess environment
│   │   ├── constrained_decoding.py # 100%-legal & repetition-aware UCI LogitsProcessor
│   │   ├── engine.py             # Stockfish helpers & failure accounting
│   │   ├── move_utils.py         # Move/FEN parsing utilities
│   │   ├── rewards.py            # Staged GRPO, WDL, and anti-hack sharp_wdl rewards
│   │   ├── train_grpo.py         # TRL GRPO/DAPO training entry point
│   │   └── evaluate.py           # Batched paired color-balanced evaluator
│   ├── training/
│   │   ├── manifest.py           # Reproducible run_manifest.json writer
│   │   ├── split.py              # Leakage-free FEN-level train/val split
│   │   └── train_sft.py          # SFT training + adapter merging
│   ├── benchmarks/phase0/        # Phase 0 model-selection benchmarks
│   └── scripts/                  # Tokenizer audit & model download helpers
├── web/                          # Zero-server GitHub Pages Coach App & Dashboard (Layer A)
├── data_generation/              # Committed sample data & manifest
└── tests/                        # Unit, integration, interp, and TRL smoke tests
```

---

## Documentation & References

- [Technical Blog Post (`docs/BLOG_POST.md`)](docs/BLOG_POST.md) — Post-Training, Reward Hacking, and Mechanistic Interpretability Under Verified Oracles
- [Verified Benchmarks (`docs/BENCHMARKS.md`)](docs/BENCHMARKS.md) — Phases I–IV.5 Local & GPU Tables
- [Experiment Log (`docs/EXPERIMENT_LOG.md`)](docs/EXPERIMENT_LOG.md) — Complete Run Ledger & Failure Autopsies
- [Design Decisions (ADR-001 to ADR-008)](docs/DESIGN_DECISIONS.md)
- [Reproducible Setup & Training Guide](docs/TRAINING_GUIDE.md)
- [CPU & Stockfish-16 Micro-Experiments](docs/EXPERIMENTS.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Phase 0 Baseline Report](docs/phase0/BASELINE_REPORT.md)
- [GRPO Paper](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — `<think>` reasoning paradigm
- [ChessLLM](https://arxiv.org/abs/2501.17186) — LLM chess via SFT

---

## License

MIT
