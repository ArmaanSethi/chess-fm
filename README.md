# ChessFM 🧠♟️

> *A 3B parameter model that plays chess by reasoning.*

[![Status](https://img.shields.io/badge/Status-Complete-green)]()
[![Model](https://img.shields.io/badge/Base-Qwen--2.5--3B-blue)]()
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-93%25-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 💡 The Idea

Most chess bots play by brute-force search. ChessFM plays by **thinking out loud**:

```xml
<think>
The opponent's queen threatens my f7 pawn.
If I castle now, I lose material.
Better to block with Nf6 first.
</think>
Nf6
```

The model produces short chess explanations alongside its moves — a trained
policy that reasons about a position before committing to a move, fine-tuned
on high-quality engine-validated traces and then refined with reinforcement
learning against Stockfish-derived rewards.

---

## 🎯 Results

The pipeline is complete end-to-end, from data generation through staged
reinforcement learning to Stockfish-benchmarked evaluation. Final checkpoints
were trained on a dedicated GPU machine; the full measurement harness is
included here.

| Metric | Status |
|--------|--------|
| **Legal Move Rate** | ≥ 95% (measured by `chessfm.rl.evaluate`) |
| **Reasoning Format** | > 95% valid `<think>` tags |
| **Elo vs Stockfish L1** | Verified via the included evaluator |

> Evaluations are reproducible: every run writes a JSON report (with git
> revision, seed, opponent level, and opening set) plus full PGNs of the
> games under `results/`.

---

## 🔬 Approach

### Phase 1: SFT Bootstrap
Train on engine-validated reasoning traces (see
[`data_generation/`](data_generation/README.md)) to teach the model chess
fundamentals and the `<think>` format. Loss is computed on completion tokens
only, and the train/validation split is done at the FEN level so no position
leaks between them.

### Phase 2: Direct GRPO (Reinforcement Learning)
Train on chess positions using verifiable rewards. Stockfish provides the
reward signal across stages:
- **Stage 0** — Legality only (`+1` legal / `-1` illegal) + format/budget bonuses
- **Stage 1** — Adds Stockfish **move quality** (how close the chosen move is to Stockfish's best)
- **Stage 2** — Adds Stockfish **eval delta** (dense per-move signal)

Stages 1 and 2 fail fast if Stockfish is unavailable — the reward objective is
never silently downgraded.

### Phase 3: Curriculum Learning
Progressive difficulty: Random → Stockfish L1 → Stockfish L3.

---

## 🛠️ Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Base Model** | [Qwen-2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Best format adherence in benchmarks |
| **Training** | [unsloth](https://github.com/unslothai/unsloth) / [trl](https://github.com/huggingface/trl) | Efficient SFT and GRPO training |
| **Chess Engine** | [Stockfish](https://stockfishchess.org/) | Reward signal + validation |
| **Hardware** | RTX 4090 (RunPod) | 24GB VRAM |

---

## 📊 Training Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   FEN Position                                             │
│        ↓                                                   │
│   ┌─────────────┐    ┌──────────────────┐   ┌───────────┐  │
│   │  SFT on     │ →  │  GRPO (Stage 0-2)│ → │  Eval     │  │
│   │  reasoning  │    │  Stockfish       │   │  vs       │  │
│   │  traces     │    │  reward signal   │   │  Stockfish│  │
│   │ (sample set)│    │                  │   │ (PGNs+JSON)│  │
│   └─────────────┘    └──────────────────┘   └───────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

```bash
# 1. Set up the environment (installs Stockfish + dependencies)
./setup_env.sh

# 2. Verify the data pipeline (dedupes by FEN, writes a manifest)
python -m chessfm.data_generation.convert_to_training

# 3. Validate data without training (no GPU needed)
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run

# 4. Train SFT (on a GPU machine)
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl

# 5. Train with GRPO stages
python -m chessfm.rl.train_grpo --stage 0 --steps 2000 --checkpoint-every 500
python -m chessfm.rl.train_grpo --stage 1 --steps 2000 --checkpoint-every 500

# 6. Evaluate against Stockfish
python -m chessfm.rl.evaluate --model checkpoints/final --games 50
```

All modules also ship as console scripts (`chessfm-train-sft`, `chessfm-evaluate`,
etc.) once installed with `pip install -e .`.

---

## 📁 Sample Data

The repo ships a **small sample** of training data for reproducibility:

| File | Contents |
|------|----------|
| `data_generation/all_sft_data.jsonl` | Raw engine-validated samples |
| `data_generation/sft_train.jsonl` | Converted `prompt`/`completion` format (deduped by FEN) |
| `data_generation/positions.txt` | 25,000 elite-game FENs |
| `data_generation/data_manifest.json` | Provenance: counts, sources, timestamp |

The committed samples are enough to verify the full pipeline (44 converted
samples, one per position); the full training set and final model checkpoints
live on the training machine.

---

## 🧪 Testing

```bash
python -m pytest                    # 112 unit + integration tests
python -m pytest --cov=chessfm      # coverage gate (≥ 80% on core modules)
ruff check .                        # lint
```

CI runs these on every push: unit/smoke tests plus a real Stockfish reward
integration test.

---

## 📚 References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) — RL algorithm
- [Qwen2.5](https://arxiv.org/abs/2409.12122) — Base model architecture
- [ChessLLM](https://arxiv.org/abs/2501.17186) — LLM chess via SFT
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Reasoning patterns

---

## 📋 Roadmap

See the full [ChessFM Roadmap](chess_fm_roadmap.md) for implementation detail.

---

## 🗂️ Project Structure

```
chess-fm/
├── README.md                 # This file
├── pyproject.toml            # Package config, ruff, pytest, coverage
├── chess_fm_roadmap.md       # Implementation plan
├── requirements.txt          # Dependencies
├── setup_env.sh              # Environment setup
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff)
│
├── src/chessfm/              # Package (install with `pip install -e .`)
│   ├── config.py             # Central configuration & prompt template
│   │
│   ├── data_generation/      # SFT data generation
│   │   ├── fetch_elite_data.py
│   │   ├── download_positions.py
│   │   ├── convert_to_training.py
│   │   └── generate_synthetic_data.py
│   │
│   ├── rl/                   # Reinforcement learning
│   │   ├── chess_env.py      # Chess environment
│   │   ├── rewards.py        # Staged reward functions
│   │   ├── engine.py         # Shared Stockfish helpers
│   │   ├── move_utils.py     # Move/FEN parsing helpers
│   │   ├── train_grpo.py     # GRPO training (TRL)
│   │   └── evaluate.py       # Evaluation harness (legal %, format %, Elo)
│   │
│   ├── training/
│   │   └── train_sft.py      # SFT training (unsloth/TRL)
│   │
│   ├── benchmarks/
│   │   └── phase0/           # Phase 0 baselines (Qwen-3B selection)
│   │       ├── BASELINE_REPORT.md
│   │       ├── STRATEGY.md
│   │       ├── benchmark_models.py
│   │       └── run_benchmark_mlx.py
│   │
│   ├── scripts/              # Utilities
│   │   ├── audit_tokenizer.py
│   │   └── download_models.py
│   │
│   └── load_model.py         # Model loading
│
├── data_generation/          # Committed sample data
│   ├── README.md
│   ├── positions.txt         # 25k elite game FENs (sample)
│   ├── all_sft_data.jsonl    # Raw samples (sample)
│   ├── sft_train.jsonl       # Converted training format (44 positions)
│   └── data_manifest.json    # Provenance manifest
│
└── tests/                    # Unit + integration tests (112 passing)
    ├── test_chess_env.py
    ├── test_rewards.py
    ├── test_convert_utils.py
    ├── test_data_generation.py
    ├── test_engine_helpers.py
    ├── test_generate_utils.py
    ├── test_data_pipeline_cli.py
    ├── test_evaluate_integration.py
    ├── test_move_utils.py
    ├── test_config.py
    └── test_scripts.py
```

---

## 📄 License

MIT
