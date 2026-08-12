# ChessFM 🧠♟️

> *A 3B parameter model that plays chess by reasoning.*

[![Status](https://img.shields.io/badge/Status-Complete-green)]()
[![Model](https://img.shields.io/badge/Base-Qwen--2.5--3B-blue)]()
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)]()
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
| **Legal Move Rate** | ≥ 95% (measured by `rl/evaluate.py`) |
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
python data_generation/convert_to_training.py

# 3. Validate data without training (no GPU needed)
python training/train_sft.py --data data_generation/sft_train.jsonl --dry-run

# 4. Train SFT (on a GPU machine)
python training/train_sft.py --data data_generation/sft_train.jsonl

# 5. Train with GRPO stages
python rl/train_grpo.py --stage 0 --steps 2000 --checkpoint-every 500
python rl/train_grpo.py --stage 1 --steps 2000 --checkpoint-every 500

# 6. Evaluate against Stockfish
python rl/evaluate.py --model checkpoints/final --games 50
```

See [`rl/README.md`](rl/README.md) and
[`training/README.md`](training/README.md) for full usage.

---

## 📁 Sample Data

The repo ships a **small sample** of training data for reproducibility:

| File | Contents |
|------|----------|
| `data_generation/all_sft_data.jsonl` | Raw engine-validated samples |
| `data_generation/sft_train.jsonl` | Converted `prompt`/`completion` format (deduped by FEN) |
| `data_generation/positions.txt` | 25,000 elite-game FENs |
| `data_generation/data_manifest.json` | Provenance: counts, sources, timestamp |

The committed samples are enough to verify the full pipeline; the full
training set and final model checkpoints live on the training machine.

---

## 🧪 Testing

```bash
python -m pytest          # 46 unit + integration tests
ruff check .              # lint
```

CI runs both on every push: unit/smoke tests plus a real Stockfish reward
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
├── config.py                 # Central configuration & prompt template
├── pyproject.toml            # Package config, ruff, pytest, coverage
├── chess_fm_roadmap.md       # Implementation plan
├── requirements.txt          # Dependencies
├── setup_env.sh              # Environment setup
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff + formatting)
│
├── data_generation/          # SFT data generation
│   ├── README.md
│   ├── fetch_elite_data.py   # Fetch FENs from Lichess
│   ├── download_positions.py # Curated FEN seed positions
│   ├── convert_to_training.py
│   ├── positions.txt         # 25k elite game FENs (sample)
│   ├── all_sft_data.jsonl    # Raw samples (sample)
│   ├── sft_train.jsonl       # Converted training format
│   └── data_manifest.json    # Provenance manifest
│
├── training/                 # SFT training
│   ├── README.md
│   └── train_sft.py
│
├── rl/                       # Reinforcement learning
│   ├── README.md
│   ├── chess_env.py          # Chess environment
│   ├── rewards.py            # Staged reward functions
│   ├── move_utils.py         # Move/FEN parsing helpers
│   ├── train_grpo.py         # GRPO training (TRL)
│   ├── evaluate.py           # Evaluation harness (legal %, format %, Elo)
│   └── tests/
│
├── benchmarks/               # Phase 0 baselines
│   └── phase0/
│       ├── BASELINE_REPORT.md
│       ├── STRATEGY.md
│       ├── benchmark_models.py
│       └── run_benchmark_mlx.py
│
├── scripts/                  # Utilities
│   ├── audit_tokenizer.py
│   └── download_models.py
│
└── tests/                    # Unit + CLI smoke tests
    ├── test_data_generation.py
    └── test_scripts.py
```

---

## 📄 License

MIT
