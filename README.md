# ChessFM

> A 3B parameter model that plays chess by reasoning.

[![CI](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmaanSethi/chess-fm/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArmaanSethi/chess-fm)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Base Model](https://img.shields.io/badge/Base-Qwen--2.5--3B--Instruct-blue)]()

---

## The Idea

Most chess bots play by brute-force search. ChessFM plays by **thinking out loud**:

```xml
<think>
The opponent's queen threatens my f7 pawn.
If I castle now, I lose material.
Better to block with Nf6 first.
</think>
Nf6
```

The model produces short chess explanations alongside its moves — a policy that
reasons about a position before committing to a move, fine-tuned on
engine-validated reasoning traces and then refined with reinforcement learning
against Stockfish-derived rewards.

---

## Status

**The full pipeline is implemented and verified end-to-end at the harness
level** (data generation → conversion → SFT → GRPO → evaluation), with unit +
integration tests and a coverage gate in CI. What is and is not in this repo:

| Component | Status | Evidence |
|-----------|--------|----------|
| Data generation (FEN sourcing, teacher-LLM traces, Stockfish validation, conversion) | Done | `data_generation/` + unit tests |
| SFT training harness | Done | `chessfm.training.train_sft` (+ config-wiring tests) |
| GRPO training harness (staged rewards) | Done | `chessfm.rl.train_grpo` (+ config-wiring tests) |
| Evaluation harness vs Stockfish/random | Done | `chessfm.rl.evaluate`, writes PGNs + JSON to `results/` |
| Trained checkpoints | **Not committed** | Trained on a dedicated GPU machine (24 GB VRAM) |
| Evaluation results | **Not committed** | Regenerate with the evaluator, or upload the JSON/PGN reports |

"Done" describes the *harness*: the data pipeline, reward math, and training
and evaluation entry points are implemented and tested. **Training runs**
(producing the actual checkpoints and eval numbers) execute on a GPU machine
and are reproducible with the commands under [Reproducibility](#reproducibility).

Nothing in the README claims a number this repo cannot produce. Every metric is
either backed by a committed artifact or by a command that generates it (see
[Reproducibility](#reproducibility)).

---

## Approach

### Phase 1: SFT Bootstrap
Train on engine-validated reasoning traces (see
[`data_generation/`](data_generation/README.md)) to teach the model chess
fundamentals and the `<think>` format. Loss is computed on completion tokens
only, and the train/validation split is done at the FEN level so no position
leaks between sets.

### Phase 2: Direct GRPO (Reinforcement Learning)
Train on chess positions using verifiable rewards. Stockfish provides the
reward signal across stages:
- **Stage 0** — Legality only (`+1` legal / `-1` illegal) + format/budget bonuses
- **Stage 1** — Adds Stockfish **move quality** (how close the chosen move is to Stockfish's best)
- **Stage 2** — Adds an **opponent-reply delta**: the evaluation after Stockfish's best response to the chosen move. This measures whether a move survives best counterplay, not just how it compares to the pre-move best.

Engine failures are treated as infrastructure errors, not as bad moves: they
are neutral in the reward, counted per run, and training aborts if the failure
rate exceeds 5% — a dying Stockfish process can never silently poison the
training signal. Stages 1 and 2 require Stockfish and fail fast if it is
unavailable.

### Phase 3: Curriculum Learning *(planned)*

Progressive difficulty: Random → Stockfish L1 → Stockfish L3. The curriculum
stages are tracked in [`chess_fm_roadmap.md`](chess_fm_roadmap.md); the staged
GRPO harness this repo ships is what each curriculum level will run against.

---

## Reproducibility

Every headline measurement has a command that produces it.

| What | Command | Artifact produced |
|------|---------|-------------------|
| Lint | `ruff check .` | Pass/fail in CI |
| Unit + integration tests | `python -m pytest` | Test report in CI |
| Coverage gate | `python -m pytest --cov=chessfm` | Coverage report (≥80% required) |
| Data pipeline (dedupe, convert, manifest) | `python -m chessfm.data_generation.convert_to_training` | `data_generation/sft_train.jsonl` + `data_manifest.json` |
| Data validation (no GPU) | `python -m chessfm.training.train_sft --dry-run` | Pass/fail |
| Full data regeneration | `python -m chessfm.data_generation.generate_synthetic_data --provider openai --model gpt-4o --positions data_generation/positions.txt` | JSONL training corpus (gitignored) |
| SFT training | `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl` | LoRA checkpoint under `checkpoints/` |
| GRPO training | `python -m chessfm.rl.train_grpo --stage 0 --steps 2000` | Checkpoint under `checkpoints/` |
| Evaluation vs Stockfish | `python -m chessfm.rl.evaluate --model <checkpoint> --games 50` | `results/eval_<timestamp>.json` + `results/pgns/eval_<timestamp>.pgn` |

Each eval run writes a JSON report with reproducibility metadata (git revision,
seed, opponent level, fixed opponent search depth, opening set) and full PGNs
of the games. The Stockfish opponent searches to a **fixed depth** (not
wall-clock time) so its strength is reproducible across machines. Games
alternate model colors by default so win/loss stats are not color-biased
(`--no-alternate-colors` to pin one color). **Model metrics (e.g. legal-move
rate, W-D-L) are reported in the eval reports — this repo commits the harness
that measures them, not the measurements themselves, because the checkpoints
are trained and evaluated on a GPU machine.**

> Note: `legal rate`, `format adherence`, and W-D-L are the primary metrics.
> The "Elo" printed alongside them is a *rough indicator only*: it comes from a
> small game sample against a Stockfish skill level whose Elo mapping is an
> informal community calibration, not a validated rating.

---

## Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Base Model** | [Qwen-2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Selected in Phase 0 benchmarks |
| **Training** | [unsloth](https://github.com/unslothai/unsloth) / [trl](https://github.com/huggingface/trl) | Efficient SFT and GRPO training |
| **Chess Engine** | [Stockfish](https://stockfishchess.org/) | Reward signal + validation |
| **Hardware** | RTX 4090 (RunPod) | 24 GB VRAM |

---

## Getting Started

```bash
# 1. Set up the environment (installs Stockfish + dependencies)
./setup_env.sh

# 2. Verify the data pipeline (dedupes by FEN, writes a manifest)
python -m chessfm.data_generation.convert_to_training

# 3. Validate data without training (no GPU needed)
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run

# 4. Train SFT (on a GPU machine)
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl

# 5. Train with GRPO stages (on a GPU machine)
python -m chessfm.rl.train_grpo --stage 0 --steps 2000 --checkpoint-every 500
python -m chessfm.rl.train_grpo --stage 1 --steps 2000 --checkpoint-every 500

# 6. Evaluate against Stockfish
python -m chessfm.rl.evaluate --model checkpoints/final --games 50
```

All modules also ship as console scripts (`chessfm-train-sft`, `chessfm-evaluate`,
etc.) once installed with `pip install -e .`.

---

## Data

The repo ships a **small committed sample** for pipeline verification. The full
training corpus is regenerated on demand and intentionally not committed
(size and provenance).

| File | Contents |
|------|----------|
| `data_generation/all_sft_data.jsonl` | Raw engine-validated samples (85 rows, no unvalidated/illegal traces) |
| `data_generation/sft_train.jsonl` | Converted `prompt`/`completion` format (44 rows, deduped by FEN) |
| `data_generation/positions.txt` | 25,000 elite-game FENs (position pool) |
| `data_generation/data_manifest.json` | Provenance: counts, sources, timestamp |

The committed samples are enough to verify the full pipeline; the full training
set and final checkpoints live on the training machine. Regenerate the full
corpus with `python -m chessfm.data_generation.generate_synthetic_data`.

---

## Testing

```bash
python -m pytest              # unit + integration tests
python -m pytest --cov=chessfm # coverage gate (≥80% required)
ruff check .                  # lint
```

CI runs these on every push: unit/smoke tests, a real Stockfish reward
integration test, and a coverage gate. Coverage is measured across the whole
package; the only modules excluded are ones whose *behavior* requires GPU
hardware, model downloads, or a network endpoint to exercise — their entry
points are smoke-tested via `--help` and their pure helpers are unit-tested
(see `[tool.coverage.run]` in `pyproject.toml`).

On top of that, `tests/test_training_smoke.py` drives **one real SFT step and
one real GRPO step** through TRL on CPU with `sshleifer/tiny-gpt2`, so trainer
wiring and TRL API drift are verified in CI rather than discovered on the GPU
machine. These tests skip gracefully when torch/trl are unavailable or the
model hub is unreachable.

---

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) — RL algorithm
- [Qwen2.5](https://arxiv.org/abs/2409.12122) — Base model architecture
- [ChessLLM](https://arxiv.org/abs/2501.17186) — LLM chess via SFT
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Reasoning patterns
- [Phase 0 Baseline Report](docs/phase0/BASELINE_REPORT.md) — model selection
- [ChessFM Roadmap](chess_fm_roadmap.md) — implementation plan

---

## Project Structure

```
chess-fm/
├── README.md                 # This file
├── pyproject.toml            # Package config, deps, ruff, pytest, coverage
├── chess_fm_roadmap.md       # Implementation plan
├── setup_env.sh              # Environment setup
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff)
│
├── src/chessfm/              # Package (install with `pip install -e .`)
│   ├── config.py             # Central configuration & prompt template
│   ├── fens.py               # Shared FEN/JSONL file readers
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
│   │   └── train_sft.py      # SFT training (TRL, unsloth-accelerated)
│   │
│   ├── benchmarks/
│   │   └── phase0/           # Phase 0 baselines (Qwen-3B selection)
│   │       ├── benchmark_models.py
│   │       └── run_benchmark_mlx.py
│   │
│   ├── scripts/              # Utilities
│   │   ├── audit_tokenizer.py
│   │   └── download_models.py
│   │
│   └── load_model.py         # Model loading
│
├── docs/
│   └── phase0/               # Phase 0 model-selection reports
│       ├── BASELINE_REPORT.md
│       └── STRATEGY.md
│
├── data_generation/          # Committed sample data
│   ├── README.md
│   ├── positions.txt         # 25k elite game FENs (position pool)
│   ├── all_sft_data.jsonl    # Raw samples (sample)
│   ├── sft_train.jsonl       # Converted training format (44 positions)
│   └── data_manifest.json    # Provenance manifest
│
└── tests/                    # Unit + integration tests
    ├── test_chess_env.py
    ├── test_config.py
    ├── test_convert_utils.py
    ├── test_data_generation.py
    ├── test_data_pipeline_cli.py
    ├── test_engine_helpers.py
    ├── test_evaluate_integration.py
    ├── test_fens.py
    ├── test_generate_utils.py
    ├── test_move_utils.py
    ├── test_rewards.py
    ├── test_scripts.py
    ├── test_split.py
    ├── test_train_grpo.py
    └── test_trainer_configs.py
```


---

## License

MIT
