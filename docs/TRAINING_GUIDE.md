# ChessFM — Reproducible Setup & Training Guide

This guide covers cloning ChessFM on a fresh machine, verifying the local CPU/Stockfish pipeline, configuring cloud GPU credentials safely via environment variables, and running both Part 1 (RL Chess Reasoning) and Part 2 (Open-Weight Coaching Distillation).

---

## 1. Local Environment Setup & Verification

```bash
git clone https://github.com/ArmaanSethi/chess-fm.git
cd chess-fm
./setup_env.sh
source venv/bin/activate
```

Run the full lint, formatting, unit/integration test suite, and local baseline checks:

```bash
venv/bin/ruff check .
venv/bin/ruff format --check .
venv/bin/python -m pytest --cov=chessfm --cov-report=term-missing
venv/bin/python -m chessfm.eval.suite --demo
```

Reproduce the verified local benchmarks and CPU micro-experiments:

```bash
venv/bin/python scripts/run_local_benchmarks.py --benchmark data_generation/eval_benchmark.jsonl --depth 6 --match-games 10
venv/bin/python scripts/run_cpu_experiments.py --positions 20 --depth 6
```

---

## 2. Cloud GPU Credentials (Environment Variables Only)

Never commit credentials or tokens to git. ChessFM reads optional runtime credentials strictly from your local shell environment and forwards them to Modal containers at runtime:

```bash
# 1. Modal CLI (one-time authentication)
pip install modal
modal setup

# 2. Optional: Hugging Face token (for uploading checkpoints/datasets to HF Hub)
# Note: Qwen3, Qwen3.5, Qwen3.6, and Gemma 4 weights are Apache-2.0 and ungated.
export HF_TOKEN="hf_..."

# 3. Optional: Weights & Biases key (for training loss / GRPO reward logging)
export WANDB_API_KEY="..."
```

---

## 3. Smoke Test (Verify GPU Stack End-to-End First)

Run a small end-to-end check (SFT merge-and-save -> GRPO Stage 0 -> Stockfish eval) before launching multi-hour jobs:

```bash
modal run scripts/modal_train.py \
  --task smoke-pipeline \
  --track v1 \
  --epochs 1 \
  --steps 20 \
  --games 4 \
  --include-ascii
```

Available `--track` values in [scripts/modal_train.py](../scripts/modal_train.py):

| Track | Model ID | Container Image | Primary Role |
|---|---|---|---|
| `v0` | `Qwen/Qwen2.5-3B-Instruct` | Stable (TRL 0.16) | Legacy control baseline |
| `v1` | `Qwen/Qwen3-4B` | Stable (TRL 0.16) | Part 1 RL backbone (fast GRPO) |
| `v1.5` | `Qwen/Qwen3.5-4B` | Modern (`transformers>=5.2`) | Part 1 hybrid reasoning backbone |
| `v2` / `coach` | `unsloth/gemma-4-E4B-it` *(or `--use-official-gemma`)* | Modern (`transformers>=5.2`) | Local / workstation coach model |
| `student` | `Qwen/Qwen3.5-0.8B` | Modern (`transformers>=5.2`) | Compact browser WebGPU coach student |
| `student-e2b` | `google/gemma-4-E2B-it` | Modern (`transformers>=5.2`) | Local / Ollama coach student |
| `teacher` | `google/gemma-4-26B-A4B-it` | vLLM + Modern | Offline open-weight prose writer |
| `teacher-qwen` | `Qwen/Qwen3.6-35B-A3B` | vLLM + Modern | Alternative open-weight prose writer |

---

## 4. Part 1 — Chess-Playing RL Pipeline (`SFT -> GRPO 0/1/2 -> Eval`)

### Step 4.1: Build / Validate Balanced Engine-Verified Corpus

```bash
venv/bin/python -m chessfm.data_generation.build_corpus \
  --output-dir data_generation \
  --depth 10 \
  --clean-sft

venv/bin/python -m chessfm.training.train_sft \
  --data data_generation/sft_train.jsonl \
  --dry-run
```

### Step 4.2: Completion-Masked SFT (Merge-and-Save)

```bash
# Fast TRL 0.16 track (Qwen/Qwen3-4B):
modal run scripts/modal_train.py \
  --task sft \
  --track v1 \
  --data data_generation/sft_train.jsonl \
  --output-subdir sft_qwen3_4b \
  --epochs 3 \
  --include-ascii

# Or modern hybrid track (Qwen/Qwen3.5-4B):
modal run scripts/modal_train.py \
  --task sft \
  --track v1.5 \
  --data data_generation/sft_train.jsonl \
  --output-subdir sft_qwen35_4b \
  --epochs 3 \
  --include-ascii
```

### Step 4.3: Staged Verifiable GRPO (`Stage 0 -> Stage 1 -> Stage 2`)

Always chain from the merged SFT checkpoint (`/vol/checkpoints/<sft_subdir>/final_merged`):

```bash
# Stage 0: Legality + <think> format (Stockfish-free)
modal run scripts/modal_train.py \
  --task grpo \
  --model /vol/checkpoints/sft_qwen3_4b/final_merged \
  --stage 0 \
  --steps 300 \
  --output-subdir grpo_qwen3_stage0 \
  --include-ascii

# Stage 1: + Stockfish move quality
modal run scripts/modal_train.py \
  --task grpo \
  --model /vol/checkpoints/grpo_qwen3_stage0/stage0_final \
  --stage 1 \
  --steps 500 \
  --output-subdir grpo_qwen3_stage1 \
  --include-ascii

# Stage 2: + Opponent-reply blunder discrimination
modal run scripts/modal_train.py \
  --task grpo \
  --model /vol/checkpoints/grpo_qwen3_stage1/stage1_final \
  --stage 2 \
  --steps 500 \
  --output-subdir grpo_qwen3_stage2 \
  --include-ascii
```

### Step 4.4: Paired Color-Balanced Evaluation

```bash
modal run scripts/modal_train.py \
  --task eval \
  --model /vol/checkpoints/grpo_qwen3_stage2/stage2_final \
  --games 50 \
  --stockfish-level 1
```

---

## 5. Part 2 — Zero-API Open-Weight Coaching Distillation

### Step 5.1: Dry-Run Locally (Verify Prompts Without GPU)

```bash
venv/bin/python -m chessfm.data_generation.distill_open_teacher \
  --input data_generation/coaching_sample.jsonl \
  --dry-run \
  --limit 5
```

### Step 5.2: Generate ACT-Eval-Verified Coaching Traces via vLLM on Modal

```bash
modal run scripts/modal_train.py \
  --task distill-teacher \
  --track teacher \
  --data data_generation/coaching_sample.jsonl \
  --output-subdir distilled_coach.jsonl \
  --max-attempts 3
```

Every output row is verified by `verify_claims()` (`0` board/move violations) and `contains_move_spoiler() == False` for Level 1 / Level 2 Socratic hints, with an audit report saved alongside the dataset (`.audit.json`).

### Step 5.3: Fine-Tune Compact Student & Export

```bash
# Fine-tune browser student (Qwen/Qwen3.5-0.8B) on verified teacher traces:
modal run scripts/modal_train.py \
  --task sft \
  --track student \
  --data /vol/data/distilled_coach.jsonl \
  --output-subdir sft_coach_qwen35_08b \
  --epochs 3

# Export GGUF + Ollama Modelfile:
modal run scripts/modal_train.py \
  --task export-gguf \
  --model /vol/checkpoints/sft_coach_qwen35_08b/final_merged \
  --output-subdir coach_qwen35_08b_gguf \
  --family qwen3
```

---

## 6. Downloading Artifacts Locally

```bash
modal volume get chessfm-artifacts results ./results
modal volume get chessfm-artifacts checkpoints ./checkpoints
modal volume get chessfm-artifacts data ./data_generation/full
modal volume get chessfm-artifacts exports ./exports
```

> All downloaded checkpoints, eval outputs, and large generated datasets live in gitignored directories (`checkpoints/`, `results/`, `data_generation/full/`). Record final verified metrics in [docs/BENCHMARKS.md](BENCHMARKS.md) alongside the exact command and `run_manifest.json` parameters.
