# AGENTS.md

ChessFM: a 3B chess model that reasons in `<think>...</think>` tags then outputs a UCI move. SFT on engine-validated traces, then staged GRPO with Stockfish rewards.

## Setup & Environment

- `./setup_env.sh` creates `venv/`, installs Stockfish, and installs dependencies editable (`pip install -e .[dev]`).
- Activate `venv` (`source venv/bin/activate`) or run via `venv/bin/python` / `venv/bin/pytest`. Bare system `python`/`pytest` commands do not resolve repo dependencies.
- Stockfish is required for integration tests (`tests/test_evaluate_integration.py` auto-skips if absent) and for GRPO stages 1/2 + evaluation.

## Commands & Verification

- **Lint & Test Pass**: `venv/bin/ruff check . && venv/bin/python -m pytest --cov=chessfm --cov-report=term-missing`
- **Coverage Gate**: CI enforces `fail_under = 80%` in `pyproject.toml`.
- **Run Single Test File**: `venv/bin/python -m pytest tests/test_rewards.py`
- **Run Focused Test**: `venv/bin/python -m pytest tests/test_rewards.py -k test_legality_reward`
- **Pre-commit**: `pre-commit run --all-files` (runs `ruff` with `--fix` and `ruff-format`).

## Module Entry Points

Package uses `src` layout (`src/chessfm/`). Entry points run as modules via `python -m`:
- Data conversion: `python -m chessfm.data_generation.convert_to_training`
- SFT dry-run (no GPU data validation): `python -m chessfm.training.train_sft --dry-run`
- SFT training: `python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl`
- GRPO training: `python -m chessfm.rl.train_grpo --stage 0 --steps 500`
- Evaluation: `python -m chessfm.rl.evaluate --model <checkpoint> --games 50`
Installed console scripts also exist (`chessfm-convert`, `chessfm-train-sft`, `chessfm-train-grpo`, `chessfm-evaluate`).

## Architecture & Conventions

- `src/chessfm/config.py` is the single source of truth: prompt templates (`build_chat_prompt`), base model (`Qwen/Qwen2.5-3B-Instruct`), `MAX_CENTIPAWN_LOSS=200`, Stockfish depth (10), and binary search paths.
- ChatML prompt template wraps the position prompt with `<think>...</think>` reasoning and UCI move format. Prompt formatting MUST stay identical across conversion, SFT, GRPO, and evaluation.
- Train/val split MUST be done at the FEN level (`training/split.py` `fen_level_split`) so no positions leak between splits.
- Do NOT commit model artifacts (`checkpoints/`), eval outputs (`results/`), or large dataset files (`data_generation/full/`). Only small sample JSONL files are committed for pipeline verification.
- GPU/hardware-dependent entry points (`load_model.py`, `scripts/*`, `training/*`, `rl/train_grpo.py`, `benchmarks/*`, `fetch_elite_data.py`) are intentionally omitted from coverage in `pyproject.toml`. Pure helper functions in these files are unit-tested directly.

## Gotchas

- `generate_synthetic_data` requires provider API keys in env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`); never put keys in files.
- GRPO Stage 0 is Stockfish-free; Stages 1 and 2 fail fast if Stockfish binary is missing.
- Evaluation runs generate JSON reports and PGN files under `results/` (gitignored). Model metrics are captured in eval reports rather than hardcoded in source.
