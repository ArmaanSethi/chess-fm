# AGENTS.md

ChessFM: a two-track chess reasoning and coaching system — Research & RL models (`Qwen/Qwen2.5-3B-Instruct` v0, `Qwen/Qwen3-4B` v1) that reason in `<think>...</think>` tags before outputting a legal UCI move via SFT + staged GRPO + constrained decoding, plus an engine-grounded AI Chess Coach (`unsloth/gemma-4-E4B-it` / `google/gemma-4-E4B-it` v2) with Stockfish MultiPV Partial Strategy Trees, Maia-2 human-Elo calibration, 3-level Socratic hints, ACT-Eval claim verification, and an offline Coach Studio UI.

## Setup & Environment

- `./setup_env.sh` creates `venv/`, installs Stockfish, and installs dependencies editable (`pip install -e .[dev]`).
- Activate `venv` (`source venv/bin/activate`) or run via `venv/bin/python` / `venv/bin/pytest`. Bare system `python`/`pytest` commands do not resolve repo dependencies.
- Stockfish is required for integration tests (`tests/test_evaluate_integration.py` auto-skips if absent), GRPO stages 1/2, evaluation, and full Coach Studio analysis.

## Commands & Verification

- **Lint & Test Pass**: `venv/bin/ruff check . && venv/bin/python -m pytest --cov=chessfm --cov-report=term-missing`
- **Format Check**: `venv/bin/ruff format --check .`
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
- Stockfish match evaluation: `python -m chessfm.rl.evaluate --model <checkpoint> --games 50`
- 4-Tier evaluation suite: `python -m chessfm.eval.suite --demo`
- Offline Coach Studio server: `python -m chessfm.coach.server --host 127.0.0.1 --port 8000`
- Cloud & local deployment scripts (`scripts/`):
  - Modal L4 runner: `modal run scripts/modal_train.py --task smoke-pipeline --include-ascii`
  - Kaggle 2x T4 runner: `python scripts/kaggle_train.py --task pipeline --resume`
  - GGUF + Ollama Modelfile export: `python scripts/export_gguf.py --model <merged_ckpt> --output <dir> --family gemma4 --dry-run`

Installed console scripts: `chessfm-convert`, `chessfm-generate-data`, `chessfm-fetch-fens`, `chessfm-train-sft`, `chessfm-train-grpo`, `chessfm-evaluate`, `chessfm-coach`, `chessfm-eval-suite`.

## Architecture & Conventions

- `src/chessfm/config.py` is the single source of truth: model tracks (`DEFAULT_BASE_MODEL`, `DEFAULT_V1_MODEL`, `DEFAULT_COACH_MODEL`, `OFFICIAL_GEMMA_COACH_MODEL`), model-agnostic prompt templates (`build_chat_prompt`, `make_prompt`, `render_ascii_board`), `MAX_CENTIPAWN_LOSS=200`, Stockfish depth (10), and binary search paths.
- Prompt formatting MUST stay identical across data conversion, SFT, GRPO, and evaluation. `build_chat_prompt` delegates to `tokenizer.apply_chat_template` when available and falls back to ChatML for Qwen-2.5.
- Train/val splits MUST be done at the FEN level (`training/split.py` `fen_level_split`, `data_generation/phase_sampler.py` `normalize_board_fen`) so no positions leak between splits.
- `data_generation/phase_sampler.py` handles opening/middlegame/endgame classification and stratified sampling; `data_generation/coaching_dataset.py` builds deterministic, engine-verified best-line SFT and coaching examples at `$0` API cost.
- `rl/constrained_decoding.py` provides a trie-based UCI `LogitsProcessor` enforcing 100% legal moves after `</think>` at inference time.
- `coach/` contains the AI Chess Coach stack: MultiPV Partial Strategy Trees & WDL classification (`classifier.py`), Maia-2 human-Elo calibration (`maia_sidecar.py`), 3-level 0%-spoiler Socratic hints (`socratic.py`), ACT-Eval deterministic claim verification (`verifier.py`), full-game PGN review & weakness profiling (`game_review.py`), Q&A orchestration (`multimodal.py`), and the stdlib HTTP server (`server.py`) serving `static/index.html`.
- `eval/suite.py` implements the 4-Tier Evaluation Suite (Static Engine Metrics, Board-State Probing / ChessQA, ACT-Eval Coaching Faithfulness, and Paired Color-Balanced Match Summary).
- `scripts/` holds top-level deployment runners (`modal_train.py`, `kaggle_train.py`, `export_gguf.py`) with pure command-builder helpers unit-tested in `tests/test_cloud_runners.py`.
- Do NOT commit model artifacts (`checkpoints/`), eval outputs (`results/`), or large dataset files (`data_generation/full/`). Only small sample JSONL files are committed for pipeline verification.
- GPU/hardware-dependent entry points (`load_model.py`, `scripts/*`, `training/*`, `rl/train_grpo.py`, `benchmarks/*`, `fetch_elite_data.py`) are intentionally omitted from coverage in `pyproject.toml`. Pure helper functions in these files are unit-tested directly.

## Gotchas

- `generate_synthetic_data` requires provider API keys in env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`); never put keys in files.
- `google/gemma-4-E4B-it` is gated and requires `HF_TOKEN`; `unsloth/gemma-4-E4B-it` is the ungated default for the v2 coach track.
- GRPO Stage 0 is Stockfish-free; Stages 1 and 2 fail fast if Stockfish binary is missing.
- Evaluation runs generate JSON reports and PGN files under `results/` (gitignored). Model metrics are captured in eval reports rather than hardcoded in source.

## Engineering & GitHub Standards

- **Commits & Pull Requests**: Use concise, scoped conventional titles (e.g. `fix(rl): ...`, `feat(coach): ...`) with clear technical rationale, design tradeoffs, and exact verification commands/results (`ruff`, `pytest --cov`).
- **Repository Hygiene**: Never commit conversational notes, scratch/teamwork artifacts (`.agents/teamwork/`), model checkpoints, raw eval outputs, or secrets/API keys.
- **Code Quality**: Maintain clean type annotations, purposeful docstrings explaining design intent, deterministic unit/integration tests, `ruff check` + `ruff format` cleanliness, and `>= 80%` test coverage.
- **Reproducibility**: Every claimed metric in documentation must be backed by a reproducible CLI command and eval manifest.

