# ChessFM Experiment & Engineering Log

Living technical log tracking every verified experiment (`EXP-NNN`) and real engineering quirk or failure mode (`QUIRK-NNN`) encountered across Part 1 (Chess-Playing RL LLM) and Part 2 (Engine-Grounded Open-Source Chess Coach).

Every metric in this log is backed by a reproducible CLI command and recorded manifest (`run_manifest.json` / `eval_manifest.json`). Raw checkpoints, PGNs, and eval dumps remain in gitignored `checkpoints/` and `results/` (mirrored to Modal volumes and Hugging Face Hub).

---

## 1. Known Engineering Quirks & Failure Ledger

| ID | Subsystem | Symptom / Verbatim Error | Root Cause | Resolution | Blog Takeaway |
|---|---|---|---|---|---|
| `QUIRK-001` | Modal container build (`scripts/modal_train.py`) | Modal image build error when chaining `.pip_install("vllm")` after `.add_local_dir(...)` on `modern_image`. | Modal requires local directory mounts (`.add_local_dir()`) to be the final layer of an image unless `copy=True` is forced, so deriving `vllm_image` from an image that already mounted the repo failed. | Split `_modern_base_image` (packages only) from `modern_image` and `vllm_image`, applying `.add_local_dir(...)` as the final step on both. | Keep Modal base images pure (APT + pip) and mount local source last so derived images (`vllm_image`) reuse cached layers cleanly. |
| `QUIRK-002` | SFT & GRPO on NVIDIA L4 (`22.08 GiB`) | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity of 22.08 GiB of which 13.88 MiB is free. Process 1 has 22.06 GiB memory in use.` at step 0/4 of `train_sft`. | Without `unsloth` in the base Modal image, `load_model.py` loads `Qwen/Qwen3-4B` in `bf16` (~8 GB weights) + LoRA + `adamw_8bit`. With `gradient_checkpointing=False`, `batch_size=4`, and longer `--include-ascii` prompts (`max_length=2048`), forward activations pushed VRAM to `22.06 / 22.08 GiB`. | Enabled `gradient_checkpointing=True` with `gradient_checkpointing_kwargs={"use_reentrant": False}` and `model.config.use_cache = False` in `train_sft.py` and `train_grpo.py`; added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `scripts/modal_train.py`; tuned L4 defaults (`batch_size=2, grad_accum=4` for SFT; `batch_size=4, num_generations=4, grad_accum=4` for GRPO); added optional `load_in_4bit=True` NF4 QLoRA safety valve in `load_model.py`. | A 4B `bf16` model leaves ~14 GB for activations on an L4 — always enable non-reentrant gradient checkpointing and `expandable_segments:True` before launching cloud runs. |
| `QUIRK-003` | `Qwen/Qwen3-4B` prompt template & constrained eval | `apply_chat_template(..., enable_thinking=False)` appends `<think>\n\n</think>\n\n` to the assistant prompt prefix. | `config.build_chat_prompt` defaulted `enable_thinking=False`. During SFT this doubled `<think>` tags when concatenated with `<think>...</think>` completions; during GRPO it suppressed `<think>` generation (conflicting with `reward_format()`); and `rl/evaluate.py` omitted `tokenizer`, `include_ascii`, and `enable_thinking` altogether while hardcoding `require_think_close=True` on `LegalMoveLogitsProcessor`. | Passed `enable_thinking=True` in `train_sft.py` and `train_grpo.py`; wired `tokenizer`, `--include-ascii`, and `--enable-thinking / --no-enable-thinking` into `rl/evaluate.py` and `eval/suite.py`; set `require_think_close=enable_thinking` on `LegalMoveLogitsProcessor` so Think-OFF constrained decoding activates immediately. | Always inspect the exact rendered chat-template tokens on Qwen3 models; `enable_thinking=False` injects an empty `<think></think>` block that silently breaks SFT loss masking, GRPO format rewards, and prefix-trie triggers. |
| `QUIRK-004` | Stockfish 16 Linux binary (`src/chessfm/rl/engine.py`) | `chess.engine.EngineTerminatedError: engine process died unexpectedly (exit code: -8)` (`SIGFPE`) during rapid sequential `depth=4` `analyse()` calls in `test_grpo_reward_variance_across_stages`. | Ubuntu's Stockfish 16 build intermittently raises `SIGFPE` under rapid shallow-depth (`depth=4`) `analyse()` bursts; previously, a dead `SimpleEngine` handle stayed dead for all subsequent reward calls. | Wrapped `open_engine()` in `ResilientEngine` (`src/chessfm/rl/engine.py`) which detects `EngineTerminatedError` / `BrokenPipeError`, cleanly closes the dead process, respawns Stockfish with all previously applied UCI options, and retries up to `max_restarts=2` times per call. | Long-running RL loops calling external UCI binaries need bounded auto-restart wrappers that preserve `configure()` state (e.g. `Threads`, `Hash`, `Skill Level`). |
| `QUIRK-005` | Paired match opening indexing (`src/chessfm/rl/evaluate.py`) | With `--alternate-colors` and `--openings`, each opening FEN was only played by a single color instead of both White and Black. | `evaluate_model` indexed `opening_fens[g % len(opening_fens)]` while alternating color by `g % 2 == 0`, so even-indexed openings were always played as White and odd-indexed openings as Black. | Changed indexing to `opening_fens[(g // 2) % len(opening_fens)]` when `alternate_colors=True` and added explicit `white` / `black` W-D-L splits plus `color_balanced` to the JSON report and `eval_manifest.json`. | Color-balanced chess benchmarks require each opening position to be played from *both* sides; indexing by `g // 2` eliminates opening-color confounding. |
| `QUIRK-006` | TRL 0.16 `GRPOTrainer` rollout speed with `gradient_checkpointing=True` | `` `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`. `` during GRPO rollouts; step 0/20 stalled >145s without completing on L4 (`ap-c0Nu3vFgTlZfryW3YUJdJv`). | TRL 0.16's `GRPOTrainer._generate_and_score_completions()` calls `unwrapped_model.generate(...)` while `model.training == True`, causing `Qwen3ForCausalLM.forward()` to forcibly disable KV caching on every autoregressive rollout token whenever `gradient_checkpointing=True`. | Subclassed `GRPOTrainer` as `_KVCacheGRPOTrainer` in `src/chessfm/rl/train_grpo.py` to switch to `model.eval()` + `model.config.use_cache = True` strictly inside `_generate_and_score_completions()`, restoring `model.train()` + `use_cache = False` before `compute_loss()`; set L4 `grad_accum=2`. | Enabling gradient checkpointing fixes backward OOMs but silently disables KV caching during TRL 0.16 rollouts unless the model is toggled to `eval()` during generation. |

---

## 2. Chronological Experiment Entries

### EXP-001 — Initial Modal L4 Smoke Pipeline (Pre-Fix OOM Baseline)

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-22 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`) |
| **Git SHA** | `12ff30a` (+ uncommitted `vllm_image` layer-order fix) |
| **Hardware & Est. Cost** | Modal NVIDIA L4 (`22.08 GiB`, `$0.80/hr`), ~6 min (~`$0.08–$0.15` incl. image build) |
| **Modal App ID** | `ap-UndgQY7pyf0tCdXrCAbfGh` |
| **W&B Run** | `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/b6vez09s` |
| **Command** | `modal run scripts/modal_train.py --task smoke-pipeline --track v1 --epochs 1 --steps 20 --games 4 --include-ascii` |

- **Hypothesis:** End-to-end Modal L4 smoke pipeline (1-epoch SFT on 40 sample rows -> 20-step GRPO Stage 0 -> 4-game Stockfish L1 eval) completes cleanly with default `batch_size=4` and `--include-ascii`.
- **Observed Result:** **FAILED at SFT Step 0/4 (`CUDA OOM`)**.
  - All 3 Modal container images (`image`, `modern_image`, `vllm_image`) built and cached cleanly (`vllm_image` in `151.49s`).
  - `Qwen/Qwen3-4B` downloaded (3 shards in `32s`) onto persistent `/vol/hf_cache` and W&B streaming connected immediately.
  - SFT loaded 40/40 valid samples (34 train / 6 val, 6 unique FENs) and OOM'd on the very first forward pass (`22.06 / 22.08 GiB` allocated, failed trying to allocate `44.00 MiB`).
- **Root Cause & Decision:** Logged as `QUIRK-002`. Enable `gradient_checkpointing=True` (`use_reentrant=False`), `use_cache=False`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and L4-safe batch defaults (`batch_size=2, grad_accum=4` for SFT; `batch_size=4, num_generations=4, grad_accum=2` for GRPO), fix `QUIRK-003`..`QUIRK-006`, then re-run as `EXP-002`.

---

### EXP-002 — Post-Fix Modal L4 End-to-End Smoke Verification (SFT -> GRPO Stage 0 -> Stockfish L1 Eval)

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-23 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`) |
| **Hardware & Est. Cost** | Modal NVIDIA L4 (`22.08 GiB`, `$0.80/hr`), ~6.5 min active GPU time (~`$0.09`) |
| **Modal App IDs** | SFT: `ap-dhCsDE6ghFWZTdDwpVw1wt` · GRPO Stage 0: `ap-yE8SoD9Fghl7RtQqpMdFpV` · Eval: `ap-DPNoASfD7PYgcWWplSHe83` |
| **W&B Runs** | SFT: `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/4y4rtckr` · GRPO: `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/my0bsikk` |
| **Commands** | `modal run --detach scripts/modal_train.py --task sft --track v1 --epochs 1 --output-subdir sft_smoke --include-ascii` -> `modal run --detach scripts/modal_train.py --task grpo --model /vol/checkpoints/sft_smoke/final_merged --stage 0 --steps 3 --output-subdir smoke_grpo_stage0 --include-ascii` -> `modal run --detach scripts/modal_train.py --task eval --model /vol/checkpoints/smoke_grpo_stage0/stage0_final --games 4 --stockfish-level 1 --include-ascii --enable-thinking` |
| **Volume Artifacts** | `/vol/checkpoints/sft_smoke/final_merged`, `/vol/checkpoints/smoke_grpo_stage0/stage0_final`, `/vol/results/eval_20261123_010847.json`, `/vol/results/pgns/eval_20261123_010847.pgn` |

- **Hypothesis:** After applying `QUIRK-002` (gradient checkpointing + `expandable_segments:True` + L4 batch defaults), `QUIRK-003` (`enable_thinking=True` prompt parity), and `QUIRK-006` (`_KVCacheGRPOTrainer`), the full SFT -> GRPO Stage 0 -> Stockfish L1 Eval pipeline completes on L4 with **0 OOMs**.
- **Measured Results:**

| Stage | Runtime | Throughput | Key Metrics | OOMs |
|---|---:|---:|---|---:|
| **SFT (1 epoch, 34 train / 6 val, `bs=2, ga=4`)** | `23.40s` | `1.453 samples/s` (`0.171 steps/s`) | `train_loss=2.7629`, `eval_loss=2.7884`, `mean_token_accuracy=0.5137`, `eval_mean_token_accuracy=0.5153` | **0** |
| **GRPO Stage 0 (3 steps, `G=4, bs=4, ga=2`)** | `283.73s` | `94.19s/step` (`16,912` tokens) | `completion_length=512.0`, `reward=-1.0`, `reward_std=0.0`, `kl=0.0` | **0** |
| **Stockfish L1 Eval (4 games, strict, `--include-ascii --enable-thinking`)** | ~`60s` | 4 games (2 W / 2 B) | `Legal Move Rate=33.3% (2/6)`, `Format Adherence=0.0% (0/6)`, `W-L-D=0-4-0` (`W 0-2-0 \| B 0-2-0`), `95% Wilson CI=[0.000, 0.490]` | **0** |

- **Diagnostic Findings (Why GRPO `reward_std == 0.0` on a 1-epoch 34-row SFT smoke checkpoint):**
  1. **Zero OOMs verified across all 3 stages** on a single 22.08 GiB L4 (`$0.80/hr`), validating our memory and KV-cache fixes.
  2. **Smoke SFT underfitting signal:** After only 4 optimizer steps on 34 sample rows, `Qwen/Qwen3-4B` had not yet learned to close `</think>` within the 512-token completion budget (`completion_length=512.0` on all rollouts, `Format Adherence=0.0%`), causing all 4 completions per group to tie at `-1.0` (`reward_std=0.0`, zero GRPO gradient).
  3. **Decision / Next Action:** Proceed to M3 — build the 2,500–3,000-row phase-balanced Stockfish-verified SFT corpus (so 3-epoch SFT reliably teaches concise `<think>...</think>` + legal UCI moves before GRPO begins) and run full Base (`Think OFF` vs `ON`, `Strict` vs `Constrained`) -> SFT -> Staged GRPO evaluations using `--detach` on Modal.
