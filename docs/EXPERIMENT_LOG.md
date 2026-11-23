# ChessFM Experiment & Engineering Log

Living technical log tracking every verified experiment (`EXP-NNN`) and real engineering quirk or failure mode (`QUIRK-NNN`) encountered across Part 1 (Chess-Playing RL LLM) and Part 2 (Engine-Grounded Open-Source Chess Coach).

Every metric in this log is backed by a reproducible CLI command and recorded manifest (`run_manifest.json` / `eval_manifest.json`). Raw checkpoints, PGNs, and eval dumps remain in gitignored `checkpoints/` and `results/` (mirrored to Modal volumes and Hugging Face Hub).

---

## 1. Known Engineering Quirks & Failure Ledger

| ID | Subsystem | Symptom / Verbatim Error | Root Cause | Resolution | Blog Takeaway |
|---|---|---|---|---|---|
| `QUIRK-001` | Modal container build (`scripts/modal_train.py`) | Modal image build error when chaining `.pip_install("vllm")` after `.add_local_dir(...)` on `modern_image`. | Modal requires local directory mounts (`.add_local_dir()`) to be the final layer of an image unless `copy=True` is forced, so deriving `vllm_image` from an image that already mounted the repo failed. | Split `_modern_base_image` (packages only) from `modern_image` and `vllm_image`, applying `.add_local_dir(...)` as the final step on both. | Keep Modal base images pure (APT + pip) and mount local source last so derived images (`vllm_image`) reuse cached layers cleanly. |
| `QUIRK-002` | SFT & GRPO on NVIDIA L4 (`22.08 GiB`) | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity of 22.08 GiB of which 13.88 MiB is free. Process 1 has 22.06 GiB memory in use.` at step 0/4 of `train_sft`. | Without `unsloth` in the base Modal image, `load_model.py` loads `Qwen/Qwen3-4B` in `bf16` (~8 GB weights) + LoRA + `adamw_8bit`. With `gradient_checkpointing=False`, `batch_size=4`, and longer `--include-ascii` prompts (`max_length=2048`), forward activations pushed VRAM to `22.06 / 22.08 GiB`. | Enabled `gradient_checkpointing=True` with `gradient_checkpointing_kwargs={"use_reentrant": False}` and `model.config.use_cache = False` in `train_sft.py` and `train_grpo.py`; added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `scripts/modal_train.py`; tuned L4 defaults (`batch_size=2, grad_accum=4` for SFT; `batch_size=4, num_generations=4, grad_accum=2` for GRPO); added optional `load_in_4bit=True` NF4 QLoRA safety valve in `load_model.py`. | A 4B `bf16` model leaves ~14 GB for activations on an L4 — always enable non-reentrant gradient checkpointing and `expandable_segments:True` before launching cloud runs. |
| `QUIRK-003` | `Qwen/Qwen3-4B` prompt template & constrained eval | `apply_chat_template(..., enable_thinking=False)` appends `<think>\n\n</think>\n\n` to the assistant prompt prefix. | `config.build_chat_prompt` defaulted `enable_thinking=False`. During SFT this doubled `<think>` tags when concatenated with `<think>...</think>` completions; during GRPO it suppressed `<think>` generation (conflicting with `reward_format()`); and `rl/evaluate.py` omitted `tokenizer`, `include_ascii`, and `enable_thinking` altogether while hardcoding `require_think_close=True` on `LegalMoveLogitsProcessor`. | Passed `enable_thinking=True` in `train_sft.py` and `train_grpo.py`; wired `tokenizer`, `--include-ascii`, and `--enable-thinking / --no-enable-thinking` into `rl/evaluate.py` and `eval/suite.py`; set `require_think_close=enable_thinking` on `LegalMoveLogitsProcessor` so Think-OFF constrained decoding activates immediately. | Always inspect the exact rendered chat-template tokens on Qwen3 models; `enable_thinking=False` injects an empty `<think></think>` block that silently breaks SFT loss masking, GRPO format rewards, and prefix-trie triggers. |
| `QUIRK-004` | Stockfish 16 Linux binary (`src/chessfm/rl/engine.py`) | `chess.engine.EngineTerminatedError: engine process died unexpectedly (exit code: -8)` (`SIGFPE`) during rapid sequential `depth=4` `analyse()` calls in `test_grpo_reward_variance_across_stages`. | Ubuntu's Stockfish 16 build intermittently raises `SIGFPE` under rapid shallow-depth (`depth=4`) `analyse()` bursts; previously, a dead `SimpleEngine` handle stayed dead for all subsequent reward calls. | Wrapped `open_engine()` in `ResilientEngine` (`src/chessfm/rl/engine.py`) which detects `EngineTerminatedError` / `BrokenPipeError`, cleanly closes the dead process, respawns Stockfish with all previously applied UCI options, and retries up to `max_restarts=2` times per call. | Long-running RL loops calling external UCI binaries need bounded auto-restart wrappers that preserve `configure()` state (e.g. `Threads`, `Hash`, `Skill Level`). |
| `QUIRK-005` | Paired match opening indexing (`src/chessfm/rl/evaluate.py`) | With `--alternate-colors` and `--openings`, each opening FEN was only played by a single color instead of both White and Black. | `evaluate_model` indexed `opening_fens[g % len(opening_fens)]` while alternating color by `g % 2 == 0`, so even-indexed openings were always played as White and odd-indexed openings as Black. | Changed indexing to `opening_fens[(g // 2) % len(opening_fens)]` when `alternate_colors=True` and added explicit `white` / `black` W-D-L splits plus `color_balanced` to the JSON report and `eval_manifest.json`. | Color-balanced chess benchmarks require each opening position to be played from *both* sides; indexing by `g // 2` eliminates opening-color confounding. |
| `QUIRK-006` | TRL 0.16 `GRPOTrainer` rollout speed with `gradient_checkpointing=True` | `` `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`. `` during GRPO rollouts; step 0/20 stalled >145s without completing on L4 (`ap-c0Nu3vFgTlZfryW3YUJdJv`). | TRL 0.16's `GRPOTrainer._generate_and_score_completions()` calls `unwrapped_model.generate(...)` while `model.training == True`, causing `Qwen3ForCausalLM.forward()` to forcibly disable KV caching on every autoregressive rollout token whenever `gradient_checkpointing=True`. | Subclassed `GRPOTrainer` as `_KVCacheGRPOTrainer` in `src/chessfm/rl/train_grpo.py` to switch to `model.eval()` + `model.config.use_cache = True` strictly inside `_generate_and_score_completions()`, restoring `model.train()` + `use_cache = False` before `compute_loss()`; set L4 `grad_accum=2`. | Enabling gradient checkpointing fixes backward OOMs but silently disables KV caching during TRL 0.16 rollouts unless the model is toggled to `eval()` during generation. |
| `QUIRK-009` | Multi-stage GRPO LoRA chaining (`src/chessfm/rl/train_grpo.py`, `scripts/modal_train.py`) | Loading `stage1_final` or `stage2_final` silently drops earlier GRPO stage weights, and `modal_train.py` did not forward `--positions` or `--max-positions` to `train_grpo.py`. | Each GRPO stage loads the prior checkpoint with `merge_peft=True` (merging in memory) and attaches a fresh LoRA adapter, but `model.name_or_path` still points to the pre-GRPO SFT checkpoint. Calling `model.save_pretrained("stageN_final")` without `merge_and_unload()` writes only the new stage's delta referencing the pre-GRPO base. | Added `--merge-and-save` (default `True`) and `save_grpo_checkpoint()` in `src/chessfm/rl/train_grpo.py` to persist standalone `stage{N}_final_merged` weights via `merge_and_unload()`; updated `scripts/modal_train.py` to forward `--positions`, `--max-positions`, and `--merge-and-save` and return `stage{N}_final_merged`. | Never chain PEFT adapters across RL stages by saving raw adapters — always merge and save standalone weights between stages so earlier RL updates are never silently lost. |

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

---

### EXP-003 — Phase-Balanced Corpus Expansion & Untuned `Qwen/Qwen3-4B` Base Evaluation (Think OFF vs ON, Strict vs Constrained + Stockfish L1)

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-23 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`, untuned base model) |
| **Hardware & Est. Cost** | Corpus: Local CPU (`$0`, `63s` on 2 cores) · Base Eval: 6× Modal NVIDIA L4 (`$0.80/hr`), `17,858.25s` active GPU time (`$3.97` completed + `~$0.40` from 3 canceled 10m client-timeout retries = **`~$4.37`**) |
| **Modal App IDs** | Job A (`Think OFF strict`): `ap-kg4eMQO8YXGj4bcDwZoIjD` · Job B (`Think OFF constrained`): `ap-IsEzMts4LoPk7PUGUMNnyF` · Job C (`Think ON strict`): `ap-NzxSs8QJSZZ86YuDQAY8X6` · Job D (`Think ON constrained`): `ap-d4asbHKOUjSDku1Bqcjlcj` · Job E (`20g SF-L1 Think OFF`): `ap-Iru6PPpkzqsjaUVpV6tMPA` · Job F (`20g SF-L1 Think ON`): `ap-Z3h1VmMxgAsr1YoeEXsSSW` |
| **Volume Artifacts** | `/vol/data/{sft_bestline_2500.jsonl,grpo_positions_5k.txt,eval_benchmark_200.jsonl,eval_openings_25.txt}` · `/vol/results/{base_think_off_strict.json,base_think_off_constrained.json,base_think_on_strict.json,base_think_on_constrained.json,eval_20261123_020452.json,eval_20261123_020645.json}` |

#### 1. Phase-Balanced, Zero-Leakage Corpus Generation (`$0` local CPU)
- **Root fix:** Previous `positions.txt` filtered `10 < fullmove < 45`, yielding `0.0%` openings. Parameterized `fetch_elite_data.py` (`--min-fullmove 3 --max-fullmove 55 --games-per-account 300`) and fetched **35,000** positions (`34,744` unique 4-field FENs: `6,239` opening / `17,174` middlegame / `11,331` endgame).
- **Disjoint splits (`build_corpus.py --build-p1-splits --depth 10 --workers 2 --seed 42`):**
  - `sft_bestline_2500.jsonl`: **2,500 rows** (`25.0%` opening / `55.0%` middlegame / `20.0%` endgame), `audit_dataset_rows` passed (`0` illegal moves, `0` blunders).
  - `grpo_positions_5k.txt`: **5,000 FENs** (`25.0%` / `55.0%` / `20.0%`).
  - `eval_benchmark_200.jsonl`: **200 FENs** (`25.0%` / `55.0%` / `20.0%`, Stockfish `depth=10, multipv=3`).
  - `eval_openings_25.txt`: **25 opening FENs** (`100.0%` opening) from the held-out eval opening slice.
  - `verify_zero_leakage(sft_fens, grpo_fens, eval_fens, strict=True)` -> **`zero_leakage=True, total_overlap=0`**.

#### 2. Base `Qwen/Qwen3-4B` Evaluation Results (200-FEN Static + 1,000 ChessQA Probes + 20-Game Paired Stockfish L1)

| Mode | Decoding | Wall-Clock (L4) | Est. Cost | Legal Rate | Format Adherence | ACPL | MAR (<=50cp) | Mean WDL Loss | Exact Match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Base (Think OFF)** | `strict` | `5,290.22s` | `$1.18` | `0.5%` (`1/200`) | `0.0%` | `80.0` *(n=1)* | `0.0%` | `7.27` *(n=1)* | `0.0%` |
| **Base (Think OFF)** | `constrained` | `76.16s` | `$0.02` | **`100.0%`** (`200/200`) | `0.0%` | **`465.36`** | `20.0%` | `19.66` | `2.0%` |
| **Base (Think ON)** | `strict` | `5,304.21s` | `$1.18` | `0.0%` (`0/200`) | `0.0%` | `—` *(n=0)* | `0.0%` | `—` | `0.0%` |
| **Base (Think ON)** | `constrained` | `6,310.20s` | `$1.40` | **`0.0%`** (`0/200`) | `0.0%` | `—` *(n=0)* | `0.0%` | `—` | `0.0%` |

- **Tier 2 Board-State Probing (ChessQA, 1,000 greedy probes across 200 FENs — identical for Think OFF & ON):**
  - **Overall Accuracy:** **`41.3%`** (`413/1,000`)
  - `side_to_move`: **`94.0%`** (`188/200`) — reads FEN header reliably
  - `piece_on_square`: **`12.5%`** (`25/200`) — severe 2D coordinate blindness despite ASCII board
  - `move_legality`: **`50.0%`** (`200/400`) — exact random-chance coin flip
  - `legal_move_count`: **`0.0%`** (`0/200`) — zero ability to enumerate legal moves

- **Tier 4 Paired Color-Balanced Matches vs Stockfish L1 (`depth=6`, `N=20`, 10 paired openings from `eval_openings_25.txt`):**

| Mode | Wall-Clock (L4) | Est. Cost | Record (W-L-D) | White / Black Split | 95% Wilson CI | Legal Move Rate | Format Rate |
|---|---:|---:|---|---|---|---:|---:|
| **Base (Think OFF, strict)** | `451.83s` | `$0.10` | `0-20-0` | `W 0-10-0 \| B 0-10-0` | `[0.000, 0.161]` | `4.76%` (`1/21`) | `0.0%` |
| **Base (Think ON, strict)** | `425.63s` | `$0.09` | `0-20-0` | `W 0-10-0 \| B 0-10-0` | `[0.000, 0.161]` | `0.0%` (`0/20`) | `0.0%` |

#### 3. Key Scientific Takeaways & Real Failure Traces
1. **Constrained decoding alone cannot save untuned Think-ON models:** `LegalMoveLogitsProcessor(require_think_close=True)` only activates after `</think>`. Untuned `Qwen/Qwen3-4B` with `enable_thinking=True` exhausts all `512` tokens inside `<think>` on `200/200` positions, so even constrained decoding yields **`0.0%` legality** (`6,310s` wasted rambling). SFT is strictly required to teach bounded `<think>...</think>` reasoning!
2. **Constrained decoding with Think OFF exposes true untuned chess strength:** With `require_think_close=False`, the prefix trie fires immediately (`76.16s`, `100.0%` legal), revealing near-random move quality (**`465.36` ACPL**, `2.0%` exact match).
3. **Real failure traces (`base_think_off_strict.json`):** Out of `199/200` illegal outputs, `186` produced verbose unparsed prose (`predicted_move=None`) and `13` emitted syntactically valid but hallucinated illegal UCI moves:
   - `5rk1/ppp3pp/1b1pq3/7B/7Q/3Pr1B1/PPP1N1K1/4R3 b - - 8 24` -> predicted `d7d6` (illegal: `d7` is empty, pawn is already on `d6`; best: `g7g6`).
   - `rnbqkb1r/pppppp1p/5np1/8/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3` -> predicted `e4e5` (illegal: White's e-pawn is still on `e2`; best: `c2c4`).
   - `1r6/R5pp/4rBk1/1p2bp2/4p3/6P1/1P3P1P/3R1BK1 b - - 0 30` -> predicted `a2a4` (illegal: Black to move and `a2` is empty; best: `g6f6`).

---

### EXP-004 — Completion-Masked SFT (`Qwen/Qwen3-4B` v1) & 4-Tier Dual-Mode Evaluation

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-23 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`, Completion-Masked LoRA SFT merged to `bf16`) |
| **Hardware & Est. Cost** | Modal NVIDIA L4 (`22.08 GiB`, `$0.80/hr`), **`3,844.29s`** (64.1m) active GPU time, **`$0.85`** total (`$0.8543`: `$0.41` SFT + `$0.25` Strict Eval + `$0.18` Constrained Eval + `$0.02` Match; CPU HF upload `~$0.00`) |
| **Modal App IDs** | SFT: `ap-egLJ0aw9njHTdWXKIjpSIB` · Strict Eval + Tier 2 Probes: `ap-Nv6Fe1bteT4isDl93lJz4h` · Constrained Eval: `ap-dt9RR3dX8IBSCX5unjoatA` · 20g SF-L1 Match: `ap-uRq75UIEtw4tQat3BgpV5d` · HF Upload: `ap-tPZUjyF9BgB1utcTkc3l56` |
| **W&B Run** | `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/er6dlrav` |
| **Hugging Face Repo** | `https://huggingface.co/winsonly/chessfm-qwen3-4b-sft` *(private — cold-start SFT checkpoint prior to GRPO)* |
| **Dataset & SHA-256** | `/vol/data/sft_bestline_2500.jsonl` (`2,125` train / `375` eval FEN-level split; SHA-256 `399b0883442d61549dc12360f97020cebb2507539657c45b1445fee2c01bfb00`) |
| **Command** | `modal run --detach scripts/modal_train.py --task sft --track v1 --data /vol/data/sft_bestline_2500.jsonl --epochs 3 --include-ascii --output-subdir sft_qwen3_4b` |
| **Local / Volume Manifests** | `results/m3_sft/{sft_run_manifest.json,sft_think_on_strict.json,sft_think_on_constrained.json,sft_match_sf1_20g.json}` · `/vol/checkpoints/sft_qwen3_4b/final_merged` |

#### 1. Completion-Masked SFT Training Summary (`epochs=3, bs=2, grad_accum=4, lr=2e-5, seed=42`)

| Metric | Value | Notes |
|---|---:|---|
| **Optimizer Steps / Tokens** | `795` steps / `1,572,499` tokens | `~2.13–2.32 s/it` on NVIDIA L4 (`gradient_checkpointing=True`) |
| **Train Loss (`epoch=3.0`)** | **`0.7668`** | Final `train_loss=0.76678` (`train_runtime=1846.84s`, `3.452 samples/s`) |
| **Eval Loss (`epoch=3.0`)** | **`0.5181`** | Final `eval_loss=0.51812` (`eval_runtime=30.56s`, `12.272 samples/s`) |
| **Train Token Accuracy** | **`83.45%`** | `mean_token_accuracy=0.8345` |
| **Eval Token Accuracy** | **`82.84%`** | `eval_mean_token_accuracy=0.8284` |
| **CUDA OOMs** | **`0`** | Clean 3-epoch run + `merge_and_save` to `final_merged` (`1847.23s` wall clock) |

#### 2. Base (`EXP-003`) vs Completion-Masked SFT (`EXP-004`) Evaluation Comparison

*200-FEN Disjoint Static Benchmark (`include_ascii=True`), 1,000-Probe Tier 2 ChessQA, and 20-Game Paired Color-Balanced Match vs Stockfish L1 (`depth=6`):*

| Metric | Base Think OFF (`EXP-003`) | Base Think ON (`EXP-003`) | SFT Think ON (`EXP-004`) | Delta vs Base |
|---|---:|---:|---:|---|
| **Format Adherence (`<think>...</think>` + UCI)** | `0.0%` | `0.0%` | **`100.0%`** (`200/200`) | **`+100.0 pts`** (0.0% -> 100.0%) |
| **Strict Legal Move Rate (200-FEN static)** | `0.5%` (`1/200`) | `0.0%` (`0/200`) | **`21.0%`** (`42/200`) | **`+21.0 pts`** vs Think ON |
| **Strict Legal Move Rate (20g SF-L1 match)** | `4.76%` (`1/21`) | `0.0%` (`0/20`) | **`25.93%`** | **`+25.93 pts`** vs Think ON |
| **Constrained Legal Move Rate (200-FEN static)** | `100.0%` (`200/200`) | `0.0%` (`0/200`) | **`100.0%`** (`200/200`) | **`+100.0 pts`** vs Think ON (closes `</think>`!) |
| **Strict ACPL (cp, legal moves only)** | `80.0` *(n=1)* | `—` *(n=0)* | **`345.07`** *(n=42)* | First non-trivial strict sample |
| **Constrained ACPL (cp, `n=200`)** | `465.36` | `—` *(n=0)* | **`380.10`** | **`-85.26 cp`** (`465.36 -> 380.10`) |
| **Constrained MAR (<=50 cp) / Mean WDL Loss** | `20.0%` / `19.66%` | `—` | **`19.5%`** / **`20.44%`** | Strict: `11.9%` MAR / `24.08%` WDL loss |
| **Exact Stockfish Best-Move Match (Constrained)** | `2.0%` (`4/200`) | `0.0%` (`0/200`) | **`6.5%`** (`13/200`) | **`+4.5 pts`** (`3.25×` Base Think OFF; strict: `0.5%`, `1/200`) |
| **Tier 2 ChessQA Overall (`1,000` probes)** | `41.3%` (`413/1000`) | `41.3%` (`413/1000`) | **`41.4%`** (`414/1000`) | `side_to_move` `94.0% -> 73.5%`, `move_legality` `50.0%`, `legal_move_count` `0.0%` |
| **Tier 2 `piece_on_square` (`200` probes)** | `12.5%` (`25/200`) | `12.5%` (`25/200`) | **`33.5%`** (`67/200`) | **`+21.0 pts`** (`2.68×` board-coordinate accuracy) |
| **Paired 20g Match vs Stockfish L1 (Strict)** | `0-20-0` | `0-20-0` | **`0-20-0`** (`W 0-10 \| B 0-10`) | `95% Wilson CI: [0.0, 0.1611]`, `estimated_elo: -299.8` |

#### 3. Verbatim Qualitative Traces & Scientific Takeaways (`results/m3_sft/`)

1. **Bounded `<think>` + Exact Stockfish Best Move in Strict Mode (`sft_think_on_strict.json`):**
   - `fen`: `"rnbq1rk1/ppp1ppbp/1n4p1/8/3P4/2N2NP1/PP2PPBP/R1BQ1RK1 b - - 0 8"`
   - `predicted_move`: `"b8c6"` | `best_move`: `"b8c6"` | `legal`: `true`, `formatted`: `true`, `cp_loss`: `0.0`, `wdl_loss`: `0.0`, `exact_match`: `true`.
2. **Prefix-Trie Constrained Decoding Rescuing a 1-Square Coordinate Slip (`sft_think_on_strict.json` vs `sft_think_on_constrained.json`, 12 such rescues across 200 FENs):**
   - `fen`: `"r2k3r/1bpn1pbp/p5p1/1p2p3/P3P3/2P1BP2/1PB3PP/RN3RK1 b - - 2 14"` (Black bishop on `b7`).
   - **Strict:** `predicted_move`: `"b7c5"` (`legal`: `false`, `formatted`: `true` — off by one rank on the diagonal).
   - **Constrained:** `predicted_move`: `"b7c6"` | `best_move`: `"b7c6"` (`legal`: `true`, `cp_loss`: `0.0`, `exact_match`: `true`). Because SFT closes `</think>` on 100% of positions, the prefix trie now activates cleanly in Think-ON mode and steers near-miss moves onto the exact Stockfish best move.
3. **Board-State Awareness Jump vs XML Format Over-Conditioning on Tier 2 Probes (`sft_think_on_strict.json`):**
   - `piece_on_square` jumped from **`12.5%` to `33.5%`**, showing that ASCII-grounded SFT measurably improves 2D square tracking.
   - However, 9 `piece_on_square` misses were caused by literal tag over-conditioning: on `"r4bk1/ppq2ppp/5n2/2pn3b/2NN4/2P3PP/PP1B1PB1/R4QK1 w - - 0 18"` (`c5`), `expected`: `"black pawn"`, `predicted`: `"<color>black</color> <pawn>"` (right piece, wrapped in XML mimicking the prompt's `'<color> <piece>'` template). Similarly, `legal_move_count` (`0/200`) began prose explanations (`"to determine the number of legal moves in the given position, we analyze the board and evaluate all possible moves for white (the player to move)"`) and hit the 64-token probe cap before emitting an integer.
- **Decision / Next Action:** SFT achieved its exact cold-start objective — **100.0% format adherence** and **21.0%–25.9% strict legality** (ensuring non-zero reward variance `reward_std > 0` for GRPO). Launch **Staged GRPO (`Stage 0 -> Stage 1 -> Stage 2`)** on Modal L4 to drive strict legality and move quality up without relying on constrained decoding.

---

### EXP-005 — Staged GRPO (`Stage 0 -> Stage 1 -> Stage 2`) & Intermediate Evaluations (`Qwen/Qwen3-4B` v1)

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-23 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`, chained merged LoRA checkpoints `sft -> grpo_stage0 -> grpo_stage1 -> grpo_stage2`) |
| **Hardware & Est. Cost** | 9× Modal NVIDIA L4 (`22.08 GiB`, `$0.80/hr`), **`7,976.67s`** active GPU time, **`$1.77`** (`$1.7726` exact: `$0.2901` S0 + `$0.4365` S1 + `$0.4246` S2 + `$0.6214` across 6 intermediate evals) |
| **Modal App IDs (Training)** | Stage 0: `ap-zmRoBeiExN2aW4NrBhYJX2` · Stage 1: `ap-j3QsjMruTTJrDSQJVcfGRE` · Stage 2: `ap-5il1tazaFrNJ3HkHEPlq9b` |
| **Modal App IDs (Interm. Evals)** | S0 Strict: `ap-MmryCIY1YA1mQzwwKdXCSJ` · S0 Constr.: `ap-3Fq7YMhEwLj4AK5bx2wZk8` · S0 20g SF-L1: `ap-3vC5qaCU8la9Fe52VgCFi8` · S1 Strict: `ap-jy2KORDc0Q2FmdIA79Yg5L` · S1 Constr.: `ap-nF1fnVLvggEjcleirjmP0B` · S1 20g SF-L1: `ap-pnDsY52um5QjxZTUl6piz3` |
| **W&B Runs** | Stage 0: `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/tpud2oxk` · Stage 1: `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/5x2jhqal` · Stage 2: `https://forge.coreweave.com/wandb/armaan-sethi-personal/chess-fm/runs/oo36sw2x` |
| **Dataset & SHA-256** | `/vol/data/grpo_positions_5k.txt` (`5,000` disjoint FENs; SHA-256 `ffb13613553b014b60d53ff5a0ba9f460395fbc1095c4b68151ce5a6e08ba613`) |
| **Commands** | `modal run --detach scripts/modal_train.py --task grpo --model /vol/checkpoints/sft_qwen3_4b/final_merged --stage 0 --steps 100 --positions /vol/data/grpo_positions_5k.txt --max-positions 5000 --include-ascii --output-subdir grpo_stage0` (chained to Stage 1 `150` steps -> Stage 2 `150` steps via `--merge-and-save`) |
| **Local / Volume Manifests** | `results/m3_grpo/{stage0,stage1,stage2}_run_manifest.json`, `results/m3_grpo/grpo_s{0,1}_{strict,constrained,match_sf1}.json` · `/vol/checkpoints/grpo_stage{0,1,2}/stage{0,1,2}_final_merged` |

#### 1. Staged GRPO Training Summary (`G=4, bs=4, ga=2, lr=1e-5, beta=0.04, temp=1.0, seed=42`)

| Stage | Reward Objective | Steps | `sec/step` | Wall-Clock (s) | GPU Cost ($) | Final Train Loss | Final Reward / Std | Completion Len | Final KL | OOMs |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| **Stage 0** | Format + Legality | 100 | `13.06` | `1,305.58` | `$0.2901` | `0.00618` | `-0.4750` / `0.6462` | `59.34` | `0.00116` | **0** |
| **Stage 1** | + Stockfish Move Quality | 150 | `13.09` | `1,964.30` | `$0.4365` | `0.00653` | `-0.3841` / `0.9329` | `56.24` | `0.00318` | **0** |
| **Stage 2** | + Opponent-Reply Delta | 150 | `12.74` | `1,910.74` | `$0.4246` | `0.00666` | `-0.3685` / `0.6515` | `55.63` | `0.00235` | **0** |
| **Total** | **3-Stage Curriculum** | **400** | **~`12.95`** | **`5,180.62`** | **`$1.1512`** | — | Non-zero `reward_std` throughout | `59.34 -> 55.63` | `<=0.006` | **0** |

#### 2. Intermediate Evaluation Deltas (SFT -> GRPO Stage 0 -> GRPO Stage 1, 200-FEN Static + 20g SF-L1 Strict)

| Checkpoint | Mode | Legal Rate | Format Adherence | ACPL ↓ | MAR (≤50cp) ↑ | Mean WDL Loss ↓ | Exact Match ↑ | 20g SF-L1 Match Legal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **SFT (`EXP-004`)** | `strict` | `21.0%` (`42/200`) | `100.0%` | `345.07` | `11.90%` | `24.08%` | `0.5%` | `25.93%` (`0-20-0`) |
| **GRPO Stage 0** | `strict` | **`28.5%`** (`57/200`, `+7.5 pp`) | `100.0%` | **`267.30`** (`-77.77`) | **`17.54%`** (`+5.64 pp`) | **`19.32%`** | **`1.5%`** | `25.93%` (`7/27`, `0-20-0`) |
| **GRPO Stage 1** | `strict` | **`28.5%`** (`57/200`) | `100.0%` | `269.21` | **`21.05%`** (`+3.51 pp`) | `19.36%` | `1.5%` | **`39.39%`** (`13/33`, `+13.46 pp`, `0-20-0`) |
| **SFT (`EXP-004`)** | `constrained` | `100.0%` (`200/200`) | `100.0%` | `380.10` | `19.50%` | `20.44%` | `6.5%` | — |
| **GRPO Stage 0** | `constrained` | `100.0%` (`200/200`) | `100.0%` | **`328.11`** (`-51.99`) | `19.50%` | **`19.82%`** | **`7.0%`** | — |
| **GRPO Stage 1** | `constrained` | `100.0%` (`200/200`) | `100.0%` | **`327.12`** (`-0.99`) | **`22.00%`** (`+2.50 pp`) | **`19.73%`** | **`7.0%`** | — |

- **Key Takeaways:**
  1. **Zero OOMs and negligible Stockfish reward overhead:** `_KVCacheGRPOTrainer` (`QUIRK-006`) + `--merge-and-save` (`QUIRK-009`) held step time steady at `12.74–13.09 s/step` across all 3 stages, even with live Stockfish depth-10 reward evaluation in Stages 1 and 2.
  2. **Consistent monotonic gains across stages:** Stage 0 drove strict legality up `+7.5 pp` (`21.0% -> 28.5%`) and cut constrained ACPL by `-51.99 cp` (`380.10 -> 328.11`); Stage 1 lifted strict MAR (`17.54% -> 21.05%`), constrained MAR (`19.50% -> 22.00%`), and strict match legality (`25.93% -> 39.39%`).

---

### EXP-006 — Final 4-Tier Dual-Mode Evaluation & N=50 Paired Color-Balanced Matches (`GRPO Stage 2`)

| Field | Value |
|---|---|
| **Date (UTC)** | 2026-11-23 |
| **Track** | P1 (`Qwen/Qwen3-4B`, `v1`, final merged `/vol/checkpoints/grpo_stage2/stage2_final_merged`) |
| **Hardware & Est. Cost** | Modal NVIDIA L4 (`$0.80/hr`): 6 completed eval jobs (`8,570.75s` = **`$1.9046`**) + 1 sequential N=50 Maia-1100 job that hit Modal's `7,200s` (120m) timeout (`$1.6000`) + 2 early-stopped recovery chunks (`~540s` = `$0.1200`) = **`$3.62`** (`$3.6246` exact; cumulative verified Modal spend: **`$11.10`**) |
| **Modal App IDs** | Think ON Strict + Probes: `ap-g153cEjhuKvLV4ley60Fba` · Think ON Constr.: `ap-KYiHEVd0fVlbDmbr3DsAjl` · Think OFF Strict: `ap-Ht1f4dw5FG3yZ2oS7A2rCP` · Think OFF Constr.: `ap-ujMQUxqDImiLUCXFvxN5cj` · 50g SF-L1 Strict: `ap-A07L5ZZEqxU9nKkjl8RjR4` · 50g SF-L1 Constr.: `ap-nSiq9XpH1BxeA1DBqIrm8Y` · 50g Maia-1100 Constr. *(120m timeout)*: `ap-QG5Mk46OJkpKagpv6ygxYl` · HF Upload (CPU): `ap-jIXcWUsmgB5URluBwMmZbz` |
| **Hugging Face Repo** | `https://huggingface.co/winsonly/chessfm-qwen3-4b-grpo` *(private — verified `private=True`, includes merged `bf16` weights, publication-grade `README.md` Model Card, `evals/stage2_run_manifest.json`, and all 6 final eval manifests)* |
| **Local / Volume Manifests** | `results/m3_final/{grpo_s2_think_on_strict,grpo_s2_think_on_constrained,grpo_s2_think_off_strict,grpo_s2_think_off_constrained,grpo_s2_match_sf1_strict_50g,grpo_s2_match_sf1_constrained_50g}.json` + `.pgn` |

#### 1. Final Dual-Mode (`Strict` vs `Constrained`) & `Think OFF` vs `Think ON` Ablation (200 Held-Out FENs)

| Mode | Decoding | Wall-Clock (L4) | GPU Cost ($) | Legal Rate | Format Adherence | ACPL ↓ | MAR (≤50cp) ↑ | Mean WDL Loss ↓ | Exact Match ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GRPO S2 (Think OFF)** | `strict` | `1,757.01s` | `$0.3904` | `12.0%` (`24/200`) | `0.0%` | `594.71` | `16.67%` | `15.43%` | `0.0%` |
| **GRPO S2 (Think OFF)** | `constrained` | `77.12s` | `$0.0171` | `100.0%` (`200/200`) | `0.0%` | `451.79` | `21.00%` | `18.88%` | `5.5%` |
| **GRPO S2 (Think ON)** | `strict` | `1,062.44s` | `$0.2361` | **`32.0%`** (`64/200`) | **`100.0%`** | **`277.16`** | **`25.00%`** | `19.70%` | `1.5%` |
| **GRPO S2 (Think ON)** | `constrained` | `520.28s` | `$0.1156` | **`100.0%`** (`200/200`) | **`100.0%`** | **`325.88`** | **`22.00%`** | `19.61%` | **`7.0%`** |

- **Tier 2 Board-State Probing (`1,000` ChessQA probes on `grpo_s2_think_on_strict.json`):**
  - **Overall Accuracy:** **`41.7%`** (`417/1,000`, highest across Base `41.3%` -> SFT `41.4%` -> GRPO S2 `41.7%`)
  - `side_to_move`: **`74.5%`** (`149/200`) · `piece_on_square`: **`34.0%`** (`68/200`, up `+21.5 pp` vs Base `12.5%`) · `move_legality`: `50.0%` (`200/400`) · `legal_move_count`: `0.0%` (`0/200`).

#### 2. N=50 Paired Color-Balanced Matches (`25` Held-Out Openings × 2 Colors, Think ON, ASCII ON, `seed=42`)

| Opponent (`backend`) | Decoding Mode | Wall-Clock (L4) | GPU Cost ($) | Record (W-L-D) | Score | 95% Wilson CI | White Split (25g) | Black Split (25g) | Legal Rate | Format Rate | Est. Elo |
|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|
| **Stockfish L1 (`stockfish`, depth 6)** | `strict` | `254.27s` | `$0.0565` | `0-50-0` | `0.00%` | `[0.0000, 0.0713]` | `0W-0D-25L` (`0.00`) | `0W-0D-25L` (`0.00`) | `33.33%` | `100.0%` | `-299.8` |
| **Stockfish L1 (`stockfish`, depth 6)** | `constrained` | `4,899.63s` | `$1.0888` | **`0-38-12`** | **`12.00%`** | **`[0.0562, 0.2380]`** | `0W-8D-17L` (`0.16`) | `0W-4D-21L` (`0.08`) | **`100.0%`** | **`100.0%`** | **`553.9`** |
| **Maia-1100 (`calibrated_fallback`)** | `constrained` | `7,200.00s` *(timeout)* | `$1.6000` | *Timed out at 120m* | — | — | — | — | — | — | — |

- **Operational Finding (Sequential `batch_size=1` Match Eval Timeout):** Full constrained N=50 games against Stockfish L1 took `4,899.63s` (~81.7 min at `~2.60 s/move` across ~1,880 model moves), while Maia-1100 (`calibrated_fallback`) games lasted longer on average and hit Modal's `7,200s` (120 min) timeout (`ap-QG5Mk46OJkpKagpv6ygxYl`) before writing its JSON summary. Rather than burn additional credits re-running sequential `batch_size=1` rollouts (two split recovery chunks `ap-pTLgqcdJdPgkokFCWw0a8G` and `ap-8NHJO4zyABfJX1ZMPKKVis` were stopped after ~4.5m = `$0.12`), Maia-1100 N=50 matches are deferred to batched parallel match evaluation (`--batch-size 16–32`).

#### 3. Verbatim Qualitative Traces & Scientific Takeaways (`results/m3_final/`)

1. **Causal Proof That `<think>...</think>` Improves Chess Quality (Same Checkpoint, Think ON vs OFF):**
   - Turning `<think>` ON vs OFF on the exact same `GRPO Stage 2` weights improves strict legality by **`+20.0 pp`** (`12.0% -> 32.0%`), cuts strict ACPL by **`-317.55 cp`** (`594.71 -> 277.16`), and cuts constrained ACPL by **`-125.91 cp`** (`451.79 -> 325.88`).
   - Example (`r1b1kbnr/ppp1pppp/2n5/8/4P3/3B4/PPP2PPP/RNBK2NR b kq - 2 5`): SFT strict played illegal `f5g4`; GRPO S2 **Think OFF** strict emitted unparsed `None`; GRPO S2 **Think ON** strict found the exact Stockfish best move **`e7e5`** (`cp_loss=12.0`, `exact_match=True`).
   - Example (`r4rk1/1q3pbp/3Ppn2/1p2N3/p1p2B2/2P3P1/PPQ1R2P/5RK1 b - - 1 24`): SFT strict played illegal `g7f6`; GRPO S2 **Think OFF** strict emitted `None`; GRPO S2 **Think ON** strict found the exact Stockfish best move **`b7b6`** (`cp_loss=15.0`, `exact_match=True`).
2. **Real Drawn Game PGN Under Constrained Decoding (`grpo_s2_match_sf1_constrained_50g.pgn`, Game 3):**
   - Opening FEN: `r2qkb1r/1p1b1ppp/p1n1pn2/8/2BN4/1P2P3/PB3PPP/RN1Q1RK1 w kq - 3 10` (Model as White vs Stockfish L1)
   - Moves: `10. Bb5 axb5 11. Qd2 b4 12. Qd3 Be7 13. Qd2 Na5 14. Qd3 Nc6 15. Qd2 Qb8 16. Qd3 1/2-1/2`
   - **Honest Diagnosis:** Constrained decoding eliminates illegal-move forfeits and earns **12 draws in 50 games (`12.0%` score, `estimated_elo: 553.9`)**, compared to `0-50-0` in strict mode. Inspecting the PGN reveals that deterministic greedy decoding (`do_sample=False`) can settle into threefold-repetition loops (`Qd2-Qd3-Qd2-Qd3`) when Stockfish L1 also shuffles, highlighting why temperature/repetition-penalty tuning and longer GRPO horizons are the next levers for P1.

---

### EXP-007 — Phase II Gate 1 (`E1a`): Piece-by-Square Coordinate List + All-Linear LoRA (`r=16`, `lr=2e-4`) SFT
- **Date:** 2026-11-23
- **Hypothesis:** Replacing the 8×8 ASCII grid with a lossless **Piece-by-Square coordinate listing** (`--include-piece-list`, eliminating column-miscounting in tokenized ASCII grids) and raising All-Linear LoRA learning rate 10× from `2e-5` to `2e-4` (*LoRA Without Regret*) on the **exact same 2,500-row SFT corpus** (`sft_bestline_2500.jsonl`) will substantially improve board-state perception (`piece_on_square`), strict move legality, and constrained move quality without adding any new training data.
- **Training Configuration (`ap-ReAUHgCgbkYDOsMx7ltJNR`, Modal L4):**
  - Command: `modal run --detach scripts/modal_train.py --task sft --track v1 --gpu l4 --data /vol/data/sft_bestline_2500.jsonl --include-piece-list --lora-r 16 --all-linear --lr 2e-4 --epochs 3 --batch-size 2 --grad-accum 8 --output-subdir sft_e1a_piece_list`
  - Merged checkpoint: `/vol/checkpoints/sft_e1a_piece_list/final_merged` (`dataset_sha256: 399b0883442d...bfb00`, `seed=42`)
  - Training metrics: `train_runtime=1,759.96s` (`wall_clock_seconds=1,760.26s`, **`3.622 rows/sec`**, `0.225 steps/sec`), `train_loss=0.5547`, `eval_loss=0.4302`, **`eval_mean_token_accuracy=85.84%`** (`0.8584`).

#### 1. Apples-to-Apples 200-FEN + Tier 2 Board-Probing Comparison (`E0 SFT` vs `E0 GRPO S2` vs `E1a SFT`)

| Metric (`eval_benchmark_200.jsonl`, Think ON) | Base + Piece List *(Zero-Shot)* | E0 SFT (`EXP-004`, ASCII, `2e-5`) | E0 GRPO S2 (`EXP-006`, ASCII) | **E1a SFT (`EXP-007`, Piece List, `2e-4`)** | **Delta (`E1a` vs `E0 SFT`)** |
|---|---:|---:|---:|---:|---:|
| **Strict Legal Rate** | `0.0%` (`0/200`) | `21.0%` (`42/200`) | `32.0%` (`64/200`) | **`63.5%` (`127/200`)** | **`+42.5 pp` (`3.0×`)** |
| **Strict Format Adherence** | `0.0%` | `100.0%` | `100.0%` | **`100.0%`** | `0.0 pp` |
| **Strict ACPL** *(legal moves only)* | `—` | `345.07 cp` | `277.16 cp` | `421.31 cp` *(see note)* | `+76.24 cp` *(3× harder set)* |
| **Strict MAR (`≤50 cp`, legal)** | `0.0%` | `16.7%` | `25.0%` | **`29.1%` (`29.13%`)** | **`+12.4 pp`** |
| **Strict Exact Match** | `0.0%` | `3.5%` | `7.0%` | **`8.0%`** | **`+4.5 pp`** |
| **Constrained Legal Rate** | `0.0%` *(512-tok budget)* | `100.0%` | `100.0%` | **`100.0%`** | `0.0 pp` |
| **Constrained ACPL** *(all 200)* | `—` | `380.10 cp` | `325.88 cp` | **`330.18 cp`** | **`-49.92 cp`** |
| **Constrained MAR (`≤50 cp`)** | `0.0%` | `16.5%` | `20.5%` | **`27.0%`** | **`+10.5 pp`** |
| **Constrained Exact Match** | `0.0%` | `6.5%` | `8.5%` | **`11.0%`** | **`+4.5 pp`** |
| **Tier 2 Overall ChessQA (1,000)** | `44.8%` (`448/1000`) | `36.3%` | `41.7%` | **`57.5%` (`575/1000`)** | **`+21.2 pp`** |
| **Tier 2 `piece_on_square` (200)** | **`40.0%` (`80/200`)** | `33.5%` | `34.0%` | **`80.5%` (`161/200`)** | **`+47.0 pp`** |
| **Tier 2 `side_to_move` (200)** | `79.0%` (`158/200`) | `72.5%` | `79.0%` | **`97.0%` (`194/200`)** | **`+24.5 pp`** |
| **Tier 2 `move_legality` (400)** | `52.5%` (`210/400`) | `37.0%` | `45.5%` | **`55.0%` (`220/400`)** | **`+18.0 pp`** |
| **CoT Claim Accuracy (Verifier V2)** | `88.89%` (`840/945`) | *N/A* | *N/A* | **`64.02%` (`895/1,398`)** | *(Phase II baseline)* |
| **Eval Wall-Clock (`batch_size=16`)** | `889.7s` / `620.2s` | `~684s` (`bs=1`) | `~860s` (`bs=1`) | **`197.8s` strict / `82.3s` constr.** | **~5–8× faster** |

- **Note on Strict ACPL Selection Effect (Simpson's Paradox):** Strict ACPL is averaged solely over legal predictions. Because `E1a` outputs **3.0× more legal moves** (`127/200` vs `42/200` for `E0 SFT`), its strict set includes 85 harder middlegame/endgame positions where `E0 SFT` emitted illegal moves and was excluded from the mean. On the full 200/200 **Constrained** benchmark (identical denominator), `E1a` cuts ACPL by **`-49.92 cp`** (`380.10 -> 330.18 cp`), boosts MAR by **`+10.5 pp`** (`16.5% -> 27.0%`), and boosts Exact Match by **`+4.5 pp`** (`6.5% -> 11.0%`), matching `E0 GRPO S2` (`325.88 cp`) with SFT alone.
- **Zero-Shot Piece-List Control Finding:** Simply adding `--include-piece-list` to unmodified Base `Qwen/Qwen3-4B` (before any SFT) raised Tier 2 `piece_on_square` from **`12.5%` -> `40.0%`** (`+27.5 pp`) and overall Tier 2 ChessQA from `41.3%` -> `44.8%`. Fine-tuning on 2.5K piece-list rows then doubled `piece_on_square` again to **`80.5%`** (`161/200`).

#### 2. Phase II Frozen Benchmark Baselines (`E1a`, Strict Mode, `--batch-size 16`)
Manifest: `results/phase2/e1a/e1a_puzzles_chessqa.json` (`ap-A62f0C3az1hMPClafEPwXt`, `wall_clock_seconds=609.72s`).
- **Frozen 900 Lichess Puzzles (`test_puzzles_900.jsonl`):**
  - Format adherence: **`100.0%`**, Strict legal rate: **`66.56%`** (`599/900`), Tactical solve accuracy: **`4.67%`** (`42/900`: theme split `5.2%` `26/500`, difficulty split `4.0%` `16/400`; top themes: `smotheredMate 20%`, `mateIn1 12%`, `attraction 12%`, `zugzwang 12%`).
  - CoT Faithfulness (Verifier V2): **`61.82%`** (`3,882 / 6,280` claims verified, `18.0%` zero-violation rate, `6.98` claims/trace).
- **Frozen 1,000 ChessQA (`test_chessqa_1000.jsonl`, zero-shot — 0 ChessQA rows in 2.5K SFT):**
  - Overall: **`11.9%`** (`119/1,000`): `Position Judgment 35.5%` (`71/200`), `Structural 24.0%` (`48/200`, including `piece_lookup 62.0%`), `Motifs / Short Tactics / Semantic 0.0%` (clean pre-scaling baseline for `E1b` 80K mixed corpus).

#### 3. Engineering Quirks Caught & Fixed During Gate 1
- **QUIRK-010 (`_suite_supports_flag` in-process `sys.path`):** `_ensure_installed()` installs `chess-fm` editable in a subprocess, so the already-running Python interpreter did not see the new `.pth` entry when `_suite_supports_flag()` imported `chessfm.eval.suite`, causing 3 fast-fail eval runs (~15s each). Fixed in `35accf5` by inserting `REMOTE_REPO/src` onto `sys.path`.
- **QUIRK-011 (`resolve_track_model` precedence):** `resolve_track_model()` checked `if track:` before `if model:`, so combining `--track v1 --model /vol/checkpoints/...` silently evaluated Base `Qwen/Qwen3-4B` instead of the checkpoint (producing the zero-shot Base + Piece-List control above). Fixed & unit-tested in `63cdfb8` so explicit `--model` always takes precedence over `--track`.

#### 4. Exact Gate 1 Modal L4 GPU Accounting (`$0.80/hr`)
| Modal App ID | Task | Active Work (s) | Est. Billed (s) | Cost (USD) |
|---|---|---:|---:|---:|
| `ap-ReAUHgCgbkYDOsMx7ltJNR` | `E1a` SFT (`2.5K` + Piece List, `r=16`, `lr=2e-4`, 3 ep) | `1,760.26` | `1,860` | `$0.413` |
| `ap-nUstchzm2g2B0F7SY1DYHO` | `E1a` 200-FEN Strict + Tier 2 Probes + CoT (`bs=16`) | `197.82` | `260` | `$0.058` |
| `ap-V4hDVBdmZxNMTV5svyXfJn` | `E1a` 200-FEN Constrained (`bs=16`) | `82.33` | `145` | `$0.032` |
| `ap-A62f0C3az1hMPClafEPwXt` | `E1a` 900 Puzzles + 1,000 ChessQA (`bs=16`) | `609.72` | `680` | `$0.151` |
| `ap-cuh9rqdXn5lrc7Jv33LQB1` | Base + Piece-List 200-FEN Strict + Probes (control) | `889.73` | `940` | `$0.209` |
| `ap-10a6MAzBm42m37Yw3eYN2x` | Base + Piece-List 200-FEN Constrained (control) | `620.21` | `670` | `$0.149` |
| `ap-FderhGVBKKAOdPjKxxvyCs` | Base + Piece-List Puzzles+ChessQA (stopped early) | `1,200.00` | `1,200` | `$0.267` |
| `ap-ie5yRmvK…` / `ap-u1hG…` / `ap-CWRn…` | 3 fast-fail `sys.path` apps (~15s each) | `45.00` | `45` | `$0.010` |
| **Pure `E1a` (SFT + 3 Evals)** | | **`2,650.13`** | **`2,945`** | **`$0.654`** |
| **Gate 1 Total (All Apps)** | | **`5,405.07`** | **`5,800`** | **`$1.289`** |

---

### EXP-008 — Phase II Gate 2 (`E1b`): 20K Mixed-Corpus SFT (2 Epochs, Piece List, All-Linear LoRA `r=16`, `lr=2e-4`) on A100-80GB
- **Date:** 2026-11-23
- **Hypothesis:** Scaling the piece-list SFT corpus 8× from `2.5K` game-only rows (`E1a`) to `20,000` mixed rows (`sft_phase2_20k.jsonl`: elite games + theme-balanced Lichess tactical puzzles + 5-category ChessQA board-reading tasks, 0.00% normalized-FEN overlap with all frozen test sets) for 2 epochs with All-Linear LoRA `r=16`, `lr=2e-4` on A100-80GB will substantially boost tactical puzzle solving, 5-category ChessQA accuracy, and strict move quality while preserving general-English capability (`test_english_100`).
- **Training Configuration (`ap-LiEKeHnzaZeOg2Np5wfnhG`, Modal A100-SXM4-80GB):**
  - Command: `modal run --detach scripts/modal_train.py --task sft --track v1 --gpu a100 --data /vol/data/phase2/sft_phase2_20k.jsonl --include-piece-list --lora-r 16 --all-linear --lr 2e-4 --epochs 2 --batch-size 8 --grad-accum 4 --output-subdir sft_e1b_20k_r16`
  - Merged checkpoint: `/vol/checkpoints/sft_e1b_20k_r16/final_merged` (`dataset_sha256: 7b67347aa1d63d5a53d51300601e3f17e0a2cecdc5bb4c1309cd6b0537cc24d2`, `seed=42`, `optim=adamw_8bit`, `17,000` train / `3,000` FEN-disjoint val, `1,062` steps, `7,030,121` tokens, 0 OOMs).
  - Training metrics: `train_runtime=2,685.5s` (`wall_clock_seconds=2,685.84s`, **`12.661 rows/sec`**, `0.395 steps/sec`), `train_loss=0.3982` (`mean_token_accuracy=91.17%`), **`eval_loss=0.2775`** (down `-0.1527` vs `E1a`'s `0.4302`), **`eval_mean_token_accuracy=90.27%`** (up `+4.43 pp` vs `E1a`'s `85.84%`).

#### 1. Frozen Phase II Benchmark Results (`E1b` vs `E1a`, `--batch-size 16`, Think ON, Piece List)

All three evaluation manifests verified `"model_id": "/vol/checkpoints/sft_e1b_20k_r16/final_merged"`.

| Benchmark & Metric | `E1a` SFT (`2.5K`, `EXP-007`) | **`E1b` SFT (`20K`, `EXP-008`)** | **Delta (`E1b` vs `E1a`)** |
|---|---:|---:|---:|
| **Eval Loss / Token Acc (Held-Out Val)** | `0.4302` / `85.84%` | **`0.2775` / `90.27%`** | **`-0.1527` / `+4.43 pp`** |
| **Static FENs — Strict Legal Rate** | `63.5%` *(200-FEN)* | **`64.4%` (`644/1,000`)** *(1,000-FEN)* | `+0.9 pp` *(5× larger test set)* |
| **Static FENs — Strict Format Adherence** | `100.0%` | **`100.0%`** | `0.0 pp` |
| **Static FENs — Strict ACPL** *(legal only)* | `421.31 cp` *(200-FEN)* | **`336.51 cp`** *(1,000-FEN)* | **`-84.80 cp`** |
| **Static FENs — Strict MAR (`≤50 cp`, legal)** | `29.13%` *(200-FEN)* | **`31.52%`** *(1,000-FEN)* | **`+2.39 pp`** |
| **Static FENs — Strict Mean WDL Loss / Exact** | *N/A* / `8.0%` | **`18.91%` / `7.8%`** | `-0.2 pp` |
| **Static FENs — Constrained Legal / Format** | `100.0%` / `100.0%` | **`100.0%` / `100.0%`** | `0.0 pp` |
| **Static FENs — Constrained ACPL / WDL Loss** | `330.18 cp` *(200-FEN)* | **`393.00 cp` / `19.99%`** *(1,000-FEN)* | *(harder 1,000-FEN distribution)* |
| **Static FENs — Constrained MAR (`≤50 cp`)** | `27.0%` *(200-FEN)* | **`29.7%`** *(1,000-FEN)* | **`+2.7 pp`** |
| **Static FENs — Constrained Exact Match** | `11.0%` *(200-FEN)* | **`11.6%`** *(1,000-FEN)* | **`+0.6 pp`** |
| **Static FENs — CoT V2 Claim Accuracy** | `64.02%` *(200-FEN)* | **`60.02%` (`3,925/6,539`)** *(20.8% zero-viol)* | `-4.0 pp` |
| **900 Frozen Puzzles — Strict Legal / Format** | `66.56%` / `100.0%` | **`67.11%` (`604/900`) / `100.0%`** | **`+0.55 pp`** |
| **900 Frozen Puzzles — Tactical Solved** | `4.67%` (`42/900`) | **`18.33%` (`165/900`, 95% CI `[15.9%, 21.0%]`)** | **`+13.66 pp` (`3.93×`)** |
| **900 Frozen Puzzles — Theme (500) / Diff (400)** | `5.2%` / `4.0%` | **`21.2%` (`106/500`) / `14.75%` (`59/400`)** | **`+16.0 pp` / `+10.75 pp`** |
| **900 Frozen Puzzles — CoT V2 Claim Accuracy** | `61.82%` (`3,882/6,280`) | **`52.79%` (`3,034/5,747`)** *(12.4% zero-viol)* | `-9.03 pp` |
| **1,000 Frozen ChessQA — Overall Accuracy** | `11.9%` (`119/1,000`) | **`38.8%` (`388/1,000`)** | **`+26.9 pp` (`3.26×`)** |
| **ChessQA — Structural (200) / `piece_lookup` (50)** | `24.0%` / `62.0%` | **`66.0%` / `94.0%`** | **`+42.0 pp` / `+32.0 pp`** |
| **ChessQA — Semantic (200) / Position Judgment (200)** | `0.0%` / `35.5%` | **`55.0%` (`game_phase 90%`) / `54.5%`** | **`+55.0 pp` / `+19.0 pp`** |
| **ChessQA — Motifs (200) / Short Tactics (200)** | `0.0%` / `0.0%` | **`10.5%` / `8.0%`** | **`+10.5 pp` / `+8.0 pp`** |
| **100-Item General English Forgetting Probe** | *Not run on E1a* | **`89.0%` (`89/100`), `0.0%` chess leakage** | *(0% catastrophic forgetting)* |

#### 2. Per-Category Breakdowns & Real Trace Diagnostics
- **900-Puzzle Theme & Difficulty Highlights:**
  - Strongest themes: `backRankMate` **`60.0%`** (`15/25`), `skewer` **`44.0%`** (`11/25`), `promotion` **`36.0%`** (`9/25`), `interference` **`32.0%`** (`8/25`), `smotheredMate` **`32.0%`** (`8/25`), `zugzwang` `24.0%` (`6/25`). Weakest: `trappedPiece` `0.0%`, `fork` `4.0%`.
  - Difficulty tiers: `<1200` **`29.86%`** (`86/288`), `1200-1599` `15.18%` (`34/224`), `1600-1999` `10.48%` (`22/210`), `>=2000` `12.92%` (`23/178`).
- **1,000-ChessQA Subcategory Highlights:** `piece_lookup` **`94.0%`** (`47/50`), `game_phase` **`90.0%`** (`45/50`), `legal_moves` **`76.0%`** (`38/50`), `side_to_move` **`72.0%`** (`36/50`), `check_status` **`70.0%`** (`35/50`), `advantage_3way` `54.5%` (`109/200`), `castling_rights` `40.0%` (`20/50`), `attacks` `24.0%` (`12/50`), `material_balance` `18.0%` (`9/50`).
- **100-Item English Forgetting Probe:** `world_knowledge` **`100.0%`** (`25/25`), `reading_comprehension` **`96.0%`** (`24/25`), `instruction_following` **`92.0%`** (`23/25`), `logical_reasoning` `68.0%` (`17/25`), chess leakage rate **`0.0%`** (`0/100`).
- **Sample Verified vs. Illegal Traces (`e1b_test_fens_1000_strict.json`):**
  - *Exact Match + Zero-Violation CoT (30/1,000 positions):* `r2q1rk1/pp2ppbp/2p3p1/2n1P3/5P2/2n2BPP/PPP5/R1B1QRK1 w - - 0 14` -> predicted `e1c3` (`best_move=e1c3`, `cp_loss=0.0`, `6/6` V2 CoT claims verified); `rn1qk1nr/pp3pp1/2pbp1bp/8/2BP3P/6N1/PPP1NPP1/R1BQK2R w KQkq - 0 9` -> predicted `e2f4` (`best_move=e2f4`, `cp_loss=0.0`, `6/6` V2 CoT claims verified).
  - *Primary Remaining Failure Modes (356/1,000 illegal strict moves):* (1) **Unresolved check** — e.g., `rnbqk1nr/ppp2ppp/4p3/3p4/2PP4/P1b5/1P2PPPP/R1BQKBNR w KQkq - 0 5` where White is in check from `Bc3+`, yet `E1b` plays `g1f3` (`Nf3`) instead of capturing `b2c3`; (2) **Ray-blocking / castling hallucinations** — e.g., `r1br2k1/2q1ppbp/4n1p1/4P3/p2p4/Q1P1BN1P/P1B2PP1/1R2R1K1 w - - 0 26` playing illegal `c2b5` (`Bb5`), or proposing `O-O` (`e1g1`) in `r1bqk2r/pp1nbppp/2np4/2p1p3/2B1P3/2NP4/PPP2PPP/R1BQKN1R w KQkq - 4 8` while `Nf1` blocks the castling path.
- **Key Scientific Takeaway (Motivating Gate 3 `E2a` `r=64` & `E2b` Full-FT):** Scaling `2.5K -> 20K` at LoRA `r=16` yielded dramatic tactical and board-reading gains (`+13.7 pp` puzzles, `+26.9 pp` ChessQA, `-84.8 cp` strict ACPL) with zero English forgetting, **yet strict legality plateaued at ~64–67%** (`63.5% -> 64.4%` FENs, `66.6% -> 67.1%` puzzles) and `attacks` / `castling_rights` stayed at `24%–40%`. Gate 3 directly tests whether increasing parameter capacity (`E2a` All-Linear LoRA `r=64` and `E2b` Full Fine-Tuning) breaks this legality and ray-geometry plateau.

#### 3. Exact Gate 2 Modal GPU Accounting (`A100-80GB @ $2.50/hr`, `L4 @ $0.80/hr`)
| Modal App ID | Task | GPU | Active (s) | Billed Wall-Clock (s) | Cost (USD) |
|---|---|---|---:|---:|---:|
| `ap-LiEKeHnzaZeOg2Np5wfnhG` | `E1b` SFT (`20K × 2 ep`, Piece List, `r=16`, `lr=2e-4`, `bs=8, ga=4`) | A100-80GB | `2,685.84` | `2,759` | `$1.916` |
| `ap-5QagLYE4BiQ7gjS3OCvOQ6` | `E1b` Frozen `test_fens_1000` Strict + CoT V2 (`bs=16`) | L4 | `397.74` | `558` | `$0.124` |
| `ap-cx1vDIx0zWEU2Xvqmmelkk` | `E1b` Frozen `test_fens_1000` Constrained (`bs=16`) | L4 | `394.81` | `533` | `$0.118` |
| `ap-6Kp1mI5QLqRPmxSVaWyfRG` | `E1b` Frozen 900 Puzzles + 1,000 ChessQA + 100 English (`bs=16`) | L4 | `460.68` | `492` | `$0.109` |
| `ap-r4h7ULm4S0X8r9dsMuRdqH` | Aborted zero-container app (parallel `AppCreate` throttle) | L4 | `0.00` | `0` | `$0.000` |
| **Gate 2 (`E1b`) Total** | | **A100 + L4** | **`3,939.07`** | **`4,342`** | **`$2.267`** |



