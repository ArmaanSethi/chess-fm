# ChessFM: 1.5B Chess Reasoning Roadmap

**Version**: 3.5 (Detailed Production Plan)
**Last Updated**: 2024-12-15

> [!NOTE]
> **Project Status**: Educational Research
> This is a personal project to explore Reinforcement Learning and reasoning in Small Language Models.
> **"FM" stands for Foundation Model, not FIDE Master.** We are not claiming to build a 2300 Elo engine, but rather a reasoning agent that can explain its moves.

---

## Table of Contents
1. [Vision & Goals](#i-vision--goals)
2. [Success Metrics & Exit Criteria](#ii-success-metrics--exit-criteria)
3. [Infrastructure](#iii-infrastructure)
4. [Board Representation & Tokenization](#iv-board-representation--tokenization)
5. [Phase 0: Baseline Measurement](#v-phase-0-baseline-measurement)
6. [Phase 1: Distillation (SFT)](#vi-phase-1-distillation-sft)
7. [Phase 2: Reinforcement Learning (GRPO)](#vii-phase-2-reinforcement-learning-grpo)
8. [Reward Engineering](#viii-reward-engineering)
9. [Evaluation Framework](#ix-evaluation-framework)
10. [Contingency Plans](#x-contingency-plans)
11. [Cost & Time Estimates](#xi-cost--time-estimates)
12. [Execution Roadmap](#xii-execution-roadmap)

---

## I. Vision & Goals

### Core Objective
Train a 1.5B parameter model to play chess at a **competent amateur level (1200 Elo)** by reasoning through positions, rather than just memorizing moves.

### Long-Term Vision: "The Pocket Tutor"
While current engines (Stockfish) are invincible calculators, they are terrible teachers. They give you a line like `+5.4 (1. e4 e5 2. Nf3)` but don't explain *why*.
Our ultimate goal is a **tiny, quantized model** that can run locally on a phone, watching your game and whispering natural language advice:
> *"Don't move the knight there! You'll leave your king open to a back-rank mate in 3 moves because the rook is controlling the d-file."*

---

## II. Success Metrics & Exit Criteria

> [!IMPORTANT]
> **This section defines what "done" looks like.** Every decision in this plan serves these metrics.

### Primary Success Metrics

| Metric | Minimum Viable | Target | Stretch Goal |
|:-------|:--------------|:-------|:-------------|
| **Illegal Move Rate** | < 5% | 0% | 0% with no constrained decoding |
| **Estimated Elo** | 800 (Beat GPT-5-nano) | 1200 (Beat Gemini 3 Pro Preview) | 1500 (Club Player) |
| **Stockfish Top-3 Agreement** | 30% | 50% | 70% |
| **Self-Play Improvement** | >55% vs T-1 | >60% vs T-1 | >70% vs T-1 |

### How Elo Will Be Measured
1. Play 500 games against Stockfish at a fixed Skill Level (e.g., Level 5, ~1500 Elo).
2. Use the [Elo Rating System formula](https://en.wikipedia.org/wiki/Elo_rating_system) to calculate model Elo based on win/draw/loss rates.
3. **Verification**: Results must be reproducible with a fixed random seed.

---

## III. Infrastructure

### Selected Stack: RunPod + RTX 4090
-   **Hardware**: 1x NVIDIA RTX 4090 (24GB VRAM).
-   **OS**: Ubuntu 22.04 (Standard for ML).
-   **Environment**:
    -   `unsloth`: For 2x faster training and 60% less VRAM.
    -   `vLLM`: For high-throughput rollout generation.
    -   `python-chess` + `stockfish`: For environment simulation and rewards.

### Verification Protocol: Infrastructure Setup
| Step | Action | Verification Command | Success Criteria |
|:-----|:-------|:---------------------|:-----------------|
| 1.1 | Run `setup_env.sh` | `nvidia-smi` | Shows RTX 4090, ~24GB VRAM. |
| 1.2 | Verify CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | Prints `True`. |
| 1.3 | Verify Stockfish | `stockfish` then type `uci` | Prints `uciok`. |
| 1.4 | Verify vLLM | `vllm serve Qwen/Qwen-2.5-Math-1.5B-Instruct --port 8000` | Server starts, no OOM. |

---

## IV. Board Representation & Tokenization

### Design Decision: Special Tokens
We will add chess pieces (`<|R|>`, `<|n|>`, etc.) as special tokens to guarantee deterministic 1-to-1 tokenization.

### "Regurgitation" Strategy
Inspired by [Dynomight](https://dynomight.substack.com/p/more-chess), we will force the model to **output the full PGN history** before generating a move. This grounds the model in the game state.

### Implementation Plan
1.  **Audit**: Run `audit_tokenizer.py` to confirm variance in base tokenizer.
2.  **Add Tokens**: Add 14 special tokens (pieces + empty + separators).
3.  **Resize**: Resize model embeddings.
4.  **Warm-up**: Train on 10k FEN-to-Description pairs to ground the new embeddings.

---

## V. Phase 0: Baseline Measurement

> [!IMPORTANT]
> **Day 0 Baseline**: Before training, we measure the base `Qwen-2.5-Math-1.5B` model on 100 FENs.

**Metrics to Record**:
- Legal Move Rate (Expected: 10-30%)
- Format Adherence (Expected: <10%)
- Stockfish Agreement (Expected: ~5%)

---

## VI. Phase 1: Distillation (SFT)

### Goal
Teach the model **how to reason** (`<think>` format) and **what legal moves look like**.

### Data Strategy: Hybrid (Open Source + Synthetic)

We will leverage existing high-quality datasets to save API costs, supplementing with synthetic data only if needed.

| Source | Dataset | Description | Role |
|:-------|:--------|:------------|:-----|
| **Primary** | `multimodal-reasoning-lab/chess` | Contains explicit "THOUGHT" process for chess moves. | **Core SFT Data** (Format alignment needed). |
| **Secondary** | `MATE` (HuggingFace) | 1M positions with expert annotations. | **Pre-training** for chess concepts. |
| **Tertiary** | Synthetic (GPT-4o) | Custom generated `<think>` traces. | **Gap filling** for specific formats. |

### Teacher Model Strategy (Synthetic)
- **Move Selection**: **Stockfish 16** (Ground Truth).
- **Reasoning Generation**: **GPT-4o**.
- **Justification**: GPT-4o has the highest puzzle accuracy (~50%), making it the best *explainer*. Stockfish ensures the move itself is perfect, mitigating GPT-4o's tendency to make illegal moves.

### Dataset Specification
- **Size**: 30,000 samples (from `multimodal-reasoning-lab/chess`).
- **Format**: Convert to our specific `<think>` XML format.
- **Prompt Augmentation**: Include full PGN history (Regurgitation) in the input to improve state tracking.

### Training Configuration
- **Base Model**: `Qwen/Qwen-2.5-Math-1.5B-Instruct`
- **LoRA Rank**: 32
- **Epochs**: 3
- **Batch Size**: 8 (Accumulation 4)

### Verification Protocol: SFT
| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| 3.1 | Prepare Dataset | Inspection | `<think>` tags present. Moves legal. |
| 3.2 | Train 1 epoch | Loss curve | Loss decreases. |
| 3.3 | Inference Test | Parse output | **Format adherence ≥ 99%**. |

---

## VII. Phase 2: Reinforcement Learning (GRPO)

### Goal
Teach the model to play **strategically good** chess through self-improvement.

### Algorithm: GRPO (Group Relative Policy Optimization)
- **Group Size**: $G=8$.
- **Rollout**: Per-Move (initially) for dense rewards.

### Staged Curriculum (The Opponent)

| Stage | Opponent | Focus | Success Criteria |
|:------|:---------|:------|:-----------------|
| **0** | None (Single Move) | Legality & Format. | Legal Rate ≥ 95%. |
| **1** | Random Mover | Basic Tactics (Mate in 1/2). | Win Rate > 90%. |
| **2** | Stockfish Skill 1 | Real Chess. | Win Rate > 60%. |
| **3** | Stockfish Skill 3 | Competitive Play. | Elo ≥ 1000. |

---

## VIII. Reward Engineering

### The Formula
$$R_{total} = 0.1 \cdot R_{format} + 1.0 \cdot R_{legality} + 1.0 \cdot R_{chess}$$

1.  **$R_{format}$**: +0.1 if `<think>` tags exist.
2.  **$R_{legality}$**: -1.0 if illegal (episode ends).
3.  **$R_{chess}$**: $\tanh(\Delta \text{Centipawns} / 100)$.

---

## IX. Evaluation Framework

### Metrics
1.  **Illegal Move Rate**: Target 0%.
2.  **Stockfish Agreement**: Target 50% (Top-3).
3.  **Estimated Elo**: Target 1200 (vs Stockfish Level 5).

### Tournament Protocol
-   **Frequency**: Every 500 steps.
-   **Match**: 100 games vs previous best checkpoint.
-   **Pass**: >55% win rate.

---

## X. Contingency Plans

| Problem | Solution |
|:--------|:---------|
| **SFT Format < 95%** | Filter dataset more aggressively. Add few-shot examples in prompt. |
| **Illegal Moves Persist** | Implement **Constrained Decoding** (mask illegal logits). |
| **Reward Hacking** | If model finds "passive" way to avoid penalty, add "Win Bonus" (+1.0). |
| **OOM Errors** | Reduce Group Size to 4. Use 8-bit quantization. |

---

## XI. Cost & Time Estimates

| Phase | Time | Cost (RunPod) |
|:------|:-----|:--------------|
| **Setup & Baseline** | 2 hr | $0.90 |
| **Tokenizer Fix** | 2 hr | $0.90 |
| **SFT Training** | 3 hr | $1.35 |
| **GRPO (Stages 0-3)** | 25 hr | $11.25 |
| **Eval & Buffer** | 10 hr | $4.50 |
| **TOTAL** | **~42 hr** | **~$19.00** |

*Note: Data costs are negligible if using Open Source datasets.*

---

## XII. Execution Roadmap

### Phase 0: Baseline & Infrastructure
- [ ] **0.1** Set up RunPod pod with RTX 4090.
- [ ] **0.2** Run `setup_env.sh`. Verify all components.
- [ ] **0.3** Run Tokenizer Audit (Pre-Fix).
- [ ] **0.4** Run Baseline Measurement (100 FENs).

### Phase 1: Tokenization & Warm-up
- [ ] **1.1** Add special tokens & resize embeddings.
- [ ] **1.2** Train Embedding Warm-up (10k samples).

### Phase 2: SFT (Distillation)
- [ ] **2.1** Download `multimodal-reasoning-lab/chess`.
- [ ] **2.2** Format data to `<think>` XML with PGN history.
- [ ] **2.3** Train SFT (3 epochs). Verify format.

### Phase 3: GRPO (RL)
- [ ] **3.1** Implement Rewards & Gym.
- [ ] **3.2** Run Stage 0 (Legality).
- [ ] **3.3** Run Stage 1 (vs Random).
- [ ] **3.4** Run Stage 2 (vs Stockfish 1).

### Phase 4: Final Eval
- [ ] **4.1** Run Elo Estimation vs Stockfish 5.
- [ ] **4.2** Compare vs Gemini 3 Pro Preview benchmark (~1050).

---

## Appendix: References
1. **Dynomight**: [Something weird is happening with LLMs and chess](https://dynomight.substack.com/p/chess)
2. **Dynomight**: [OK, I can partly explain the LLM chess weirdness now](https://dynomight.substack.com/p/more-chess)
3. **Dubesor.de**: [AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard)
4. **GRPO Paper**: DeepSeekMath (arXiv:2402.03300)

---

*ChessFM Roadmap v3.5 (Detailed)*
