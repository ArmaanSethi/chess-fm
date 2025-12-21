# ChessFM: 1.5B Chess Reasoning Roadmap

**Version**: 4.0 (Reasoning Era Edition)
**Last Updated**: 2025-12-21

> [!NOTE]
> **Project Status**: Educational Learning Project
> This roadmap is structured in two tiers:
> - **Standard Plan** ✅ — Get the fundamentals working first
> - **Bonus Upgrades** 🚀 — Advanced techniques to add after v1 works
>
> Complete the Standard Plan before attempting Bonus steps!

---

## Table of Contents
1. [Vision & Goals](#i-vision--goals)
2. [Success Metrics](#ii-success-metrics)
3. [Infrastructure](#iii-infrastructure)
4. [Thought Format](#iv-thought-format)
5. [Phase 0: Baseline](#v-phase-0-baseline-measurement)
6. [Phase 1: Distillation (SFT)](#vi-phase-1-distillation-sft)
7. [Phase 2: Reinforcement Learning](#vii-phase-2-reinforcement-learning-grpo)
8. [Reward Engineering](#viii-reward-engineering)
9. [Execution Roadmap](#ix-execution-roadmap)
10. [Bonus Upgrades Reference](#x-bonus-upgrades-reference)
11. [References](#xi-references)

---

## I. Vision & Goals

### Core Objective
Train a 1.5B parameter model to play chess at **1200 Elo** by reasoning through positions, not just memorizing moves.

### What We're Building

```
Input: [FEN position]
Output: <think>...reasoning...</think> e4
```

The model explains *why* it's making a move, like a chess tutor.

### 🚀 Bonus Vision: "System 2" Thinking
*After v1 works*, we can upgrade to structured verification:
```
<think>
    <threat_scan>...</threat_scan>
    <candidates>...</candidates>
    <verification>...</verification>
</think>
```
See [Bonus Upgrades](#x-bonus-upgrades-reference) for details.

---

## II. Success Metrics

### Standard Plan Metrics ✅

| Metric | Target | How to Measure |
|:-------|:-------|:---------------|
| **Illegal Move Rate** | < 5% | Parse outputs, validate with python-chess |
| **Format Adherence** | > 95% | `<think>` tags present and parseable |
| **Estimated Elo** | ~1000 | 500 games vs Stockfish Level 3 |

### 🚀 Bonus Metrics (Add Later)

| Metric | Target | What It Measures |
|:-------|:-------|:-----------------|
| **Illegal Move Rate** | **0%** | No constrained decoding needed |
| **Structure Adherence** | > 98% | All XML tags present |
| **"Aha!" Recovery Rate** | > 20% | Model catches its own mistakes |
| **Elo** | 1200+ | Beat Gemini Pro (~1050) |

### How Elo is Calculated

Using the [Elo Rating System](https://en.wikipedia.org/wiki/Elo_rating_system):

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

After $N$ games: $R'_A = R_A + K \cdot (S_A - E_A)$

Where $S_A$ = actual score (1=win, 0.5=draw, 0=loss), $K=32$.

---

## III. Infrastructure

### Stack
| Component | Tool | Purpose |
|:----------|:-----|:--------|
| Hardware | RTX 4090 (RunPod) | 24GB VRAM |
| Training | [unsloth](https://github.com/unslothai/unsloth) | 2x faster, 60% less VRAM |
| Inference | [vLLM](https://github.com/vllm-project/vllm) | Fast rollouts |
| Chess | [python-chess](https://python-chess.readthedocs.io/) + [Stockfish](https://stockfishchess.org/) | Validation & rewards |

### Base Model: Qwen-2.5-Math-1.5B

Per [Qwen2.5-Math paper](https://arxiv.org/abs/2409.12122), math models have better logic/reasoning pre-training.

---

## IV. Thought Format

### Standard Plan Format ✅ (Simple)

```xml
<think>
The position shows White with a strong center. 
Black's knight on c6 is blocking the c-pawn.
I should play e4 to control more space.
</think>
e4
```

**Requirements**:
- `<think>` and `</think>` tags
- Free-form reasoning inside
- Legal move at the end

### 🚀 Bonus Format: Socratic Structure

*Upgrade after v1 works* — enforces structured reasoning:

```xml
<think>
    <threat_scan>
        Is my King safe? Yes.
        Enemy threats: Rook eyes d7 pawn.
    </threat_scan>
    <candidates>Nf3, e4, c4</candidates>
    <verification>
        Nf3: Safe, develops piece.
        e4: <error>Loses pawn to Nxe4!</error>
        c4: Controls center. ✓
    </verification>
    <eval>+0.3</eval>
</think>
c4
```

Why this matters: The **Structure Hypothesis** says that valid logic *shape* helps small models reason better. See [Bonus Reference](#1-socratic-structure-the-structure-hypothesis).

---

## V. Phase 0: Baseline Measurement

### Goal
Pick the best base model and establish "Day 0" performance.

### Steps ✅

1. **Test 3 Models**:
   - [Qwen-2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
   - [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   - [Qwen-2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

2. **Benchmark**: 1,000 positions from [Lichess Elite Database](https://database.lichess.org/)

3. **Measure**: Legal rate, format adherence, Stockfish agreement

4. **Pick winner** → proceed to SFT

---

## VI. Phase 1: Distillation (SFT)

### Goal
Teach the model the `<think>` format and what good moves look like.

### Standard Plan ✅

#### Data Source Options

| Option | Size | Cost | Quality |
|:-------|:-----|:-----|:--------|
| [multimodal-reasoning-lab/chess](https://huggingface.co/datasets/multimodal-reasoning-lab/chess) | 30k | Free | Good, needs format conversion |
| Synthetic (GPT-4o-mini + Stockfish) | 15k | ~$15 | Excellent, exact format |

**Recommendation**: Start with open-source data. Add synthetic if format adherence < 90%.

#### Training Config
- **Method**: LoRA (rank 32)
- **Epochs**: 3
- **Loss**: Standard cross-entropy

$$\mathcal{L}_{SFT} = -\sum_{t} \log p_\theta(y_t | y_{<t}, x)$$

### 🚀 Bonus: Negative Data

*Add after v1 works* — teach model to recognize bad moves:

```xml
<think>
    Considering f3... <error>Weakens king diagonal, allows Qh4#!</error>
    Backtracking to e4.
</think>
e4
```

This is **contrastive learning** — teaching what NOT to do. See [Bonus Reference](#2-negative-data-contrastive-learning).

---

## VII. Phase 2: Reinforcement Learning (GRPO)

### Goal
Make the model actually *win games*, not just imitate the teacher.

### Algorithm: GRPO

[Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) — uses groups instead of a critic:

$$\mathcal{L}_{GRPO} = -\mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \hat{A}_i \cdot \log \pi_\theta(y_i | x) \right]$$

Where advantage is normalized within group:
$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G}$$

**Hyperparameters**:
- Group size $G = 8$
- KL penalty $\beta = 0.1$

### Curriculum ✅

| Stage | Opponent | When to Move On |
|:------|:---------|:----------------|
| 1 | Random mover | Win rate > 90% |
| 2 | Stockfish Skill 1 | Win rate > 60% |
| 3 | Stockfish Skill 3 | Win rate > 50% |

### 🚀 Bonus Curriculum

*Add after basic curriculum works*:

| Stage | Focus | Details |
|:------|:------|:--------|
| Puzzles | Tactics | Train on [Lichess puzzles](https://database.lichess.org/#puzzles) |
| Self-play | General | Model plays against older versions |

---

## VIII. Reward Engineering

### Standard Plan Reward ✅

$$R_{total} = R_{format} + R_{legal} + R_{chess}$$

| Component | Value | When |
|:----------|:------|:-----|
| $R_{format}$ | +0.1 | `<think>` tags present |
| $R_{legal}$ | -1.0 | Illegal move (ends episode) |
| $R_{chess}$ | $\tanh(\Delta\text{cp}/100)$ | Per-move quality |

The centipawn delta: $\Delta\text{cp} = \text{eval}(s') - \text{eval}(s)$

### 🚀 Bonus Rewards (Add Later)

| Component | Value | Purpose |
|:----------|:------|:--------|
| $R_{structure}$ | +0.2 | All Socratic tags present |
| $R_{threat}$ | +0.2 | `<threat_scan>` identifies real threats |
| $R_{candidate}$ | +0.2 | Final move was in `<candidates>` |
| $R_{budget}$ | -0.1 | Penalize < 50 or > 1024 tokens |

See [Bonus Reference](#3-advanced-rewards) for formulas.

---

## IX. Execution Roadmap

### Standard Plan ✅ (Do This First)

#### Phase 0: Setup & Baseline
- [ ] **0.1** Set up RunPod with RTX 4090
- [ ] **0.2** Run `setup_env.sh`, verify CUDA/Stockfish
- [ ] **0.3** Benchmark 3 models on 1k positions
- [ ] **0.4** Select best model

#### Phase 1: SFT
- [ ] **1.1** Download [chess dataset](https://huggingface.co/datasets/multimodal-reasoning-lab/chess)
- [ ] **1.2** Convert to `<think>` format
- [ ] **1.3** Train for 3 epochs
- [ ] **1.4** Verify format adherence > 95%

#### Phase 2: GRPO
- [ ] **2.1** Implement ChessGym environment
- [ ] **2.2** Implement reward function
- [ ] **2.3** Train Stage 1 (vs random)
- [ ] **2.4** Train Stage 2 (vs Stockfish 1)
- [ ] **2.5** Train Stage 3 (vs Stockfish 3)

#### Phase 3: Evaluation
- [ ] **3.1** Play 500 games vs Stockfish Level 3
- [ ] **3.2** Calculate Elo
- [ ] **3.3** 🎉 **Celebrate v1!**

---

### 🚀 Bonus Upgrades (After v1 Works)

#### Upgrade 1: Socratic Structure
- [ ] **B1.1** Create Socratic data synthesizer
- [ ] **B1.2** Generate 15k structured samples
- [ ] **B1.3** Re-train SFT with new format
- [ ] **B1.4** Verify structure adherence > 98%

#### Upgrade 2: Negative Data
- [ ] **B2.1** Generate 5k error-correction samples
- [ ] **B2.2** Mix into training data
- [ ] **B2.3** Test for self-correction behavior

#### Upgrade 3: Advanced Rewards
- [ ] **B3.1** Add $R_{structure}$ component
- [ ] **B3.2** Add $R_{budget}$ (length constraints)
- [ ] **B3.3** Re-run GRPO with new rewards

#### Upgrade 4: Puzzle Training
- [ ] **B4.1** Download [Lichess puzzles](https://database.lichess.org/#puzzles)
- [ ] **B4.2** Create tactical curriculum
- [ ] **B4.3** Train on mate-in-1, mate-in-2

#### Upgrade 5: TIC-GRPO
- [ ] **B5.1** Study [DAPO paper](https://arxiv.org/abs/2503.14476) for bias correction
- [ ] **B5.2** Implement trajectory importance weights
- [ ] **B5.3** Compare to standard GRPO

---

## X. Bonus Upgrades Reference

Detailed explanations of all advanced concepts.

---

### 1. Socratic Structure (The Structure Hypothesis)

**What**: Replace free-form `<think>` with structured XML tags.

**Why**: The [Structure Hypothesis](https://arxiv.org/abs/2412.xxxxx) states that **valid logic shape is a prerequisite for valid answers**. Small models following correct reasoning templates outperform larger models with unstructured outputs.

**The Tags**:
| Tag | Purpose | Verification |
|:----|:--------|:-------------|
| `<threat_scan>` | Situational awareness | Compare to Stockfish threats |
| `<candidates>` | Hypothesis generation | Final move should be listed |
| `<verification>` | Error detection | Look for `<error>` tags |
| `<eval>` | Value prediction | Compare to Stockfish eval |

**When to Add**: After v1 achieves > 90% format adherence with simple `<think>`.

---

### 2. Negative Data (Contrastive Learning)

**What**: Train on examples where the model considers a bad move, then rejects it.

**Why**: Based on [Contrastive Learning](https://arxiv.org/abs/2010.05113) — teaching "what NOT to do" builds clearer decision boundaries.

**Example**:
```xml
<think>
    <candidates>f3, e4</candidates>
    <verification>
        f3: <error>Weakens king's diagonal, allows Qh4#!</error>
        e4: Safe. ✓
    </verification>
</think>
e4
```

**The "Aha!" Metric**: Measures how often the model:
1. Proposes a candidate
2. Rejects it in verification
3. Picks something better

$$\text{Aha! Rate} = \frac{\text{corrections that improved position}}{\text{total outputs}}$$

**When to Add**: After v1 works, generate 5k error samples (~$3 API cost).

---

### 3. Advanced Rewards

**What**: Add structure and length-based reward components.

#### Structure Reward ($R_{shape}$)

$$R_{shape} = 0.1 \cdot \mathbb{1}[\text{all tags}] + 0.2 \cdot \mathbb{1}[\text{threat correct}] + 0.2 \cdot \mathbb{1}[a \in C]$$

Where $C$ is the candidate set and $a$ is the final move.

#### Budget Forcing ($R_{budget}$)

Prevents degenerate behaviors (guessing or rambling):

$$R_{budget} = \begin{cases} -0.1 & |\text{think}| < 50 \\ -0.1 & |\text{think}| > 1024 \\ 0 & \text{otherwise} \end{cases}$$

Reference: [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)

**When to Add**: After standard rewards produce stable training.

---

### 4. TIC-GRPO (Trajectory Importance Correction)

**What**: Bias correction for GRPO that accounts for off-policy data.

**Why**: Standard GRPO can be biased when groups don't represent the true policy distribution.

**Upgrade Path**:
1. Run standard GRPO first
2. Monitor for training instability or reward hacking
3. If issues arise, implement importance weights per [DAPO paper](https://arxiv.org/abs/2503.14476)

**When to Add**: Only if you observe bias issues (reward hacking, unstable loss).

---

### 5. Value Function Distillation

**What**: Train model to predict position evaluation inside `<eval>` tag.

**Why**: Per [AlphaZero](https://arxiv.org/abs/1712.01815), a value estimate helps guide search. We embed this IN the thought process.

**How**:
1. Include Stockfish eval in training data: `<eval>+0.35</eval>`
2. Optionally add reward: $R_{eval} = +0.1$ if $|\text{pred} - \text{SF}| < 100\text{cp}$

**When to Add**: After Socratic structure is working well.

---

## XI. References

### Core Papers
| Paper | Link | Relevance |
|:------|:-----|:----------|
| **GRPO** | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | Our RL algorithm |
| **Qwen2.5-Math** | [arXiv:2409.12122](https://arxiv.org/abs/2409.12122) | Base model choice |
| **DeepSeek-R1** | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) | Self-correction patterns |
| **AlphaZero** | [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) | Value function inspiration |

### Technical Resources
| Resource | Link | Use |
|:---------|:-----|:----|
| **Dynomight Chess** | [Part 1](https://dynomight.substack.com/p/chess), [Part 2](https://dynomight.substack.com/p/more-chess) | Regurgitation technique |
| **AI Chess Leaderboard** | [dubesor.de](https://dubesor.de/chess/chess-leaderboard) | LLM Elo comparisons |
| **Lichess Database** | [database.lichess.org](https://database.lichess.org/) | Training/eval data |

### Foundational Concepts
| Topic | Link |
|:------|:-----|
| Knowledge Distillation | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| Contrastive Learning | [arXiv:2010.05113](https://arxiv.org/abs/2010.05113) |
| Elo Rating System | [Wikipedia](https://en.wikipedia.org/wiki/Elo_rating_system) |

---

## Cost Estimates

### Standard Plan ✅
| Phase | Time | Cost |
|:------|:-----|:-----|
| Setup & Baseline | 4 hr | $1.80 |
| SFT | 4 hr | $1.80 |
| GRPO | 20 hr | $9.00 |
| **Total** | **~28 hr** | **~$13** |

### With All Bonuses 🚀
| Addition | Time | Cost |
|:---------|:-----|:-----|
| Socratic Data Synthesis | 4 hr | +$15 (API) |
| Negative Data | 2 hr | +$3 (API) |
| Advanced GRPO | 10 hr | +$4.50 |
| **Total** | **~44 hr** | **~$35** |

---

*ChessFM Roadmap v4.0 — Start simple, add complexity progressively!*
