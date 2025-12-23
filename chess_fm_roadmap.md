# ChessFM: 1.5B Chess Reasoning Roadmap

**Version**: 4.0 (Direct RL Approach)
**Last Updated**: 2024-12-20

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

| Candidate | Why? |
|:----------|:-----|
| **Qwen-2.5-Math-1.5B** | Strong reasoning/math pre-training. Likely best at logic. |
| **DeepSeek-Coder-1.3B** | Code models often handle structured notation (like PGN) well. |
| **Gemma-2-2B-it** | Google's latest small model, instruction-tuned, strong performance. |

2. **Benchmark**: 1,000 positions from [Lichess Elite Database](https://database.lichess.org/)

3. **Measure**: Legal rate, format adherence, Stockfish agreement

4. **Pick winner** → proceed to SFT

---

## VI. Phase 1: Direct RL Training (GRPO)

> [!IMPORTANT]
> **Updated Approach**: We skip SFT and go directly to RL. Chess has **verifiable rewards** (legal/illegal, win/lose), making direct GRPO training viable. SFT can be added later for reasoning traces.

### Why Skip SFT?
1. **Verifiable Environment**: Chess moves are objectively legal or illegal—no subjective labeling needed.
2. **Dense Rewards**: Stockfish provides per-move evaluation (centipawn scores).
3. **Simpler Pipeline**: No synthetic data generation, no API costs.
4. **Research Support**: DeepSeekMath proved GRPO works for reasoning without extensive SFT.

### Implementation: Direct GRPO

#### Training Framework
- **Library**: `unsloth` + `trl` (for GRPO implementation)
- **Hardware**: 1x RTX 4090 (24GB) or Apple Silicon (MPS)
- **Quantization**: 4-bit QLoRA for memory efficiency

#### GRPO Configuration
```python
grpo_config = {
    "group_size": 8,           # Generate 8 responses per prompt
    "learning_rate": 1e-5,
    "kl_coef": 0.05,           # Prevent policy collapse
    "max_grad_norm": 1.0,
    "temperature": 0.7,        # Diverse responses for group
    "max_new_tokens": 10,      # Chess moves are short
}
```

#### Staged Curriculum (Detailed)

| Stage | Opponent | Reward Signal | Steps | Success Criteria |
|:------|:---------|:--------------|:------|:-----------------|
| **0** | None (single move) | `+1` legal, `-1` illegal | 2,000 | Legal rate ≥ 50% |
| **1** | Random mover | Win: `+1`, Lose: `-1` | 5,000 | Win rate > 80% |
| **2** | Stockfish Level 1 | Win/Draw/Lose + Δcp bonus | 10,000 | Win rate > 50% |
| **3** | Stockfish Level 3 | Same + Stockfish agreement | 10,000 | Elo ≥ 1000 |

#### Stage 0: Legality Training (Critical Bootstrap)
```python
def reward_stage_0(fen, generated_move):
    board = chess.Board(fen)
    try:
        board.parse_san(generated_move)
        return +1.0  # Legal move
    except:
        return -1.0  # Illegal move
```

---

## VII. Phase 1.5: SFT for Reasoning (OPTIONAL)

> [!NOTE]
> **This phase is OPTIONAL**. Complete it if you want the model to produce `<think>` reasoning traces. The RL-trained model will know *what* moves to play; SFT teaches it *how to explain* those moves.

### When to Add SFT
- After RL achieves ≥80% legal move rate
- If you want natural language explanations
- If you get free API credits (Gemini, GPT-4o, etc.)

### Standard Plan ✅

| Source | Dataset | Description | Role |
|:-------|:--------|:------------|:-----|
| **Primary** | `multimodal-reasoning-lab/chess` | Contains explicit "THOUGHT" process. | **Core SFT Data**. |
| **Secondary** | `MATE` (HuggingFace) | 1M positions with expert annotations. | **Augmentation**. |
| **Tertiary** | `laion/strategic-game-chess` | ChessGPT game-language dataset. | **Augmentation**. |
| **Synthetic** | Gemini 2.0 Flash / GPT-4o | Custom `<think>` trace generation. | **Gap filling**. |

### Synthetic Data Generation Options

| API | Cost | Quality | Notes |
|:----|:-----|:--------|:------|
| **Gemini 2.0 Flash** | Free tier: 1500 req/day | Good | Best for free credits |
| **Gemini 1.5 Pro** | $1.25/1M input tokens | Excellent | If budget allows |
| **GPT-4o** | $5/1M input tokens | Excellent | Highest puzzle accuracy (~50%) |
| **Claude 3.5 Sonnet** | $3/1M input tokens | Very Good | Strong reasoning |

### Synthetic Data Pipeline
1. **Get Move from Stockfish**: Ensures legal, optimal move.
2. **Generate Reasoning from LLM**: Explain *why* the move is good.
3. **Format**: Convert to `<think>` XML format.

```python
prompt = f"""
FEN: {fen}
Stockfish Best Move: {stockfish_move}

Explain why this is a good move in 2-3 sentences.
Consider: piece activity, king safety, pawn structure, tactics.
"""
# LLM generates explanation
# Wrap in <think>...</think> format
```

### Training Configuration
- **Base Model**: RL-trained checkpoint (or base Qwen)
- **LoRA Rank**: 32
- **Epochs**: 3
- **Batch Size**: 8 (Accumulation 4)
- **Dataset Size**: 10,000-30,000 samples

### Verification Protocol: SFT
| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| S.1 | Prepare Dataset | Inspection | `<think>` tags present. Moves legal. |
| S.2 | Train 1 epoch | Loss curve | Loss decreases. |
| S.3 | Inference Test | Parse output | **Format adherence ≥ 99%**. |

---

## VIII. Reward Engineering

### The Formula (Direct RL)
$$R_{total} = R_{legality} + R_{outcome} + R_{quality}$$

| Component | Signal | Value |
|:----------|:-------|:------|
| **$R_{legality}$** | Legal move | `+0.1` legal, `-1.0` illegal (episode ends) |
| **$R_{outcome}$** | Game result | `+1.0` win, `+0.3` draw, `-0.5` lose |
| **$R_{quality}$** | Stockfish Δcp | `tanh(Δcp / 100)` per move |

### Why This Works
1. **Legality is binary**: Model quickly learns move format.
2. **Outcome is sparse but strong**: Drives strategic play.
3. **Δcp is dense**: Provides signal every move to avoid reward sparsity.

---

## IX. Evaluation Framework

---

## IX. Execution Roadmap

### Standard Plan ✅ (Do This First)

#### Phase 0: Setup & Baseline
- [ ] **0.1** Set up RunPod with RTX 4090
- [ ] **0.2** Run `setup_env.sh`, verify CUDA/Stockfish
- [/] **0.3** Benchmark 3 models on 1k positions *(in progress)*
- [ ] **0.4** Select best model

#### Phase 1: SFT Data Preparation
- [x] **1.1** Create data generation scripts (`data_generation/`)
- [x] **1.2** Create position database (500 positions ready)
- [x] **1.3** Create format converter for `<think>` format
- [ ] **1.4** Generate 15k samples
- [ ] **1.5** Train for 3 epochs
- [ ] **1.6** Verify format adherence > 95%

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

### Phase 0: Baseline & Infrastructure ✅
> [!NOTE]
> **Maintenance Instruction**: After completing each step in this roadmap, mark the item as `[x]` and update the date/version at the top if necessary.

- [x] **0.1** Set up environment (local Mac or RunPod RTX 4090).
- [x] **0.2** Run `setup_env.sh`. Verify all components.
- [x] **0.3** Run Tokenizer Audit (Pre-Fix).
- [x] **0.4** **Model Tournament**: Run benchmark on Qwen/Llama/DeepSeek.
- [x] **0.5** **Prompt Search**: Test Zero-shot vs Regurgitation vs CoT.
- [x] **0.6** Download all candidate models locally.

### Phase 1: Tokenization & Direct RL
- [ ] **1.1** Add special tokens & resize embeddings.
- [ ] **1.2** Implement GRPO training loop with `unsloth` + `trl`.
- [ ] **1.3** Run Stage 0: Legality training (2k steps).
- [ ] **1.4** Run Stage 1: vs Random Mover (5k steps).
- [ ] **1.5** Run Stage 2: vs Stockfish Level 1 (10k steps).
- [ ] **1.6** Run Stage 3: vs Stockfish Level 3 (10k steps).

### Phase 1.5: SFT for Reasoning (OPTIONAL)
- [ ] **1.5.1** Acquire API credits (Gemini 2.0 Flash free tier or similar).
- [ ] **1.5.2** Download `multimodal-reasoning-lab/chess` dataset.
- [ ] **1.5.3** Generate synthetic `<think>` traces if needed.
- [ ] **1.5.4** Train SFT on RL checkpoint (3 epochs).

### Phase 2: Final Eval
- [ ] **2.1** Run Elo Estimation vs Stockfish 5.
- [ ] **2.2** Compare vs Gemini 3 Pro Preview benchmark (~1050).
- [ ] **2.3** Test reasoning quality (if SFT completed).

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
