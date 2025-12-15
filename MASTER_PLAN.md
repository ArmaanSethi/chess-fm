# Verifiable-Zero: The Ultimate Master Plan

**Version**: 3.0 (Production-Ready, Fully Verifiable)
**Last Updated**: 2024-12-15

---

## Table of Contents
1. [Executive Summary](#i-executive-summary)
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

## I. Executive Summary

### Project Goal
Train a 1.5B parameter reasoning model to play chess at a **measurable, competitive level** using Group Relative Policy Optimization (GRPO) and Verifiable Rewards.

### Learning Objectives
This project is a structured curriculum to master four pillars of modern AI Engineering:
1. **Synthetic Data Pipelines**: Creating high-quality distillation datasets from stronger models.
2. **System 2 Reasoning**: Implementing Chain-of-Thought (CoT) via reinforcement learning.
3. **Reward Engineering**: Designing dense, scalar signals for complex sequential environments.
4. **Hardware Optimization**: Fitting training loops into consumer hardware (24GB VRAM).

### Why This Matters
- LLMs struggle with chess because they lack spatial reasoning and long-horizon planning.
- Current frontier models (GPT-4, Claude) achieve only ~1200-1350 Elo.
- Successfully training a 1.5B model to play *legal, strategic* chess would demonstrate that RL + verifiable rewards can unlock structured reasoning in small models.

---

## II. Success Metrics & Exit Criteria

> [!IMPORTANT]
> **This section defines what "done" looks like.** Every decision in this plan serves these metrics.

### Primary Success Metrics

| Metric | Minimum Viable | Target | Stretch Goal |
|:-------|:--------------|:-------|:-------------|
| **Illegal Move Rate** | < 5% | 0% | 0% with no constrained decoding |
| **Estimated Elo** | 800 | 1200 | 1500 |
| **Stockfish Top-3 Agreement** | 30% | 50% | 70% |
| **Self-Play Improvement** | >55% vs T-1 | >60% vs T-1 | >70% vs T-1 |

### How Elo Will Be Measured
1. Play 500 games against Stockfish at a fixed Skill Level (e.g., Level 5, ~1500 Elo).
2. Use the [Elo Rating System formula](https://en.wikipedia.org/wiki/Elo_rating_system) to calculate model Elo based on win/draw/loss rates.
3. **Verification**: Results must be reproducible with a fixed random seed.

### Exit Criteria (When Are We Done?)
- **Phase 1 (SFT) Complete**: Format adherence ≥ 99%. Legal move rate ≥ 90%.
- **Phase 2 (GRPO) Complete**: Legal move rate = 100%. Elo ≥ 800.
- **Project Complete**: Elo ≥ 1200 OR diminishing returns observed over 5 consecutive evaluation checkpoints.

---

## III. Infrastructure

### Design Analysis: Where to Run?

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. Google Colab (Free/Pro)** | Jupyter-based GPU access | Low cost ($0-10/mo). Easy start. | Background processes (vLLM) break frequently. I/O timeouts on GDrive. 24hr session limit kills long runs. No `tmux`. | **REJECTED** |
| **B. Local Mac (M2/M3 Max)** | Apple Silicon GPU | Free. Persistent storage. | `unsloth` and `vLLM` are CUDA-optimized. MPS backend is experimental and slow. Debugging library compatibility is painful. | **REJECTED** |
| **C. Modal / Lambda Labs** | Serverless GPU functions | Pay-per-second. Good for burst. | Less control over environment. Cold start latency. Not ideal for long RL loops. | **CONSIDERED** |
| **D. RunPod (Persistent Pod)** | Full Linux VM with GPU | Full SSH access. `tmux` for background jobs. Native CUDA. Pay only for active hours (~$0.45/hr for 4090). | Requires manual setup. Data stored on pod (must snapshot). | **SELECTED** |

### Why RunPod Wins
1. **Full Control**: Real Linux terminal with `tmux` for managing `vllm serve` and `python train.py` simultaneously.
2. **Unsloth Compatibility**: CUDA-native kernels work out of the box.
3. **Cost Efficiency**: $0.45/hr for RTX 4090. 20 hours of focused training = $9.

### Selected Stack
| Component | Choice | Purpose |
|:----------|:-------|:--------|
| **GPU** | NVIDIA RTX 4090 (24GB VRAM) | Consumer-grade GPU with best price/performance. |
| **OS** | Ubuntu 22.04 | Standard for ML. |
| **Training Library** | `unsloth` | 2x faster training, 60% less VRAM via optimized kernels. |
| **Inference Library** | `vLLM` (0.6.x+) | High-throughput batched inference for GRPO rollouts. |
| **Chess Engine** | Stockfish 16+ | Ground truth for reward calculation. |
| **Chess Library** | `python-chess` | Board state, legal move generation, FEN parsing. |

### Verification Protocol: Infrastructure Setup

| Step | Action | Verification Command | Success Criteria |
|:-----|:-------|:---------------------|:-----------------|
| 1.1 | Run `setup_env.sh` | `nvidia-smi` | Shows RTX 4090, ~24GB VRAM. |
| 1.2 | Verify CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | Prints `True`. |
| 1.3 | Verify Stockfish | `stockfish` then type `uci` | Prints `uciok`. |
| 1.4 | Verify vLLM | `vllm serve Qwen/Qwen-2.5-Math-1.5B-Instruct --port 8000` | Server starts, no OOM. |
| 1.5 | Verify Unsloth | `python -c "from unsloth import FastLanguageModel"` | No import errors. |

---

## IV. Board Representation & Tokenization

### The Problem
Standard LLM tokenizers merge common substrings. For example:
- `rnbqkbnr` might become 2-3 tokens instead of 8.
- This forces the model to "un-merge" internally, wasting capacity.
- Token counts vary between FENs, making learning inconsistent.

### Design Analysis: Board Representation

| Option | Example Input | Pros | Cons | Verdict |
|:-------|:--------------|:-----|:-----|:--------|
| **A. Raw FEN** | `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1` | Compact. Standard notation. | Tokenizer merges pieces. High variance in token counts. | **REJECTED** |
| **B. Spaced FEN** | `r n b q k b n r / p p p ...` | Forces character-level tokens. No tokenizer modification needed. | Increases sequence length significantly (~2x). Still may merge `/` or numbers. | **CONSIDERED (Backup)** |
| **C. 8x8 Grid (ASCII Art)** | `r n b q k b n r\np p p p p p p p\n. . . . . . . .\n...` | Leverages model's spatial understanding from code. No tokenizer modification. | Increases sequence length. Requires custom parsing. | **CONSIDERED** |
| **D. Special Tokens** | Each piece (`R`, `n`, etc.) as a special token. | Guarantees 1-to-1 mapping. Minimal sequence length. Deterministic tokenization. | Requires tokenizer modification. New embeddings are randomly initialized (need warm-up). | **SELECTED** |

### Why Special Tokens Win
1. **Deterministic**: Every FEN tokenizes to the exact same structure.
2. **Compact**: Minimal sequence length compared to alternatives.
3. **Learnable**: With proper embedding warm-up, the model can learn chess semantics.

### Implementation Plan

#### Step 1: Tokenizer Audit (Pre-Fix)
```python
# audit_tokenizer.py
# Run on 100 random FENs, measure token count variance.
```
- **Expected Result**: Variance > 0 (tokenizer is merging).

#### Step 2: Add Special Tokens
```python
special_tokens = [
    '<|P|>', '<|N|>', '<|B|>', '<|R|>', '<|Q|>', '<|K|>',  # White pieces
    '<|p|>', '<|n|>', '<|b|>', '<|r|>', '<|q|>', '<|k|>',  # Black pieces
    '<|.|>',  # Empty square
    '<|/|>',  # Rank separator
]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))
```

#### Step 3: Embedding Warm-up (CRITICAL)

> [!CAUTION]
> **New special token embeddings are randomly initialized.** The model has *zero* understanding of what `<|R|>` means. You MUST warm up these embeddings before SFT.

**Warm-up Dataset**: 10,000 samples of FEN → Board Description pairs.
```
Input: <|r|><|n|><|b|><|q|>... (special token FEN)
Output: "The black rook is on a8. The black knight is on b8. ..."
```

**Training**: 1 epoch of SFT on this dataset.

**Purpose**: Grounds the new token embeddings in chess semantics.

### Verification Protocol: Tokenization

| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| 2.1 | Run `audit_tokenizer.py` (Pre-Fix) | Check variance | Variance > 0 (confirms problem exists). |
| 2.2 | Add special tokens | `len(tokenizer)` before/after | Token count increases by 14. |
| 2.3 | Resize embeddings | `model.get_input_embeddings().weight.shape` | Shape matches new tokenizer size. |
| 2.4 | Run `audit_tokenizer.py` (Post-Fix) | Check variance | **Variance = 0** (all FENs same token structure). |
| 2.5 | Run Embedding Warm-up | Train on 10k samples | Loss decreases. Model can describe board given special token FEN. |

---

## V. Phase 0: Baseline Measurement

> [!IMPORTANT]
> **You cannot measure progress without a starting point.** Before ANY training, we must establish how the base model performs.

### Why This Matters
- If the base Qwen-Math model already achieves 50% legal moves, SFT is easier.
- If it achieves 0% legal moves, we know SFT has more work to do.
- This is our "Day 0" measurement.

### Baseline Protocol

#### Test Set
- 100 FENs from varied game phases (20 openings, 40 middlegames, 40 endgames).
- Include tactical positions (forks, pins, mates).

#### Zero-Shot Prompt
```
You are a chess grandmaster. Given the following position in FEN notation:
{FEN}

Analyze the position and provide the best move in Standard Algebraic Notation (SAN).
Wrap your reasoning in <think></think> tags before giving your final answer.

Example format:
<think>The opponent's queen is undefended on d4. I can capture it with my knight from f3.</think>
Nxd4
```

#### Metrics to Record
| Metric | Description |
|:-------|:------------|
| **Legal Move Rate** | % of outputs that are legal moves in the given position. |
| **Format Adherence** | % of outputs that contain `<think>` tags. |
| **Stockfish Agreement** | % of legal moves that match Stockfish's top-1 move. |
| **Top-3 Agreement** | % of legal moves that are in Stockfish's top-3 moves. |

### Verification Protocol: Baseline

| Step | Action | Success Criteria |
|:-----|:-------|:-----------------|
| 0.1 | Generate 100 FEN test set | FENs cover all game phases. |
| 0.2 | Run inference with zero-shot prompt | All 100 FENs processed. |
| 0.3 | Parse outputs | Extract move from each output. |
| 0.4 | Validate legality | Use `python-chess` to check. |
| 0.5 | Record all metrics | Store in `baseline_results.json`. |

**Expected Baseline Results (Hypothesis)**:
- Legal Move Rate: 10-30%
- Format Adherence: 0-10% (model may not know `<think>` format)
- Stockfish Agreement: ~5-10%

---

## VI. Phase 1: Distillation (SFT)

### Goal
Teach the model **how to reason** (`<think>` format) and **what legal moves look like**. This is NOT about teaching optimal chess strategy—that comes from RL.

### Design Analysis: SFT Data Source

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. PGN Database** | Train on historical chess games (e.g., Lichess DB). | Huge dataset (millions of games). Free. | No reasoning trace. Teaches "what" not "why". Model memorizes openings. | **REJECTED** |
| **B. Stockfish Analysis** | Use Stockfish to generate best moves + simple explanations. | Accurate moves. Cheap. | Stockfish explanations are robotic/non-existent. Doesn't teach CoT. | **CONSIDERED** |
| **C. Strong LLM Distillation** | Use GPT-4/Gemini/DeepSeek to generate reasoning traces. | High-quality CoT. Teaches reasoning style. | API costs. May contain errors (LLMs aren't perfect at chess). | **SELECTED** |
| **D. Hybrid** | Use Stockfish for ground truth move, LLM for reasoning trace. | Best of both: accurate moves + rich reasoning. | Complex pipeline. | **SELECTED (Enhanced)** |

### Selected Approach: Hybrid Distillation

1. **Step 1**: Sample diverse FENs from game databases.
2. **Step 2**: Use Stockfish to determine the **ground truth best move**.
3. **Step 3**: Prompt a strong LLM (Gemini 1.5 Pro / DeepSeek-V3) to explain **why** that move is best.
4. **Step 4**: Filter out samples where the LLM's explanation doesn't match the Stockfish move.

### Dataset Specification

| Parameter | Minimum Viable | Target | Stretch |
|:----------|:--------------|:-------|:--------|
| **Dataset Size** | 10,000 samples | 30,000 samples | 100,000 samples |
| **Game Phase Distribution** | 30% opening, 40% middlegame, 30% endgame | Even distribution | Overweight tactical positions |
| **Teacher Model** | Gemini 1.5 Flash | Gemini 1.5 Pro | DeepSeek-V3 |

### Prompt Template for Teacher
```
You are a chess grandmaster analyzing a position.

Position (FEN): {FEN}
Best Move (from Stockfish): {BEST_MOVE}

Explain WHY this is the best move. Consider:
1. Immediate tactical threats or opportunities
2. Positional factors (piece activity, pawn structure, king safety)
3. What the opponent's best response might be

Wrap your entire analysis in <think></think> tags, then state the move.

Format:
<think>
[Your detailed analysis here]
</think>
{BEST_MOVE}
```

### Training Configuration

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| **Base Model** | `Qwen/Qwen-2.5-Math-1.5B-Instruct` | Math-trained, strong CoT priors. |
| **Adapter** | LoRA | Memory efficient. Can merge later. |
| **LoRA Rank (r)** | 32 | Balance between capacity and efficiency. |
| **LoRA Alpha** | 64 | Standard 2x rank. |
| **Target Modules** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | All linear layers for maximum learning. |
| **Learning Rate** | 2e-4 | Standard for LoRA SFT. |
| **Epochs** | 3 | Prevent overfitting. |
| **Batch Size** | 8 (with gradient accumulation) | Fit in 24GB VRAM. |

### Verification Protocol: SFT

| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| 3.1 | Generate 100 samples | Manual inspection | `<think>` tags present. Moves are legal. Reasoning is coherent. |
| 3.2 | Train for 1 epoch | Monitor loss | Loss decreases smoothly. |
| 3.3 | Inference on test set (100 FENs) | Parse outputs | **Format adherence ≥ 99%**. |
| 3.4 | Validate legality | `python-chess` | **Legal move rate ≥ 90%**. |
| 3.5 | Compare to baseline | All metrics | Improvement over Phase 0 baseline on all metrics. |

---

## VII. Phase 2: Reinforcement Learning (GRPO)

### Goal
Teach the model to play **strategically good** chess through self-improvement, using Stockfish as a verifiable reward signal.

### Design Analysis: RL Algorithm

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. PPO** | Proximal Policy Optimization | Well-studied. Stable. | Requires 4 models in memory (Policy, Reference, Reward Model, Value/Critic). Impossible on 24GB without extreme quantization. | **REJECTED** |
| **B. DPO** | Direct Preference Optimization | Simple. No RL loop. | Requires preference pairs (better/worse). Doesn't naturally fit chess (need win/loss pairs, hard to generate). | **CONSIDERED** |
| **C. REINFORCE** | Vanilla Policy Gradient | Simple. Low memory. | High variance. Slow convergence. | **REJECTED** |
| **D. GRPO** | Group Relative Policy Optimization | No Critic model (saves 50% VRAM). Uses group normalization for stable updates. Fits 1.5B on 24GB with Unsloth. | Newer algorithm, fewer tutorials. | **SELECTED** |

### Why GRPO Wins
1. **Memory Efficient**: No value network. Only Policy + Reference (and Reference can be CPU offloaded or LoRA-disabled).
2. **Proven for Reasoning**: DeepSeek-R1 and DeepSeekMath used GRPO for CoT improvements.
3. **Unsloth Support**: Recent Unsloth releases support GRPO with <7GB VRAM for 1.5B models.

### GRPO: How It Works

1. **Sample**: Given a FEN, generate $G$ candidate responses (e.g., 8 moves).
2. **Score**: Evaluate each response with the reward function (Stockfish delta).
3. **Normalize**: Compute mean and std of rewards within the group.
4. **Advantage**: $A_i = \frac{R_i - \mu}{\sigma}$ (relative advantage).
5. **Update**: Policy gradient update weighted by advantage.

### Critical Design Decision: Rollout Horizon

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. Per-Move** | Generate $G$ moves for a single FEN. Score each move. Update. | Fast iteration. Dense rewards. Clear learning signal. | Doesn't learn long-term planning. | **SELECTED (Initial)** |
| **B. Per-Game** | Generate $G$ complete games from a starting position. Score final outcome. | Learns long-term strategy. | Slow (games can be 40+ moves). Sparse rewards. High variance. | **CONSIDERED (Future)** |
| **C. Hybrid** | Per-move for first N steps, then per-game. | Best of both worlds. | Complex implementation. | **CONSIDERED (v2)** |

**Decision**: Start with **Per-Move** for fast iteration and dense rewards. Transition to Per-Game or Hybrid once Per-Move converges.

### Critical Design Decision: Opponent

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. No Opponent (Single Move)** | Model just predicts best move for given FEN. Stockfish scores the move. | Simplest. Fast. | Doesn't learn from opponent responses. | **SELECTED (Stage 1)** |
| **B. Random Mover** | Model plays against random legal moves. | Easy to beat. Dense positive rewards. | Doesn't learn real chess (opponent is terrible). | **SELECTED (Stage 2)** |
| **C. Weak Stockfish (Skill 1-5)** | Model plays against Stockfish at low skill. | Real chess opponent. Meaningful games. | Harder to win. May be too difficult early. | **SELECTED (Stage 3)** |
| **D. Self-Play** | Model plays against a copy of itself. | No external dependency. AlphaZero-style. | Prone to mode collapse. Needs careful regularization. | **CONSIDERED (v2)** |

**Decision**: Implement a **Staged Curriculum** (see Section IX).

### Training Configuration

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| **Group Size ($G$)** | 8 | Balance between diversity and VRAM. |
| **Learning Rate** | 5e-6 | Lower than SFT for stability. |
| **KL Coefficient** | 0.1 | Prevents policy drift from reference model. |
| **Max Steps** | 10,000 | Evaluate every 500 steps. |
| **Gradient Accumulation** | 4 | Effective batch = 32. |
| **vLLM Integration** | Direct (Unsloth native) | Avoid double memory copies. |

### Verification Protocol: GRPO

| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| 4.1 | Run 1-step training with dummy rewards | `nvidia-smi` | VRAM < 22GB. |
| 4.2 | Run 10 steps with real rewards | Monitor reward curve | Rewards are in expected range (-1 to 1). |
| 4.3 | Run 500 steps | Evaluate on test set | Legal move rate improves. Elo improves. |
| 4.4 | Self-play tournament (Checkpoint 500 vs Checkpoint 0) | Win rate | Checkpoint 500 wins > 55%. |

---

## VIII. Reward Engineering

### Goal
Provide a dense, informative signal that teaches the model *what good chess looks like*.

### Design Analysis: Reward Functions

| Option | Description | Pros | Cons | Verdict |
|:-------|:------------|:-----|:-----|:--------|
| **A. Sparse (Win/Loss/Draw)** | +1 for win, 0 for draw, -1 for loss. | Simple. Clear objective. | **Useless for learning.** Chess games are 40+ moves. A single loss punishes all prior good moves. Credit assignment is impossible. | **REJECTED** |
| **B. Binary Stockfish Match** | +1 if move matches Stockfish top-1, else 0. | Simple reward signals. | **Too rigid.** Second-best moves that are still winning get 0 reward. Discourages creative play. Model may overfit to Stockfish style. | **REJECTED** |
| **C. Scalar Delta (Centipawns)** | Reward = change in Stockfish evaluation before/after move. | Dense signal. Rewards improvements, punishes blunders. Continuous gradient. | Requires Stockfish eval each step (adds latency). | **SELECTED** |
| **D. Multi-Objective** | Combine format, legality, and chess quality. | Comprehensive. Allows curriculum weighting. | More complex. Needs careful tuning. | **SELECTED (Enhanced)** |

### The Reward Function (Final)

$$R_{total} = \alpha \cdot R_{format} + \beta \cdot R_{legality} + \gamma \cdot R_{chess}$$

Where:
- $\alpha = 0.1$ (small bonus for format compliance)
- $\beta = 1.0$ (strong penalty for illegal moves)
- $\gamma = 1.0$ (main learning signal)

#### Component 1: Format Reward ($R_{format}$)
```python
def reward_format(response: str) -> float:
    if "<think>" in response and "</think>" in response:
        return 0.1
    return 0.0
```
- **Purpose**: Maintain CoT structure learned in SFT.

#### Component 2: Legality Reward ($R_{legality}$)
```python
def reward_legality(move: str, board: chess.Board) -> float:
    try:
        parsed_move = board.parse_san(move)
        if parsed_move in board.legal_moves:
            return 0.0  # Neutral (don't reward legality, just don't punish)
    except:
        pass
    return -1.0  # Illegal move: strong punishment
```
- **Purpose**: Hard constraint. Illegal moves end the episode.

#### Component 3: Chess Quality Reward ($R_{chess}$)
```python
def reward_chess(board_before: chess.Board, move: str, engine: chess.engine) -> float:
    # Get eval before move
    info_before = engine.analyse(board_before, chess.engine.Limit(depth=15))
    score_before = info_before["score"].relative.score(mate_score=10000)
    
    # Apply move
    board_after = board_before.copy()
    board_after.push_san(move)
    
    # Get eval after move
    info_after = engine.analyse(board_after, chess.engine.Limit(depth=15))
    score_after = info_after["score"].relative.score(mate_score=10000)
    
    # Delta (from the player's perspective)
    delta = score_after - score_before
    
    # Normalize to [-1, 1] using tanh
    return math.tanh(delta / 100)
```
- **Purpose**: Reward moves that improve the position.

### Handling Edge Cases

| Case | Handling |
|:-----|:---------|
| **Checkmate delivered** | $R_{chess} = +1.0$ (maximum reward). |
| **Checkmate received** | $R_{chess} = -1.0$ (maximum penalty). |
| **Stalemate** | $R_{chess} = -0.1$ (slight penalty—we want wins). |
| **Draw by repetition/50-move** | $R_{chess} = -0.05$ (minor penalty). |
| **Illegal move** | $R_{legality} = -1.0$, episode ends immediately. |

### Verification Protocol: Rewards

| Step | Action | Verification | Success Criteria |
|:-----|:-------|:-------------|:-----------------|
| 5.1 | Unit test: Brilliant move (e.g., Queen sac leading to mate) | Compute reward | $R_{chess} > 0.5$ |
| 5.2 | Unit test: Blunder (e.g., Hang queen) | Compute reward | $R_{chess} < -0.5$ |
| 5.3 | Unit test: Neutral move (e.g., Developing knight) | Compute reward | $R_{chess} \approx 0$ |
| 5.4 | Unit test: Illegal move | Compute reward | $R_{legality} = -1.0$ |
| 5.5 | Unit test: Missing `<think>` tags | Compute reward | $R_{format} = 0$ |

---

## IX. Evaluation Framework

### Why Multiple Metrics?
Self-play alone doesn't tell you if your model is *good* at chess—just if it's *better than before*. We need external benchmarks.

### Evaluation Metrics

| Metric | Description | How to Measure | Target |
|:-------|:------------|:---------------|:-------|
| **Illegal Move Rate** | % of outputs that are illegal moves. | Parse with `python-chess`. | 0% |
| **Format Adherence** | % of outputs with correct `<think>` structure. | Regex check. | 99% |
| **Stockfish Top-1 Agreement** | % of moves matching Stockfish's best move. | Compare move strings. | 30% |
| **Stockfish Top-3 Agreement** | % of moves in Stockfish's top 3. | Stockfish multi-PV. | 50% |
| **Self-Play Win Rate** | Win rate against previous checkpoint. | 100 games, alternate colors. | >55% |
| **Estimated Elo** | Rating based on performance vs fixed opponent. | 500 games vs Stockfish Skill 5. | 1200 |

### Self-Play Tournament Protocol

1. **Freeze Checkpoint A** (e.g., Step 0).
2. **Train to Checkpoint B** (e.g., Step 500).
3. **Match**: 100 games (50 as White, 50 as Black).
4. **Time Control**: 1 second per move (model) vs instant (Stockfish for eval only).
5. **Result**: Record wins, draws, losses.
6. **Success**: B wins ≥ 55% of games against A.

### Elo Estimation Protocol

1. **Opponent**: Stockfish at Skill Level 5 (~1500 Elo).
2. **Games**: 500.
3. **Calculate**: Use win/draw/loss counts in Elo formula.
4. **Baseline**: Random mover has ~100 Elo.

### Staged Curriculum (Training Progression)

| Stage | Opponent | Focus | Success Criteria |
|:------|:---------|:------|:-----------------|
| **0** | None (single move prediction) | Learn format + legality. | Legal move rate ≥ 95%. |
| **1** | Random Mover | Learn basic tactics. | Win rate vs Random > 90%. |
| **2** | Stockfish Skill 1 | Learn real chess. | Win rate > 60%. |
| **3** | Stockfish Skill 3 | Improve Elo. | Estimated Elo ≥ 1000. |
| **4** | Stockfish Skill 5 | Stretch goal. | Estimated Elo ≥ 1200. |

---

## X. Contingency Plans

> [!WARNING]
> **Things WILL go wrong.** This section defines what to do when they do.

### Contingency Table

| Problem | Symptom | Root Cause | Solution |
|:--------|:--------|:-----------|:---------|
| **SFT format adherence < 95%** | Model ignores `<think>` tags. | Insufficient data or weak teacher signal. | Double dataset size. Add explicit formatting examples. Increase SFT epochs. |
| **SFT legal move rate < 80%** | Model outputs illegal moves. | FENs are confusing. Tokenization issue. | Verify tokenizer audit passed. Add more tactical positions to training. |
| **GRPO reward goes to -∞** | All moves get negative reward. | Model is outputting illegal moves or blundering. | Reduce LR. Increase KL penalty. Add more warmup steps. |
| **GRPO reward plateaus** | No improvement after 2000 steps. | Model stuck in local optimum. | Increase group size. Add exploration noise. Advance to next curriculum stage. |
| **VRAM exceeded (OOM)** | CUDA out of memory. | Batch size or sequence length too large. | Reduce batch size. Increase gradient accumulation. Use 8-bit quantization. |
| **Illegal moves persist in GRPO** | Legal move rate never reaches 100%. | RL not learning from legality penalty. | Implement constrained decoding (mask illegal move tokens). |
| **Self-play shows no improvement** | New checkpoint doesn't beat old. | Training is not working or overfitting. | Check reward function. Add regularization. Increase evaluation games for statistical power. |
| **Stockfish eval too slow** | Training throughput < 10 samples/sec. | Stockfish depth too high. | Reduce depth from 15 to 10. Use caching for repeated positions. |

### Emergency Fallback: Constrained Decoding

If illegal move rate cannot be reduced to 0% via RL:

1. At generation time, parse the current board state.
2. Get list of legal moves from `python-chess`.
3. Compute token IDs for each legal move.
4. Apply a **logit mask** that sets all non-legal-move tokens to $-\infty$.
5. Model can only generate legal moves.

**Downside**: This is a "crutch" that bypasses true learning. Use only if RL fails.

---

## XI. Cost & Time Estimates

### Compute Costs

| Phase | Estimated Time | GPU Hours | Cost (@ $0.45/hr) |
|:------|:---------------|:----------|:-------------------|
| **Baseline Measurement** | 30 min | 0.5 | $0.23 |
| **Tokenizer Audit + Fix** | 1 hr | 1 | $0.45 |
| **Embedding Warm-up** | 1 hr | 1 | $0.45 |
| **SFT (30k samples, 3 epochs)** | 3 hr | 3 | $1.35 |
| **GRPO Stage 0 (Legality)** | 5 hr | 5 | $2.25 |
| **GRPO Stage 1 (vs Random)** | 5 hr | 5 | $2.25 |
| **GRPO Stage 2 (vs SF Skill 1)** | 10 hr | 10 | $4.50 |
| **GRPO Stage 3 (vs SF Skill 3)** | 10 hr | 10 | $4.50 |
| **Evaluation & Tournaments** | 2 hr | 2 | $0.90 |
| **Contingency Buffer (20%)** | 7 hr | 7 | $3.15 |
| **TOTAL** | ~45 hr | 45 | **~$20** |

### API Costs (SFT Data Generation)

| Dataset Size | Tokens (~500/sample) | Cost (Gemini 1.5 Flash @ $0.075/1M input, $0.30/1M output) |
|:-------------|:---------------------|:------------------------------------------------------------|
| 10,000 | 5M input, 5M output | ~$2 |
| 30,000 | 15M input, 15M output | ~$5 |
| 100,000 | 50M input, 50M output | ~$20 |

### Total Estimated Budget

| Category | Low Estimate | Target | High Estimate |
|:---------|:-------------|:-------|:--------------|
| **Compute (RunPod)** | $15 | $20 | $40 |
| **Data Generation (API)** | $2 | $5 | $20 |
| **TOTAL** | **$17** | **$25** | **$60** |

---

## XII. Execution Roadmap

### Phase 0: Baseline & Infrastructure
- [ ] **0.1** Set up RunPod pod with RTX 4090.
- [ ] **0.2** Run `setup_env.sh`. Verify all components.
- [ ] **0.3** Run Tokenizer Audit (Pre-Fix). Record variance.
- [ ] **0.4** Run Baseline Measurement on 100 FENs. Record all metrics.
  - **Deliverable**: `baseline_results.json`

### Phase 1: Tokenization & Embedding Warm-up
- [ ] **1.1** Add special tokens to tokenizer.
- [ ] **1.2** Resize model embeddings.
- [ ] **1.3** Run Tokenizer Audit (Post-Fix). Verify variance = 0.
- [ ] **1.4** Generate 10k FEN→Description warm-up dataset.
- [ ] **1.5** Train Embedding Warm-up for 1 epoch.
  - **Verification**: Model can describe a board given special token FEN.

### Phase 2: SFT Data Generation
- [ ] **2.1** Sample 30k FENs from Lichess database.
- [ ] **2.2** Run Stockfish to get ground truth moves.
- [ ] **2.3** Generate reasoning traces with Gemini API.
- [ ] **2.4** Filter and validate dataset.
  - **Deliverable**: `sft_dataset_30k.jsonl`

### Phase 3: Supervised Fine-Tuning
- [ ] **3.1** Configure LoRA and training hyperparameters.
- [ ] **3.2** Train for 3 epochs.
- [ ] **3.3** Evaluate on test set.
  - **Verification**: Format adherence ≥ 99%, Legal move rate ≥ 90%.
  - **Deliverable**: `sft_checkpoint/`

### Phase 4: GRPO Training (Staged Curriculum)
- [ ] **4.1** Implement reward function. Run unit tests.
- [ ] **4.2** GRPO Stage 0: Single-move prediction. Legal move rate → 100%.
- [ ] **4.3** GRPO Stage 1: vs Random Mover. Win rate → 90%.
- [ ] **4.4** GRPO Stage 2: vs Stockfish Skill 1. Win rate → 60%.
- [ ] **4.5** GRPO Stage 3: vs Stockfish Skill 3. Elo → 1000.
  - **Deliverable**: Checkpoints at each stage.

### Phase 5: Final Evaluation
- [ ] **5.1** Run Self-Play Tournament (Final vs SFT baseline).
- [ ] **5.2** Run Elo Estimation vs Stockfish Skill 5.
- [ ] **5.3** Record all final metrics.
  - **Deliverable**: `final_evaluation_report.md`

---

## Appendix A: File Structure

```
chess-fm/
├── MASTER_PLAN.md          # This file
├── README.md               # Project overview
├── .gitignore              # Ignore checkpoints, data, etc.
├── setup_env.sh            # Environment setup script
├── requirements.txt        # Python dependencies
├── src/
│   ├── tokenizer/
│   │   ├── audit_tokenizer.py
│   │   └── add_special_tokens.py
│   ├── data/
│   │   ├── generate_warmup_data.py
│   │   ├── generate_sft_data.py
│   │   └── sample_fens.py
│   ├── training/
│   │   ├── train_warmup.py
│   │   ├── train_sft.py
│   │   └── train_grpo.py
│   ├── rewards/
│   │   ├── reward_function.py
│   │   └── test_rewards.py
│   └── evaluation/
│       ├── baseline.py
│       ├── tournament.py
│       └── elo_estimation.py
├── data/                   # (gitignored) Generated datasets
│   ├── warmup_10k.jsonl
│   └── sft_30k.jsonl
├── checkpoints/            # (gitignored) Model checkpoints
│   ├── warmup/
│   ├── sft/
│   └── grpo/
└── results/                # Evaluation results
    ├── baseline_results.json
    └── final_evaluation_report.md
```

---

## Appendix B: Key References

1. **GRPO Paper**: DeepSeekMath (arXiv:2402.03300) - Original GRPO algorithm.
2. **DeepSeek-R1**: Incentivizing Reasoning in LLMs via RL.
3. **Unsloth GRPO Guide**: https://unsloth.ai/blog/grpo
4. **LLM Chess Leaderboard**: https://github.com/adamkarvonen/chess_llm_benchmark
5. **Python-Chess Docs**: https://python-chess.readthedocs.io/
6. **Stockfish Docs**: https://official-stockfish.github.io/docs/

---

*This plan is Version 3.0. Last updated: 2024-12-15.*
