# ChessFM: 1.5B Chess Reasoning Roadmap

**Version**: 5.0 (Research-Backed Multi-Path Plan)  
**Last Updated**: 2025-12-27

> [!NOTE]
> This roadmap presents **multiple viable approaches** with research-backed justifications. Decision points are clearly marked with criteria for choosing between paths.

---

## Table of Contents
1. [Vision & Goals](#i-vision--goals)
2. [Key Research Insights](#ii-key-research-insights)
3. [The SFT Quality Problem](#iii-the-sft-quality-problem)
4. [Decision Framework](#iv-decision-framework)
5. [Training Approaches](#v-training-approaches)
6. [Thought Format & Control](#vi-thought-format--control)
7. [Reward Engineering](#vii-reward-engineering)
8. [Execution Checklist](#viii-execution-checklist)
9. [References](#ix-references)

---

## I. Vision & Goals

### Core Objective
Train a language model to play chess by **reasoning through positions** — producing visible Chain-of-Thought before each move.

### Output Format
```xml
<think>
Black just played Nf6, attacking my e4 pawn.
Options: Nc3 (defend + develop), e5 (aggressive push), Bd3 (prepare castling).
Nc3 is solid and keeps options open.
</think>
b1c3
```

### Success Metrics

| Metric | Target | Measurement |
|:-------|:-------|:------------|
| **Legal Move Rate** | > 90% | Parse outputs with python-chess |
| **Format Adherence** | > 95% | `<think>` tags present and parseable |
| **Elo Rating** | 1000-1200 | 500 games vs Stockfish Level 3 |

---

## II. Key Research Insights

This section summarizes the research that informs our approach.

### 1. LLM Chess Capabilities

| Model | Elo | Source |
|:------|:----|:-------|
| GPT-4o | ~1050 | [AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard) |
| Gemini Pro | ~1050 | [AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard) |
| Claude Sonnet | ~1000 | [AI Chess Leaderboard](https://dubesor.de/chess/chess-leaderboard) |
| ChessLLM (8B) | 1788 | [ChessLLM Paper](https://arxiv.org/abs/2501.17186) |

**Insight**: Base LLMs are mediocre at chess (~1000 Elo). Significant training is required to exceed this. ChessLLM achieved 1788 Elo but required 15 million games and an 8B model.

### 2. Small Models and Reasoning

**The "Logic Drift" Problem** ([DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)):
> Small models (<7B) can mimic the *style* of reasoning without maintaining logical coherence. They produce connectives ("Therefore", "However") but lose track of premises over multiple steps.

**Implication**: Our `<think>` traces must be short and structured to avoid drift. Long-form reasoning may degrade quality in 3B models.

### 3. RL from Scratch vs. SFT Bootstrap

**DeepSeek-R1 Findings**:
> GRPO can bootstrap reasoning from scratch in math domains, but requires the model to occasionally produce correct answers by chance to generate learning signal.

**Chess-Specific Challenge**: Our benchmark shows Qwen-3B produces **4.2% legal moves** out of the box. This is significantly lower than math tasks where models have ~20-30% baseline accuracy.

**Implication**: Pure RL may struggle. A minimal SFT bootstrap to teach move notation is likely necessary.

### 4. The Teacher Model Problem

**Problem**: If we use GPT-4o (~1050 Elo) to generate reasoning traces, we're teaching our model to think like a ~1050 Elo player.

**Research on Knowledge Distillation** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)):
> The student can only learn what the teacher knows. Distilling from a weak teacher caps the student's capability at the teacher's level.

**Implication**: LLM-generated reasoning should be treated as *format training*, not *chess training*. The actual chess knowledge must come from Stockfish or RL.

---

## III. The SFT Quality Problem

### The Fundamental Tradeoff

| Approach | Reasoning Quality | Chess Quality | Risk |
|:---------|:-----------------|:--------------|:-----|
| **LLM generates reasoning** | Creative, readable | ~1000 Elo (capped by teacher) | Hallucinated logic |
| **Stockfish PV as reasoning** | Mechanical, accurate | Optimal | Less "human-like" |
| **Minimal SFT + Heavy RL** | Emergent | Unknown | May not converge |

### Recommendation: Hybrid Approach

Use **Stockfish for chess correctness** and **LLM for linguistic fluency**:

```python
# Step 1: Stockfish provides the analysis
pv = stockfish.get_principal_variation()  # e.g., ["Bxf7+", "Kxf7", "Ng5+"]
eval_change = stockfish.get_eval()  # +4.5

# Step 2: LLM converts to natural language (not chess reasoning)
prompt = f"""
Convert this chess analysis to natural language:
- Best move: Bxf7+
- Following line: Kxf7, Ng5+, Ke8, Qxd8
- Evaluation: White is winning (+4.5)

Output a brief explanation (2-3 sentences) of what happens.
Do NOT add chess analysis beyond what's provided.
"""
```

This separates concerns:
- **Stockfish** → Chess correctness
- **LLM** → Linguistic formatting only

---

## IV. Decision Framework

### Decision Point 1: SFT vs Pure RL

```
                    ┌─────────────────────────┐
                    │ Run Stage 0 GRPO        │
                    │ (Legality only, 2k steps)│
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Legal rate after 2k?  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
         < 10%            10-30%            > 30%
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ PATH A:         │ │ PATH B:         │ │ PATH C:         │
    │ Minimal SFT     │ │ Continue RL     │ │ Full RL         │
    │ (2-5k samples)  │ │ (longer)        │ │ (as planned)    │
    │ Then resume RL  │ │ Check at 5k     │ │                 │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Justification**: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) showed that if a model has *some* baseline capability, RL can amplify it. But if baseline is near-zero, there's no signal to amplify.

### Decision Point 2: Model Size

```
                    ┌─────────────────────────┐
                    │ Train on 3B model       │
                    │ (Qwen-2.5-3B-Instruct)  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Achieved Elo?         │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
         < 900           900-1100          > 1100
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Diagnose:       │ │ Try scaling:    │ │ Success!        │
    │ Data issue?     │ │ 7B model        │ │ Iterate on      │
    │ Reward issue?   │ │ More data       │ │ reasoning       │
    │ Format issue?   │ │ Longer training │ │ quality         │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Justification**: [ChessLLM](https://arxiv.org/abs/2501.17186) used 8B parameters. Smaller models may hit a capability ceiling. Start small for iteration speed, scale up for final quality.

### Decision Point 3: Reasoning Source

| Option | When to Use | Tradeoff |
|:-------|:------------|:---------|
| **A: LLM-generated** | Want creative, human-like reasoning | May hallucinate tactics |
| **B: Stockfish PV** | Want accurate tactical reasoning | Less natural language |
| **C: Hybrid** | Best of both worlds | More complex pipeline |
| **D: Minimal + RL** | Trust emergent behavior | Less control over output |

---

## V. Training Approaches

### Approach A: SFT-First (Conservative)

**Intuition**: Teach the model chess notation and format first, then refine with RL.

**Pipeline**:
```
1. Generate 5,000+ SFT samples
   - Stockfish provides best move
   - LLM provides explanation (or Stockfish PV)
   
2. SFT for 3 epochs
   - Target: >50% legal move rate
   - Target: 100% format adherence
   
3. GRPO curriculum
   - Stage 1: Win/lose vs random mover
   - Stage 2: Win/lose vs Stockfish L1
   - Stage 3: Win/lose + eval vs Stockfish L3
```

**Research Support**: This is how [AlphaGo](https://www.nature.com/articles/nature24270) was trained — supervised learning on human games first, then self-play RL.

**Pros**: Most likely to produce a working model  
**Cons**: Capped by teacher quality, more data needed

---

### Approach B: RL-First (Experimental)

**Intuition**: Chess has verifiable rewards (legal/illegal, win/lose). Let the model discover chess through trial and error.

**Pipeline**:
```
1. Stage 0: Legality training (2k steps)
   - Reward: +1 legal, -1 illegal
   - No game playing, just move validation
   
2. Decision point: Check legal rate
   - If < 10%: Pivot to Approach A
   - If > 10%: Continue
   
3. Stage 1: vs Random Mover (5k steps)
   - Full games, win/lose reward
   
4. Stage 2: vs Stockfish (10k steps)
   - Add Stockfish eval delta reward
```

**Research Support**: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) showed GRPO can bootstrap reasoning without SFT in math domains.

**Pros**: No SFT data needed, potentially more creative  
**Cons**: May not converge, slower iteration

---

### Approach C: Hybrid (Balanced)

**Intuition**: Use SFT for format/notation only, RL for chess quality.

**Pipeline**:
```
1. Minimal SFT (~1,000 samples)
   - Focus: teach <think> format and UCI notation
   - NOT focused on chess quality
   - Use simple positions (openings, basic tactics)
   
2. GRPO curriculum (full)
   - Let RL discover what good moves look like
   - Reward includes format bonus
   
3. Optional: Post-training SFT
   - Add reasoning quality AFTER RL
   - Model knows good moves, now learn to explain them
```

**Research Support**: This separates concerns as recommended by [curriculum learning research](https://arxiv.org/abs/2012.09841) — teach prerequisites first, then build complexity.

**Pros**: Best balance of reliability and efficiency  
**Cons**: Requires careful staging

---

### Approach D: Scale-Up (Ambitious)

**Intuition**: If 3B isn't enough, go bigger.

**Pipeline**:
```
1. Prototype on 3B
   - Get pipeline working
   - Iterate on rewards/format
   
2. Apply to 7B+ model
   - Qwen-2.5-7B-Instruct or Llama-3.1-8B
   - Same pipeline, more capacity
   
3. Best-of-N at inference
   - Generate 10 candidate moves
   - Stockfish picks best
   - Boosts Elo significantly
```

**Research Support**: [ChessLLM](https://arxiv.org/abs/2501.17186) achieved 1788 Elo with 8B + best-of-10. This is the proven path to high performance.

**Pros**: Highest ceiling  
**Cons**: More compute, slower iteration

---

## VI. Thought Format & Control

### Basic Format
```xml
<think>
[2-5 sentences of reasoning]
</think>
[UCI move, e.g., e2e4]
```

### Advanced Format (Bonus Upgrade)
```xml
<think>
    <threats>King safe. Enemy rook targets d7.</threats>
    <candidates>Nf3, e4, c4</candidates>
    <analysis>
        Nf3: Develops, safe.
        e4: Loses pawn to Nxe4.
        c4: Controls center.
    </analysis>
</think>
c2c4
```

**Why Structure Helps** ([Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)):
> Structured reasoning reduces the chance of logical drift. Small models benefit from explicit scaffolding.

### Controlling Thought Length

**Training-time**: Add budget reward component

```python
def reward_budget(think_tokens: int) -> float:
    if think_tokens < 20:
        return -0.1   # Too short, probably not reasoning
    elif think_tokens > 200:
        return -0.05  # Rambling, penalize slightly
    else:
        return 0.0    # Good length
```

**Inference-time**: Control via generation parameters

```python
output = model.generate(
    prompt,
    max_new_tokens=150,  # Limits total output
    stop_sequences=["</think>"],  # Alternative: stop after think
)
```

---

## VII. Reward Engineering

### Component Rewards

| Component | Formula | When Applied |
|:----------|:--------|:-------------|
| **Legality** | `+1` legal, `-1` illegal | Always |
| **Outcome** | `+1` win, `+0.3` draw, `-0.5` loss | Game end (Stage 1+) |
| **Eval Delta** | `tanh(Δcp / 100)` | Per-move (Stage 2+) |
| **Format** | `+0.1` if `<think>` present | Always |
| **Budget** | `-0.1` if < 20 or > 200 tokens | Always |

### Staged Application

```python
def get_reward(stage: int, move: str, output: str, ...) -> float:
    r = 0.0
    
    # Always check legality
    r += reward_legality(fen, move)
    if r < 0:
        return r  # Illegal = fail fast
    
    # Always check format
    r += reward_format(output)
    r += reward_budget(output)
    
    # Stage-specific rewards
    if stage >= 1:
        r += reward_outcome(result, color)
    
    if stage >= 2:
        r += reward_eval_delta(fen_before, fen_after)
    
    return r
```

### Potential Issues & Mitigations

| Issue | Symptom | Mitigation |
|:------|:--------|:-----------|
| **Reward hacking** | Model finds exploit | Use multiple reward signals |
| **Mode collapse** | Always plays e2e4 | Diverse position sampling |
| **Format gaming** | Empty `<think></think>` | Budget minimum penalty |
| **Reasoning disconnect** | Good moves, bad explanations | Separate reasoning eval |

---

## VIII. Execution Checklist

### Phase 0: Setup & Baseline ✅
- [x] Set up environment (local Mac or RunPod)
- [x] Run tokenizer audit
- [x] Benchmark candidate models
- [x] Select base model (Qwen-3B)
- [x] Generate initial positions (25k FENs)
- [x] Generate initial SFT data (71 samples — NEED MORE)

### Phase 1: Bootstrap Decision
- [ ] Run Stage 0 GRPO (legality only, 2k steps)
- [ ] **Decision Point**: Check legal rate
  - If < 10%: Generate 2-5k SFT samples, train, then resume
  - If 10-30%: Continue to 5k steps, reassess
  - If > 30%: Proceed to Stage 1

### Phase 2: SFT (If Needed)
- [ ] Choose reasoning source:
  - [ ] Option A: LLM-generated (creative)
  - [ ] Option B: Stockfish PV-based (accurate)
  - [ ] Option C: Hybrid
- [ ] Generate 5,000+ samples
- [ ] Train for 3 epochs
- [ ] Verify legal rate > 50%

### Phase 3: RL Curriculum
- [ ] Stage 1: vs Random Mover (5k steps)
- [ ] Stage 2: vs Stockfish Level 1 (10k steps)
- [ ] Stage 3: vs Stockfish Level 3 (10k steps)
- [ ] Add format + budget rewards throughout

### Phase 4: Evaluation
- [ ] Play 500 games vs Stockfish L3
- [ ] Calculate Elo
- [ ] Evaluate reasoning quality (manual inspection)

### Phase 5: Scaling (Optional)
- [ ] Train on 7B model with same pipeline
- [ ] Implement best-of-N inference
- [ ] Target higher Elo

---

## IX. References

### Core Papers
| Paper | Link | Relevance |
|:------|:-----|:----------|
| **GRPO** | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | Our RL algorithm |
| **DeepSeek-R1** | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) | RL for reasoning, cold start problem |
| **ChessLLM** | [arXiv:2501.17186](https://arxiv.org/abs/2501.17186) | State-of-the-art LLM chess (1788 Elo) |
| **Knowledge Distillation** | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | Why teacher quality matters |
| **Chain-of-Thought** | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) | Structured reasoning helps |

### Technical Resources
| Resource | Link | Use |
|:---------|:-----|:----|
| **AI Chess Leaderboard** | [dubesor.de](https://dubesor.de/chess/chess-leaderboard) | LLM Elo benchmarks |
| **Lichess Database** | [database.lichess.org](https://database.lichess.org/) | Position data |
| **Dynomight Chess** | [Part 1](https://dynomight.substack.com/p/chess), [Part 2](https://dynomight.substack.com/p/more-chess) | Regurgitation technique |

---

## Cost Estimates

| Phase | GPU Hours | RunPod Cost |
|:------|:----------|:------------|
| Setup & Baseline | 4 hr | $1.80 |
| SFT (if needed) | 4 hr | $1.80 |
| GRPO Training | 20 hr | $9.00 |
| Evaluation | 2 hr | $0.90 |
| **Total (3B)** | **~30 hr** | **~$13** |
| **Adding 7B Scale-up** | +20 hr | +$9 |

---

*ChessFM Roadmap v5.0 — Research-backed, multi-path, decision-driven.*
