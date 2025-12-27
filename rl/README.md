# RL Training for ChessFM

## 🧪 Experiment: RL Before SFT

This folder implements **direct reinforcement learning** without supervised fine-tuning first.
We start with GRPO (Group Relative Policy Optimization) but the infrastructure supports other algorithms.

### Hypothesis

**We expect this experiment to struggle** because:

1. **Zero Baseline**: Our benchmark shows Qwen-2.5-Math-1.5B produces **0% legal moves** out of the box
2. **Research Warning**: "Small models trained with GRPO for chess have demonstrated limitations, suggesting that a foundational understanding of piece movement is crucial for RL to be effective"
3. **Sparse Signal**: With 0% legal moves, every trajectory gets -1 reward initially

**However, we're trying it anyway because:**

1. Chess has **verifiable rewards** (legal/illegal is binary)
2. DeepSeek-R1 showed GRPO can bootstrap reasoning from scratch
3. Stage 0 focuses purely on legality, which is a simpler objective than winning
4. If it works, we skip expensive SFT data generation

### Success Criteria

| Stage | Metric | Target | Pivot Point |
|:------|:-------|:-------|:------------|
| Stage 0 | Legal Move Rate | ≥50% | If <5% after 2k steps, add minimal SFT |
| Stage 1 | Win Rate vs Random | ≥80% | - |
| Stage 2 | Elo vs Stockfish 3 | ≥1000 | - |

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify Stockfish is installed
stockfish --version  # Should show version
# If not: brew install stockfish (Mac) or apt install stockfish (Linux)
```

### 2. Run Verification Tests

```bash
# Run all unit tests first
python -m pytest tests/ -v

# Verify reward function works
python -c "from rewards import reward_legality; print('✅ Rewards OK')"

# Verify chess environment works
python -c "from chess_env import ChessEnv; print('✅ Env OK')"
```

### 3. Run Stage 0 (Legality Training)

```bash
# Quick test (100 steps, ~5 min)
python train_grpo.py --stage 0 --steps 100 --checkpoint-every 50

# Full Stage 0 (2000 steps, ~2 hours on RTX 4090)
python train_grpo.py --stage 0 --steps 2000 --checkpoint-every 500
```

### 4. Evaluate Progress

```bash
# Check legal move rate
python evaluate.py --checkpoint checkpoints/stage0_step_2000 --samples 100
```

---

## File Structure

```
grpo/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── train_grpo.py       # Main training script
├── chess_env.py        # Chess environment wrapper
├── rewards.py          # Reward functions
├── evaluate.py         # Evaluation script
├── tests/
│   ├── test_rewards.py      # Unit tests for rewards
│   ├── test_chess_env.py    # Unit tests for environment
│   └── test_integration.py  # Integration tests
└── checkpoints/        # Saved model checkpoints
```

---

## Hardware Requirements

| Setup | VRAM | Time for Stage 0 |
|:------|:-----|:-----------------|
| RTX 4090 | 24GB | ~2 hours |
| RTX 3090 | 24GB | ~3 hours |
| Apple M1/M2 | 16GB unified | ~4 hours |
| CPU only | - | Not recommended |

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch-size` from 8 to 4
- Enable gradient checkpointing: `--gradient-checkpointing`

### "Stockfish not found"
```bash
# Mac
brew install stockfish

# Linux
sudo apt install stockfish

# Manual: Download from https://stockfishchess.org/download/
```

### "0% legal moves after 2000 steps"
This confirms our hypothesis. Pivot to minimal SFT:
```bash
cd ../data_generation
python generate_sft_data_proxy.py --samples 2000 --model gemini-3-pro
```

---

## Curriculum Learning: Why Staged Training Works

### The Intuition

You can't learn calculus before arithmetic. Similarly, an LLM can't learn *good* chess moves before learning *legal* chess moves.

```
Stage 0: "Is e2e4 a valid string?" → Learn chess notation
Stage 1: "Does e2e4 help me win?" → Learn basic strategy  
Stage 2: "Is e2e4 the BEST move?" → Learn tactics & optimization
```

### The Math

Each stage uses a different reward signal:

| Stage | Reward Function | Signal Density |
|:------|:---------------|:--------------|
| **0** | `+1` legal, `-1` illegal | Dense (every move) |
| **1** | `+1` win, `-0.5` lose, `+0.3` draw | Sparse (end of game) |
| **2** | `tanh(Δcp/100)` (Stockfish delta) | Dense (every move) |

### Why This Works (AlphaGo Did It Too)

AlphaGo used **curriculum learning** with progressively harder opponents:
1. First learned from human games (SFT equivalent)
2. Then self-played against itself (Stage 1 equivalent)
3. Finally refined against stronger versions (Stage 2 equivalent)

We're testing if we can skip step 1 entirely by starting with pure RL.

---

## Alternative Approaches Considered

This section documents **all approaches we considered**, including ones we chose not to pursue. Each includes justification for why it's a good or bad idea.

---

### 1. Training Methodologies

#### 1.1 Pure Supervised Fine-Tuning (SFT)
| Aspect | Details |
|:-------|:--------|
| **Description** | Train exclusively on (position, move) pairs from human/engine games |
| **Pros** | Simple, proven to work (ChessLLM got 1788 Elo), stable training |
| **Cons** | Requires massive dataset (15M+ games), no exploration, limited to training data quality |
| **Research** | ChessLLM (2025) used this approach with best-of-N sampling |
| **Our Assessment** | ✅ **Good baseline** - We have SFT infrastructure ready as fallback |

#### 1.2 Pure Reinforcement Learning (Our Current Approach)
| Aspect | Details |
|:-------|:--------|
| **Description** | Learn from scratch via trial and error with reward signals |
| **Pros** | No labeled data needed, can discover novel strategies, cheap |
| **Cons** | May not converge, needs model to have some baseline capability |
| **Research** | DeepSeek-R1 showed GRPO can work for reasoning; but chess papers warn it struggles for small models |
| **Our Assessment** | ⚠️ **Risky but worth trying** - We're testing this hypothesis |

#### 1.3 Behavior Cloning + RL (Two-Phase)
| Aspect | Details |
|:-------|:--------|
| **Description** | First SFT to imitate, then RL to improve beyond training data |
| **Pros** | Stable bootstrap, then improvement beyond human level |
| **Cons** | More complex pipeline, needs both datasets |
| **Research** | This is how AlphaGo was trained (supervised warmstart → self-play) |
| **Our Assessment** | ✅ **Likely best approach** - Will pivot to this if pure RL fails |

#### 1.4 Direct Preference Optimization (DPO)
| Aspect | Details |
|:-------|:--------|
| **Description** | Train on (winning move, losing move) pairs without RL loop |
| **Pros** | Simpler than PPO, doesn't need on-policy rollouts |
| **Cons** | Needs paired preference data, may be less effective than PPO |
| **Research** | 2024 studies show PPO often beats DPO when properly tuned |
| **Our Assessment** | 🤔 **Interesting alternative** - Easy to implement with Stockfish rankings |

#### 1.5 Proximal Policy Optimization (PPO)
| Aspect | Details |
|:-------|:--------|
| **Description** | Classic RL algorithm with clipped objective for stability |
| **Pros** | Well-studied, stable, works well with sparse rewards |
| **Cons** | More complex than GRPO, needs value function |
| **Research** | Standard in RLHF, but GRPO is newer and simpler |
| **Our Assessment** | 🔄 **Backup option** - Try if GRPO has stability issues |

#### 1.6 RLHF (RL from Human Feedback)
| Aspect | Details |
|:-------|:--------|
| **Description** | Train reward model from human preferences, then optimize |
| **Pros** | Aligns with human play style, not just winning |
| **Cons** | Expensive (needs human annotators), overkill for chess |
| **Research** | Used for Maia-2 to create human-like chess AI |
| **Our Assessment** | ❌ **Overkill** - We have Stockfish as objective oracle |

---

### 2. Search Integration

#### 2.1 LLM + MCTS (Monte Carlo Tree Search)
| Aspect | Details |
|:-------|:--------|
| **Description** | Use LLM as policy/value network within search tree |
| **Pros** | Combines LLM reasoning with systematic search |
| **Cons** | Slow inference, complex implementation, not "pure" LLM |
| **Research** | Multiple 2024 papers show MCTS+LLM improves reasoning |
| **Our Assessment** | 🚀 **Future upgrade** - Great for inference, but not our current goal |

#### 2.2 Best-of-N Sampling
| Aspect | Details |
|:-------|:--------|
| **Description** | Generate N moves, pick best according to Stockfish |
| **Pros** | Simple, effective, ChessLLM used N=10 to boost Elo significantly |
| **Cons** | Expensive at inference time (N forward passes) |
| **Research** | ChessLLM: 1788 Elo with N=10 vs 1500 Elo with N=1 |
| **Our Assessment** | ✅ **Will use for evaluation** - Easy win for final Elo |

#### 2.3 Beam Search with Stockfish Reranking
| Aspect | Details |
|:-------|:--------|
| **Description** | Generate multiple candidates, rerank by Stockfish eval |
| **Pros** | Better than greedy, cheaper than best-of-N |
| **Cons** | Still needs search at inference |
| **Research** | Common technique in code generation |
| **Our Assessment** | 🔄 **Alternative to best-of-N** - May be more efficient |

---

### 3. Output Constraints

#### 3.1 Constrained Decoding (Grammar Enforcement)
| Aspect | Details |
|:-------|:--------|
| **Description** | Force model to only output legal moves at decode time |
| **Pros** | Guarantees 100% legal moves, no retraining needed |
| **Cons** | Inference-time hack, model doesn't truly "learn" legality |
| **Research** | Used in some benchmarks, but seen as cheating |
| **Our Assessment** | ⚠️ **Last resort** - We want model to learn legality intrinsically |

#### 3.2 Move Retry Loop
| Aspect | Details |
|:-------|:--------|
| **Description** | If illegal, regenerate until legal |
| **Pros** | Simple, works at inference |
| **Cons** | Expensive, hides model's true capability |
| **Research** | Common in LLM chess benchmarks |
| **Our Assessment** | 🔄 **Acceptable for eval** - But train for intrinsic legality |

#### 3.3 Legal Move Vocabulary Restriction
| Aspect | Details |
|:-------|:--------|
| **Description** | Dynamically restrict vocabulary to legal moves per position |
| **Pros** | Guaranteed legal, doesn't waste compute on illegal tokens |
| **Cons** | Complex implementation, position-dependent vocab |
| **Research** | Theoretically sound but rarely implemented |
| **Our Assessment** | 🤔 **Interesting research direction** - Complex engineering |

---

### 4. Data Strategies

#### 4.1 Full Game Training
| Aspect | Details |
|:-------|:--------|
| **Description** | Train on complete games from opening to endgame |
| **Pros** | Learns full game context, natural curriculum |
| **Cons** | Long sequences, expensive |
| **Research** | ChessLLM used 15M complete games |
| **Our Assessment** | ✅ **Standard approach** - For SFT phase |

#### 4.2 Puzzle Training
| Aspect | Details |
|:-------|:--------|
| **Description** | Train on tactical puzzles (mate-in-1, mate-in-2, etc.) |
| **Pros** | Dense reward, clear objective, Lichess has millions |
| **Cons** | Only tactics, misses strategic play |
| **Research** | LLM puzzle accuracy is a common benchmark |
| **Our Assessment** | 🚀 **Great for Stage 2** - After basic legality works |

#### 4.3 Opening-Only Training
| Aspect | Details |
|:-------|:--------|
| **Description** | Focus on first 10-15 moves only |
| **Pros** | Bounded complexity, well-studied theory |
| **Cons** | Doesn't generalize to middlegame/endgame |
| **Research** | No specific papers, but reasonable curriculum |
| **Our Assessment** | 🤔 **Curriculum option** - Could help generalization |

#### 4.4 Endgame-Only Training
| Aspect | Details |
|:-------|:--------|
| **Description** | Train on simple endgames (KRK, KQK, etc.) |
| **Pros** | Simpler positions, tablebases for ground truth |
| **Cons** | Different skills than opening/middlegame |
| **Research** | Traditional chess engines start with endgame tablebases |
| **Our Assessment** | 🤔 **Curriculum option** - Clear win/lose signals |

#### 4.5 Synthetic Data Generation
| Aspect | Details |
|:-------|:--------|
| **Description** | Use stronger LLMs (GPT-4o, Gemini) to generate reasoning traces |
| **Pros** | Get `<think>` traces without human labeling |
| **Cons** | API costs, may propagate LLM errors |
| **Research** | Our `generate_sft_data_proxy.py` implements this |
| **Our Assessment** | ✅ **Ready to use** - Infrastructure built |

---

### 5. Reward Engineering

#### 5.1 Sparse Game Outcome
| Aspect | Details |
|:-------|:--------|
| **Description** | Only reward at end: +1 win, 0 draw, -1 loss |
| **Pros** | Simple, objective |
| **Cons** | Very sparse, hard to learn from |
| **Research** | AlphaGo used this but with MCTS |
| **Our Assessment** | ⚠️ **Too sparse alone** - Need dense signals too |

#### 5.2 Dense Stockfish Evaluation
| Aspect | Details |
|:-------|:--------|
| **Description** | Reward based on centipawn evaluation change |
| **Pros** | Dense signal every move |
| **Cons** | Expensive (Stockfish call per move), may cause reward hacking |
| **Research** | Common in chess RL papers |
| **Our Assessment** | ✅ **Stage 2 reward** - After legality works |

#### 5.3 Stockfish Agreement
| Aspect | Details |
|:-------|:--------|
| **Description** | +1 if move matches Stockfish top-N |
| **Pros** | Imitation signal, dense |
| **Cons** | Just imitation, no exploration |
| **Research** | Used in some benchmarks |
| **Our Assessment** | 🤔 **Alternative to delta-eval** - Simpler |

#### 5.4 Material Counting
| Aspect | Details |
|:-------|:--------|
| **Description** | Reward based on piece captures (pawn=1, knight=3, etc.) |
| **Pros** | Fast (no Stockfish needed), intuitive |
| **Cons** | Doesn't capture positional play |
| **Research** | Classical heuristic, rarely used in ML |
| **Our Assessment** | 🤔 **Cheap alternative** - For resource-constrained setup |

#### 5.5 Legality-Only (Stage 0)
| Aspect | Details |
|:-------|:--------|
| **Description** | +1 legal, -1 illegal, game ends on illegal |
| **Pros** | Very dense, clear objective |
| **Cons** | Doesn't teach strategy |
| **Research** | Our innovation for curriculum |
| **Our Assessment** | ✅ **Our Stage 0** - Currently implemented |

---

### 6. Architecture Variations

#### 6.1 Standard Transformer (1-3B params)
| Aspect | Details |
|:-------|:--------|
| **Description** | Off-the-shelf decoder-only LLM |
| **Pros** | Proven architecture, lots of pretrained checkpoints |
| **Cons** | Not optimized for chess |
| **Research** | All LLM chess papers use this |
| **Our Assessment** | ✅ **Our approach** - Using Qwen-2.5-Math-1.5B |

#### 6.2 Specialized Chess Tokenizer
| Aspect | Details |
|:-------|:--------|
| **Description** | Train tokenizer specifically for chess notation |
| **Pros** | More efficient encoding, less fragmentation |
| **Cons** | Loses general language ability, needs retraining |
| **Research** | Some papers suggest this helps |
| **Our Assessment** | 🤔 **Future experiment** - Worth testing |

#### 6.3 Multimodal (Board Image + Text)
| Aspect | Details |
|:-------|:--------|
| **Description** | Input board as image, output move as text |
| **Pros** | Leverages visual pattern recognition |
| **Cons** | Much more complex, needs vision encoder |
| **Research** | Some 2024 papers explore this |
| **Our Assessment** | 🚀 **Future work** - Interesting research direction |

#### 6.4 Dual Encoder (FEN + PGN)
| Aspect | Details |
|:-------|:--------|
| **Description** | Encode current position AND move history separately |
| **Pros** | May improve long-game understanding |
| **Cons** | More complex architecture |
| **Research** | Not well-explored for LLMs |
| **Our Assessment** | 🤔 **Research idea** - Novel contribution potential |

---

### 7. Training Infrastructure

#### 7.1 Single GPU (RTX 4090)
| Aspect | Details |
|:-------|:--------|
| **Description** | 24GB VRAM, 4-bit quantization, LoRA |
| **Pros** | Affordable, sufficient for 1.5B model |
| **Cons** | Limited batch size, slower |
| **Research** | unsloth makes this viable |
| **Our Assessment** | ✅ **Our plan** - RunPod RTX 4090 |

#### 7.2 Multi-GPU Training
| Aspect | Details |
|:-------|:--------|
| **Description** | 4-8x GPUs with FSDP/DeepSpeed |
| **Pros** | Much faster, larger batch sizes |
| **Cons** | Expensive, complex setup |
| **Research** | Standard for serious training |
| **Our Assessment** | 🔄 **Scale-up option** - If experiments succeed |

#### 7.3 Apple Silicon (MPS)
| Aspect | Details |
|:-------|:--------|
| **Description** | Train on M1/M2/M3 macs |
| **Pros** | Quiet, efficient, always available |
| **Cons** | Slower than RTX 4090, some library issues |
| **Research** | unsloth partially supports MPS |
| **Our Assessment** | 🤔 **Local dev option** - For quick tests |

---

## Summary: Our Chosen Path

| Phase | Approach | Reason |
|:------|:---------|:-------|
| **Stage 0** | Pure RL for legality | Testing hypothesis that chess rules can be learned from reward |
| **Stage 1** | RL with win/lose reward | Build strategy once legality works |
| **Stage 2** | RL with Stockfish eval | Refine move quality |
| **Fallback** | Minimal SFT (2k samples) | If Stage 0 fails after 2k steps |
| **Evaluation** | Best-of-N with Stockfish | Boost final Elo score |

---

## Research References

- [ChessLLM](https://arxiv.org/abs/2501.17186): Achieved 1788 Elo with SFT on 15M games
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948): GRPO for reasoning without SFT
- [AlphaGo](https://www.nature.com/articles/nature24270): Curriculum learning in games
- [Maia-2](https://arxiv.org/abs/2409.20553): Behavior cloning for human-like chess
- [PPO vs DPO](https://proceedings.mlr.press/v235/xu24l.html): Comprehensive comparison (ICML 2024)
- [LLM+MCTS](https://arxiv.org/abs/2406.07394): Integrating search with language models
- [Dynomight Chess](https://dynomight.substack.com/p/chess): Regurgitation technique analysis
- [Constrained Decoding](https://arxiv.org/abs/2305.13971): Grammar-based output constraints
