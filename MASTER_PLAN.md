# Verifiable-Zero: The Ultimate Master Plan

**Version**: 2.1 (Deep Dive & Verification Focused)
**Goal**: Train a 1.5B parameter reasoning model to play high-level chess using Group Relative Policy Optimization (GRPO) and Verifiable Rewards.

---

## I. Executive Summary
This project is a structured curriculum to master the four pillars of modern AI Engineering:
1.  **Synthetic Data Pipelines**: Creating high-quality distillation datasets.
2.  **System 2 Reasoning**: Implementing Chain-of-Thought (CoT) via reinforcement learning.
3.  **Reward Engineering**: Designing dense, scalar signals for complex environments.
4.  **Hardware Optimization**: Fitting massive training loops into consumer hardware (24GB VRAM).

---

## II. Infrastructure (Where to Run?)

### Design Analysis
| Option | Pros | Cons | Verdict |
| :--- | :--- | :--- | :--- |
| **A. Google Colab** | Free/Cheap ($10/mo) | **The Trap**. Background processes (vLLM) are painful to manage. I/O timeouts on GDrive. 24h limit. | **REJECTED** |
| **B. Local Mac (M2/M3)** | Free, Persistent | **The Struggle**. Poor support for `unsloth` and `vLLM` (CUDA optimized). Debugging hell. | **REJECTED** |
| **C. Cloud GPU (RunPod)** | Full Control, Powerful | **The Winner**. Full SSH (Linux). Native CUDA support. `tmux` for background tasks. Cost efficient (~$0.45/hr). | **SELECTED** |

### Selected Stack: RunPod + RTX 4090
-   **Hardware**: 1x NVIDIA RTX 4090 (24GB VRAM).
-   **OS**: Ubuntu 22.04 (Standard for ML).
-   **Environment**:
    -   `unsloth`: For 2x faster training and 60% less VRAM.
    -   `vLLM`: For high-throughput rollout generation (essential for GRPO).
    -   `python-chess` + `stockfish`: For environment simulation and rewards.

### Verification Protocol
**Step 1.1: Environment Setup**
-   **Action**: Run `setup_env.sh`.
-   **Verification**:
    1.  Run `nvidia-smi` -> Must show RTX 4090.
    2.  Run `python -c "import torch; print(torch.cuda.is_available())"` -> Must print `True`.
    3.  Run `stockfish` -> Must start the engine shell.
    4.  Run `vllm serve Qwen/Qwen-2.5-Math-1.5B-Instruct --port 8000` -> Must start without crashing.

---

## III. Data & Tokenization (How to See?)

### Design Analysis
| Option | Input Example | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| **A. Raw FEN** | `rnbqkbnr...` | **REJECTED** | **The Failure**. Tokenizers merge common substrings (e.g., `rnbqk` = 1 token). The model wastes capacity "un-merging" this to understand the board. |
| **B. Spaced FEN** | `r n b q k ...` | **Improvement** | Forces character-level tokens, but increases sequence length significantly. |
| **C. Special Tokens** | `<|R|><|n|>...` | **SELECTED** | **The Senior Solution**. Adding chess pieces as *Special Tokens* guarantees 1-to-1 mapping, minimizes sequence length, and prevents merging. |

### Implementation Details
-   **Base Model**: `Qwen/Qwen-2.5-Math-1.5B-Instruct`.
-   **Tokenizer Modification**: Add tokens `[ 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', '/', '1'...'8' ]`.
-   **Embedding Resize**: Resize model embeddings to accommodate new tokens.

### Verification Protocol
**Step 2.1: Tokenizer Audit**
-   **Action**: Run `audit_tokenizer.py` (Pre-Fix).
-   **Verification**: Observe variance in token counts for random FENs.
-   **Action**: Apply Special Tokens fix.
-   **Verification**: Run `audit_tokenizer.py` (Post-Fix).
    -   **Success Criteria**: Every FEN must have a **deterministic** token count (e.g., exactly 64 squares + slashes + active color). Variance must be **0**.

---

## IV. Phase 1: Distillation (The Setup)

### Design Analysis
| Option | Strategy | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| **A. Train on Games** | PGN Database | **REJECTED** | Teaches *moves*, not *reasoning*. The model will just memorize openings. |
| **B. SFT on CoT** | Distill Strong Model | **SELECTED** | **The Foundation**. We need to teach the model *how* to think before we teach it *how to win*. |

### Implementation Details
-   **Teacher**: DeepSeek-V3 (via API) or Gemini 1.5 Pro.
-   **Prompt**: "Analyze this position. Wrap your reasoning in `<think>` tags. End with the best move."
-   **Dataset Size**: 2,000 high-quality samples.
-   **Training**: LoRA SFT (Supervised Fine-Tuning).
    -   **Rank (r)**: 16 or 32.
    -   **Target Modules**: All linear layers.

### Verification Protocol
**Step 3.1: Data Generation**
-   **Action**: Generate 5 samples.
-   **Verification**: Manually inspect. Do they contain `<think>` tags? Is the reasoning sound? Is the move legal?

**Step 3.2: SFT Training**
-   **Action**: Train for 1 epoch.
-   **Verification**: Run inference on a held-out FEN.
    -   **Success Criteria**: Output **must** start with `<think>` and end with `</think>`. Format adherence > 99%.

---

## V. Phase 2: RL Algorithms (The Core)

### Design Analysis
| Option | Algorithm | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| **A. PPO** | Proximal Policy Optimization | **REJECTED** | **Too Heavy**. Requires 4 models in memory (Policy, Ref, Reward, Value). Impossible on 24GB VRAM without extreme quantization. |
| **B. GRPO** | Group Relative Policy Optimization | **SELECTED** | **The Efficiency King**. No Critic model (saves 50% VRAM). Uses group scoring (mean/std) to normalize rewards. Fits 1.5B model + batch size 8 on 24GB. |

### Implementation Details
-   **Group Size ($G$)**: 8-16.
-   **Rollout Strategy**: Use `vLLM` for fast asynchronous rollouts.
-   **Reference Model**: Frozen copy of SFT model (can be offloaded to CPU if needed, or just use LoRA disable trick).

### Verification Protocol
**Step 4.1: RL Loop Test**
-   **Action**: Run training loop for 1 step with dummy rewards.
-   **Verification**: Check VRAM usage. Must be < 22GB. Check throughput (tokens/sec).

---

## VI. Reward Engineering (The Teacher)

### Design Analysis
| Option | Reward Function | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| **A. Sparse (Win/Loss)** | `+1` / `-1` | **REJECTED** | **Model won't learn**. Chess is too long. A loss after 40 good moves punishes the good moves. |
| **B. Binary Stockfish** | Match Top Move | **REJECTED** | **Too Rigid**. Punishes "Second Best" moves that are still winning. Discourages creative play. |
| **C. Scalar Delta** | $\Delta$ Centipawns | **SELECTED** | **The Best Signal**. Rewards *improving* the position and punishes *worsening* it, regardless of the final game outcome. |

### The Formula
$$R_{total} = R_{format} + R_{legality} + R_{chess}$$

1.  **$R_{format}$**: $+0.1$ (Gatekeeper). If missing, reward is 0 for everything else.
2.  **$R_{legality}$**:
    -   Illegal Move: $-1.0$ (Immediate death).
    -   Legal Move: $0.0$.
3.  **$R_{chess}$**:
    $$R_{chess} = \tanh\left(\frac{\text{Score}_{after} - \text{Score}_{before}}{100}\right)$$
    -   **Normalization**: $\tanh$ squashes the reward between -1 and 1.
    -   **Mate**: $\pm 10,000$ cp.

### Verification Protocol
**Step 5.1: Reward Function Unit Test**
-   **Action**: Create a test script `test_rewards.py`.
    -   Case A: Blunder (Queen hang). Expected: Negative reward near -1.
    -   Case B: Good move. Expected: Positive reward.
    -   Case C: Illegal move. Expected: -1.0.
-   **Verification**: All assertions pass.

---

## VII. Evaluation (The Proof)

### Design Analysis
| Option | Method | Verdict | Rationale |
| :--- | :--- | :--- | :--- |
| **A. Training Loss** | Cross-Entropy | **REJECTED** | **Meaningless in RL**. Loss often increases as the model explores new, high-reward strategies. |
| **B. Stockfish Level 0** | vs Random | **REJECTED** | **Participation Trophy**. Proves nothing. |
| **C. Self-Play** | Tournament | **SELECTED** | **The Gold Standard**. Elo is relative. The model must beat its past self to prove progress. |

### Tournament Protocol
1.  **Baseline**: Checkpoint $T_0$.
2.  **Challenger**: Checkpoint $T_{current}$.
3.  **Match**: 100 games (50 white, 50 black).
4.  **Success**: Challenger win rate > 55%.

### Verification Protocol
**Step 6.1: Tournament Script**
-   **Action**: Run `tournament.py` with two identical models.
-   **Verification**: Win rate should be ~50% (within statistical noise).
-   **Action**: Run `tournament.py` with Model A vs Random Mover.
-   **Verification**: Model A should win 100%.

---

## VIII. Execution Roadmap

### Phase 1: Infrastructure & Data
- [ ] **1.1** Setup RunPod environment (`setup_env.sh`). -> *Verify: `nvidia-smi`, `vllm` running.*
- [ ] **1.2** Audit Tokenizer (`audit_tokenizer.py`). -> *Verify: Variance > 0.*
- [ ] **1.3** Fix Tokenizer (Add Special Tokens). -> *Verify: Variance == 0.*

### Phase 2: Distillation (SFT)
- [ ] **2.1** Generate 2k SFT samples (`generate_data.py`). -> *Verify: `<think>` tags present.*
- [ ] **2.2** Train SFT Model (`train_sft.py`). -> *Verify: Inference follows format.*

### Phase 3: Reinforcement Learning (GRPO)
- [ ] **3.1** Implement Rewards (`rewards.py`). -> *Verify: Unit tests pass.*
- [ ] **3.2** Train GRPO (`train_grpo.py`). -> *Verify: Reward curve goes up.*

### Phase 4: Final Eval
- [ ] **4.1** Run Self-Play Tournament. -> *Verify: Win rate > Baseline.*
