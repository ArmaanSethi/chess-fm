# ChessFM 🧠♟️

> *A 1.5B parameter model that plays chess by reasoning, not memorizing.*

[![Status](https://img.shields.io/badge/Status-Research-orange)]()
[![Model](https://img.shields.io/badge/Base-Qwen--2.5--3B-blue)]()
[![Target](https://img.shields.io/badge/Target-1200%20Elo-green)]()

---

## 💡 The Idea

Most chess bots play by brute-force search. ChessFM plays by **thinking out loud**:

```xml
<think>
The opponent's queen threatens my f7 pawn.
If I castle now, I lose material.
Better to block with Nf6 first.
</think>
Nf6
```

The model explains *why* it's making a move — like a chess tutor, not a calculator.

---

## 🎯 Goals

| Metric | Target |
|--------|--------|
| **Elo Rating** | 1200+ (beat most LLMs) |
| **Illegal Move Rate** | < 5% |
| **Reasoning Format** | > 95% valid `<think>` tags |

### Benchmarks

| Model | Elo |
|-------|-----|
| GPT-4o | ~1050 |
| Gemini Pro | ~1050 |
| Claude Sonnet | ~1000 |
| **ChessFM (target)** | **1200** |

---

## 🔬 Approach

### Phase 1: SFT Bootstrap
Train on reasoning traces to teach the model chess fundamentals and `<think>` format.

### Phase 2: Direct GRPO (Reinforcement Learning)
Train directly on chess games using verifiable rewards (legal/illegal, win/lose).
Stockfish provides the reward signal for curriculum learning.

### Phase 3: Curriculum Learning
Progressive difficulty: Random → Stockfish L1 → Stockfish L3

---

## 🛠️ Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Base Model** | [Qwen-2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Best format adherence in benchmarks |
| **Training** | [unsloth](https://github.com/unslothai/unsloth) | 2x faster, 60% less VRAM |
| **Inference** | [vLLM](https://github.com/vllm-project/vllm) | Fast game rollouts |
| **Chess Engine** | [Stockfish 16](https://stockfishchess.org/) | Reward signal + validation |
| **Hardware** | RTX 4090 (RunPod) | 24GB VRAM |

---

## 📊 Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   FEN Position                                              │
│        ↓                                                    │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │  SFT on     │ →   │   GRPO vs   │ →   │   Elo       │  │
│   │  reasoning  │     │  Stockfish  │     │   Eval      │  │
│   │  traces     │     │  curriculum │     │   (500      │  │
│   │  (185 smpl) │     │             │     │   games)    │  │
│   └─────────────┘     └─────────────┘     └─────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Bonus Features (After v1)

- **Socratic Structure** — Force reasoning into `<threat_scan>`, `<candidates>`, `<verification>` tags
- **Negative Data** — Train on mistake-then-correction examples
- **Puzzle Training** — Tactical curriculum from Lichess puzzles

---

## 💰 Cost Estimate

| Phase | Time | Cost |
|-------|------|------|
| Setup & Baseline | 4 hr | $1.80 |
| SFT Training | 4 hr | $1.80 |
| GRPO Training | 20 hr | $9.00 |
| **Total** | **~28 hr** | **~$13** |

*Yes, you can train a chess-playing LLM for the price of lunch.*

---

## 📚 References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) — Our RL algorithm
- [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) — Base model architecture
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Self-correction patterns
- [Dynomight Chess](https://dynomight.substack.com/p/chess) — Regurgitation technique

---

## 📋 Roadmap

See the full [ChessFM Roadmap](chess_fm_roadmap.md) for detailed implementation steps.

---

## 🗂️ Project Structure

```
chess-fm/
├── README.md                 # This file
├── chess_fm_roadmap.md       # Detailed implementation plan
├── requirements.txt          # All dependencies
├── setup_env.sh              # Environment setup script
│
├── data_generation/          # SFT data generation
│   ├── README.md
│   ├── fetch_elite_data.py   # Fetch FENs from Lichess
│   ├── download_positions.py # Generate diverse positions
│   ├── convert_to_training.py
│   ├── positions.txt         # 25k elite FENs
│   └── all_sft_data.jsonl    # 185 deduplicated samples
│
├── training/                 # SFT training
│   ├── README.md
│   └── train_sft.py
│
├── rl/                       # Reinforcement learning
│   ├── README.md
│   ├── chess_env.py          # Chess environment
│   ├── rewards.py            # Reward functions
│   ├── train_grpo.py         # GRPO training
│   └── tests/
│
├── benchmarks/               # Evaluation & baselines
│   └── phase0/
│       ├── BASELINE_REPORT.md
│       ├── STRATEGY.md
│       ├── benchmark_models.py
│       └── run_benchmark_mlx.py
│
├── scripts/                  # Utilities
│   ├── audit_tokenizer.py
│   └── download_models.py
│
└── tests/                    # Unit tests
    └── test_data_generation.py
```

---

## 📄 License

MIT
