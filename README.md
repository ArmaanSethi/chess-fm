# ChessFM 🧠♟️

> *A 1.5B parameter model that plays chess by reasoning, not memorizing.*

[![Status](https://img.shields.io/badge/Status-Research-orange)]()
[![Model](https://img.shields.io/badge/Base-Qwen--2.5--Math--1.5B-blue)]()
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

### Phase 1: Distillation (SFT)
Train the model to imitate GPT-4o + Stockfish reasoning traces.

### Phase 2: Reinforcement Learning (GRPO)
Reward moves that actually win games, not just look good.

### Phase 3: Curriculum Learning
Progressive difficulty: Random → Stockfish L1 → Stockfish L3

---

## 🛠️ Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Base Model** | [Qwen-2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) | Better reasoning pre-training |
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
│   │   SFT on    │ →   │   GRPO vs   │ →   │   Elo       │  │
│   │   GPT-4o    │     │  Stockfish  │     │   Eval      │  │
│   │   traces    │     │  curriculum │     │   (500      │  │
│   │             │     │             │     │   games)    │  │
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

## � Cost Estimate

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
- [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) — Base model
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Self-correction patterns
- [Dynomight Chess](https://dynomight.substack.com/p/chess) — Regurgitation technique

---

## 📋 Roadmap

See the full [ChessFM Roadmap](chess_fm_roadmap.md) for detailed implementation steps.

---

## 📄 License

MIT
