# ChessFM SFT Data Generation

Generate high-quality reasoning traces for chess using FREE AI credits.

> **Note**: SFT is optional! The main training uses Direct GRPO. But SFT data can improve reasoning quality and can be generated in parallel while GRPO runs.

---

## 🏆 Recommended Models

| Provider | Model | Quality | Setup |
|----------|-------|---------|-------|
| **Antigravity** | `gemini-3-pro` | ⭐⭐⭐⭐⭐ | Already have it! (this IDE) |
| **Kiro** | `claude-sonnet-4.5` | ⭐⭐⭐⭐⭐ | [kiro.dev](https://kiro.dev) - 500 free credits |
| **Qwen Code** | `qwen3-coder-plus` | ⭐⭐⭐⭐ | Alibaba Cloud account |

---

## 🚀 Quick Start

### Step 1: Set Up Proxy

```bash
git clone https://github.com/justlovemaki/AIClient-2-API
cd AIClient-2-API
chmod +x install-and-run.sh && ./install-and-run.sh
```

Then open http://localhost:3000 and configure your providers.

### Step 2: Install Dependencies

```bash
pip install openai python-chess
brew install stockfish  # macOS (or apt install stockfish on Linux)
```

### Step 3: Generate Data

```bash
cd data_generation
python generate_sft_data_proxy.py --model gemini-3-pro --samples 1000
```

### Step 4: Convert to Training Format

```bash
python convert_to_training.py
```

---

## 📂 File Structure

```
data_generation/
├── generate_sft_data_proxy.py  # Generate raw data
├── convert_to_training.py      # Convert to HuggingFace format
├── positions.txt               # Your FEN positions (create this)
├── data/
│   ├── sft_data.jsonl          # Raw generated data
│   └── sft_train.jsonl         # Training-ready data
└── README.md                   # This file
```

---

## 📊 Output Formats

### Raw Format (`sft_data.jsonl`)

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite played e4...\n</think>",
  "move": "e7e5",
  "model": "gemini-3-pro",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "stockfish_best": "e7e5"
}
```

### Training Format (`sft_train.jsonl`)

```json
{
  "instruction": "Position (FEN): ... Side to move: Black\n\nAnalyze this position and choose the best move.",
  "output": "<think>\nWhite played e4...\n</think>\ne7e5"
}
```

This matches the roadmap format:
```xml
<think>
reasoning here
</think>
e2e4
```

---

## 📋 Usage Examples

```bash
# Generate 500 samples with Gemini 3 Pro
python generate_sft_data_proxy.py --model gemini-3-pro --samples 500

# Generate with Claude
python generate_sft_data_proxy.py --model claude-sonnet-4.5

# Lower rate limit for stability
python generate_sft_data_proxy.py --model qwen3-coder-plus --rpm 20

# Convert to training format (only high-quality moves)
python convert_to_training.py --only-reasonable
```

---

## 🧮 Throughput Estimates

| Model | Approx. RPM | Samples/Hour | Samples/Day |
|-------|-------------|--------------|-------------|
| gemini-3-pro | ~30 | ~1,800 | ~40,000 |
| claude-sonnet-4.5 | ~20 | ~1,200 | ~25,000 |
| qwen3-coder-plus | ~20 | ~1,200 | ~25,000 |

**Target**: 15,000 samples is enough for initial SFT.

---

## ⚠️ Parallel with GRPO

While this generates SFT data, you can simultaneously:
1. Run GRPO training on the base model (no data needed)
2. Once SFT data is ready, optionally fine-tune for better reasoning

The output format is designed to work with:
- `unsloth` SFT training
- HuggingFace `SFTTrainer`
- Any instruction-following fine-tuning setup

---

## 🛠️ Troubleshooting

### "Connection refused"
→ Start the AIClient-2-API proxy first

### "Stockfish not found"
```bash
# macOS
brew install stockfish
# Update STOCKFISH_PATH in generate_sft_data_proxy.py if needed
```

### High "illegal move" rate
→ Normal! Some models make mistakes. The script filters them out.

### Rate limited
→ Reduce `--rpm` parameter, or wait and retry
