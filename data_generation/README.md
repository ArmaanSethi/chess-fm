# ChessFM SFT Data Generation

Generate high-quality reasoning traces for chess using FREE AI credits.

## 🏆 Recommended Models

| Provider | Model | Quality | How to Get |
|----------|-------|---------|------------|
| **Antigravity** | `gemini-3-pro` | ⭐⭐⭐⭐⭐ | You already have it! (this IDE) |
| **Antigravity** | `claude-sonnet-4.5` | ⭐⭐⭐⭐⭐ | Same as above |
| **Kiro** | `claude-sonnet-4.5` | ⭐⭐⭐⭐⭐ | [kiro.dev](https://kiro.dev) - 500 free credits |
| **Kiro** | `claude-opus-4.5` | ⭐⭐⭐⭐⭐ | Best model if you have credits |
| **Qwen Code** | `qwen3-coder-plus` | ⭐⭐⭐⭐ | Alibaba Cloud account |

---

## 🚀 Quick Start

### Step 1: Set Up the Proxy

```bash
# Clone and run AIClient-2-API
git clone https://github.com/justlovemaki/AIClient-2-API
cd AIClient-2-API
chmod +x install-and-run.sh && ./install-and-run.sh
```

### Step 2: Configure Your Providers

Open http://localhost:3000 and configure:

- **Antigravity**: Should already be authorized if you're using this IDE
- **Kiro**: Download from [kiro.dev](https://kiro.dev), login once
- **Qwen**: Login via Alibaba Cloud

### Step 3: Generate Data

```bash
cd data_generation
pip install openai python-chess
python generate_sft_data_proxy.py --model gemini-3-pro
```

---

## 📋 Usage Examples

```bash
# Use Gemini 3 Pro via Antigravity (RECOMMENDED)
python generate_sft_data_proxy.py --model gemini-3-pro

# Use Claude Sonnet via Kiro
python generate_sft_data_proxy.py --model claude-sonnet-4.5

# Use Qwen Code Plus
python generate_sft_data_proxy.py --model qwen3-coder-plus

# Generate 500 samples
python generate_sft_data_proxy.py --model gemini-3-pro --samples 500

# Custom rate limit (requests per minute)
python generate_sft_data_proxy.py --model claude-sonnet-4.5 --rpm 20
```

---

## 📂 File Structure

```
data_generation/
├── generate_sft_data_proxy.py   # Main script (uses proxy)
├── generate_sft_data.py         # Direct API version (backup)
├── positions.txt                # Your FEN positions (one per line)
├── data/
│   └── sft_data.jsonl           # Generated training data
└── README.md                    # This file
```

---

## 📁 Adding Positions

Create `positions.txt` with FEN positions (one per line):

```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
# Comments start with #
```

**Where to get positions:**
- [Lichess Elite Database](https://database.lichess.org/) (high-rated games)
- [Lichess Puzzles](https://database.lichess.org/#puzzles) (tactical positions)

---

## 📊 Output Format

Data is saved to `data/sft_data.jsonl`. Each line:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite has played e4, claiming central space...\n</think>",
  "move": "e7e5",
  "model": "gemini-3-pro",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "stockfish_best": "e7e5",
  "generated_at": "2024-12-22T21:00:00"
}
```

---

## 🧮 Estimated Throughput

| Model | Approx. RPM | Samples/Hour | Samples/Day |
|-------|-------------|--------------|-------------|
| gemini-3-pro | ~30 | ~1,800 | ~40,000 |
| claude-sonnet-4.5 | ~20 | ~1,200 | ~25,000 |
| qwen3-coder-plus | ~20 | ~1,200 | ~25,000 |

**With multiple providers**: Pool them in AIClient-2-API for even higher throughput!

---

## 🛠️ Troubleshooting

### "Connection refused at localhost:3000"
→ Make sure AIClient-2-API proxy is running

### "Stockfish not found"
```bash
# macOS
brew install stockfish

# Linux  
sudo apt install stockfish
```

### "Rate limited"
→ Reduce `--rpm` parameter or wait a bit

### "Illegal move" errors
→ This is expected for some responses; the script skips them automatically
