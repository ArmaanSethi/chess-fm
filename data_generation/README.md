# ChessFM SFT Data Generation

Generate reasoning traces for chess positions using FREE AI credits.

## 🎁 FREE AI Credits Sources

| Provider | Models | Free Limit | How to Get |
|----------|--------|------------|------------|
| **Gemini CLI** | gemini-2.5-pro, gemini-2.5-flash | 60 RPM, 1000/day | Install [Gemini CLI](https://github.com/google-gemini/gemini-cli) |
| **Antigravity** | gemini-3-pro, claude-sonnet-4.5 | Generous | Comes with this IDE! |
| **Kiro** | claude-sonnet-4.5, opus-4.5 | 500 credits on signup | [kiro.dev](https://kiro.dev) |
| **Qwen Code** | qwen3-coder-plus | Generous | Alibaba Cloud account |
| **Google AI Studio** | gemini-1.5-flash, gemini-1.5-pro | 1500/day, 50/day | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **OpenRouter** | Various free models | Limited | [openrouter.ai](https://openrouter.ai) |

**BEST COMBO**: Gemini CLI + Kiro + AI Studio = **Thousands of requests per day FREE!**

---

## 🚀 Quick Start

### Option 1: Direct API (Simple)

```bash
pip install google-generativeai python-chess
export GOOGLE_API_KEY="your-key"  # Get from aistudio.google.com
python generate_sft_data.py
```

### Option 2: AIClient-2-API Proxy (Recommended for High Volume)

```bash
# 1. Clone and run the proxy
git clone https://github.com/justlovemaki/AIClient-2-API
cd AIClient-2-API
chmod +x install-and-run.sh && ./install-and-run.sh

# 2. Open http://localhost:3000 and configure your credentials

# 3. Run the generator
pip install openai python-chess
python generate_sft_data_proxy.py
```

---

## 📋 Scripts

| Script | Description | Requirements |
|--------|-------------|--------------|
| `generate_sft_data.py` | Direct Gemini API | `GOOGLE_API_KEY` |
| `generate_sft_data_proxy.py` | Via AIClient-2-API proxy | Proxy running on :3000 |

---

## 🔧 Configuration

### Change the model:
```bash
python generate_sft_data_proxy.py --model gemini-3-pro
```

### Generate specific number:
```bash
python generate_sft_data_proxy.py --samples 500
```

### Custom proxy URL:
```bash
python generate_sft_data_proxy.py --proxy-url http://localhost:8080/v1
```

---

## 📁 Adding More Positions

Create `positions.txt` with one FEN per line:

```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
```

Get positions from:
- [Lichess Elite Database](https://database.lichess.org/)
- [Lichess Puzzles](https://database.lichess.org/#puzzles)

---

## 📊 Output Format

Each line in `sft_data.jsonl`:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite has played e4...\n</think>",
  "move": "e7e5",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "best_move": "e7e5",
  "model": "gemini-2.5-flash"
}
```

---

## 🧮 Cost & Time Estimates

| Method | Samples/Day | Time for 15k | Cost |
|--------|-------------|--------------|------|
| Direct API | ~1,000 | 15 days | FREE |
| AIClient-2-API | ~5,000+ | 3 days | FREE |
| Multiple accounts | ~10,000+ | 1-2 days | FREE |

---

## 🛠️ Troubleshooting

### Stockfish not found
```bash
# macOS
brew install stockfish

# Linux
sudo apt install stockfish

# Update path in script if needed
STOCKFISH_PATH = "/usr/bin/stockfish"
```

### Proxy connection failed
- Make sure AIClient-2-API is running on port 3000
- Check http://localhost:3000 in browser
- Configure at least one provider in the Web UI

### Rate limited
- Use multiple providers in the proxy
- Reduce REQUESTS_PER_MINUTE in script
- Wait and retry later
