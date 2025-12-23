# ChessFM SFT Data Generation

Generate high-quality reasoning traces for chess move selection using large language models.

> **Note**: SFT is optional! The main training uses Direct GRPO. SFT data can improve reasoning quality and can be generated in parallel.

---

## 📊 Data Goals

| Metric | Target |
|--------|--------|
| **Total Samples** | 15,000 - 20,000 |
| **Position Source** | 25,000 elite FENs (Magnus, Stockfish, Leela games) |
| **Quality Filter** | Stockfish-validated legal moves only |

---

## 🤖 Current Models

| Provider | Model | Notes |
|----------|-------|-------|
| **Gemini** | `gemini-3-pro-preview` | Google's latest, strong reasoning |
| **Claude** | `claude-sonnet-4-5` | Anthropic via Kiro |
| **Qwen** | `qwen3-coder-plus` | Alibaba, code-optimized |

### Future Models (Planned)
- **DeepSeek 3.2 Thinking** - Extended reasoning capabilities
- **Kimi K2** - Strong multilingual reasoning

---

## 📂 File Structure

```
data_generation/
├── fetch_elite_data.py      # Fetch FENs from elite Lichess games
├── convert_to_training.py   # Convert to HuggingFace format
├── positions.txt            # 25,000 elite game FENs
├── data/
│   ├── combined_sft_data.jsonl   # All merged samples
│   ├── gemini_sft_data.jsonl     # Gemini-generated
│   ├── kiro_claude_sft_data.jsonl # Claude-generated
│   └── qwen_sft_data.jsonl       # Qwen-generated
└── legacy/                  # Archived older runs
```

---

## 📋 Output Format

Each sample follows the ChessFM reasoning format:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite played e4, controlling the center...\n</think>",
  "move": "e7e5",
  "model": "gemini-3-pro-preview",
  "is_reasonable": true,
  "centipawn_loss": 15
}
```

Training format (after conversion):
```xml
<think>
reasoning here
</think>
e2e4
```

---

## 🎯 Quality Metrics

- **Legal moves only** (validated by python-chess)
- **Reasonable moves** (centipawn loss < 200)
- **Diverse positions** (mid-game focus, moves 10-45)

---

## 🔄 Integration with Training

The generated SFT data works with:
- `unsloth` SFT training
- HuggingFace `SFTTrainer`
- Any instruction-following fine-tuning setup

Run conversion before training:
```bash
python convert_to_training.py
```
