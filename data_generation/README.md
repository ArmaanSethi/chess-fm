# ChessFM Data Generation

Generate high-quality reasoning traces for chess move selection using large language models.

> **Note**: SFT is optional! The main training uses Direct GRPO. SFT data can improve reasoning quality and can be generated in parallel.

---

## 📊 Data Status

| Metric | Current |
|--------|---------|
| **Total Samples** | 185 unique (deduplicated) |
| **Position Source** | 25,000 elite FENs (Magnus, Stockfish, Leela games) |
| **Quality Filter** | Stockfish-validated legal moves only |

---

## 🤖 Models Used

| Provider | Model | Samples |
|----------|-------|---------|
| **Qwen** | `qwen3-coder-plus` | 63 |
| **Claude** | `claude-sonnet-4-5` | 62 |
| **Gemini** | `gemini-3-pro-preview` | 56 |

---

## 📂 File Structure

```
data_generation/
├── README.md                 # This file
├── fetch_elite_data.py       # Fetch FENs from elite Lichess games
├── download_positions.py     # Generate diverse FEN positions
├── convert_to_training.py    # Convert to HuggingFace format
├── positions.txt             # 25,000 elite game FENs
└── all_sft_data.jsonl        # All merged SFT samples (deduplicated)
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

## 🔄 Workflow

### 1. Generate Positions (if needed)
```bash
python fetch_elite_data.py --count 25000
```

### 2. Convert to Training Format
```bash
python convert_to_training.py --input all_sft_data.jsonl --output sft_train.jsonl
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

Run training with:
```bash
cd ../training
python train_sft.py --data ../data_generation/sft_train.jsonl
```
