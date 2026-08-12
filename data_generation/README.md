# ChessFM Data Generation

Engine-validated reasoning traces for chess move selection, and the scripts
that produce them.

> The committed JSONL files are a **small reproducibility sample**. The full
> training set was generated with the same pipeline on the training machine;
> the committed files are enough to verify every step of the pipeline works.

---

## 📊 Data Status

| Metric | Value |
|--------|-------|
| **Raw samples** | 185 |
| **Unique FENs (after dedup)** | 71 |
| **Converted training samples** | 71 (one per position) |
| **Position Source** | 25,000 elite FENs (Magnus, Stockfish, Leela games) |
| **Quality Filter** | Stockfish-validated legal moves only (centipawn loss < 200 for "reasonable") |
| **Provenance** | `data_manifest.json` written on every conversion |

---

## 📂 File Structure

```
data_generation/
├── README.md                 # This file
├── fetch_elite_data.py       # Fetch FENs from elite Lichess games
├── download_positions.py     # Curated FEN seed positions
├── convert_to_training.py    # Convert to HuggingFace format
├── positions.txt             # 25,000 elite game FENs (sample)
├── all_sft_data.jsonl        # Raw samples (sample)
├── sft_train.jsonl           # Converted training format (deduped by FEN)
└── data_manifest.json        # Provenance manifest (counts, sources, timestamp)
```

---

## 📋 Output Format

Each raw sample follows the ChessFM reasoning format:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite played e4, controlling the center...\n</think>",
  "move": "e7e5",
  "model": "gemini-3-pro-preview",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "stockfish_best": "e7e5",
  "generated_at": "..."
}
```

Converted format (`sft_train.jsonl`) — uses TRL `prompt`/`completion`
columns so SFT loss is computed on completion tokens only:

```json
{
  "prompt": "Position (FEN): ... Side to move: White\n\nAnalyze ...",
  "completion": "<think>\n...reasoning...\n</think>\ne2e4",
  "fen": "...",
  "move": "e2e4",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "stockfish_best": "e2e4"
}
```

---

## 🔄 Workflow

### 1. Generate Positions (if needed)
```bash
python fetch_elite_data.py --count 25000
```

### 2. Convert to Training Format
```bash
# From the project root
python data_generation/convert_to_training.py

# Only include reasonable moves (centipawn loss < 200)
python data_generation/convert_to_training.py --only-reasonable
```

Conversion:
- **Deduplicates by FEN**, keeping the highest-quality sample per position
  (reasonable first, then lowest centipawn loss).
- **Balances `<think>` tags** on every trace.
- **Preserves metadata** (`centipawn_loss`, `stockfish_best`, `is_reasonable`).
- **Writes `data_manifest.json`** with raw/unique/output counts + timestamp.

---

## 🎯 Quality Metrics

- **Legal moves only** (validated by python-chess + Stockfish)
- **Reasonable moves** (centipawn loss < 200)
- **Diverse positions** (mid-game focus, moves 10-45)

---

## 🔄 Integration with Training

The converted SFT data works with:

```bash
# Validate without training (no GPU needed)
python training/train_sft.py --data data_generation/sft_train.jsonl --dry-run

# Train
python training/train_sft.py --data data_generation/sft_train.jsonl
```
