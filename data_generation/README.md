# ChessFM Data Generation

Engine-validated reasoning traces for chess move selection, and the scripts
that produce them.

> The committed JSONL files are a **small committed sample** for pipeline
> verification. The full training corpus is regenerated with the same pipeline
> (`generate_synthetic_data`) and intentionally not committed — it is large and
> its provenance is tracked by the scripts, not by git.

---

## Committed Sample

| File | Contents |
|------|----------|
| `positions.txt` | 25,000 elite-game FENs (the position pool) |
| `all_sft_data.jsonl` | 85 raw samples (teacher-LLM traces, Stockfish-validated) |
| `sft_train.jsonl` | 44 converted training rows (one per FEN, reasonable only) |
| `data_manifest.json` | Provenance: counts, sources, timestamp |

The sample is enough to verify every step of the pipeline. To produce a
training-sized corpus:

```bash
python -m chessfm.data_generation.generate_synthetic_data \
    --provider openai --model <teacher> \
    --positions data_generation/positions.txt --limit 5000 \
    --output data_generation/full/all_sft_data.jsonl
```

(`data_generation/full/` is gitignored; keys come from environment variables —
never commit API keys.)

---

## Raw Sample Format

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

## Converted Format (`sft_train.jsonl`)

Uses TRL `prompt`/`completion` columns so SFT loss is computed on completion
tokens only:

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

## Workflow

### 1. Source positions (if needed)

```bash
# Elite games from Lichess (25,000 positions by default)
python -m chessfm.data_generation.fetch_elite_data --count 25000

# Curated seed positions (small, deterministic)
python -m chessfm.data_generation.download_positions --count 1000
```

### 2. Generate teacher traces (full corpus)

```bash
python -m chessfm.data_generation.generate_synthetic_data \
    --provider openai --model gpt-4o --positions data_generation/positions.txt
```

Each proposed move is validated against Stockfish and recorded with a
`centipawn_loss`. Runs are resumable and deterministic given a seed.

### 3. Convert to training format

```bash
python -m chessfm.data_generation.convert_to_training
```

Conversion:
- **Deduplicates by FEN**, keeping the highest-quality sample per position
  (reasonable first, then lowest centipawn loss).
- **Balances `<think>` tags** on every trace.
- **Filters to reasonable moves** (centipawn loss < 200) by default;
  pass `--include-unreasonable` to keep everything.
- **Writes `data_manifest.json`** with raw/unique/output counts + timestamp.

---

## Quality

- **Legal moves only** — every trace is validated with python-chess + Stockfish.
- **No unvalidated rows** — traces whose move could not be analyzed by the
  engine (or that were illegal) are skipped at write time, never recorded.
- **Reasonable moves** — `is_reasonable` means centipawn loss < 200.
- **Diverse positions** — midgame focus (fullmoves 11–44).
- **Negative losses are clamped to 0** at conversion (a move cannot beat
  Stockfish's best, and legacy traces predate the clamp).

---

## Integration with Training

```bash
# Validate the committed sample without training (no GPU needed)
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl --dry-run

# Train
python -m chessfm.training.train_sft --data data_generation/sft_train.jsonl
```
