# ChessFM SFT Data Generation

Generate reasoning traces for chess positions using Gemini API (free tier).

## Setup

```bash
# Install dependencies
pip install google-generativeai python-chess

# Install Stockfish (for move validation)
brew install stockfish

# Get your free API key
# Go to: https://aistudio.google.com/app/apikey

# Set the key
export GOOGLE_API_KEY="your-key-here"
```

## Usage

```bash
cd data_generation
python generate_sft_data.py
```

## Free Tier Limits

| Access Method | Requests/Min | Requests/Day |
|---------------|--------------|--------------|
| **Gemini CLI** | 60 | 1,000 |
| AI Studio (Flash) | 15 | 1,500 |

At 1,000/day, you can generate **15,000 samples in 2 weeks** for free.

## Output Format

The script generates a JSONL file where each line is:

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  "color": "Black",
  "reasoning": "<think>\nWhite has played e4, claiming central space...\n</think>",
  "move": "e7e5",
  "is_reasonable": true,
  "centipawn_loss": 15,
  "best_move": "e7e5"
}
```

## Adding More Positions

Create a `positions.txt` file with one FEN per line:

```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
...
```

Get positions from [Lichess Elite Database](https://database.lichess.org/).
