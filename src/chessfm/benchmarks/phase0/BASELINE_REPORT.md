# ChessFM Baseline Benchmark Report
**Date:** 2025-12-27
**Hardware:** MacBook Pro (Apple Silicon MPS)

## Summary
We benchmarked 4 local models to select the best baseline for ChessFM. The goal was to find a model that can at least output UCI format (`e2e4`) correctly, even if the moves are illegal.

| Model | Size | Quant | Sample Size | Legal Move Rate | Format Score | Speed (pos/s) | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Qwen2.5-3B-MLX** | 3B | 4-bit | 1000 | **4.2%** | **100.0%** | 4.60 | 🏆 **Winner.** Perfect format, non-zero legality. |
| **Phi-3.5-mini-instruct** | 3.8B | 4-bit | 100 | 0.0% | 98.0% | 3.40 | High reasoning score didn't translate to chess. |
| **Gemma-2-2B-IT** | 2B | fp16 | 20 | 0.0% | 90.0% | 3.45 | Hallucinated `e4` for every move (illegal). |
| **DeepSeek-Coder-1.3B**| 1.3B | fp16 | 20 | 0.0% | 0.0% | **5.71** | Fastest, but failed format (outputted text). |
| **Qwen-2.5-Math-1.5B** | 1.5B | fp16 | 20 | 0.0% | 0.0% | 2.30 | Failed format (outputted "To determine..."). |

## Detailed Findings

### 1. Qwen2.5-3B-Instruct-4bit (MLX)
- **Status:** ✅ Selected Base Model
- **Performance:** 42/1000 legal moves (4.2%)
- **Pros:**
    - Optimized for Mac (MLX format)
    - Perfect instruction following for UCI format
    - Fast enough for local development (4.6 pos/sec)
- **Cons:**
    - Still fails legality 95.8% of the time (needs training!)

### 2. Gemma-2-2B-IT
- **Status:** ❌ Rejected
- **Performance:** 0/20 legal moves
- **Failure Mode:** Stubbornly outputted "e4" for almost every position, regardless of whether it was legal or black's turn.

### 3. DeepSeek-Coder-1.3B
- **Status:** ❌ Rejected
- **Performance:** 0/20 legal moves
- **Failure Mode:** Outputted conversational text/code instead of just the move. Likely needs stronger prompting or few-shot examples.

### 4. Qwen-2.5-Math-1.5B
- **Status:** ❌ Rejected
- **Performance:** 0/20 legal moves
- **Failure Mode:** Outputted CoT-style text ("To determine the best move...") despite "OUTPUT ONLY MOVE" instruction.

## Interesting Observations

### 🤖 The "Gemma Stubbornness" Effect
Gemma-2-2B-IT showed a fascinating failure mode: it outputted `e4` for **every single position**, regardless of whether it was:
- White's turn (often valid opening)
- Black's turn (illegal notation, should be `...e5` or just piece move)
- A position where `e4` was blocked or capturing own piece

This suggests strict instruction tuning ("Move: ") can sometimes lead to **mode collapse** where the model defaults to the most statistically probable token in its training data (opening move e4) rather than analyzing the specific FEN.

### 📉 The "Reasoning Trap" (Qwen 1.5B)
Qwen-1.5B-Math failed precisely *because* it's a math/reasoning model. Even when told asking for `ONLY UCI format`, it couldn't resist outputting:
> "To determine the best move, we must analyze..."

This validates the need for **<think>** tags in our training. The model *wants* to think. When we force it to output just a move (`e2e4`) without thinking space, performance collapses. Our architecture will embrace this by allowing `<think>...</think>` before the move.

### 🐍 The Code vs Chess Gap (DeepSeek-Coder)
We hypothesized that DeepSeek-Coder would be better at UCI notation (which looks like code/syntax).
**Result:** Failed.
**Lesson:** Code models are trained to write *functions*, not single tokens. It likely expected to write a `def next_move(board):` function rather than playing the move itself.

## Conclusion & Next Steps
We will proceed with **Qwen2.5-3B-Instruct-4bit** as our base model because:
1. It respects the output format (100%)
2. It has non-zero random legality (4.2%)
3. It runs efficiently on Apple Silicon (MLX 4-bit)

**Next Step:** Generate SFT data to raise legality from 4.2% → >90%.
