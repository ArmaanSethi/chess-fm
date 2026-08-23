# Strategy: The "Reasoning Mirage" & Model Selection

**Date:** 2025-12-27
**Status:** Strategic Pivot

## The Core Question
*"If thinking makes models smarter, why isn't everyone doing it on small models?"*

We explored starting with **Thinking Models** (DeepSeek-R1) vs **Standard Models** (Qwen-3B). This document outlines why small models struggle with reasoning and our strategy to overcome it.

## 1. The "Reasoning Mirage" (a.k.a. Logic Drift)
Small models (<7B) suffer from a phenomenon known as the **Coherence Gap**.

*   **Big Models (70B+):** Can maintain a logical thread ("Premise A → B → C").
*   **Small Models (<3B):** Start well ("Premise A is true...") but suffer from **Logic Drift**. By step 3, they often hallucinate or forget the original constraint.
*   **Chess Example:**
    > *Model:* "The knight is under attack. I must save it. But moving the pawn controls the center. Therefore, I will sacrifice the Queen for no reason."
    
    The reasoning *sounds* smart (connectives like "Therefore", "However"), but the logic is flawed. This mimics the style of reasoning without the substance.

## 2. Thinkers vs. Non-Thinkers (Cold Start Problem)

| Feature | **Thinker (DeepSeek-R1-Distill)** | **Non-Thinker (Qwen-3B-Instruct)** |
|:---|:---|:---|
| **Mechanism** | Native `<think>` tags pre-trained via distillation. | Standard instruction following. |
| **Pros** | Knows *how* to structure a thought process. | **Fast** (4.6 pos/s). 100% Format Adherence. |
| **Cons** | **Extremely Slow** (0.06 pos/s). Hard to steer. | **Legality = 4.2%**. Needs data to learn logic. |
| **Risk** | **Hallucinated Logic:** Might argue itself into illegal moves. | **Randomness:** Needs SFT to escape random guessing. |

## 3. Our Strategy: "Bootstrap & Scale"

We cannot "Skip SFT" completely on a 3B model because 4.2% legality is too low for RL to bite. RL needs a signal better than "everything is wrong."

### Phase A: Bootstrap (3B Model)
*   **Model:** Qwen2.5-3B-Instruct (Non-Thinker).
*   **Method:** Generate **1,000 - 5,000 SFT samples** using a "Teacher" (Gemini/Claude).
*   **Goal:** Teach the model *specific* chess reasoning patterns, avoiding the "Mirage" by overfitting to valid chess logic.
*   **Logic:** "Fake it 'til you make it." We force it to memorize valid reasoning paths until it generalizes.

### Phase B: Scale (7B+ Model)
*   **Model:** Qwen2.5-7B or Llama-3.1-8B.
*   **Method:** Pure RL (GRPO) or minimal SFT.
*   **Goal:** Unleash *emergent* reasoning.
*   **Logic:** At 7B+, effective context tracking allows for genuine novel reasoning.

## 4. Why DeepSeek-R1-Distill is (Currently) Out
Although DeepSeek-R1-Distill-1.5B is a "distilled thinker" (copying a genius teacher), our benchmarks showed it was **too slow** (0.06 pos/s) for efficient RL training loops where thousands of game rollouts are needed. Even with cloud GPUs, the inference speed bottleneck makes it impractical for our curriculum learning approach.

**Decision:** We proceed with **Qwen-3B + SFT Bootstrap**. We will "distill" intelligence into it via our generated dataset, then use GRPO for RL training.

