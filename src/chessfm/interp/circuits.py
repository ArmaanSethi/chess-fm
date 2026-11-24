"""Logit-lens, causal CoT faithfulness sabotage, and attention-head circuit helpers.

Provides deterministic pure functions for:
- INTERP #3: Sampling `<think>` checkpoints, summarizing token-position logit lens
  trajectories, and constructing corrupted-`<think>` prefill prompts (Lanham/Turpin
  causal faithfulness intervention).
- INTERP #4: Locating prompt token spans for board squares, scoring `(layer, head)`
  attention from output move tokens back to tactical squares, and sampling
  layer-matched random heads as a causal ablation control.
"""

from __future__ import annotations

import random
import re
from typing import Any

import chess


def sample_think_checkpoints(
    num_think_tokens: int,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[int]:
    """Return 0-indexed token offsets inside ``<think>`` corresponding to ``fractions``."""
    if num_think_tokens <= 0:
        return [0 for _ in fractions]
    last = num_think_tokens - 1
    return [min(last, max(0, int(round(f * last)))) for f in fractions]


def build_corrupted_think_prefill(
    fen: str,
    target_move_uci: str,
    distractor_move_uci: str | None = None,
) -> tuple[str, str]:
    """Construct a plausible corrupted ``<think>...</think>`` prefill steering to a legal distractor.

    Returns ``(prefill_text, distractor_uci)``. If ``distractor_move_uci`` is not
    provided, deterministically picks a legal non-target move from ``fen``.
    """
    board = chess.Board(fen)
    legal_ucis = sorted(m.uci() for m in board.legal_moves)
    candidates = [u for u in legal_ucis if u != target_move_uci]
    if not candidates:
        raise ValueError(f"Position has no legal distractor move besides {target_move_uci}")

    chosen = distractor_move_uci if distractor_move_uci in candidates else candidates[0]
    move_obj = chess.Move.from_uci(chosen)
    chosen_san = board.san(move_obj)
    target_san = target_move_uci
    try:
        t_obj = chess.Move.from_uci(target_move_uci)
        if t_obj in board.legal_moves:
            target_san = board.san(t_obj)
    except ValueError:
        pass

    prefill = (
        f"<think>\nWait: {target_san} ({target_move_uci}) actually runs into a refutation "
        f"and loses material. Instead, {chosen_san} ({chosen}) is the cleanest winning move "
        f"here, keeping control and avoiding tactics. I will play {chosen}.\n</think>\n"
    )
    return prefill, chosen


def summarize_logit_lens(
    records: list[dict[str, Any]],
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    """Aggregate per-puzzle logit-lens trajectories across ``<think>`` progress fractions."""
    if not records:
        return {
            "num_puzzles": 0,
            "mean_prob_by_fraction": {str(f): 0.0 for f in fractions},
            "top1_match_by_fraction": {str(f): 0.0 for f in fractions},
            "mean_prob_gain_0_to_100": 0.0,
        }

    mean_prob: dict[str, float] = {}
    top1_rate: dict[str, float] = {}
    for f in fractions:
        key = str(f)
        probs = [
            float(r["prob_by_fraction"][key])
            for r in records
            if key in r.get("prob_by_fraction", {})
        ]
        hits = [1.0 if r.get("top1_by_fraction", {}).get(key) else 0.0 for r in records]
        mean_prob[key] = round(sum(probs) / max(len(probs), 1), 5)
        top1_rate[key] = round(sum(hits) / max(len(hits), 1), 4)

    p0 = mean_prob.get(str(fractions[0]), 0.0)
    p1 = mean_prob.get(str(fractions[-1]), 0.0)
    return {
        "num_puzzles": len(records),
        "mean_prob_by_fraction": mean_prob,
        "top1_match_by_fraction": top1_rate,
        "mean_prob_gain_0_to_100": round(p1 - p0, 5),
    }


_SQUARE_SPAN_RE = re.compile(r"(?<![A-Za-z0-9])[KQRBNP]?([a-h][1-8])(?![A-Za-z0-9])")


def find_square_token_indices(
    token_strings: list[str],
    target_squares: set[str],
) -> tuple[list[int], list[int]]:
    """Split prompt token indices into ``(target_indices, other_square_indices)``.

    Uses character-span matching over the joined token stream so both single-token
    (``"Rb3"``) and BPE-split (``[" R", "b", "3"]``) square coordinates map to
    every overlapping token index.
    """
    joined_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tok in token_strings:
        # Insert a space only when tokens have no leading/trailing whitespace and
        # look like separate words (unit tests pass whitespace-stripped lists).
        need_sep = (
            cursor > 0
            and joined_parts[-1]
            and not joined_parts[-1][-1].isspace()
            and tok
            and not tok[0].isspace()
            and joined_parts[-1][-1] in ",:|"
        )
        if need_sep:
            joined_parts.append(" ")
            cursor += 1
        start = cursor
        joined_parts.append(tok)
        cursor += len(tok)
        spans.append((start, cursor))

    full_text = "".join(joined_parts)
    norm_targets = {sq.lower() for sq in target_squares}
    target_set: set[int] = set()
    other_set: set[int] = set()

    for m in _SQUARE_SPAN_RE.finditer(full_text):
        sq = m.group(1).lower()
        m_start, m_end = m.span(0)
        dest = target_set if sq in norm_targets else other_set
        for i, (t_start, t_end) in enumerate(spans):
            if t_end > m_start and t_start < m_end:
                dest.add(i)

    other_set -= target_set
    return sorted(target_set), sorted(other_set)


def score_attention_heads(
    attn_weights: list[Any],
    query_pos: int,
    target_indices: list[int],
    background_indices: list[int],
) -> dict[tuple[int, int], float]:
    """Score each ``(layer, head)`` by excess mean attention to target vs. background squares.

    Args:
        attn_weights: Sequence of per-layer tensors shaped ``(num_heads, seq_len, seq_len)``
            or ``(1, num_heads, seq_len, seq_len)``.
        query_pos: Token index of the generated UCI move token.
        target_indices: Prompt token indices for moving/target/king squares.
        background_indices: Prompt token indices for non-involved board squares.
    """
    if not target_indices:
        return {}
    scores: dict[tuple[int, int], float] = {}
    for layer_idx, layer_attn in enumerate(attn_weights):
        t = layer_attn[0] if getattr(layer_attn, "ndim", 3) == 4 else layer_attn
        n_heads = int(t.shape[0])
        for h in range(n_heads):
            row = t[h, query_pos]
            tgt_mean = float(row[target_indices].float().mean().item())
            bg_mean = (
                float(row[background_indices].float().mean().item()) if background_indices else 0.0
            )
            scores[(layer_idx, h)] = tgt_mean - bg_mean
    return scores


def rank_piece_tracker_heads(
    per_example_scores: list[dict[tuple[int, int], float]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Average head selectivity scores across examples and return top-k Piece-Tracker Heads."""
    accum: dict[tuple[int, int], list[float]] = {}
    for ex in per_example_scores:
        for key, val in ex.items():
            accum.setdefault(key, []).append(float(val))

    ranked = [
        {
            "layer": layer,
            "head": head,
            "mean_selectivity": round(sum(vals) / len(vals), 6),
            "n_examples": len(vals),
        }
        for (layer, head), vals in accum.items()
        if vals
    ]
    ranked.sort(key=lambda x: x["mean_selectivity"], reverse=True)
    return ranked[:top_k]


def sample_layer_matched_random_heads(
    top_heads: list[dict[str, Any]],
    num_heads_per_layer: int,
    *,
    seed: int = 42,
) -> list[dict[str, int]]:
    """Sample random control heads from the exact same layers as ``top_heads`` (disjoint)."""
    rng = random.Random(seed)
    excluded = {(int(h["layer"]), int(h["head"])) for h in top_heads}
    controls: list[dict[str, int]] = []
    chosen = set(excluded)
    for item in top_heads:
        layer = int(item["layer"])
        avail = [h for h in range(num_heads_per_layer) if (layer, h) not in chosen]
        if not avail:
            avail = [h for h in range(num_heads_per_layer) if (layer, h) not in excluded]
        pick = rng.choice(avail)
        chosen.add((layer, pick))
        controls.append({"layer": layer, "head": pick})
    return controls
