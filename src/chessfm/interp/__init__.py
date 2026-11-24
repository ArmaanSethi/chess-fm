"""Mechanistic interpretability tools for ChessFM models.

Provides four scientifically controlled interpretability analyses:
1. Linear board-state probes and causal activation steering (``probes.py``).
2. Exact LoRA weight-diff SVD geometry and activation steering helpers (``lora_svd.py``).
3. Token-position logit lens and causal CoT prefill sabotage helpers (``circuits.py``).
4. Attention-head piece-tracker scoring and matched random-head controls (``circuits.py``).
"""

from chessfm.interp.circuits import (
    build_corrupted_think_prefill,
    find_square_token_indices,
    rank_piece_tracker_heads,
    sample_layer_matched_random_heads,
    sample_think_checkpoints,
    score_attention_heads,
    summarize_logit_lens,
)
from chessfm.interp.lora_svd import (
    analyze_lora_adapter,
    compare_lora_profiles,
    compute_exact_lora_svd,
    parse_lora_module_key,
)
from chessfm.interp.probes import (
    PIECE_TO_CLASS,
    build_random_control_vector,
    build_square_steering_vector,
    fen_to_board_targets,
    train_board_linear_probe,
)

__all__ = [
    "PIECE_TO_CLASS",
    "analyze_lora_adapter",
    "build_corrupted_think_prefill",
    "build_random_control_vector",
    "build_square_steering_vector",
    "compare_lora_profiles",
    "compute_exact_lora_svd",
    "fen_to_board_targets",
    "find_square_token_indices",
    "parse_lora_module_key",
    "rank_piece_tracker_heads",
    "sample_layer_matched_random_heads",
    "sample_think_checkpoints",
    "score_attention_heads",
    "summarize_logit_lens",
    "train_board_linear_probe",
]
