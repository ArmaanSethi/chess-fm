"""ChessFM coaching sidecar and move-classification utilities."""

from chessfm.coach.classifier import (
    build_partial_strategy_tree,
    classify_move,
    win_prob_from_cp,
)
from chessfm.coach.maia_sidecar import MaiaSidecar

__all__ = [
    "MaiaSidecar",
    "build_partial_strategy_tree",
    "classify_move",
    "win_prob_from_cp",
]
