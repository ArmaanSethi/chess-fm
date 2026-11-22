"""ChessFM coaching sidecar, move classification, Socratic hints, verification, and game review."""

from chessfm.coach.classifier import (
    build_partial_strategy_tree,
    classify_move,
    win_prob_from_cp,
)
from chessfm.coach.game_review import build_weakness_profile, review_pgn_game
from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.coach.multimodal import CoachOrchestrator
from chessfm.coach.socratic import contains_move_spoiler, generate_socratic_hint
from chessfm.coach.verifier import verify_and_repair, verify_claims

__all__ = [
    "CoachOrchestrator",
    "MaiaSidecar",
    "build_partial_strategy_tree",
    "build_weakness_profile",
    "classify_move",
    "contains_move_spoiler",
    "generate_socratic_hint",
    "review_pgn_game",
    "verify_and_repair",
    "verify_claims",
    "win_prob_from_cp",
]
