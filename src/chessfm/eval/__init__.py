"""4-Tier Evaluation Suite for ChessFM."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chessfm.eval.suite import (
        evaluate_board_probes,
        evaluate_coaching_faithfulness,
        evaluate_static_positions,
        generate_board_probes,
        run_demo_suite,
        summarize_paired_match,
    )

__all__ = [
    "evaluate_board_probes",
    "evaluate_coaching_faithfulness",
    "evaluate_static_positions",
    "generate_board_probes",
    "run_demo_suite",
    "summarize_paired_match",
]


def __getattr__(name: str) -> Any:
    """Lazily re-export suite helpers so ``python -m chessfm.eval.suite`` stays warning-free."""
    if name in __all__:
        suite = import_module("chessfm.eval.suite")
        return getattr(suite, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
