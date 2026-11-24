"""Linear board-state probes and causal activation steering helpers.

Tests whether post-trained chess LLMs build a linearly decodable 64-square
internal board representation across layers (inspired by Othello-GPT and
Karvonen 2024), and constructs matched causal steering vs. random control
vectors for activation surgery.
"""

from __future__ import annotations

from typing import Any

import chess
import torch
import torch.nn as nn

# 0 = empty, 1..6 = White P/N/B/R/Q/K, 7..12 = Black p/n/b/r/q/k
PIECE_TO_CLASS: dict[str, int] = {
    ".": 0,
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}
NUM_BOARD_CLASSES = 13
NUM_SQUARES = 64


def fen_to_board_targets(fen: str) -> list[int]:
    """Encode a FEN board state into 64 integer class labels (a1=0 .. h8=63)."""
    board = chess.Board(fen)
    targets: list[int] = []
    for sq in range(NUM_SQUARES):
        piece = board.piece_at(sq)
        targets.append(PIECE_TO_CLASS[piece.symbol()] if piece is not None else 0)
    return targets


def train_board_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    epochs: int = 80,
    lr: float = 1e-2,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a 64-square x 13-class linear probe on residual-stream activations.

    Args:
        x_train: Float tensor of shape ``(N_train, hidden_dim)``.
        y_train: Long tensor of shape ``(N_train, 64)`` with class IDs in ``[0, 12]``.
        x_test: Float tensor of shape ``(N_test, hidden_dim)``.
        y_test: Long tensor of shape ``(N_test, 64)`` with class IDs in ``[0, 12]``.

    Returns:
        Dictionary containing overall 64-square accuracy, occupied-square accuracy,
        empty-square accuracy, per-class macro accuracy, and probe weight tensor
        ``weight`` of shape ``(64, 13, hidden_dim)`` on CPU.
    """
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("x_train and x_test must be 2D tensors (N, hidden_dim)")
    if y_train.shape != (x_train.shape[0], NUM_SQUARES) or y_test.shape != (
        x_test.shape[0],
        NUM_SQUARES,
    ):
        raise ValueError("y_train and y_test must have shape (N, 64)")

    torch.manual_seed(seed)
    device = x_train.device
    hidden_dim = x_train.shape[1]

    # Center and normalize by global RMS using training statistics only (zero leakage).
    # Global RMS preserves relative signal-to-noise geometry instead of amplifying dead/noise dims.
    mean = x_train.float().mean(dim=0, keepdim=True)
    centered_tr = x_train.float() - mean
    rms = centered_tr.pow(2).mean().sqrt().clamp(min=1e-5)
    x_tr = centered_tr / rms
    x_te = (x_test.float() - mean) / rms
    y_tr = y_train.long().to(device)
    y_te = y_test.long().to(device)

    probe = nn.Linear(hidden_dim, NUM_SQUARES * NUM_BOARD_CLASSES).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x_tr).view(-1, NUM_SQUARES, NUM_BOARD_CLASSES)
        loss = criterion(logits.reshape(-1, NUM_BOARD_CLASSES), y_tr.reshape(-1))
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(x_te).view(-1, NUM_SQUARES, NUM_BOARD_CLASSES)
        preds = test_logits.argmax(dim=-1)
        correct = preds == y_te
        overall_acc = float(correct.float().mean().item())

        occupied_mask = y_te > 0
        empty_mask = y_te == 0
        occupied_acc = (
            float(correct[occupied_mask].float().mean().item()) if occupied_mask.any() else 0.0
        )
        empty_acc = float(correct[empty_mask].float().mean().item()) if empty_mask.any() else 0.0

        # Recover probe weights in original (unstandardized) activation coordinates
        # so steering vectors apply directly to raw residual-stream hidden states.
        w_std = probe.weight.detach().view(NUM_SQUARES, NUM_BOARD_CLASSES, hidden_dim)
        w_orig = (w_std / rms).cpu()

    return {
        "overall_accuracy": round(overall_acc, 4),
        "occupied_accuracy": round(occupied_acc, 4),
        "empty_accuracy": round(empty_acc, 4),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "weight": w_orig,
    }


def build_square_steering_vector(
    probe_weight: torch.Tensor,
    square_idx: int,
    from_class: int,
    to_class: int = 0,
) -> torch.Tensor:
    """Build a unit-norm steering direction flipping ``square_idx`` from ``from_class`` to ``to_class``."""
    if not (0 <= square_idx < NUM_SQUARES):
        raise ValueError(f"square_idx must be in [0, 63], got {square_idx}")
    if not (0 <= from_class < NUM_BOARD_CLASSES and 0 <= to_class < NUM_BOARD_CLASSES):
        raise ValueError("from_class and to_class must be in [0, 12]")
    direction = (
        probe_weight[square_idx, to_class].float() - probe_weight[square_idx, from_class].float()
    )
    norm = direction.norm().clamp(min=1e-8)
    return direction / norm


def build_random_control_vector(
    reference: torch.Tensor,
    *,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a norm-matched random unit vector orthogonal to ``reference``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randn(reference.shape, generator=gen, dtype=torch.float32)
    ref = reference.detach().cpu().float()
    ref_unit = ref / ref.norm().clamp(min=1e-8)
    # Gram-Schmidt orthogonalization so control has zero projection on the probe axis.
    orth = rand - torch.dot(rand, ref_unit) * ref_unit
    orth_unit = orth / orth.norm().clamp(min=1e-8)
    return orth_unit * ref.norm()
