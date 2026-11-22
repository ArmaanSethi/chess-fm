"""Human-calibrated move prediction and trap detection sidecar (Maia-2 + fallback).

Provides skill-calibrated probability distributions over legal moves for a given
student Elo bracket. When the optional ``maia2`` neural package and weights are
installed, uses official Maia-2 inference; otherwise uses a deterministic,
fast, Elo-calibrated heuristic (+ optional Stockfish) fallback and explicitly
reports ``backend: "calibrated_fallback"``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import chess
import chess.engine

logger = logging.getLogger(__name__)

ELO_BRACKETS: tuple[int, ...] = (
    600,
    800,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2200,
    2400,
    2600,
)

_PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

_CENTER_SQUARES = frozenset({chess.D4, chess.E4, chess.D5, chess.E5})
_EXTENDED_CENTER = frozenset(
    {
        chess.C3,
        chess.D3,
        chess.E3,
        chess.F3,
        chess.C4,
        chess.F4,
        chess.C5,
        chess.F5,
        chess.C6,
        chess.D6,
        chess.E6,
        chess.F6,
    }
)


def snap_elo_bracket(elo: int) -> int:
    """Snap an arbitrary Elo rating to the nearest supported bracket."""
    return min(ELO_BRACKETS, key=lambda b: abs(b - int(elo)))


def _elo_skill_factor(student_elo: int) -> float:
    """Map student Elo to a normalized skill factor in ``[0.0, 1.0]``."""
    clamped = max(ELO_BRACKETS[0], min(ELO_BRACKETS[-1], int(student_elo)))
    return (clamped - ELO_BRACKETS[0]) / float(ELO_BRACKETS[-1] - ELO_BRACKETS[0])


def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
    """Compute a numerically stable softmax over move scores."""
    if not scores:
        return {}
    temp = max(1e-6, float(temperature))
    scaled = {uci: val / temp for uci, val in scores.items()}
    max_val = max(scaled.values())
    exps = {uci: math.exp(val - max_val) for uci, val in scaled.items()}
    total = sum(exps.values())
    return {uci: val / total for uci, val in exps.items()}


def _heuristic_move_score(
    board: chess.Board,
    move: chess.Move,
    skill: float,
) -> float:
    """Compute a deterministic human-plausibility score for a single legal move.

    Higher ``skill`` (closer to 1.0) increases weight on tactical safety
    (avoiding hanging pieces) and positional development, while lower ``skill``
    leaves superficial features (checks, captures) relatively more attractive.
    """
    score = 0.0
    moving_piece = board.piece_at(move.from_square)
    moving_val = _PIECE_VALUES.get(moving_piece.piece_type, 1.0) if moving_piece else 1.0

    # Captures (MVV-LVA): tempting across all ratings.
    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_val = 1.0
        else:
            captured = board.piece_at(move.to_square)
            victim_val = _PIECE_VALUES.get(captured.piece_type, 1.0) if captured else 1.0
        score += 0.9 + 0.35 * victim_val - 0.08 * moving_val

    # Promotions (especially to a queen).
    if move.promotion is not None:
        promo_val = _PIECE_VALUES.get(move.promotion, 1.0)
        score += 0.5 + 0.25 * promo_val

    # Castling is a natural human move at every level.
    if board.is_castling(move):
        score += 1.0 + 0.4 * skill

    # Center control.
    if move.to_square in _CENTER_SQUARES:
        score += 0.55 + 0.25 * skill
    elif move.to_square in _EXTENDED_CENTER:
        score += 0.25 + 0.1 * skill

    # Developing knights and bishops off the back rank.
    if moving_piece and moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        back_rank = 0 if board.turn == chess.WHITE else 7
        if chess.square_rank(move.from_square) == back_rank:
            score += 0.45 + 0.25 * skill

    # Evaluate post-move checks and tactical hanging-piece risk.
    mover_color = board.turn
    board.push(move)
    try:
        if board.is_checkmate():
            score += 4.0 + 2.0 * skill
        elif board.is_check():
            # Checks are tempting to lower-rated players even when not best.
            score += 0.85 - 0.25 * skill

        # Penalize leaving the moved piece attacked by the opponent.
        opp_attackers = board.attackers(not mover_color, move.to_square)
        if opp_attackers:
            defenders = board.attackers(mover_color, move.to_square)
            min_attacker_val = min(
                _PIECE_VALUES.get(board.piece_at(sq).piece_type, 1.0)
                for sq in opp_attackers
                if board.piece_at(sq) is not None
            )
            if not defenders:
                # Undefended piece: higher-rated players spot this much more reliably.
                score -= (0.25 + 0.95 * skill) * min(moving_val, 5.0)
            elif min_attacker_val < moving_val:
                # Attacked by a cheaper piece.
                score -= (0.2 + 0.75 * skill) * (moving_val - min_attacker_val)
    finally:
        board.pop()

    return score


class MaiaSidecar:
    """Human-like move distribution predictor and trap detector.

    Attempts to initialize the official ``maia2`` neural model when
    ``use_neural=True``. If ``maia2`` or its pretrained weights are unavailable,
    transparently falls back to a deterministic, Elo-calibrated predictor and
    reports ``backend == "calibrated_fallback"``.
    """

    def __init__(
        self,
        use_neural: bool = True,
        model_type: str = "rapid",
        device: str = "cpu",
        engine: chess.engine.SimpleEngine | None = None,
        engine_depth: int = 6,
    ) -> None:
        self.backend: Literal["maia2", "calibrated_fallback"] = "calibrated_fallback"
        self._maia_model: Any = None
        self._maia_prepared: Any = None
        self._maia_inference: Any = None
        self._engine = engine
        self._engine_depth = int(engine_depth)

        if use_neural:
            self._try_load_maia2(model_type=model_type, device=device)

    def _try_load_maia2(self, model_type: str, device: str) -> None:
        """Attempt to import and initialize official ``maia2`` weights."""
        try:
            from maia2 import inference, model  # type: ignore[import-not-found]

            self._maia_model = model.from_pretrained(type=model_type, device=device)
            self._maia_prepared = inference.prepare()
            self._maia_inference = inference
            self.backend = "maia2"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Maia-2 neural backend unavailable (%s); using calibrated fallback.", exc)
            self._maia_model = None
            self._maia_prepared = None
            self._maia_inference = None
            self.backend = "calibrated_fallback"

    def _engine_move_scores(self, board: chess.Board, legal_ucis: list[str]) -> dict[str, float]:
        """Query optional Stockfish engine for top move centipawn evaluations."""
        if self._engine is None or not legal_ucis:
            return {}
        multipv = min(8, len(legal_ucis))
        limit = chess.engine.Limit(depth=self._engine_depth)
        raw = self._engine.analyse(board, limit, multipv=multipv)
        infos = raw if isinstance(raw, list) else [raw]
        cp_by_move: dict[str, float] = {}
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            uci = pv[0].uci()
            cp = score.pov(board.turn).score(mate_score=10000)
            if cp is not None:
                cp_by_move[uci] = float(cp)
        return cp_by_move

    def _predict_fallback(
        self,
        board: chess.Board,
        student_elo: int,
    ) -> dict[str, float]:
        """Compute deterministic Elo-calibrated legal move probabilities."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        skill = _elo_skill_factor(student_elo)
        scores: dict[str, float] = {
            move.uci(): _heuristic_move_score(board, move, skill) for move in legal_moves
        }

        # Blend optional Stockfish scores with weight increasing by Elo.
        if self._engine is not None:
            cp_map = self._engine_move_scores(board, list(scores.keys()))
            if cp_map:
                best_cp = max(cp_map.values())
                engine_weight = 0.4 + 1.8 * skill
                for uci, cp in cp_map.items():
                    if uci in scores:
                        # Penalize cp loss relative to best move in pawn units.
                        loss_pawns = max(0.0, (best_cp - cp) / 100.0)
                        scores[uci] += engine_weight * max(-3.0, 1.5 - loss_pawns)

        # Temperature decreases monotonically with student Elo (sharper at higher Elo).
        temperature = 1.8 - 1.15 * skill
        return _softmax(scores, temperature=temperature)

    def _predict_maia2(
        self,
        board: chess.Board,
        fen: str,
        student_elo: int,
        opponent_elo: int,
    ) -> dict[str, float]:
        """Query loaded Maia-2 model and renormalize over strictly legal moves."""
        legal_ucis = {move.uci() for move in board.legal_moves}
        if not legal_ucis:
            return {}

        result = self._maia_inference.inference_each(
            self._maia_model,
            self._maia_prepared,
            fen,
            student_elo,
            opponent_elo,
        )
        raw_probs = result[0] if isinstance(result, tuple) else result
        filtered = {
            uci: max(0.0, float(prob)) for uci, prob in raw_probs.items() if uci in legal_ucis
        }
        total = sum(filtered.values())
        if total <= 0.0:
            return self._predict_fallback(board, student_elo)
        return {uci: prob / total for uci, prob in filtered.items()}

    def predict_human_moves(
        self,
        fen: str,
        student_elo: int = 1200,
        opponent_elo: int | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Predict human move probabilities for ``fen`` at ``student_elo``.

        Returns a dictionary containing ``fen``, ``student_elo``, ``backend``,
        ``move_probs`` (summing to 1.0 over legal UCI moves), and ``top_moves``
        (sorted ``list[tuple[str, float]]``).
        """
        board = chess.Board(fen)
        opp_elo = int(opponent_elo) if opponent_elo is not None else int(student_elo)

        if board.legal_moves.count() == 0:
            move_probs: dict[str, float] = {}
        elif self.backend == "maia2":
            move_probs = self._predict_maia2(board, fen, int(student_elo), opp_elo)
        else:
            move_probs = self._predict_fallback(board, int(student_elo))

        sorted_moves = sorted(move_probs.items(), key=lambda item: (-item[1], item[0]))
        top_moves = sorted_moves[: max(0, int(top_k))]

        return {
            "fen": fen,
            "student_elo": int(student_elo),
            "opponent_elo": opp_elo,
            "elo_bracket": snap_elo_bracket(student_elo),
            "backend": self.backend,
            "move_probs": move_probs,
            "top_moves": top_moves,
        }

    def detect_human_trap(
        self,
        fen: str,
        played_move: str,
        best_move: str,
        cp_loss: float,
        student_elo: int = 1200,
    ) -> dict[str, Any]:
        """Determine whether a costly mistake is a common human trap at ``student_elo``.

        Flags ``is_common_human_trap=True`` when ``cp_loss >= 100`` and the
        predicted human probability for ``played_move`` is at least ``0.15``.
        Otherwise categorizes the move as ``"unusual_oversight"`` (if
        ``cp_loss >= 100``) or ``"sound_move"``.
        """
        prediction = self.predict_human_moves(fen, student_elo=student_elo)
        move_probs: dict[str, float] = prediction["move_probs"]
        played_prob = float(move_probs.get(played_move, 0.0))
        best_prob = float(move_probs.get(best_move, 0.0))
        loss = float(cp_loss)

        if loss >= 100.0 and played_prob >= 0.15:
            category = "common_human_trap"
            is_common_human_trap = True
            coaching_context = (
                f"Common human trap at ~{student_elo} Elo: {played_move} is very tempting "
                f"({played_prob:.0%} predicted human frequency) but drops {loss:.0f} cp. "
                f"Look for {best_move} instead."
            )
        elif loss >= 100.0:
            category = "unusual_oversight"
            is_common_human_trap = False
            coaching_context = (
                f"Unusual oversight at ~{student_elo} Elo: {played_move} "
                f"({played_prob:.0%} predicted frequency) drops {loss:.0f} cp; "
                f"{best_move} keeps the position on track."
            )
        else:
            category = "sound_move"
            is_common_human_trap = False
            coaching_context = (
                f"Sound choice at ~{student_elo} Elo: {played_move} loses only "
                f"{loss:.0f} cp compared to {best_move}."
            )

        return {
            "fen": fen,
            "played_move": played_move,
            "best_move": best_move,
            "cp_loss": loss,
            "student_elo": int(student_elo),
            "backend": prediction["backend"],
            "played_move_prob": round(played_prob, 6),
            "best_move_prob": round(best_prob, 6),
            "is_common_human_trap": is_common_human_trap,
            "category": category,
            "coaching_context": coaching_context,
        }
