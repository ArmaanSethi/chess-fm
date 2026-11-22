#!/usr/bin/env python3
"""SM-2 spaced-repetition blunder drill engine for ChessFM Coach.

Turns recurring mistakes from ``build_weakness_profile`` into a persistent deck
of spaced-repetition tactical and strategic drills scheduled with the standard
SuperMemo-2 (SM-2) algorithm and graded against Stockfish centipawn loss.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import chess

from chessfm.data_generation.phase_sampler import normalize_board_fen
from chessfm.rl.engine import best_cp, eval_after_move

# SM-2 bounds and defaults.
SM2_MIN_EASE_FACTOR = 1.3
SM2_DEFAULT_EASE_FACTOR = 2.5

# Centipawn loss thresholds when grading non-exact legal moves with Stockfish.
ACCEPTABLE_CP_LOSS = 30.0
INACCURATE_CP_LOSS = 100.0


def _today_iso() -> str:
    """Return today's date as an ISO-8601 ``YYYY-MM-DD`` string."""
    return date.today().isoformat()


def _make_card_id(normalized_fen: str) -> str:
    """Return a deterministic short identifier for a normalized board FEN."""
    digest = hashlib.sha256(normalized_fen.encode("utf-8")).hexdigest()[:12]
    return f"drill_{digest}"


@dataclass
class DrillCard:
    """A single spaced-repetition chess blunder drill card."""

    card_id: str
    fen: str
    best_move: str
    category: str
    student_mistake: str
    ease_factor: float = SM2_DEFAULT_EASE_FACTOR
    interval_days: int = 0
    repetitions: int = 0
    due_date: str = field(default_factory=_today_iso)
    attempts: int = 0
    successes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the card to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrillCard:
        """Reconstruct a ``DrillCard`` from a dictionary."""
        return cls(
            card_id=str(data["card_id"]),
            fen=str(data["fen"]),
            best_move=str(data["best_move"]),
            category=str(data.get("category", "uncategorized")),
            student_mistake=str(data.get("student_mistake", "")),
            ease_factor=float(data.get("ease_factor", SM2_DEFAULT_EASE_FACTOR)),
            interval_days=int(data.get("interval_days", 0)),
            repetitions=int(data.get("repetitions", 0)),
            due_date=str(data.get("due_date", _today_iso())),
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
        )


def sm2_update(card: DrillCard, quality: int, today: date | None = None) -> DrillCard:
    """Apply the standard SuperMemo-2 update rule and return an updated card.

    Args:
        card: Existing ``DrillCard`` to schedule.
        quality: Recall quality score in ``0..5``.
        today: Reference date for computing ``due_date`` (defaults to ``date.today()``).

    Returns:
        A new ``DrillCard`` with updated ``ease_factor``, ``interval_days``,
        ``repetitions``, ``due_date``, ``attempts``, and ``successes``.

    Raises:
        ValueError: If ``quality`` is not an integer in ``0..5``.
    """
    if isinstance(quality, bool) or not isinstance(quality, int) or not (0 <= quality <= 5):
        raise ValueError(f"SM-2 quality must be an integer in 0..5, got {quality!r}")

    ref_date = today if today is not None else date.today()

    if quality < 3:
        new_repetitions = 0
        new_interval = 1
    else:
        if card.repetitions == 0:
            new_interval = 1
        elif card.repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(card.interval_days * card.ease_factor))
        new_repetitions = card.repetitions + 1

    ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    new_ef = max(SM2_MIN_EASE_FACTOR, card.ease_factor + ef_delta)
    new_due = (ref_date + timedelta(days=new_interval)).isoformat()

    return replace(
        card,
        ease_factor=new_ef,
        interval_days=new_interval,
        repetitions=new_repetitions,
        due_date=new_due,
        attempts=card.attempts + 1,
        successes=card.successes + (1 if quality >= 3 else 0),
    )


def grade_drill_attempt(
    fen: str,
    played_move_uci: str,
    best_move_uci: str,
    engine: Any = None,
    depth: int = 8,
) -> dict[str, Any]:
    """Grade a student's drill attempt and map it to an SM-2 quality score.

    Grading rules:
    - Illegal or unparseable UCI move -> ``quality=0``, ``passed=False``, ``cp_loss=None``.
    - Exact match with ``best_move_uci`` -> ``quality=5``, ``passed=True``, ``cp_loss=0.0``.
    - Legal non-best move with Stockfish ``engine``:
      - ``cp_loss <= 30`` -> ``quality=4``, ``passed=True``.
      - ``cp_loss <= 100`` -> ``quality=2``, ``passed=False``.
      - ``cp_loss > 100`` -> ``quality=1``, ``passed=False``.
    - Legal non-best move without ``engine`` (or on engine error):
      ``quality=1``, ``passed=False``, ``cp_loss=None``.
    """
    board = chess.Board(fen)
    played = (played_move_uci or "").strip().lower()
    best = (best_move_uci or "").strip().lower()

    try:
        move = chess.Move.from_uci(played)
    except ValueError:
        move = None

    if move is None or move not in board.legal_moves:
        return {
            "quality": 0,
            "passed": False,
            "cp_loss": None,
            "played_move": played,
            "best_move": best,
            "status": "illegal",
        }

    if played == best:
        return {
            "quality": 5,
            "passed": True,
            "cp_loss": 0.0,
            "played_move": played,
            "best_move": best,
            "status": "exact",
        }

    if engine is not None:
        best_eval = best_cp(engine, board, depth=depth)
        played_eval = eval_after_move(engine, board, played, depth=depth)
        if best_eval is not None and played_eval is not None:
            cp_loss = round(max(0.0, float(best_eval) - float(played_eval)), 2)
            if cp_loss <= ACCEPTABLE_CP_LOSS:
                return {
                    "quality": 4,
                    "passed": True,
                    "cp_loss": cp_loss,
                    "played_move": played,
                    "best_move": best,
                    "status": "acceptable",
                }
            if cp_loss <= INACCURATE_CP_LOSS:
                return {
                    "quality": 2,
                    "passed": False,
                    "cp_loss": cp_loss,
                    "played_move": played,
                    "best_move": best,
                    "status": "inaccurate",
                }
            return {
                "quality": 1,
                "passed": False,
                "cp_loss": cp_loss,
                "played_move": played,
                "best_move": best,
                "status": "missed",
            }

    return {
        "quality": 1,
        "passed": False,
        "cp_loss": None,
        "played_move": played,
        "best_move": best,
        "status": "missed",
    }


class DrillDeck:
    """Manages a deduplicated collection of SM-2 chess blunder drill cards."""

    def __init__(self, cards: list[DrillCard] | None = None) -> None:
        self._cards: dict[str, DrillCard] = {}
        self._fen_index: dict[str, str] = {}
        if cards:
            for card in cards:
                norm_fen = normalize_board_fen(card.fen)
                if norm_fen in self._fen_index:
                    continue
                self._cards[card.card_id] = card
                self._fen_index[norm_fen] = card.card_id

    def __len__(self) -> int:
        return len(self._cards)

    @property
    def cards(self) -> list[DrillCard]:
        """Return all cards currently in the deck."""
        return list(self._cards.values())

    def get_card(self, card_id: str) -> DrillCard | None:
        """Return a card by ``card_id``, or ``None`` if absent."""
        return self._cards.get(card_id)

    def add_card(
        self,
        fen: str,
        best_move: str,
        category: str = "uncategorized",
        student_mistake: str = "",
        today: date | None = None,
        card_id: str | None = None,
    ) -> tuple[DrillCard, bool]:
        """Add a drill card if its normalized board FEN is not already present.

        Returns:
            ``(card, added)`` where ``added`` is ``True`` if a new card was
            inserted and ``False`` if an existing card matched the normalized FEN.
        """
        norm_fen = normalize_board_fen(fen)
        existing_id = self._fen_index.get(norm_fen)
        if existing_id is not None:
            return self._cards[existing_id], False

        resolved_id = card_id or _make_card_id(norm_fen)
        ref_date = today if today is not None else date.today()
        card = DrillCard(
            card_id=resolved_id,
            fen=fen.strip(),
            best_move=best_move.strip().lower(),
            category=category,
            student_mistake=student_mistake.strip().lower(),
            due_date=ref_date.isoformat(),
        )
        self._cards[resolved_id] = card
        self._fen_index[norm_fen] = resolved_id
        return card, True

    def add_from_weakness_profile(
        self,
        profile: dict[str, Any],
        today: date | None = None,
    ) -> int:
        """Import drill candidates from a ``build_weakness_profile`` dictionary.

        Deduplicates by normalized board FEN and returns the count of newly
        added cards.
        """
        added_count = 0
        for item in profile.get("drills", []):
            fen = item.get("fen")
            best_move = item.get("best_move")
            if not fen or not best_move:
                continue
            try:
                chess.Board(fen)
            except ValueError:
                continue
            _, added = self.add_card(
                fen=fen,
                best_move=best_move,
                category=str(item.get("category") or "uncategorized"),
                student_mistake=str(item.get("student_mistake") or ""),
                today=today,
            )
            if added:
                added_count += 1
        return added_count

    def due_cards(
        self,
        today: date | None = None,
        limit: int | None = None,
    ) -> list[DrillCard]:
        """Return cards due on or before ``today``, ordered by due date then ease."""
        ref_date = today if today is not None else date.today()
        due = [
            card for card in self._cards.values() if date.fromisoformat(card.due_date) <= ref_date
        ]
        due.sort(key=lambda c: (c.due_date, c.ease_factor, c.card_id))
        if limit is not None:
            return due[: max(0, limit)]
        return due

    def record_attempt(
        self,
        card_id: str,
        played_move_uci: str,
        engine: Any = None,
        today: date | None = None,
        depth: int = 8,
    ) -> dict[str, Any]:
        """Grade an attempt on ``card_id``, apply SM-2 scheduling, and return the result."""
        if card_id not in self._cards:
            raise KeyError(f"Unknown drill card_id: {card_id!r}")

        card = self._cards[card_id]
        grade = grade_drill_attempt(
            fen=card.fen,
            played_move_uci=played_move_uci,
            best_move_uci=card.best_move,
            engine=engine,
            depth=depth,
        )
        updated = sm2_update(card, grade["quality"], today=today)
        self._cards[card_id] = updated

        return {
            "card_id": card_id,
            **grade,
            "card": updated.to_dict(),
        }

    def stats(self, today: date | None = None) -> dict[str, Any]:
        """Return summary statistics for the drill deck."""
        cards = self.cards
        total_cards = len(cards)
        due_count = len(self.due_cards(today=today))
        mastered = sum(1 for c in cards if c.repetitions >= 3 and c.interval_days >= 14)
        total_attempts = sum(c.attempts for c in cards)
        total_successes = sum(c.successes for c in cards)
        success_rate = round(total_successes / total_attempts, 4) if total_attempts else 0.0
        avg_ease = round(sum(c.ease_factor for c in cards) / total_cards, 4) if total_cards else 0.0

        by_category: dict[str, int] = {}
        for card in cards:
            by_category[card.category] = by_category.get(card.category, 0) + 1

        return {
            "total_cards": total_cards,
            "due_cards": due_count,
            "mastered_cards": mastered,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "success_rate": success_rate,
            "average_ease_factor": avg_ease,
            "by_category": by_category,
        }

    def save(self, path: Path | str) -> Path:
        """Persist the deck as JSON to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "cards": [card.to_dict() for card in self.cards],
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> DrillDeck:
        """Load a ``DrillDeck`` from a JSON file created by ``save``."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw_cards = payload.get("cards", [])
        cards = [DrillCard.from_dict(item) for item in raw_cards]
        return cls(cards=cards)
