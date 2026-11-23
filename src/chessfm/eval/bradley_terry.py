"""Multi-opponent Bradley-Terry Elo MLE solver with stratified bootstrap CIs.

Pure ``numpy`` + standard library implementation (no ``scipy`` required).
Estimates a model's Elo rating against a pool of fixed-anchor opponents by
maximizing the strictly concave 1D Bradley-Terry log-likelihood via safeguarded
Newton-Raphson, with Jeffreys regularization for 0% / 100% score boundaries
and stratified bootstrap confidence intervals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_ANCHOR_RATINGS: dict[str, float] = {
    "random": 200.0,
    "sf_skill_0": 800.0,
    "sf_skill_1": 950.0,
    "maia_1100": 1100.0,
    "sf_uci_1320": 1320.0,
}

_LOG10_OVER_400: float = math.log(10.0) / 400.0
_ELO_CLAMP_MARGIN: float = 800.0
_RESULT_TEXT_TO_SCORE: dict[str, float] = {
    "win": 1.0,
    "w": 1.0,
    "draw": 0.5,
    "d": 0.5,
    "1/2-1/2": 0.5,
    "loss": 0.0,
    "l": 0.0,
}


def expected_score(rating: float, anchor_elo: float) -> float:
    """Logistic Elo expected score ``1 / (1 + 10^((anchor_elo - rating) / 400))``."""
    z = _LOG10_OVER_400 * (rating - anchor_elo)
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class OpponentMatchResult:
    """Per-opponent game outcomes against a fixed-anchor rating."""

    opponent: str
    anchor_elo: float
    outcomes: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.opponent or not isinstance(self.opponent, str):
            raise ValueError("opponent must be a non-empty string")
        if not math.isfinite(self.anchor_elo):
            raise ValueError(f"anchor_elo must be finite, got {self.anchor_elo!r}")
        normalized: list[float] = []
        for idx, outcome in enumerate(self.outcomes):
            val = float(outcome)
            if not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise ValueError(
                    f"outcomes[{idx}] must be in [0.0, 1.0], got {outcome!r} for {self.opponent!r}"
                )
            normalized.append(val)
        object.__setattr__(self, "anchor_elo", float(self.anchor_elo))
        object.__setattr__(self, "outcomes", tuple(normalized))

    @classmethod
    def from_wdl(
        cls,
        opponent: str,
        anchor_elo: float,
        wins: int,
        draws: int,
        losses: int,
    ) -> OpponentMatchResult:
        """Construct an ``OpponentMatchResult`` from integer W/D/L counts."""
        if wins < 0 or draws < 0 or losses < 0:
            raise ValueError(
                f"W/D/L counts must be non-negative, got wins={wins}, draws={draws}, losses={losses}"
            )
        outcomes = (1.0,) * int(wins) + (0.5,) * int(draws) + (0.0,) * int(losses)
        return cls(opponent=opponent, anchor_elo=float(anchor_elo), outcomes=outcomes)

    @property
    def games(self) -> int:
        return len(self.outcomes)

    @property
    def score(self) -> float:
        return float(sum(self.outcomes))

    @property
    def score_rate(self) -> float:
        return self.score / self.games if self.games else 0.0


@dataclass(frozen=True)
class BradleyTerryEstimate:
    """Point estimate, stratified bootstrap CI, and per-opponent breakdown."""

    elo: float
    ci_low: float
    ci_high: float
    std_error: float
    ci_level: float
    n_bootstrap: int
    total_games: int
    total_score: float
    score_rate: float
    regularized: bool
    bound_type: str
    per_opponent: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize estimate to a JSON-compatible dictionary."""
        return {
            "elo": round(self.elo, 2),
            "ci_low": round(self.ci_low, 2),
            "ci_high": round(self.ci_high, 2),
            "std_error": round(self.std_error, 2),
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "total_games": self.total_games,
            "total_score": round(self.total_score, 4),
            "score_rate": round(self.score_rate, 6),
            "regularized": self.regularized,
            "bound_type": self.bound_type,
            "per_opponent": self.per_opponent,
        }


def _solve_1d_mle(
    anchors: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
    *,
    lo: float,
    hi: float,
    max_iter: int = 100,
    tol: float = 1e-11,
) -> float:
    """Safeguarded Newton-Raphson solver for the strictly concave 1D BT objective.

    Finds ``R`` in ``[lo, hi]`` maximizing
    ``sum_j [s_j ln p_j(R) + (n_j - s_j) ln(1 - p_j(R))]``.
    """

    def eval_derivatives(rating: float) -> tuple[float, float, float]:
        z = np.clip(_LOG10_OVER_400 * (rating - anchors), -60.0, 60.0)
        probs = 1.0 / (1.0 + np.exp(-z))
        residual = float(np.sum(scores - counts * probs))
        grad = _LOG10_OVER_400 * residual
        hess = float(-(_LOG10_OVER_400**2) * np.sum(counts * probs * (1.0 - probs)))
        return residual, grad, hess

    _, g_lo, _ = eval_derivatives(lo)
    if g_lo <= 0.0:
        return lo
    _, g_hi, _ = eval_derivatives(hi)
    if g_hi >= 0.0:
        return hi

    # Game-weighted anchor mean is a well-conditioned starting point.
    total_n = float(np.sum(counts))
    r = float(np.sum(counts * anchors) / total_n) if total_n > 0 else 0.5 * (lo + hi)
    r = min(max(r, lo), hi)

    left, right = lo, hi
    step_tol = max(tol, 1e-9)
    for _ in range(max_iter):
        residual, grad, hess = eval_derivatives(r)
        if (right - left) <= tol:
            break
        if grad > 0.0:
            left = max(left, r)
        else:
            right = min(right, r)

        if hess < -1e-18:
            candidate = r - grad / hess
        else:
            candidate = 0.5 * (left + right)

        if not (left < candidate < right):
            candidate = 0.5 * (left + right)

        step = abs(candidate - r)
        if (abs(residual) <= tol and step <= step_tol) or step <= tol:
            r = candidate
            break
        r = candidate

    return float(min(max(r, lo), hi))


def solve_bradley_terry_elo(
    opponents: Sequence[OpponentMatchResult],
    *,
    prior_strength: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-11,
) -> tuple[float, bool, str]:
    """Compute the 1D Bradley-Terry Elo MLE against fixed-anchor opponents.

    Parameters
    ----------
    opponents:
        Non-empty sequence of ``OpponentMatchResult`` with at least one game.
    prior_strength:
        Symmetric Jeffreys pseudo-count added (``prior_strength`` wins and
        ``prior_strength`` losses at the game-weighted mean anchor) when total
        score is at the 0% or 100% boundary.
    max_iter:
        Maximum Newton-Raphson / bisection iterations.
    tol:
        Convergence tolerance in gradient / Elo units.

    Returns
    -------
    tuple[float, bool, str]
        ``(elo, regularized, bound_type)`` where ``bound_type`` is ``"exact"``,
        ``"upper"`` (0% score), or ``"lower"`` (100% score).
    """
    if not opponents:
        raise ValueError("opponents must be non-empty")
    if prior_strength <= 0.0 or not math.isfinite(prior_strength):
        raise ValueError(f"prior_strength must be positive and finite, got {prior_strength!r}")

    active = [res for res in opponents if res.games > 0]
    if not active:
        raise ValueError("opponents must contain at least one played game")

    anchors = np.array([res.anchor_elo for res in active], dtype=np.float64)
    counts = np.array([float(res.games) for res in active], dtype=np.float64)
    scores = np.array([res.score for res in active], dtype=np.float64)

    total_games = float(np.sum(counts))
    total_score = float(np.sum(scores))

    min_anchor = float(np.min(anchors))
    max_anchor = float(np.max(anchors))
    lo = min_anchor - _ELO_CLAMP_MARGIN
    hi = max_anchor + _ELO_CLAMP_MARGIN

    regularized = False
    bound_type = "exact"

    if total_score <= 0.0 or total_score >= total_games:
        regularized = True
        bound_type = "upper" if total_score <= 0.0 else "lower"
        mean_anchor = float(np.sum(counts * anchors) / total_games)
        anchors = np.append(anchors, mean_anchor)
        counts = np.append(counts, 2.0 * prior_strength)
        scores = np.append(scores, prior_strength)

    elo = _solve_1d_mle(anchors, counts, scores, lo=lo, hi=hi, max_iter=max_iter, tol=tol)
    return elo, regularized, bound_type


def estimate_bradley_terry_elo(
    results: Sequence[OpponentMatchResult],
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    prior_strength: float = 0.5,
) -> BradleyTerryEstimate:
    """Estimate multi-opponent Bradley-Terry Elo and stratified bootstrap CI."""
    if not results:
        raise ValueError("results must be non-empty")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be non-negative, got {n_bootstrap}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level!r}")

    seen_opponents: set[str] = set()
    for res in results:
        if res.opponent in seen_opponents:
            raise ValueError(f"Duplicate opponent name in results: {res.opponent!r}")
        seen_opponents.add(res.opponent)

    active = [res for res in results if res.games > 0]
    if not active:
        raise ValueError("results must contain at least one played game")

    elo, regularized, bound_type = solve_bradley_terry_elo(active, prior_strength=prior_strength)

    total_games = sum(res.games for res in active)
    total_score = float(sum(res.score for res in active))
    score_rate = total_score / total_games

    if n_bootstrap <= 1:
        ci_low = elo
        ci_high = elo
        std_error = 0.0
    else:
        rng = np.random.default_rng(seed)
        anchors = np.array([res.anchor_elo for res in active], dtype=np.float64)
        counts = np.array([float(res.games) for res in active], dtype=np.float64)
        outcome_arrays = [np.asarray(res.outcomes, dtype=np.float64) for res in active]
        min_anchor = float(np.min(anchors))
        max_anchor = float(np.max(anchors))
        lo = min_anchor - _ELO_CLAMP_MARGIN
        hi = max_anchor + _ELO_CLAMP_MARGIN
        mean_anchor = float(np.sum(counts * anchors) / float(total_games))

        boot_elos = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            boot_scores = np.empty(len(active), dtype=np.float64)
            for idx, arr in enumerate(outcome_arrays):
                n_i = arr.size
                sample_idx = rng.integers(0, n_i, size=n_i)
                boot_scores[idx] = float(np.sum(arr[sample_idx]))

            b_total = float(np.sum(boot_scores))
            if b_total <= 0.0 or b_total >= total_games:
                b_anchors = np.append(anchors, mean_anchor)
                b_counts = np.append(counts, 2.0 * prior_strength)
                b_scores = np.append(boot_scores, prior_strength)
            else:
                b_anchors = anchors
                b_counts = counts
                b_scores = boot_scores

            boot_elos[b] = _solve_1d_mle(b_anchors, b_counts, b_scores, lo=lo, hi=hi)

        alpha = 1.0 - ci_level
        ci_low, ci_high = (
            float(x) for x in np.percentile(boot_elos, [alpha * 50.0, (1.0 - alpha / 2.0) * 100.0])
        )
        std_error = float(np.std(boot_elos, ddof=1))

    per_opponent: dict[str, dict[str, float | int]] = {}
    for res in active:
        per_opponent[res.opponent] = {
            "anchor_elo": round(res.anchor_elo, 2),
            "games": res.games,
            "score": round(res.score, 4),
            "score_rate": round(res.score_rate, 6),
            "expected_score_rate": round(expected_score(elo, res.anchor_elo), 6),
        }

    return BradleyTerryEstimate(
        elo=elo,
        ci_low=ci_low,
        ci_high=ci_high,
        std_error=std_error,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        total_games=total_games,
        total_score=total_score,
        score_rate=score_rate,
        regularized=regularized,
        bound_type=bound_type,
        per_opponent=per_opponent,
    )


def results_from_game_records(
    records: Iterable[Mapping[str, Any]],
    anchor_ratings: Mapping[str, float] | None = None,
) -> list[OpponentMatchResult]:
    """Group per-game record dicts into ``OpponentMatchResult`` objects.

    Each record must provide ``opponent`` and either numeric ``score`` (in
    ``[0.0, 1.0]``) or textual ``result`` (``"win"``, ``"draw"``, ``"loss"``).
    Anchor ratings are taken from record-level ``anchor_elo`` if present,
    otherwise looked up in ``anchor_ratings`` (falling back to
    ``DEFAULT_ANCHOR_RATINGS``).
    """
    anchors_map: dict[str, float] = dict(DEFAULT_ANCHOR_RATINGS)
    if anchor_ratings is not None:
        anchors_map.update({k: float(v) for k, v in anchor_ratings.items()})

    grouped_outcomes: dict[str, list[float]] = defaultdict(list)
    resolved_anchors: dict[str, float] = {}

    for idx, rec in enumerate(records):
        opponent = rec.get("opponent")
        if not opponent or not isinstance(opponent, str):
            raise ValueError(f"Record {idx} is missing a valid 'opponent' string")

        if "score" in rec and rec["score"] is not None:
            score = float(rec["score"])
        elif "result" in rec and rec["result"] is not None:
            raw_res = str(rec["result"]).strip().lower()
            if raw_res not in _RESULT_TEXT_TO_SCORE:
                raise ValueError(f"Record {idx} has unrecognized result {rec['result']!r}")
            score = _RESULT_TEXT_TO_SCORE[raw_res]
        else:
            raise ValueError(f"Record {idx} must include 'score' or 'result'")

        if "anchor_elo" in rec and rec["anchor_elo"] is not None:
            anchor = float(rec["anchor_elo"])
        elif opponent in anchors_map:
            anchor = float(anchors_map[opponent])
        else:
            raise KeyError(
                f"No anchor_elo found for opponent {opponent!r}; pass anchor_ratings or record['anchor_elo']"
            )

        if opponent in resolved_anchors and not math.isclose(resolved_anchors[opponent], anchor):
            raise ValueError(
                f"Conflicting anchor_elo for opponent {opponent!r}: "
                f"{resolved_anchors[opponent]} vs {anchor}"
            )
        resolved_anchors[opponent] = anchor
        grouped_outcomes[opponent].append(score)

    return [
        OpponentMatchResult(
            opponent=opp,
            anchor_elo=resolved_anchors[opp],
            outcomes=tuple(outcomes),
        )
        for opp, outcomes in grouped_outcomes.items()
    ]
