#!/usr/bin/env python3
"""Game importers and multi-game player profiling for ChessFM Coach.

Supports importing recent games from Lichess and Chess.com public APIs as well
as local multi-game PGN bundles, auto-detecting the student's color per game,
running ``review_pgn_game`` across the batch, and aggregating recurring themes
via ``build_weakness_profile``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import chess.pgn
import requests

from chessfm.coach.game_review import build_weakness_profile, review_pgn_game
from chessfm.rl.engine import close_engine, open_engine

DEFAULT_USER_AGENT = "ChessFM/0.1 (+https://github.com/ArmaanSethi/chess-fm)"
LICHESS_GAMES_API = (
    "https://lichess.org/api/games/user/{username}?max={max_games}&evals=false&opening=true"
)
CHESSCOM_ARCHIVES_API = "https://api.chess.com/pub/player/{username}/games/archives"


class ImporterError(RuntimeError):
    """Raised when fetching or parsing remote games fails."""


def split_pgn_games(pgn_bundle: str, max_games: int | None = None) -> list[str]:
    """Split a multi-game PGN bundle into individual PGN strings.

    Uses ``chess.pgn.read_game`` so headers, comments, variations, and blank
    lines between games are preserved cleanly.

    Args:
        pgn_bundle: Raw text containing zero, one, or many PGN games.
        max_games: Optional cap on the number of games returned. Values ``<= 0``
            return an empty list.

    Returns:
        List of individual PGN strings in original order.
    """
    if not isinstance(pgn_bundle, str) or not pgn_bundle.strip():
        return []
    if max_games is not None and max_games <= 0:
        return []

    stream = io.StringIO(pgn_bundle)
    games: list[str] = []

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        exported = game.accept(exporter).strip()
        if exported:
            games.append(exported)
            if max_games is not None and len(games) >= max_games:
                break

    return games


def detect_student_color(pgn_text: str, username: str) -> str:
    """Detect whether ``username`` played White or Black in ``pgn_text``.

    Performs a case-insensitive match against the PGN ``White`` and ``Black``
    headers, defaulting to ``"white"`` when the username is blank, missing, or
    does not match either header.
    """
    target = username.strip().lower() if isinstance(username, str) else ""
    if not target or not isinstance(pgn_text, str) or not pgn_text.strip():
        return "white"

    headers = chess.pgn.read_headers(io.StringIO(pgn_text))
    if headers is None:
        return "white"

    white_player = headers.get("White", "").strip().lower()
    if white_player == target:
        return "white"

    black_player = headers.get("Black", "").strip().lower()
    if black_player == target:
        return "black"

    return "white"


def _check_http_response(response: requests.Response, platform: str, username: str) -> None:
    """Raise a clear ``ImporterError`` for non-2xx HTTP responses."""
    status = response.status_code
    if status == 404:
        raise ImporterError(f"{platform} user {username!r} not found (HTTP 404).")
    if status == 429:
        raise ImporterError(f"{platform} API rate limit exceeded for {username!r} (HTTP 429).")
    if status >= 400:
        raise ImporterError(f"{platform} API returned HTTP {status} for {username!r}.")


def fetch_lichess_games(username: str, max_games: int = 10, timeout: int = 15) -> list[str]:
    """Fetch recent games for a Lichess user via the public unauthenticated API.

    Args:
        username: Lichess username.
        max_games: Maximum number of recent games to fetch.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of individual PGN strings.
    """
    clean_user = username.strip() if isinstance(username, str) else ""
    if not clean_user:
        raise ValueError("username must be a non-empty string.")
    if max_games <= 0:
        return []

    url = LICHESS_GAMES_API.format(
        username=quote(clean_user, safe=""),
        max_games=max_games,
    )
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/x-chess-pgn",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ImporterError(f"Failed to fetch Lichess games for {clean_user!r}: {exc}") from exc

    _check_http_response(response, "Lichess", clean_user)
    return split_pgn_games(response.text, max_games=max_games)


def fetch_chesscom_games(username: str, max_games: int = 10, timeout: int = 15) -> list[str]:
    """Fetch recent games for a Chess.com user via the public monthly archives API.

    Walks monthly archives from newest to oldest and collects up to ``max_games``
    PGNs in reverse-chronological order.

    Args:
        username: Chess.com username (case-insensitive).
        max_games: Maximum number of recent games to fetch.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of individual PGN strings.
    """
    clean_user = username.strip() if isinstance(username, str) else ""
    if not clean_user:
        raise ValueError("username must be a non-empty string.")
    if max_games <= 0:
        return []

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
    }
    archives_url = CHESSCOM_ARCHIVES_API.format(username=quote(clean_user.lower(), safe=""))

    try:
        archives_resp = requests.get(archives_url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ImporterError(
            f"Failed to fetch Chess.com archives for {clean_user!r}: {exc}"
        ) from exc

    _check_http_response(archives_resp, "Chess.com", clean_user)

    try:
        archives_payload = archives_resp.json()
    except ValueError as exc:
        raise ImporterError(
            f"Invalid JSON in Chess.com archives response for {clean_user!r}."
        ) from exc

    archive_urls = archives_payload.get("archives", [])
    if not isinstance(archive_urls, list) or not archive_urls:
        return []

    collected: list[str] = []
    for month_url in reversed(archive_urls):
        if not isinstance(month_url, str) or not month_url.strip():
            continue
        try:
            month_resp = requests.get(month_url, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            raise ImporterError(
                f"Failed to fetch Chess.com archive {month_url!r} for {clean_user!r}: {exc}"
            ) from exc

        _check_http_response(month_resp, "Chess.com", clean_user)

        try:
            month_payload = month_resp.json()
        except ValueError as exc:
            raise ImporterError(
                f"Invalid JSON in Chess.com monthly archive {month_url!r}."
            ) from exc

        month_games = month_payload.get("games", [])
        if not isinstance(month_games, list):
            continue

        for game_entry in reversed(month_games):
            if not isinstance(game_entry, dict):
                continue
            pgn = game_entry.get("pgn")
            if isinstance(pgn, str) and pgn.strip():
                collected.append(pgn.strip())
                if len(collected) >= max_games:
                    return collected

    return collected


def profile_player_games(
    pgns: list[str],
    username: str = "",
    engine: Any | None = None,
    student_elo: int = 1200,
    depth: int = 6,
) -> dict[str, Any]:
    """Review a batch of PGNs and aggregate a player's weakness profile.

    Args:
        pgns: List of individual PGN strings to analyze.
        username: Optional player handle used to auto-detect whether the student
            played White or Black in each game. Defaults to ``"white"`` when empty.
        engine: Optional open Stockfish engine; falls back to the heuristic
            analyzer when ``None``.
        student_elo: Approximate student Elo rating passed to ``review_pgn_game``.
        depth: Stockfish search depth when ``engine`` is provided.

    Returns:
        Dictionary with keys ``username``, ``games_analyzed``, ``mean_accuracy``,
        ``reviews``, and ``weakness_profile``.
    """
    clean_user = username.strip() if isinstance(username, str) else ""
    reviews: list[dict[str, Any]] = []

    for pgn_text in pgns:
        if not isinstance(pgn_text, str) or not pgn_text.strip():
            continue
        student_color = detect_student_color(pgn_text, clean_user) if clean_user else "white"
        review = review_pgn_game(
            pgn_text,
            engine=engine,
            student_color=student_color,
            student_elo=student_elo,
            depth=depth,
        )
        reviews.append(review)

    weakness_profile = build_weakness_profile(reviews)
    mean_accuracy = (
        round(sum(float(r["accuracy"]) for r in reviews) / len(reviews), 1) if reviews else 0.0
    )

    return {
        "username": clean_user,
        "games_analyzed": len(reviews),
        "mean_accuracy": mean_accuracy,
        "reviews": reviews,
        "weakness_profile": weakness_profile,
    }


def format_profile_report(profile: dict[str, Any]) -> str:
    """Format a ``profile_player_games`` result into a concise terminal summary."""
    user_label = profile.get("username") or "Student"
    games_analyzed = profile.get("games_analyzed", 0)
    mean_accuracy = float(profile.get("mean_accuracy", 0.0))
    wp = profile.get("weakness_profile", {})

    lines = [
        f"ChessFM Player Profile — {user_label}",
        f"Games Analyzed: {games_analyzed} | Mean Accuracy: {mean_accuracy:.1f}%",
    ]

    top_priorities = wp.get("top_priorities", [])
    if top_priorities:
        lines.append("Top Study Priorities:")
        for idx, item in enumerate(top_priorities, start=1):
            lines.append(
                f"  {idx}. {item['category']} "
                f"(count={item['count']}, avg_cp_loss={item['avg_cp_loss']:.1f}) — "
                f"{item['recommendation']}"
            )
    else:
        lines.append("Top Study Priorities: none detected")

    drills = wp.get("drills", [])
    lines.append(f"Practice Drills Extracted: {len(drills)}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``python -m chessfm.coach.importers``."""
    parser = argparse.ArgumentParser(
        prog="python -m chessfm.coach.importers",
        description=(
            "Import games from Lichess, Chess.com, or a local PGN file and "
            "generate an aggregated ChessFM weakness profile."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("lichess", "chesscom"),
        default=None,
        help="Remote platform to import games from (required unless --pgn-file is set).",
    )
    parser.add_argument(
        "--username",
        default="",
        help="Player username (required for remote imports; optional for --pgn-file color detection).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Maximum number of games to import and analyze (default: 5).",
    )
    parser.add_argument(
        "--pgn-file",
        type=Path,
        default=None,
        help="Optional local PGN file (single or multi-game) to profile offline.",
    )
    parser.add_argument(
        "--elo",
        type=int,
        default=1200,
        help="Approximate student Elo rating (default: 1200).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Stockfish search depth when available (default: 6).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="HTTP request timeout in seconds for remote imports (default: 15).",
    )
    parser.add_argument(
        "--no-engine",
        action="store_true",
        help="Skip Stockfish and use the lightweight heuristic analyzer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output the structured player profile as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for importing and profiling player games."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.pgn_file is not None:
        if not args.pgn_file.is_file():
            print(f"Error: PGN file not found: {args.pgn_file}", file=sys.stderr)
            return 1
        pgn_text = args.pgn_file.read_text(encoding="utf-8")
        pgns = split_pgn_games(pgn_text, max_games=args.games)
    else:
        if not args.platform:
            print(
                "Error: specify --platform (lichess|chesscom) or --pgn-file.",
                file=sys.stderr,
            )
            return 2
        if not args.username.strip():
            print(
                "Error: --username is required when importing from a remote platform.",
                file=sys.stderr,
            )
            return 2
        try:
            if args.platform == "lichess":
                pgns = fetch_lichess_games(
                    args.username, max_games=args.games, timeout=args.timeout
                )
            else:
                pgns = fetch_chesscom_games(
                    args.username, max_games=args.games, timeout=args.timeout
                )
        except (ImporterError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    engine = None if args.no_engine else open_engine(threads=1, hash_mb=16)
    try:
        profile = profile_player_games(
            pgns,
            username=args.username,
            engine=engine,
            student_elo=args.elo,
            depth=args.depth,
        )
    finally:
        close_engine(engine)

    if args.as_json:
        print(json.dumps(profile, indent=2))
    else:
        print(format_profile_report(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
