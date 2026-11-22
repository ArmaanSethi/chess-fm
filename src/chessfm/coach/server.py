#!/usr/bin/env python3
"""Zero-dependency HTTP server and REST API for ChessFM Coach Studio.

Built on Python's stdlib :class:`http.server.ThreadingHTTPServer` so it runs
out of the box in the base ``venv`` without requiring FastAPI/uvicorn.

Supports:
- Stockfish MultiPV + WDL classification + Partial Strategy Tree analysis
- Maia-2 human move distribution & trap detection (neural or calibrated fallback)
- 3-level Socratic hint engine with strict 0%-spoiler verification
- Full ``CoachOrchestrator`` Q&A with ACT-Eval deterministic claim verification
- Optional local Ollama LLM backend via ``OLLAMA_HOST`` + ``CHESSFM_OLLAMA_MODEL``
  with automatic fallback to deterministic Stockfish+Maia coaching
- Full PGN game review and recurring weakness profiling with clickable drills
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chess
import requests

from chessfm.coach.classifier import (
    build_partial_strategy_tree,
    classify_move,
    win_prob_from_cp,
)
from chessfm.coach.game_review import (
    _analyze_position,
    build_weakness_profile,
    review_pgn_game,
)
from chessfm.coach.maia_sidecar import MaiaSidecar
from chessfm.coach.multimodal import CoachOrchestrator
from chessfm.coach.socratic import contains_move_spoiler, generate_socratic_hint
from chessfm.coach.verifier import verify_claims
from chessfm.rl.engine import close_engine, open_engine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_OLLAMA_TIMEOUT_S = 15.0
MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured Ollama endpoint cannot be reached or fails."""


def _flatten_messages_for_ollama(messages: list[dict[str, Any]]) -> str:
    """Flatten Gemma 4 multimodal message dicts into a plain-text prompt."""
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            body = "\n".join(p for p in text_parts if p)
        else:
            body = str(content)
        if body.strip():
            blocks.append(f"[{role}]\n{body.strip()}")
    return "\n\n".join(blocks)


def build_ollama_llm_fn(
    host: str,
    model: str,
    timeout: float = DEFAULT_OLLAMA_TIMEOUT_S,
) -> Callable[[list[dict[str, Any]]], str]:
    """Create a callable wrapping Ollama's ``/api/generate`` endpoint."""
    endpoint = f"{host.rstrip('/')}/api/generate"

    def _call_ollama(messages: list[dict[str, Any]]) -> str:
        prompt = _flatten_messages_for_ollama(messages)
        try:
            resp = requests.post(
                endpoint,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

        text = str(payload.get("response", "")).strip()
        if not text:
            raise OllamaUnavailableError("Ollama returned an empty response.")
        return text

    return _call_ollama


def _parse_fen(fen: Any) -> chess.Board:
    """Validate and parse a FEN string into a :class:`chess.Board`."""
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("Field 'fen' must be a non-empty FEN string.")
    try:
        return chess.Board(fen.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid FEN string: {exc}") from exc


def _parse_int(value: Any, name: str, default: int, *, lo: int, hi: int) -> int:
    """Parse and range-check an optional integer parameter."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Field '{name}' must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{name}' must be an integer.") from exc
    if parsed < lo or parsed > hi:
        raise ValueError(f"Field '{name}' must be between {lo} and {hi}.")
    return parsed


def _heuristic_strategy_tree(
    board: chess.Board,
    depth: int,
    student_move: str | None = None,
) -> dict[str, Any]:
    """Build a Partial Strategy Tree without Stockfish using the heuristic evaluator."""
    fen = board.fen()
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    if board.legal_moves.count() == 0:
        return {
            "fen": fen,
            "side_to_move": side_to_move,
            "best_move": None,
            "top_lines": [],
            "student_branch": None,
        }

    best_uci, best_cp, _ = _analyze_position(board, None, depth, board.turn)
    top_lines: list[dict[str, Any]] = []
    if best_uci is not None:
        best_obj = chess.Move.from_uci(best_uci)
        best_san = board.san(best_obj)
        top_lines.append(
            {
                "move": best_uci,
                "san": best_san,
                "eval_cp": int(round(best_cp)),
                "pv_uci": [best_uci],
                "pv_san": [best_san],
            }
        )

    student_branch: dict[str, Any] | None = None
    if student_move is not None:
        try:
            move_obj = chess.Move.from_uci(student_move)
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move: {student_move!r}") from exc
        if move_obj not in board.legal_moves:
            raise ValueError(f"Illegal move {student_move!r} for FEN {fen!r}")

        san = board.san(move_obj)
        board_after = board.copy()
        board_after.push(move_obj)
        if board_after.is_checkmate():
            student_cp = 10000.0
        elif board_after.is_game_over():
            student_cp = 0.0
        else:
            _, student_cp, _ = _analyze_position(board_after, None, depth, board.turn)

        student_branch = {
            "move": student_move,
            "san": san,
            "eval_cp": int(round(student_cp)),
            "pv_uci": [student_move],
            "pv_san": [san],
            "classification": classify_move(
                eval_before_cp=best_cp,
                eval_after_cp=student_cp,
                is_best_move=(student_move == best_uci),
            ),
        }

    return {
        "fen": fen,
        "side_to_move": side_to_move,
        "best_move": best_uci,
        "top_lines": top_lines,
        "student_branch": student_branch,
    }


def _enrich_maia_moves(
    board: chess.Board,
    top_moves: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    """Attach SAN notation and rounded probabilities to Maia top move tuples."""
    enriched: list[dict[str, Any]] = []
    for uci, prob in top_moves:
        try:
            mv = chess.Move.from_uci(uci)
            san = board.san(mv) if mv in board.legal_moves else uci
        except ValueError:
            san = uci
        enriched.append({"uci": uci, "san": san, "prob": round(float(prob), 4)})
    return enriched


@dataclass
class CoachServerContext:
    """Shared server state holding engine, Maia sidecar, LLM backend, and lock."""

    engine: Any | None = None
    owns_engine: bool = False
    maia: MaiaSidecar = field(default_factory=lambda: MaiaSidecar(use_neural=False))
    llm_fn: Callable[[list[dict[str, Any]]], str] | None = None
    llm_backend: str = "deterministic_fallback"
    default_depth: int = 8
    static_dir: Path = STATIC_DIR
    quiet: bool = False
    engine_lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        """Close the Stockfish engine if owned by this server context."""
        if self.owns_engine and self.engine is not None:
            close_engine(self.engine)
            self.engine = None
            self.owns_engine = False

    def health(self) -> dict[str, Any]:
        """Return health status and active backend identifiers."""
        return {
            "status": "ok",
            "stockfish": self.engine is not None,
            "maia_backend": self.maia.backend,
            "llm_backend": self.llm_backend,
        }

    def handle_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run Stockfish MultiPV + WDL + Partial Strategy Tree + Maia-2 analysis."""
        board = _parse_fen(payload.get("fen"))
        fen = board.fen()
        student_elo = _parse_int(payload.get("student_elo"), "student_elo", 1200, lo=400, hi=3200)
        depth = _parse_int(payload.get("depth"), "depth", self.default_depth, lo=1, hi=22)
        multipv = _parse_int(payload.get("multipv"), "multipv", 3, lo=1, hi=5)

        raw_student_move = payload.get("student_move")
        student_move: str | None = None
        if raw_student_move is not None and str(raw_student_move).strip():
            student_move = str(raw_student_move).strip().lower()

        with self.engine_lock:
            if self.engine is not None:
                tree = build_partial_strategy_tree(
                    fen=fen,
                    engine=self.engine,
                    depth=depth,
                    multipv=multipv,
                    pv_plies=6,
                    student_move=student_move,
                )
                engine_source = "stockfish"
            else:
                tree = _heuristic_strategy_tree(board, depth=depth, student_move=student_move)
                engine_source = "heuristic"

            maia_pred = self.maia.predict_human_moves(fen, student_elo=student_elo, top_k=5)

            human_trap: dict[str, Any] | None = None
            student_branch = tree.get("student_branch")
            if student_branch is not None and tree.get("best_move"):
                human_trap = self.maia.detect_human_trap(
                    fen=fen,
                    played_move=student_branch["move"],
                    best_move=tree["best_move"],
                    cp_loss=student_branch["classification"]["cp_loss"],
                    student_elo=student_elo,
                )

        top_lines = tree["top_lines"]
        best_eval_cp = float(top_lines[0]["eval_cp"]) if top_lines else 0.0
        best_san = top_lines[0]["san"] if top_lines else None
        win_prob = round(win_prob_from_cp(best_eval_cp), 2)
        # White-perspective eval for UI eval bar rendering.
        eval_cp_white = best_eval_cp if board.turn == chess.WHITE else -best_eval_cp

        legal_moves = [
            {
                "uci": mv.uci(),
                "san": board.san(mv),
                "from": chess.square_name(mv.from_square),
                "to": chess.square_name(mv.to_square),
            }
            for mv in board.legal_moves
        ]

        return {
            "fen": fen,
            "side_to_move": tree["side_to_move"],
            "in_check": board.is_check(),
            "is_game_over": board.is_game_over(),
            "engine_source": engine_source,
            "best_move": tree["best_move"],
            "best_san": best_san,
            "eval_cp": round(best_eval_cp, 1),
            "eval_cp_white": round(eval_cp_white, 1),
            "win_prob": win_prob,
            "win_prob_white": round(win_prob_from_cp(eval_cp_white), 2),
            "strategy_tree": tree,
            "classification": student_branch["classification"] if student_branch else None,
            "human_trap": human_trap,
            "maia": {
                "backend": maia_pred["backend"],
                "student_elo": maia_pred["student_elo"],
                "elo_bracket": maia_pred["elo_bracket"],
                "top_moves": _enrich_maia_moves(board, maia_pred["top_moves"]),
            },
            "legal_moves": legal_moves,
        }

    def handle_hint(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate a 3-level Socratic hint with 0%-spoiler verification."""
        board = _parse_fen(payload.get("fen"))
        fen = board.fen()
        level = _parse_int(payload.get("level"), "level", 1, lo=1, hi=3)
        depth = _parse_int(payload.get("depth"), "depth", self.default_depth, lo=1, hi=22)

        if board.legal_moves.count() == 0:
            raise ValueError("Position has no legal moves (game is already over).")

        raw_best = payload.get("best_move")
        pv_ucis: list[str] | None = None
        if raw_best is not None and str(raw_best).strip():
            best_move_uci = str(raw_best).strip().lower()
            try:
                mv = chess.Move.from_uci(best_move_uci)
            except ValueError as exc:
                raise ValueError(f"Invalid UCI best_move: {best_move_uci!r}") from exc
            if mv not in board.legal_moves:
                raise ValueError(f"Illegal best_move {best_move_uci!r} for FEN.")
        else:
            with self.engine_lock:
                if self.engine is not None:
                    tree = build_partial_strategy_tree(
                        fen=fen,
                        engine=self.engine,
                        depth=depth,
                        multipv=1,
                        pv_plies=4,
                    )
                    best_move_uci = tree["best_move"]
                    if tree["top_lines"]:
                        pv_ucis = tree["top_lines"][0]["pv_uci"]
                else:
                    best_move_uci, _, _ = _analyze_position(board, None, depth, board.turn)
            if not best_move_uci:
                raise ValueError("Could not determine a best move for this position.")

        result = generate_socratic_hint(
            fen=fen,
            best_move_uci=best_move_uci,
            level=level,
            pv_ucis=pv_ucis,
        )
        has_spoiler = contains_move_spoiler(result["hint"], best_move_uci, fen)
        spoiler_verified = (not has_spoiler) if level in (1, 2) else True

        return {
            "fen": fen,
            **result,
            "spoiler_verified": spoiler_verified,
        }

    def handle_coach(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run full ``CoachOrchestrator`` pipeline + ACT-Eval claim verification badge."""
        board = _parse_fen(payload.get("fen"))
        fen = board.fen()
        student_elo = _parse_int(payload.get("student_elo"), "student_elo", 1200, lo=400, hi=3200)
        depth = _parse_int(payload.get("depth"), "depth", self.default_depth, lo=1, hi=22)

        hint_level_raw = payload.get("hint_level")
        hint_level = (
            _parse_int(hint_level_raw, "hint_level", 1, lo=1, hi=3)
            if hint_level_raw is not None
            else None
        )

        question = str(payload.get("question") or "").strip()
        student_move_raw = payload.get("student_move")
        student_move = str(student_move_raw).strip().lower() if student_move_raw else None
        image_path = str(payload["image_path"]).strip() if payload.get("image_path") else None
        audio_transcript = (
            str(payload["audio_transcript"]).strip() if payload.get("audio_transcript") else None
        )

        student_move_analysis: dict[str, Any] | None = None
        if student_move is not None:
            try:
                mv = chess.Move.from_uci(student_move)
            except ValueError as exc:
                raise ValueError(f"Invalid UCI student_move: {student_move!r}") from exc
            if mv not in board.legal_moves:
                raise ValueError(f"Illegal student_move {student_move!r} for FEN.")
            student_san = board.san(mv)
            if not question:
                question = f"Explain why {student_san} ({student_move}) is good or bad here."

        if not question:
            question = "Explain the key strategic and tactical ideas in this position."

        llm_fallback_used = False
        active_llm_backend = self.llm_backend

        with self.engine_lock:
            if student_move is not None:
                if self.engine is not None:
                    tree = build_partial_strategy_tree(
                        fen=fen,
                        engine=self.engine,
                        depth=depth,
                        multipv=3,
                        pv_plies=4,
                        student_move=student_move,
                    )
                else:
                    tree = _heuristic_strategy_tree(board, depth=depth, student_move=student_move)
                student_move_analysis = tree.get("student_branch")

            orchestrator = CoachOrchestrator(engine=self.engine, llm_fn=self.llm_fn)
            try:
                coach_result = orchestrator.coach(
                    fen=fen,
                    user_question=question,
                    student_elo=student_elo,
                    hint_level=hint_level,
                    image_path=image_path,
                    audio_transcript=audio_transcript,
                    depth=depth,
                )
            except OllamaUnavailableError as exc:
                logger.warning("Ollama unavailable (%s); falling back to deterministic coach.", exc)
                llm_fallback_used = True
                active_llm_backend = "deterministic_fallback"
                fallback_orch = CoachOrchestrator(engine=self.engine, llm_fn=None)
                coach_result = fallback_orch.coach(
                    fen=fen,
                    user_question=question,
                    student_elo=student_elo,
                    hint_level=hint_level,
                    image_path=image_path,
                    audio_transcript=audio_transcript,
                    depth=depth,
                )

        response_text = coach_result["response"]
        if student_move_analysis is not None:
            cls = student_move_analysis["classification"]
            move_note = (
                f" Move evaluation for {student_move_analysis['san']} "
                f"({student_move_analysis['move']}): {cls['label']} "
                f"(cp loss: {cls['cp_loss']:.0f}, win prob drop: {cls['win_prob_drop']:.1f}%)."
            )
            response_text = f"{response_text}{move_note}"

        verification = verify_claims(fen, response_text, engine=None)

        return {
            "fen": fen,
            "mode": coach_result["mode"],
            "llm_backend": active_llm_backend,
            "llm_fallback_used": llm_fallback_used,
            "response": response_text,
            "hint": coach_result["hint"],
            "analysis": coach_result["analysis"],
            "student_move_analysis": student_move_analysis,
            "verification": {
                "passed": verification["passed"],
                "factuality_score": round(float(verification["factuality_score"]), 4),
                "total_claims": verification["total_claims"],
                "verified_claims": verification["verified_claims"],
                "violations": verification["violations"],
            },
        }

    def handle_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run full PGN game review and aggregate recurring weakness profile."""
        pgn_text = payload.get("pgn")
        if not isinstance(pgn_text, str) or not pgn_text.strip():
            raise ValueError("Field 'pgn' must be a non-empty PGN string.")

        student_color = str(payload.get("student_color") or "white").strip().lower()
        student_elo = _parse_int(payload.get("student_elo"), "student_elo", 1200, lo=400, hi=3200)
        depth = _parse_int(payload.get("depth"), "depth", self.default_depth, lo=1, hi=18)
        max_critical = _parse_int(payload.get("max_critical"), "max_critical", 5, lo=1, hi=20)

        with self.engine_lock:
            review = review_pgn_game(
                pgn_text=pgn_text,
                engine=self.engine,
                student_color=student_color,
                student_elo=student_elo,
                depth=depth,
                max_critical=max_critical,
            )

        profile = build_weakness_profile([review])
        return {
            "review": review,
            "weakness_profile": profile,
        }


class CoachHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying a shared :class:`CoachServerContext`."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        ctx: CoachServerContext,
    ) -> None:
        self.ctx = ctx
        super().__init__(server_address, CoachRequestHandler)

    def server_close(self) -> None:
        """Close the listening socket and any owned Stockfish engine."""
        try:
            super().server_close()
        finally:
            self.ctx.close()


class CoachRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler dispatching ChessFM Coach Studio UI and JSON API routes."""

    server: CoachHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress noisy stderr request logs when ``ctx.quiet`` is enabled."""
        if not self.server.ctx.quiet:
            logger.info("%s - %s", self.address_string(), format % args)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html_bytes: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(html_bytes)

    def _read_json_body(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return {}
        try:
            content_length = int(length_header)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length < 0 or content_length > MAX_REQUEST_BODY_BYTES:
            raise ValueError(f"Request body exceeds limit ({MAX_REQUEST_BODY_BYTES} bytes).")
        if content_length == 0:
            return {}

        raw = self.rfile.read(content_length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed JSON request body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object.")
        return parsed

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Respond to CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        """Serve the Coach Studio single-page UI and ``/api/health``."""
        path = urlparse(self.path).path
        ctx = self.server.ctx

        if path in ("/", "/index.html"):
            index_path = ctx.static_dir / "index.html"
            if not index_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "UI index.html not found."})
                return
            self._send_html(HTTPStatus.OK, index_path.read_bytes())
            return

        if path == "/api/health":
            self._send_json(HTTPStatus.OK, ctx.health())
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        """Dispatch JSON API endpoints."""
        path = urlparse(self.path).path
        ctx = self.server.ctx

        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "/api/analyze": ctx.handle_analyze,
            "/api/hint": ctx.handle_hint,
            "/api/coach": ctx.handle_coach,
            "/api/review": ctx.handle_review,
        }
        handler = routes.get(path)
        if handler is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
            return

        try:
            payload = self._read_json_body()
            result = handler(payload)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in %s", path)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, result)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    engine: Any | None = None,
    *,
    auto_open_engine: bool = True,
    maia_sidecar: MaiaSidecar | None = None,
    llm_fn: Callable[[list[dict[str, Any]]], str] | None = None,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    depth: int = 8,
    static_dir: Path | None = None,
    quiet: bool = False,
) -> CoachHTTPServer:
    """Create and configure a :class:`CoachHTTPServer` instance.

    Args:
        host: Bind address (defaults to ``127.0.0.1``).
        port: TCP port (pass ``0`` for an ephemeral OS-assigned port in tests).
        engine: Optional pre-opened Stockfish engine (or compatible mock).
        auto_open_engine: When ``engine is None`` and ``True``, attempts to open
            a local Stockfish binary via :func:`chessfm.rl.engine.open_engine`.
        maia_sidecar: Optional pre-initialized :class:`MaiaSidecar`.
        llm_fn: Optional custom LLM generation callable ``(messages) -> str``.
        ollama_host: Optional Ollama base URL (defaults to ``OLLAMA_HOST`` env).
        ollama_model: Optional Ollama model tag (defaults to ``CHESSFM_OLLAMA_MODEL`` env).
        depth: Default Stockfish analysis depth.
        static_dir: Override directory containing ``index.html``.
        quiet: Suppress per-request log output when ``True``.
    """
    owns_engine = False
    active_engine = engine
    if active_engine is None and auto_open_engine:
        active_engine = open_engine()
        owns_engine = active_engine is not None

    sidecar = (
        maia_sidecar
        if maia_sidecar is not None
        else MaiaSidecar(use_neural=True, engine=active_engine)
    )

    resolved_host = ollama_host if ollama_host is not None else os.environ.get("OLLAMA_HOST")
    resolved_model = (
        ollama_model if ollama_model is not None else os.environ.get("CHESSFM_OLLAMA_MODEL")
    )

    active_llm_fn = llm_fn
    if active_llm_fn is not None:
        llm_backend = "custom"
    elif resolved_host and resolved_model:
        active_llm_fn = build_ollama_llm_fn(resolved_host, resolved_model)
        llm_backend = f"ollama:{resolved_model}"
    else:
        llm_backend = "deterministic_fallback"

    ctx = CoachServerContext(
        engine=active_engine,
        owns_engine=owns_engine,
        maia=sidecar,
        llm_fn=active_llm_fn,
        llm_backend=llm_backend,
        default_depth=depth,
        static_dir=static_dir if static_dir is not None else STATIC_DIR,
        quiet=quiet,
    )
    return CoachHTTPServer((host, int(port)), ctx)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``python -m chessfm.coach.server``."""
    parser = argparse.ArgumentParser(description="Run the ChessFM Coach Studio HTTP server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to listen on.")
    parser.add_argument("--depth", type=int, default=8, help="Default Stockfish search depth.")
    parser.add_argument(
        "--no-stockfish",
        action="store_true",
        help="Disable Stockfish auto-discovery and use heuristic fallback.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI entry point for launching ChessFM Coach Studio."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    server = create_server(
        host=args.host,
        port=args.port,
        auto_open_engine=not args.no_stockfish,
        depth=args.depth,
    )
    bound_host, bound_port = server.server_address[:2]
    logger.info(
        "ChessFM Coach Studio running at http://%s:%s (health: %s)",
        bound_host,
        bound_port,
        server.ctx.health(),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down ChessFM Coach Studio.")
    finally:
        server.server_close()


if __name__ == "__main__":  # pragma: no cover
    main()
