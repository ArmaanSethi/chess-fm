"""Unit and integration tests for ``chessfm.coach.server``."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest
import requests

from chessfm.coach.server import (
    CoachHTTPServer,
    OllamaUnavailableError,
    _flatten_messages_for_ollama,
    build_ollama_llm_fn,
    create_server,
    parse_args,
)
from chessfm.coach.socratic import contains_move_spoiler

START_FEN = chess.STARTING_FEN
TACTICAL_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
SAMPLE_PGN = (
    '[Event "Test Game"]\n'
    '[White "Student"]\n'
    '[Black "Opponent"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. Ng5 Qxg5 5. d3 Qxg2 6. Rf1 d6"
)


class FakeEngine:
    """Minimal deterministic Stockfish stand-in for fast server tests."""

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int = 1,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        moves = list(board.legal_moves)
        if not moves:
            return []
        # Prefer checkmate moves first so tactical tests are deterministic.
        ordered = sorted(
            moves,
            key=lambda m: (
                not _gives_mate(board, m),
                not board.is_capture(m),
                m.uci(),
            ),
        )
        infos: list[dict[str, Any]] = []
        for idx, mv in enumerate(ordered[: max(1, multipv)]):
            score_cp = 9900 if _gives_mate(board, mv) else 45 - idx * 20
            infos.append(
                {
                    "pv": [mv],
                    "score": chess.engine.PovScore(chess.engine.Cp(score_cp), board.turn),
                }
            )
        return infos if multipv > 1 else infos[0]


def _gives_mate(board: chess.Board, move: chess.Move) -> bool:
    sim = board.copy(stack=False)
    sim.push(move)
    return sim.is_checkmate()


@contextmanager
def running_server(**kwargs: Any) -> Iterator[tuple[CoachHTTPServer, str]]:
    """Start a ``CoachHTTPServer`` on an ephemeral port in a background thread."""
    kwargs.setdefault("host", "127.0.0.1")
    kwargs.setdefault("port", 0)
    kwargs.setdefault("auto_open_engine", False)
    kwargs.setdefault("quiet", True)
    server = create_server(**kwargs)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Shared heuristic-mode Coach server for fast endpoint tests."""
    with running_server(auto_open_engine=False, depth=4) as (_, base_url):
        yield base_url


@pytest.fixture(scope="module")
def engine_server_url() -> Iterator[str]:
    """Shared Coach server backed by ``FakeEngine``."""
    with running_server(engine=FakeEngine(), auto_open_engine=False, depth=4) as (_, base_url):
        yield base_url


# ---------------------------------------------------------------------------
# UI & Health Endpoints
# ---------------------------------------------------------------------------


def test_get_root_serves_coach_studio_html(server_url: str) -> None:
    resp = requests.get(f"{server_url}/", timeout=5)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["Content-Type"]
    assert "ChessFM Coach Studio" in resp.text
    assert "Game Review" in resp.text

    resp_idx = requests.get(f"{server_url}/index.html", timeout=5)
    assert resp_idx.status_code == 200


def test_get_root_missing_static_returns_404(tmp_path: Path) -> None:
    with running_server(static_dir=tmp_path) as (_, base_url):
        resp = requests.get(f"{base_url}/", timeout=5)
        assert resp.status_code == 404
        assert "error" in resp.json()


def test_options_cors_preflight(server_url: str) -> None:
    resp = requests.options(f"{server_url}/api/health", timeout=5)
    assert resp.status_code == 204
    assert resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_health_endpoint_heuristic_and_engine(server_url: str, engine_server_url: str) -> None:
    r1 = requests.get(f"{server_url}/api/health", timeout=5)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "ok"
    assert d1["stockfish"] is False
    assert isinstance(d1["maia_backend"], str)
    assert d1["llm_backend"] == "deterministic_fallback"

    r2 = requests.get(f"{engine_server_url}/api/health", timeout=5)
    assert r2.status_code == 200
    assert r2.json()["stockfish"] is True


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------


def test_analyze_start_position_heuristic(server_url: str) -> None:
    resp = requests.post(
        f"{server_url}/api/analyze",
        json={"fen": START_FEN, "student_elo": 1400},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fen"] == START_FEN
    assert data["engine_source"] == "heuristic"
    assert data["best_move"] is not None
    assert 0.0 <= data["win_prob"] <= 100.0
    assert len(data["strategy_tree"]["top_lines"]) >= 1
    assert len(data["maia"]["top_moves"]) == 5
    assert data["maia"]["student_elo"] == 1400
    assert len(data["legal_moves"]) == 20


def test_analyze_with_student_move_and_fake_engine(engine_server_url: str) -> None:
    resp = requests.post(
        f"{engine_server_url}/api/analyze",
        json={"fen": TACTICAL_FEN, "student_move": "h5f7", "student_elo": 1200},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_source"] == "stockfish"
    assert data["best_move"] == "h5f7"
    assert data["classification"]["label"] == "Best"
    assert data["strategy_tree"]["student_branch"]["move"] == "h5f7"
    assert data["human_trap"] is not None
    assert data["human_trap"]["played_move"] == "h5f7"


def test_analyze_with_student_move_heuristic(server_url: str) -> None:
    resp = requests.post(
        f"{server_url}/api/analyze",
        json={"fen": START_FEN, "student_move": "e2e4", "student_elo": 1000},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"] is not None
    assert data["strategy_tree"]["student_branch"]["move"] == "e2e4"


def test_analyze_checkmate_position(server_url: str) -> None:
    resp = requests.post(f"{server_url}/api/analyze", json={"fen": CHECKMATE_FEN}, timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_game_over"] is True
    assert data["best_move"] is None
    assert data["strategy_tree"]["top_lines"] == []


# ---------------------------------------------------------------------------
# POST /api/hint (3-Level Socratic + 0% Spoiler Verification)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_hint_levels_with_explicit_best_move(server_url: str, level: int) -> None:
    best_move = "h5f7"
    resp = requests.post(
        f"{server_url}/api/hint",
        json={"fen": TACTICAL_FEN, "level": level, "best_move": best_move},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["level"] == level
    assert isinstance(data["hint"], str) and data["hint"]
    assert data["spoiler_verified"] is True

    if level in (1, 2):
        assert data["spoiler_free"] is True
        assert data["move_uci"] is None
        assert not contains_move_spoiler(data["hint"], best_move, TACTICAL_FEN)
    else:
        assert data["spoiler_free"] is False
        assert data["move_uci"] == best_move
        assert data["move_san"] == "Qxf7#"


def test_hint_auto_best_move_engine_and_heuristic(server_url: str, engine_server_url: str) -> None:
    r_eng = requests.post(
        f"{engine_server_url}/api/hint",
        json={"fen": TACTICAL_FEN, "level": 1},
        timeout=5,
    )
    assert r_eng.status_code == 200
    assert r_eng.json()["spoiler_free"] is True
    assert not contains_move_spoiler(r_eng.json()["hint"], "h5f7", TACTICAL_FEN)

    r_heu = requests.post(
        f"{server_url}/api/hint",
        json={"fen": START_FEN, "level": 2},
        timeout=5,
    )
    assert r_heu.status_code == 200
    assert r_heu.json()["spoiler_free"] is True


# ---------------------------------------------------------------------------
# POST /api/coach + Verification Badge + Ollama Fallback
# ---------------------------------------------------------------------------


def test_coach_deterministic_with_verification_badge(engine_server_url: str) -> None:
    resp = requests.post(
        f"{engine_server_url}/api/coach",
        json={
            "fen": TACTICAL_FEN,
            "question": "Why not play d3 here?",
            "student_move": "d2d3",
            "student_elo": 1200,
            "image_path": "board_photo.png",
            "audio_transcript": "Should I push d3?",
        },
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "template_fallback"
    assert data["llm_fallback_used"] is False
    assert isinstance(data["response"], str) and data["response"]
    assert data["student_move_analysis"]["move"] == "d2d3"
    badge = data["verification"]
    assert badge["passed"] is True
    assert 0.0 <= badge["factuality_score"] <= 1.0
    assert badge["violations"] == []


def test_coach_default_question_and_hint_level(server_url: str) -> None:
    resp = requests.post(
        f"{server_url}/api/coach",
        json={"fen": START_FEN, "student_move": "e2e4", "hint_level": 1},
        timeout=5,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hint"]["level"] == 1
    assert data["verification"]["passed"] is True


def test_coach_ollama_success_and_fallback() -> None:
    real_post = requests.post

    # 1. Successful mocked Ollama call
    mock_ok = MagicMock()
    mock_ok.raise_for_status.return_value = None
    mock_ok.json.return_value = {"response": "Play Qxf7# (h5f7) to deliver checkmate."}

    def _post_ok(url: str, *args: Any, **kwargs: Any) -> Any:
        if url.endswith("/api/generate"):
            return mock_ok
        return real_post(url, *args, **kwargs)

    with patch("chessfm.coach.server.requests.post", side_effect=_post_ok):
        with running_server(
            engine=FakeEngine(),
            ollama_host="http://127.0.0.1:11434",
            ollama_model="gemma4:e4b",
        ) as (_, base_url):
            health = requests.get(f"{base_url}/api/health", timeout=5).json()
            assert health["llm_backend"] == "ollama:gemma4:e4b"

            r = real_post(
                f"{base_url}/api/coach",
                json={"fen": TACTICAL_FEN, "question": "What is the best move?"},
                timeout=5,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["mode"] == "llm"
            assert body["llm_fallback_used"] is False
            assert "Qxf7#" in body["response"]
            assert body["verification"]["passed"] is True

    # 2. Automatic fallback when Ollama is unreachable
    def _post_fail(url: str, *args: Any, **kwargs: Any) -> Any:
        if url.endswith("/api/generate"):
            raise requests.ConnectionError("connection refused")
        return real_post(url, *args, **kwargs)

    with patch("chessfm.coach.server.requests.post", side_effect=_post_fail):
        with running_server(
            engine=FakeEngine(),
            ollama_host="http://127.0.0.1:11434",
            ollama_model="gemma4:e4b",
        ) as (_, base_url):
            r = real_post(
                f"{base_url}/api/coach",
                json={"fen": TACTICAL_FEN, "question": "What should I play?"},
                timeout=5,
            )
            assert r.status_code == 200
            body = r.json()
            assert body["mode"] == "template_fallback"
            assert body["llm_fallback_used"] is True
            assert body["llm_backend"] == "deterministic_fallback"
            assert body["verification"]["passed"] is True


def test_ollama_helper_empty_response_and_flatten() -> None:
    flattened = _flatten_messages_for_ollama(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "Hi"}],
            },
            {"role": "model", "content": "Hello"},
        ]
    )
    assert "[USER]" in flattened and "Hi" in flattened and "[MODEL]" in flattened

    mock_empty = MagicMock()
    mock_empty.raise_for_status.return_value = None
    mock_empty.json.return_value = {"response": "   "}
    llm_fn = build_ollama_llm_fn("http://localhost:11434", "gemma4")
    with patch("chessfm.coach.server.requests.post", return_value=mock_empty):
        with pytest.raises(OllamaUnavailableError):
            llm_fn([{"role": "user", "content": "Hi"}])


def test_custom_llm_fn_backend() -> None:
    with running_server(llm_fn=lambda _msgs: "Develop with Nf3 (g1f3).") as (_, base_url):
        health = requests.get(f"{base_url}/api/health", timeout=5).json()
        assert health["llm_backend"] == "custom"
        r = requests.post(f"{base_url}/api/coach", json={"fen": START_FEN}, timeout=5)
        assert r.status_code == 200
        assert "Nf3" in r.json()["response"]


# ---------------------------------------------------------------------------
# POST /api/review
# ---------------------------------------------------------------------------


def test_review_pgn_game_and_weakness_profile(server_url: str) -> None:
    resp = requests.post(
        f"{server_url}/api/review",
        json={"pgn": SAMPLE_PGN, "student_color": "white", "student_elo": 1100},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    review = data["review"]
    profile = data["weakness_profile"]

    assert 0.0 <= review["accuracy"] <= 100.0
    assert review["student_color"] == "white"
    assert review["total_plies"] == 12
    assert profile["games_analyzed"] == 1
    assert isinstance(profile["top_priorities"], list)
    assert isinstance(profile["drills"], list)
    if profile["drills"]:
        drill = profile["drills"][0]
        assert {"fen", "best_move", "category", "student_mistake"}.issubset(drill.keys())


# ---------------------------------------------------------------------------
# Error Handling (400 / 404) & CLI Arg Parser
# ---------------------------------------------------------------------------


def test_404_unknown_routes(server_url: str) -> None:
    assert requests.get(f"{server_url}/api/does-not-exist", timeout=5).status_code == 404
    assert requests.post(f"{server_url}/api/does-not-exist", json={}, timeout=5).status_code == 404


def test_400_bad_requests(server_url: str) -> None:
    # Malformed JSON
    r_bad_json = requests.post(
        f"{server_url}/api/analyze",
        data="{not-json",
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert r_bad_json.status_code == 400

    # Non-object JSON
    r_list = requests.post(
        f"{server_url}/api/analyze",
        data=json.dumps([1, 2, 3]),
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert r_list.status_code == 400

    # Invalid FEN
    assert (
        requests.post(f"{server_url}/api/analyze", json={"fen": "bad fen"}, timeout=5).status_code
        == 400
    )
    assert requests.post(f"{server_url}/api/analyze", json={}, timeout=5).status_code == 400

    # Illegal student_move
    assert (
        requests.post(
            f"{server_url}/api/analyze",
            json={"fen": START_FEN, "student_move": "e2e5"},
            timeout=5,
        ).status_code
        == 400
    )
    assert (
        requests.post(
            f"{server_url}/api/coach",
            json={"fen": START_FEN, "student_move": "notamove"},
            timeout=5,
        ).status_code
        == 400
    )
    assert (
        requests.post(
            f"{server_url}/api/coach",
            json={"fen": START_FEN, "student_move": "e2e5"},
            timeout=5,
        ).status_code
        == 400
    )

    # Hint validation errors
    assert (
        requests.post(
            f"{server_url}/api/hint", json={"fen": START_FEN, "level": 9}, timeout=5
        ).status_code
        == 400
    )
    assert (
        requests.post(
            f"{server_url}/api/hint", json={"fen": START_FEN, "level": "x"}, timeout=5
        ).status_code
        == 400
    )
    assert (
        requests.post(
            f"{server_url}/api/hint", json={"fen": CHECKMATE_FEN, "level": 1}, timeout=5
        ).status_code
        == 400
    )
    assert (
        requests.post(
            f"{server_url}/api/hint",
            json={"fen": START_FEN, "level": 1, "best_move": "e2e5"},
            timeout=5,
        ).status_code
        == 400
    )

    # Review validation errors
    assert requests.post(f"{server_url}/api/review", json={"pgn": ""}, timeout=5).status_code == 400
    assert (
        requests.post(
            f"{server_url}/api/review",
            json={"pgn": SAMPLE_PGN, "student_color": "purple"},
            timeout=5,
        ).status_code
        == 400
    )


def test_parse_args_defaults_and_overrides() -> None:
    defaults = parse_args([])
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8765
    assert defaults.no_stockfish is False

    custom = parse_args(["--host", "0.0.0.0", "--port", "9000", "--depth", "6", "--no-stockfish"])
    assert custom.host == "0.0.0.0"
    assert custom.port == 9000
    assert custom.depth == 6
    assert custom.no_stockfish is True
