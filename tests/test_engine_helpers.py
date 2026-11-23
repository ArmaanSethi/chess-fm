#!/usr/bin/env python3
"""Unit tests for the shared Stockfish analysis helpers."""

import chess
import pytest

from chessfm.rl.engine import best_cp, close_engine, eval_after_move, open_engine


class _FakeScore:
    def __init__(self, cp, mate=None):
        self._cp = cp
        self._mate = mate

    def is_mate(self):
        return self._mate is not None

    def mate(self):
        return self._mate

    def white(self):
        return self

    def score(self):
        return self._cp


class _MockEngine:
    """Returns a fixed white-centric eval; the helper handles the flip."""

    def __init__(self, white_cp=50):
        self.white_cp = white_cp

    def analyse(self, board, limit):
        return {"score": _FakeScore(self.white_cp)}


class _MockEngineWithMate:
    """Reports a forced mate for White."""

    def analyse(self, board, limit):
        return {"score": _FakeScore(0, mate=2)}


def test_best_cp_white_perspective():
    board = chess.Board()  # white to move
    assert best_cp(_MockEngine(white_cp=50), board) == 50.0


def test_best_cp_black_perspective_is_flipped():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = chess.Board(fen)  # black to move
    assert best_cp(_MockEngine(white_cp=50), board) == -50.0


def test_best_cp_for_color_override():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    board = chess.Board(fen)
    # Ask for white's perspective explicitly despite black to move.
    assert best_cp(_MockEngine(white_cp=50), board, for_color=chess.WHITE) == 50.0


def test_eval_after_move_is_from_mover_perspective():
    board = chess.Board()  # white to move
    # After e2e4 it's black's turn, but the eval must stay from white's view.
    assert eval_after_move(_MockEngine(white_cp=50), board, "e2e4") == 50.0


def test_eval_after_move_illegal_move_returns_none():
    board = chess.Board()
    assert eval_after_move(_MockEngine(white_cp=50), board, "e2e5") is None


def test_best_cp_none_when_engine_missing():
    assert best_cp(None, chess.Board()) is None
    assert eval_after_move(None, chess.Board(), "e2e4") is None


def test_best_cp_positive_mate_uses_10000():
    assert best_cp(_MockEngineWithMate(), chess.Board()) == 10000.0


def test_best_cp_negative_mate_from_black_perspective():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert best_cp(_MockEngineWithMate(), chess.Board(fen)) == -10000.0


class _FailingEngine:
    def analyse(self, board, limit):
        # A genuine engine error (not a code bug) must map to None.
        raise chess.engine.EngineError("engine crashed")


def test_best_cp_returns_none_on_engine_error():
    assert best_cp(_FailingEngine(), chess.Board()) is None


def test_best_cp_does_not_mask_code_bugs():
    """A non-engine exception (a bug) must propagate, not become None."""

    class _BuggyEngine:
        def analyse(self, board, limit):
            raise RuntimeError("this is a bug, not an engine error")

    with pytest.raises(RuntimeError):
        best_cp(_BuggyEngine(), chess.Board())


def test_eval_after_move_returns_none_on_engine_error():
    assert eval_after_move(_FailingEngine(), chess.Board(), "e2e4") is None


class _NeverQuits:
    def quit(self):
        raise chess.engine.EngineError("refusing to quit")


def test_close_engine_handles_failure():
    close_engine(_NeverQuits())  # must not raise
    close_engine(None)  # must not raise
    close_engine(object())


def test_open_engine_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr("chessfm.rl.engine.STOCKFISH_PATHS", ["/nonexistent/stockfish"])
    assert open_engine() is None


class _PovScoreEngine:
    """Returns a real python-chess PovScore for testing mate/MateGiven semantics."""

    def __init__(self, pov_score):
        self._pov_score = pov_score

    def analyse(self, board, limit):
        return {"score": self._pov_score}


def test_white_delivered_mate_scores_positive_10000():
    # White just checkmated Black: Black is to move and mated -> PovScore(Mate(0), BLACK).
    eng = _PovScoreEngine(chess.engine.PovScore(chess.engine.Mate(0), chess.BLACK))
    board = chess.Board()
    assert best_cp(eng, board, for_color=chess.WHITE) == 10000.0
    assert best_cp(eng, board, for_color=chess.BLACK) == -10000.0


def test_black_delivered_mate_scores_positive_10000():
    # Black just checkmated White: White is to move and mated -> PovScore(Mate(0), WHITE).
    eng = _PovScoreEngine(chess.engine.PovScore(chess.engine.Mate(0), chess.WHITE))
    board = chess.Board()
    assert best_cp(eng, board, for_color=chess.BLACK) == 10000.0
    assert best_cp(eng, board, for_color=chess.WHITE) == -10000.0


def test_mate_given_explicit_pov():
    eng_w = _PovScoreEngine(chess.engine.PovScore(chess.engine.MateGiven, chess.WHITE))
    assert best_cp(eng_w, chess.Board(), for_color=chess.WHITE) == 10000.0
    assert best_cp(eng_w, chess.Board(), for_color=chess.BLACK) == -10000.0

    eng_b = _PovScoreEngine(chess.engine.PovScore(chess.engine.MateGiven, chess.BLACK))
    assert best_cp(eng_b, chess.Board(), for_color=chess.BLACK) == 10000.0
    assert best_cp(eng_b, chess.Board(), for_color=chess.WHITE) == -10000.0


def test_live_stockfish_checkmate_delivery_scores_positive_10000():
    eng = open_engine()
    if eng is None:
        pytest.skip("Stockfish not available")
    try:
        # Scholar's mate: White delivers mate with h5f7.
        board_w = chess.Board()
        for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"]:
            board_w.push_uci(uci)
        assert eval_after_move(eng, board_w, "h5f7", depth=5) == 10000.0
        board_w.push_uci("h5f7")
        assert best_cp(eng, board_w, depth=5, for_color=chess.WHITE) == 10000.0
        assert best_cp(eng, board_w, depth=5, for_color=chess.BLACK) == -10000.0

        # Fool's mate: Black delivers mate with d8h4.
        board_b = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4"]:
            board_b.push_uci(uci)
        assert eval_after_move(eng, board_b, "d8h4", depth=5) == 10000.0
        board_b.push_uci("d8h4")
        assert best_cp(eng, board_b, depth=5, for_color=chess.BLACK) == 10000.0
        assert best_cp(eng, board_b, depth=5, for_color=chess.WHITE) == -10000.0
    finally:
        close_engine(eng)


def test_resilient_engine_restarts_on_sigfpe_and_reapplies_options(monkeypatch):
    import chess.engine

    from chessfm.rl.engine import ResilientEngine

    configured_calls: list[dict] = []
    quit_count = 0

    class _FlakyEngine:
        def __init__(self, should_crash: bool):
            self.should_crash = should_crash

        def configure(self, opts):
            configured_calls.append(dict(opts))

        def analyse(self, board, limit):
            if self.should_crash:
                raise chess.engine.EngineTerminatedError("engine process died (exit code: -8)")
            return {"score": _FakeScore(42)}

        def quit(self):
            nonlocal quit_count
            quit_count += 1

    fresh_engine = _FlakyEngine(should_crash=False)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _p: fresh_engine)

    resilient = ResilientEngine(
        "/usr/games/stockfish",
        _FlakyEngine(should_crash=True),
        threads=1,
        hash_mb=16,
        max_restarts=2,
    )
    resilient.configure({"Skill Level": 3})

    cp = best_cp(resilient, chess.Board(), depth=4)
    assert cp == 42.0
    assert resilient.restart_count == 1
    assert quit_count == 1
    # Restart re-applied initial Threads/Hash plus custom Skill Level option.
    assert configured_calls[-1] == {"Threads": 1, "Hash": 16, "Skill Level": 3}


def test_resilient_engine_stops_after_max_restarts(monkeypatch):
    import chess.engine

    from chessfm.rl.engine import ResilientEngine

    class _AlwaysCrashEngine:
        def configure(self, opts):
            pass

        def analyse(self, board, limit):
            raise chess.engine.EngineTerminatedError("SIGFPE")

        def quit(self):
            pass

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda _p: _AlwaysCrashEngine())

    resilient = ResilientEngine(
        "/usr/games/stockfish",
        _AlwaysCrashEngine(),
        max_restarts=2,
    )
    # best_cp converts EngineTerminatedError (subclass of EngineError) to None after bounded retries.
    assert best_cp(resilient, chess.Board(), depth=4) is None
    assert resilient.restart_count == 2


def test_resilient_engine_configure_does_not_poison_options_on_error():
    import chess.engine

    from chessfm.rl.engine import ResilientEngine

    class _RejectingEngine:
        def configure(self, opts):
            if "BadOption" in opts:
                raise chess.engine.EngineError("unknown option")

        def quit(self):
            pass

    resilient = ResilientEngine(
        "/usr/games/stockfish",
        _RejectingEngine(),
        threads=1,
        hash_mb=16,
    )
    with pytest.raises(chess.engine.EngineError):
        resilient.configure({"BadOption": 999})

    assert "BadOption" not in resilient._options
    assert resilient._options == {"Threads": 1, "Hash": 16}


def test_open_engine_with_explicit_stockfish_path_wraps_resilient(monkeypatch):
    import chess.engine

    from chessfm.rl.engine import ResilientEngine

    opened_paths: list[str] = []

    class _StubEngine:
        def configure(self, opts):
            pass

        def quit(self):
            pass

    def _fake_popen(path):
        opened_paths.append(path)
        return _StubEngine()

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", _fake_popen)
    engine = open_engine(threads=2, hash_mb=32, stockfish_path="/custom/sf")
    assert isinstance(engine, ResilientEngine)
    assert opened_paths == ["/custom/sf"]
    assert engine._options == {"Threads": 2, "Hash": 32}
    close_engine(engine)
