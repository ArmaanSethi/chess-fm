"""Unit and live Stockfish tests for chessfm.coach.multimodal."""

from __future__ import annotations

import chess
import chess.engine
import pytest

from chessfm.coach.multimodal import (
    CoachOrchestrator,
    build_gemma4_coach_messages,
    verify_chess_claims,
)
from chessfm.rl.engine import close_engine, open_engine

START_FEN = chess.STARTING_FEN
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
TRAP_FEN = "r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3"
CASTLE_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture(scope="module")
def stockfish_engine():
    engine = open_engine(threads=1, hash_mb=16)
    if engine is None:
        pytest.skip("Stockfish binary not available")
    yield engine
    close_engine(engine)


class TestBuildGemma4CoachMessages:
    def test_basic_message_structure_and_roles(self):
        context = {
            "student_elo": 1350,
            "hint_level": 2,
            "stockfish": {
                "best_move": "h5f7",
                "best_san": "Qxf7#",
                "eval_cp": 10000.0,
                "win_prob": 100.0,
                "pv": ["h5f7"],
                "top_moves": [{"uci": "h5f7", "san": "Qxf7#", "cp": 10000.0}],
            },
            "maia": {
                "predicted_human_move": "h5f7",
                "human_moves": [{"move": "h5f7", "prob": 0.65}],
                "traps": [
                    {"uci": "h5e5", "san": "Qxe5+", "cp_loss": 800.0, "reason": "hangs queen"}
                ],
            },
            "history": [
                {"role": "user", "content": "What should I look for?"},
                {"role": "assistant", "content": "Check the f7 square."},
            ],
        }
        messages = build_gemma4_coach_messages(
            fen=SCHOLARS_MATE_FEN,
            user_question="Is there a tactic here?",
            context=context,
            image_path="/tmp/board.png",
            audio_transcript="I think I can take on e5?",
        )

        # Roles must strictly be "user" / "model" (assistant normalized to model)
        assert [m["role"] for m in messages] == ["user", "model", "user"]
        last_content = messages[-1]["content"]
        assert last_content[0] == {"type": "image", "image": "/tmp/board.png"}
        text = last_content[1]["text"]
        assert SCHOLARS_MATE_FEN in text
        assert "Student Elo: 1350" in text
        assert "Qxf7#" in text
        assert "Qxe5+" in text
        assert "I think I can take on e5?" in text
        assert "Is there a tactic here?" in text

    def test_without_image_or_context(self):
        messages = build_gemma4_coach_messages(
            fen=START_FEN,
            user_question="Best opening move?",
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "text"
        assert START_FEN in messages[0]["content"][0]["text"]

    def test_invalid_fen_raises(self):
        with pytest.raises(ValueError):
            build_gemma4_coach_messages("not a valid fen", "Help?")


class TestVerifyChessClaims:
    def test_verified_legal_moves_and_pieces(self):
        text = (
            "With the white queen on h5 and bishop on c4, you can play Qxf7# (h5f7) for checkmate."
        )
        result = verify_chess_claims(
            SCHOLARS_MATE_FEN,
            text,
            analysis={"stockfish": {"best_move": "h5f7"}},
        )
        assert result["verified"] is True
        assert result["illegal_moves"] == []
        assert result["hallucinated_pieces"] == []
        assert "h5f7" in result["verified_moves"]
        assert result["best_move_mentioned"] is True

    def test_catches_illegal_uci_and_san_and_hallucinated_pieces(self):
        text = (
            "Play e2e5 or Qh8# since your rook on d4 and black queen on h5 "
            "and knight on a1 control the board."
        )
        result = verify_chess_claims(SCHOLARS_MATE_FEN, text)
        assert result["verified"] is False
        assert "e2e5" in result["illegal_moves"]
        assert "Qh8#" in result["illegal_moves"]
        assert len(result["hallucinated_pieces"]) == 3
        assert len(result["issues"]) >= 5


class TestCoachOrchestratorCPUFallback:
    def test_end_to_end_cpu_fallback_without_engine_or_llm(self):
        coach = CoachOrchestrator()
        result = coach.coach(
            SCHOLARS_MATE_FEN,
            "What is the best move?",
            student_elo=1100,
        )
        assert result["mode"] == "template_fallback"
        assert result["analysis"]["stockfish"]["source"] == "heuristic"
        assert result["analysis"]["stockfish"]["best_move"] == "h5f7"
        assert result["verification"]["verified"] is True
        assert "h5f7" in result["response"]
        assert result["messages"][-1]["role"] == "user"

    def test_three_level_socratic_hints(self):
        coach = CoachOrchestrator()
        analysis = coach.analyze_position(SCHOLARS_MATE_FEN, student_elo=1000)

        h1 = coach.generate_socratic_hint(SCHOLARS_MATE_FEN, level=1, analysis=analysis)
        h2 = coach.generate_socratic_hint(SCHOLARS_MATE_FEN, level=2, analysis=analysis)
        h3 = coach.generate_socratic_hint(SCHOLARS_MATE_FEN, level=3, analysis=analysis)

        assert h1["level"] == 1 and h1["reveals_move"] is False
        assert "h5f7" not in h1["hint"]
        assert h2["level"] == 2 and h2["reveals_move"] is False
        assert "queen" in h2["hint"].lower()
        assert "h5f7" not in h2["hint"]
        assert h3["level"] == 3 and h3["reveals_move"] is True
        assert "h5f7" in h3["hint"]

        with pytest.raises(ValueError, match="hint_level"):
            coach.generate_socratic_hint(SCHOLARS_MATE_FEN, level=4, analysis=analysis)

    def test_coach_with_hint_level_and_audio_transcript(self):
        coach = CoachOrchestrator()
        result = coach.coach(
            TRAP_FEN,
            "Should I take on e5?",
            student_elo=1000,
            hint_level=2,
            audio_transcript="Can my queen capture on e5 safely?",
        )
        assert result["hint"]["level"] == 2
        assert "Hint (Level 2)" in result["response"]
        assert "Can my queen capture on e5 safely?" in result["response"]
        # Trap detector should flag Qxe5+ (h5e5) as a blunder losing the queen
        trap_ucis = [t["uci"] for t in result["analysis"]["maia"]["traps"]]
        assert "h5e5" in trap_ucis

    def test_llm_fn_verified_and_hallucination_repair(self):
        captured_messages: list[list[dict]] = []

        def good_llm(messages: list[dict]) -> str:
            captured_messages.append(messages)
            return "Great spotting the tactic! Play Qxf7# (h5f7) with your white queen on h5."

        coach_good = CoachOrchestrator(llm_fn=good_llm)
        res_good = coach_good.coach(
            SCHOLARS_MATE_FEN,
            "How do I finish?",
            image_path="/tmp/pos.png",
        )
        assert res_good["mode"] == "llm"
        assert res_good["verification"]["verified"] is True
        assert "[Coach Verifier Note:" not in res_good["response"]
        assert captured_messages[0][-1]["content"][0]["type"] == "image"

        def hallucinating_llm(_messages: list[dict]) -> str:
            return "Move your knight on a1 via e2e5 to win immediately."

        coach_bad = CoachOrchestrator(llm_fn=hallucinating_llm)
        res_bad = coach_bad.coach(SCHOLARS_MATE_FEN, "How do I finish?")
        assert res_bad["mode"] == "llm"
        assert res_bad["verification"]["verified"] is False
        assert "[Coach Verifier Note:" in res_bad["response"]
        assert "h5f7" in res_bad["response"]

    def test_custom_hooks_and_context_manager(self):
        closed = {"flag": False}

        class DummyEngine:
            def quit(self):
                closed["flag"] = True

        with CoachOrchestrator(
            engine=DummyEngine(),
            owns_engine=True,
            analyzer_fn=lambda fen, depth: {
                "source": "custom",
                "best_move": "e1g1",
                "best_san": "O-O",
                "eval_cp": 45.0,
                "win_prob": 54.0,
                "pv": ["e1g1"],
                "top_moves": [],
            },
            maia_fn=lambda fen, elo: {
                "student_elo": elo,
                "predicted_human_move": "e1g1",
                "human_moves": [{"move": "e1g1", "prob": 0.5}],
                "traps": [],
            },
            hint_fn=lambda fen, lvl, analysis: {
                "level": lvl,
                "hint": "Custom hint: secure your king.",
                "reveals_move": False,
            },
            verifier_fn=lambda fen, text, analysis: {
                "verified": True,
                "illegal_moves": [],
                "hallucinated_pieces": [],
                "verified_moves": [],
                "best_move_mentioned": False,
                "issues": [],
            },
        ) as coach:
            res = coach.coach(CASTLE_FEN, "What now?", hint_level=1)
            assert res["analysis"]["stockfish"]["source"] == "custom"
            assert "Custom hint" in res["response"]

        assert closed["flag"] is True

    def test_review_game_delegation(self):
        coach = CoachOrchestrator()
        review = coach.review_game("1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0")
        assert review["total_plies"] == 7
        assert review["engine_used"] == "heuristic"


class TestCoachOrchestratorLiveStockfish:
    def test_live_stockfish_analysis_and_coaching(self, stockfish_engine):
        coach = CoachOrchestrator(engine=stockfish_engine, owns_engine=False)
        res = coach.coach(
            SCHOLARS_MATE_FEN,
            "Find the winning move.",
            student_elo=1200,
            hint_level=3,
            depth=6,
        )
        assert res["analysis"]["stockfish"]["source"] == "stockfish"
        assert res["analysis"]["stockfish"]["best_move"] == "h5f7"
        assert res["analysis"]["stockfish"]["eval_cp"] >= 9000.0
        assert res["verification"]["verified"] is True
        assert "Qxf7#" in res["response"]

    def test_live_stockfish_broken_engine_fallback(self):
        class BrokenEngine:
            def analyse(self, *args, **kwargs):
                raise chess.engine.EngineError("pipe broken")

        coach = CoachOrchestrator(engine=BrokenEngine())
        analysis = coach.analyze_position(SCHOLARS_MATE_FEN, depth=6)
        assert analysis["stockfish"]["source"] == "heuristic"
        assert analysis["stockfish"]["best_move"] == "h5f7"
