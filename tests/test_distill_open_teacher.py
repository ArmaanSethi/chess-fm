"""Unit tests for open-weight rejection-sampled coaching distillation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chessfm.data_generation.distill_open_teacher import (
    OpenAICompatibleTeacher,
    build_teacher_prompt,
    distill_rows,
    infer_task_type,
    main,
    strip_reasoning_blocks,
    verify_candidate,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class TestStripAndInfer:
    def test_strip_qwen_and_gemma4_reasoning_blocks(self):
        qwen = "<think>\nCheck center control.\n</think>\nDevelop a piece toward the center."
        assert strip_reasoning_blocks(qwen) == "Develop a piece toward the center."

        gemma = "<|channel>thought\nInspect e4 vs d4.\n<channel|>Control the center early."
        assert strip_reasoning_blocks(gemma) == "Control the center early."

    def test_infer_task_type_variants(self):
        assert infer_task_type({"task_type": "socratic_l1"}) == "socratic_l1"
        assert infer_task_type({"mode": "socratic_l2"}) == "socratic_l2"
        assert infer_task_type({"hint_level": 1}) == "socratic_l1"
        assert infer_task_type({"hint_level": 2}) == "socratic_l2"
        assert infer_task_type({"hint_level": 3}) == "socratic_l3"
        assert infer_task_type({}) == "postmortem"


class TestPromptAndVerification:
    @pytest.mark.parametrize(
        "task_type", ["postmortem", "socratic_l1", "socratic_l2", "socratic_l3"]
    )
    def test_build_teacher_prompt_all_tasks(self, task_type: str):
        prompt = build_teacher_prompt(
            {
                "fen": START_FEN,
                "best_move_uci": "e2e4",
                "best_move_san": "e4",
                "student_elo": 1100,
                "principal_variation": ["e4", "e5", "Nf3"],
            },
            task_type=task_type,
        )
        assert START_FEN in prompt
        assert f"Task ({task_type})" in prompt

    def test_build_teacher_prompt_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unsupported task_type"):
            build_teacher_prompt({"fen": START_FEN}, task_type="invalid")

    def test_verify_candidate_accepts_valid_coaching(self):
        decision = verify_candidate(
            START_FEN,
            "<think>Evaluate opening moves.</think>Playing e4 claims central space and opens lines for the bishop and queen.",
            task_type="postmortem",
            best_move_uci="e2e4",
        )
        assert decision.accepted is True
        assert decision.reason == "accepted"
        assert "<think>" not in decision.cleaned_text

    def test_verify_candidate_rejects_short_output(self):
        decision = verify_candidate(START_FEN, "Too short.", task_type="postmortem")
        assert decision.accepted is False
        assert decision.reason == "empty_or_short"

    def test_verify_candidate_rejects_illegal_chess_claim(self):
        # Queen cannot legally move to h5 from the starting position.
        decision = verify_candidate(
            START_FEN,
            "You should play Qh5 right away to attack the kingside.",
            task_type="postmortem",
            best_move_uci="e2e4",
        )
        assert decision.accepted is False
        assert decision.reason == "claim_violation"
        assert decision.violations

    def test_verify_candidate_rejects_socratic_spoiler(self):
        decision = verify_candidate(
            START_FEN,
            "Look closely at pushing the pawn to e4 to seize the center immediately.",
            task_type="socratic_l1",
            best_move_uci="e2e4",
        )
        assert decision.accepted is False
        assert decision.reason == "spoiler_leak"

    def test_verify_candidate_accepts_non_spoiler_socratic_l1(self):
        decision = verify_candidate(
            START_FEN,
            "Focus on contesting the central squares and opening lines for rapid piece development.",
            task_type="socratic_l1",
            best_move_uci="e2e4",
        )
        assert decision.accepted is True


class TestDistillRowsAndClient:
    def test_distill_rows_retries_on_violation_and_error(self):
        responses = iter(
            [
                RuntimeError("temporary timeout"),
                "Play Qh5 immediately to threaten checkmate.",  # illegal claim
                "Claiming central space opens diagonals for your minor pieces smoothly.",  # clean
            ]
        )

        def fake_teacher(_sys: str, _usr: str) -> str:
            nxt = next(responses)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        accepted, audit = distill_rows(
            [
                {"fen": START_FEN, "best_move_uci": "e2e4", "task_type": "socratic_l1"},
                {"no_fen_key": True},
            ],
            fake_teacher,
            teacher_model="google/gemma-4-26B-A4B-it",
            max_attempts=3,
        )
        assert len(accepted) == 1
        assert accepted[0]["attempts_used"] == 3
        assert accepted[0]["verified"] is True
        assert audit["input_rows"] == 1
        assert audit["accepted_rows"] == 1
        assert audit["rejection_counts"]["backend_error"] == 1
        assert audit["rejection_counts"]["claim_violation"] == 1

    def test_distill_rows_invalid_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            distill_rows([], lambda _s, _u: "", max_attempts=0)

    def test_openai_compatible_teacher_posts_expected_payload(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Solid central development."}}]
        }
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("chessfm.data_generation.distill_open_teacher.requests.post", mock_post)

        client = OpenAICompatibleTeacher(
            base_url="http://127.0.0.1:8001/v1/",
            model="google/gemma-4-26B-A4B-it",
            api_key="EMPTY",
        )
        out = client("sys", "usr")
        assert out == "Solid central development."
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "http://127.0.0.1:8001/v1/chat/completions"

    def test_cli_dry_run_and_write_output(self, tmp_path: Path, monkeypatch, capsys):
        in_path = tmp_path / "seed.jsonl"
        out_path = tmp_path / "out.jsonl"
        in_path.write_text(
            json.dumps({"fen": START_FEN, "best_move_uci": "e2e4", "task_type": "postmortem"})
            + "\n",
            encoding="utf-8",
        )

        rc_dry = main(["--input", str(in_path), "--dry-run", "--limit", "1"])
        assert rc_dry == 0
        assert "[dry-run]" in capsys.readouterr().out
        assert not out_path.exists()

        monkeypatch.setattr(
            OpenAICompatibleTeacher,
            "__call__",
            lambda self, _s, _u: (
                "Developing centrally gives your pieces active diagonals right away."
            ),
        )
        rc = main(["--input", str(in_path), "--output", str(out_path), "--limit", "1"])
        assert rc == 0
        assert out_path.is_file()
        audit_path = out_path.with_suffix(".audit.json")
        assert audit_path.is_file()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["accepted_rows"] == 1
