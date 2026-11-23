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
        assert "timestamp_utc" in audit
        assert "git_sha" in audit
        assert "wall_clock_seconds" in audit
        assert "per_task_type" in audit


class TestLayerBSchemaConcurrencyAndProvenance:
    def test_why_not_task_type_and_prompt(self):
        assert infer_task_type({"mode": "why_not"}) == "why_not"
        assert infer_task_type({"task_type": "why_not"}) == "why_not"
        prompt = build_teacher_prompt(
            {
                "fen": START_FEN,
                "best_move": "e2e4",
                "student_move": "a2a3",
                "severity": "inaccuracy",
                "maia_prob": 0.0,
                "cp_loss": 45,
                "phase": "opening",
                "best_motifs": ["center_control"],
                "student_motifs": ["passive_wing_move"],
            },
            task_type="why_not",
        )
        assert "Task (why_not):" in prompt
        facts_json = prompt.split("Engine Facts (JSON):\n", 1)[1].split("\n\nTask", 1)[0]
        facts = json.loads(facts_json)
        assert facts["best_move_san"] == "e4"
        assert facts["played_move"] == "a3"
        assert facts["classification"] == "inaccuracy"
        assert facts["maia_human_prob"] == 0.0
        assert facts["phase"] == "opening"
        assert facts["best_motifs"] == ["center_control"]
        assert facts["student_motifs"] == ["passive_wing_move"]

    def test_distill_rows_skips_engine_best_line_and_emits_messages_chatml(self):
        rows = [
            {
                "fen": START_FEN,
                "source": "engine_best_line",
                "move": "e2e4",
                "stockfish_best": "e2e4",
            },
            {
                "fen": START_FEN,
                "mode": "socratic_l1",
                "best_move": "e2e4",
                "student_move": "a2a3",
                "severity": "inaccuracy",
                "maia_prob": 0.0,
            },
            {
                "fen": START_FEN,
                "mode": "why_not",
                "best_move": "e2e4",
                "student_move": "a2a3",
                "severity": "inaccuracy",
            },
        ]

        def teacher(_sys: str, user_prompt: str) -> str:
            if "Task (socratic_l1)" in user_prompt:
                return "Think about fighting for central space and activating your pieces early."
            return (
                "Pushing a3 is too slow, whereas e4 immediately claims the center and opens lines."
            )

        accepted, audit = distill_rows(rows, teacher, workers=1)
        assert audit["skipped_non_coaching_rows"] == 1
        assert audit["input_rows"] == 2
        assert audit["accepted_rows"] == 2
        assert audit["per_task_type"] == {"socratic_l1": 1, "why_not": 1}
        assert audit["workers"] == 1
        assert isinstance(audit["wall_clock_seconds"], float)
        assert audit["timestamp_utc"].endswith("Z")
        assert len(audit["git_sha"]) == 40 or audit["git_sha"] == "unknown"

        l1_row, wn_row = accepted
        assert l1_row["level"] == 1
        assert l1_row["best_move"] == "e2e4"
        assert l1_row["act_eval_passed"] is True
        assert l1_row["spoiler_free"] is True
        assert [m["role"] for m in l1_row["messages"]] == ["system", "user", "assistant"]
        assert l1_row["prompt"].startswith("<|im_start|>system\n")
        assert l1_row["prompt"].endswith("<|im_start|>assistant\n")
        assert l1_row["completion"].endswith("<|im_end|>")
        assert "level" not in wn_row
        assert wn_row["task_type"] == "why_not"

    def test_distill_rows_concurrent_workers_preserve_deterministic_order(self):
        import time

        rows = [
            {
                "fen": START_FEN,
                "mode": mode,
                "best_move": "e2e4",
                "student_elo": 1000 + idx * 100,
            }
            for idx, mode in enumerate(
                ["postmortem", "socratic_l1", "socratic_l2", "socratic_l3", "why_not"]
            )
        ]

        def slow_teacher(_sys: str, user_prompt: str) -> str:
            # Deliberately reverse sleep durations so later items finish faster.
            if "1000" in user_prompt:
                time.sleep(0.03)
            elif "1100" in user_prompt:
                time.sleep(0.02)
            return "Focus on controlling the center and developing your minor pieces smoothly."

        accepted, audit = distill_rows(rows, slow_teacher, workers=4)
        assert audit["workers"] == 4
        assert audit["accepted_rows"] == 5
        assert [r["student_elo"] for r in accepted] == [1000, 1100, 1200, 1300, 1400]
        assert [r["task_type"] for r in accepted] == [
            "postmortem",
            "socratic_l1",
            "socratic_l2",
            "socratic_l3",
            "why_not",
        ]
        assert accepted[1]["level"] == 1
        assert accepted[2]["level"] == 2
        assert accepted[3]["level"] == 3

    def test_openai_compatible_teacher_session_pooling_and_none_content(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": None}}]}
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        monkeypatch.setattr(
            "chessfm.data_generation.distill_open_teacher.requests.Session",
            lambda: mock_session,
        )

        client = OpenAICompatibleTeacher(base_url="http://127.0.0.1:8001/v1")
        assert client("sys", "usr") == ""
        assert client("sys", "usr2") == ""
        assert mock_session.post.call_count == 2


class TestTrainSftMessagesFormatting:
    def test_format_sft_example_messages_priority_over_include_ascii(self):
        from chessfm.training.train_sft import _is_valid_sft_sample, format_sft_example

        captured_kwargs: list[dict] = []

        class FakeTokenizer:
            chat_template = "dummy"
            eos_token = "<|im_end|>"

            def apply_chat_template(
                self, messages, *, tokenize=False, add_generation_prompt=False, enable_thinking=True
            ):
                captured_kwargs.append(
                    {
                        "n": len(messages),
                        "add_generation_prompt": add_generation_prompt,
                        "enable_thinking": enable_thinking,
                    }
                )
                out = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
                if add_generation_prompt:
                    out += "<assistant>"
                return out

        sample = {
            "fen": START_FEN,
            "messages": [
                {"role": "system", "content": "Coach system"},
                {"role": "user", "content": "Explain why e4 is strong."},
                {"role": "assistant", "content": "e4 claims central space."},
            ],
        }
        assert _is_valid_sft_sample(sample) is True

        # Even with include_ascii=True, messages MUST take priority over P1 FEN prompt.
        row_text = format_sft_example(
            sample, FakeTokenizer(), include_ascii=True, use_prompt_completion=False
        )
        assert "text" in row_text
        assert "Explain why e4 is strong." in row_text["text"]
        assert captured_kwargs[-1]["enable_thinking"] is False

        # Dual-TRL prompt/completion split mode for TRL >= 1.18.
        row_pc = format_sft_example(
            sample, FakeTokenizer(), include_ascii=True, use_prompt_completion=True
        )
        assert set(row_pc.keys()) == {"prompt", "completion"}
        assert "Explain why e4 is strong." in row_pc["prompt"]
        assert "e4 claims central space." in row_pc["completion"]
        assert all(call["enable_thinking"] is False for call in captured_kwargs)

    def test_format_sft_example_p1_backwards_compatibility(self):
        from chessfm.training.train_sft import _is_valid_sft_sample, format_sft_example

        sample = {
            "fen": START_FEN,
            "prompt": "<|im_start|>user\nFEN: start<|im_end|>\n<|im_start|>assistant\n",
            "completion": "<think>Analyze</think>\ne2e4<|im_end|>",
        }
        assert _is_valid_sft_sample(sample) is True
        assert _is_valid_sft_sample({**sample, "is_reasonable": False}) is False

        out = format_sft_example(sample, None, include_ascii=False, use_prompt_completion=False)
        assert out == {"text": sample["prompt"] + sample["completion"]}

        out_ascii = format_sft_example(
            sample, None, include_ascii=True, use_prompt_completion=False
        )
        assert "a b c d e f g h" in out_ascii["text"]
        assert "<think>Analyze</think>\ne2e4<|im_end|>" in out_ascii["text"]
