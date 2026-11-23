"""Unit tests for cloud runners (Modal, Kaggle) and GGUF / Ollama export scripts."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_gguf = _load_script_module("export_gguf", "export_gguf.py")
kaggle_train = _load_script_module("kaggle_train", "kaggle_train.py")


# ---------------------------------------------------------------------------
# export_gguf tests
# ---------------------------------------------------------------------------


class TestExportGGUF:
    def test_build_ollama_modelfile_gemma4(self):
        content = export_gguf.build_ollama_modelfile(
            "./coach-q4_k_m.gguf", model_family="gemma4", temperature=0.2
        )
        assert "FROM ./coach-q4_k_m.gguf" in content
        assert "SYSTEM" in content
        assert "<think>...</think>" in content
        assert "PARAMETER temperature 0.2" in content
        assert 'PARAMETER stop "<turn|>"' in content
        assert 'PARAMETER stop "<|turn>"' in content
        assert 'PARAMETER stop "<eos>"' in content

    def test_build_ollama_modelfile_qwen3(self):
        content = export_gguf.build_ollama_modelfile(
            "./qwen3-q4_k_m.gguf", model_family="qwen3", temperature=0.15
        )
        assert "FROM ./qwen3-q4_k_m.gguf" in content
        assert "PARAMETER temperature 0.15" in content
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|im_start|>"' in content
        assert 'PARAMETER stop "<|endoftext|>"' in content

    def test_build_ollama_modelfile_qwen25(self):
        content = export_gguf.build_ollama_modelfile(
            "./qwen25-q4_k_m.gguf", model_family="qwen25", temperature=0.0
        )
        assert "FROM ./qwen25-q4_k_m.gguf" in content
        assert "PARAMETER temperature 0.0" in content
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|im_start|>"' in content

    def test_build_ollama_modelfile_invalid_family_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_family"):
            export_gguf.build_ollama_modelfile("./m.gguf", model_family="llama3")

    def test_build_ollama_modelfile_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            export_gguf.build_ollama_modelfile("./m.gguf", model_family="gemma4", temperature=-0.1)

    def test_cli_dry_run_does_not_write_files(self, tmp_path: Path):
        out_dir = tmp_path / "export_out"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "export_gguf.py"),
                "--model",
                str(tmp_path / "final_merged"),
                "--output",
                str(out_dir),
                "--quant",
                "q4_k_m",
                "--family",
                "gemma4",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "[dry-run]" in result.stdout
        assert 'PARAMETER stop "<turn|>"' in result.stdout
        assert not out_dir.exists()

    def test_cli_writes_modelfile_without_llama_cpp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        monkeypatch.setattr(export_gguf, "find_llama_cpp_converter", lambda _dir=None: None)
        out_dir = tmp_path / "gguf_out"
        rc = export_gguf.main(
            [
                "--model",
                str(tmp_path / "sft_merged"),
                "--output",
                str(out_dir),
                "--quant",
                "q4_k_m",
                "--family",
                "qwen3",
            ]
        )
        assert rc == 0
        modelfile = out_dir / "Modelfile"
        assert modelfile.is_file()
        text = modelfile.read_text(encoding="utf-8")
        assert "FROM ./sft_merged-q4_k_m.gguf" in text
        assert 'PARAMETER stop "<|im_end|>"' in text
        captured = capsys.readouterr().out
        assert "convert_hf_to_gguf.py" in captured
        assert "ollama create chessfm-qwen3" in captured


# ---------------------------------------------------------------------------
# kaggle_train tests
# ---------------------------------------------------------------------------


class TestKaggleTrain:
    def test_build_sft_cmd_defaults_and_include_ascii(self):
        cmd = kaggle_train.build_sft_cmd(
            model="unsloth/gemma-4-E4B-it",
            data="data_generation/sft_train.jsonl",
            output="/kaggle/working/checkpoints/sft",
            epochs=2,
            include_ascii=True,
            merge_and_save=True,
            python_bin="python",
        )
        assert cmd[:3] == ["python", "-m", "chessfm.training.train_sft"]
        assert "--model" in cmd and "unsloth/gemma-4-E4B-it" in cmd
        assert "--output" in cmd and "/kaggle/working/checkpoints/sft" in cmd
        assert "--merge-and-save" in cmd
        assert "--include-ascii" in cmd

    def test_build_sft_cmd_without_ascii_or_merge(self):
        cmd = kaggle_train.build_sft_cmd(
            include_ascii=False,
            merge_and_save=False,
            python_bin="python",
        )
        assert "--no-merge-and-save" in cmd
        assert "--include-ascii" not in cmd

    def test_build_grpo_cmd_resume_and_include_ascii(self):
        cmd = kaggle_train.build_grpo_cmd(
            model="/kaggle/working/checkpoints/sft/final_merged",
            stage=1,
            steps=300,
            output="/kaggle/working/checkpoints/grpo_stage1",
            resume=True,
            include_ascii=True,
            python_bin="python",
        )
        assert cmd[:3] == ["python", "-m", "chessfm.rl.train_grpo"]
        assert "--stage" in cmd and cmd[cmd.index("--stage") + 1] == "1"
        assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "300"
        assert "--resume" in cmd
        assert "--include-ascii" in cmd

    def test_build_grpo_cmd_no_resume(self):
        cmd = kaggle_train.build_grpo_cmd(resume=False, include_ascii=False, python_bin="python")
        assert "--resume" not in cmd
        assert "--include-ascii" not in cmd

    def test_build_eval_cmd_stockfish_and_random(self):
        sf_cmd = kaggle_train.build_eval_cmd(
            model="/kaggle/working/checkpoints/stage0_final",
            games=10,
            stockfish_level=2,
            random_opponent=False,
            python_bin="python",
        )
        assert sf_cmd[:3] == ["python", "-m", "chessfm.rl.evaluate"]
        assert "--games" in sf_cmd and sf_cmd[sf_cmd.index("--games") + 1] == "10"
        assert (
            "--stockfish-level" in sf_cmd and sf_cmd[sf_cmd.index("--stockfish-level") + 1] == "2"
        )
        assert "--random-opponent" not in sf_cmd

        rand_cmd = kaggle_train.build_eval_cmd(random_opponent=True, python_bin="python")
        assert "--random-opponent" in rand_cmd

    def test_ensure_usr_games_on_path(self):
        env = kaggle_train.ensure_usr_games_on_path({"PATH": "/usr/bin:/bin"})
        assert env["PATH"].split(":")[0] == "/usr/games"

        # Idempotent when already present
        env2 = kaggle_train.ensure_usr_games_on_path({"PATH": "/usr/local/bin:/usr/games:/usr/bin"})
        assert env2["PATH"].count("/usr/games") == 1

    def test_require_stockfish_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(kaggle_train, "find_stockfish", lambda _env=None: None)
        with pytest.raises(RuntimeError, match="Stockfish binary not found"):
            kaggle_train.require_stockfish()

    def test_cli_pipeline_dry_run(self, capsys):
        kaggle_train.main(
            [
                "--task",
                "pipeline",
                "--model",
                "Qwen/Qwen3-4B",
                "--stage",
                "0",
                "--steps",
                "50",
                "--output",
                "/kaggle/working/checkpoints",
                "--resume",
                "--include-ascii",
                "--dry-run",
            ]
        )
        out = capsys.readouterr().out
        assert "chessfm.training.train_sft" in out
        assert "chessfm.rl.train_grpo" in out
        assert "final_merged" in out
        assert "--resume" in out
        assert "--include-ascii" in out
        assert "chessfm.rl.evaluate" in out


# ---------------------------------------------------------------------------
# modal_train AST & structure tests
# ---------------------------------------------------------------------------


class TestModalTrainAST:
    def _parse_modal_source(self) -> tuple[str, ast.Module]:
        source = (SCRIPTS_DIR / "modal_train.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="scripts/modal_train.py")
        return source, tree

    def test_valid_python_syntax_and_expected_functions(self):
        _, tree = self._parse_modal_source()
        fn_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for expected in (
            "requires_modern_stack",
            "resolve_track_model",
            "run_sft",
            "run_sft_modern",
            "run_grpo",
            "run_grpo_modern",
            "run_eval",
            "run_eval_modern",
            "run_static_eval",
            "run_static_eval_modern",
            "run_upload_hf",
            "run_distill_teacher",
            "run_coach_eval",
            "run_export_gguf",
            "main",
        ):
            assert expected in fn_names, f"Missing function {expected} in scripts/modal_train.py"

    def test_expected_tasks_and_tracks_present(self):
        source, tree = self._parse_modal_source()
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for task in (
            "smoke-pipeline",
            "sft",
            "grpo",
            "eval",
            "static-eval",
            "coach-eval",
            "distill-teacher",
            "export-gguf",
            "upload-hf",
        ):
            assert task in string_literals, f"Task {task!r} not found in scripts/modal_train.py"

        for model_id in (
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3.5-4B",
            "unsloth/gemma-4-E4B-it",
            "google/gemma-4-E4B-it",
            "Qwen/Qwen3.5-0.8B",
            "google/gemma-4-E2B-it",
            "google/gemma-4-26B-A4B-it",
            "Qwen/Qwen3.6-35B-A3B",
        ):
            assert model_id in string_literals, f"Track model {model_id!r} missing"

        assert "PYTORCH_CUDA_ALLOC_CONF" in string_literals
        assert "expandable_segments:True" in string_literals
        assert "--include-ascii" in string_literals
        assert "--merge-and-save" in string_literals
        assert "--decoding-mode" in string_literals
        assert "--enable-thinking" in string_literals
        assert "--maia-elo" in string_literals
        assert "--positions" in string_literals
        assert "--max-positions" in string_literals
        assert "final_merged" in source
        assert "_final_merged" in source
        assert "scripts/export_gguf.py" in string_literals
        assert "scripts/upload_hf.py" in string_literals
        assert "chessfm.eval.suite" in string_literals
        assert "chessfm.data_generation.distill_open_teacher" in string_literals

    def test_grpo_functions_accept_positions_and_merge_and_save(self):
        _, tree = self._parse_modal_source()
        fn_args = {
            node.name: {arg.arg for arg in node.args.args}
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for fn_name in ("_exec_grpo", "run_grpo", "run_grpo_a100", "run_grpo_modern"):
            assert "positions" in fn_args[fn_name]
            assert "merge_and_save" in fn_args[fn_name]
            assert "reward_mode" in fn_args[fn_name]
            assert "filter_zero_variance" in fn_args[fn_name]
        for fn_name in ("_exec_sft", "run_sft", "run_sft_a100", "run_sft_h100", "run_sft_modern"):
            assert "lora_r" in fn_args[fn_name]
            assert "full_ft" in fn_args[fn_name]
            assert "packing" in fn_args[fn_name]
            assert "include_piece_list" in fn_args[fn_name]
        for flag in (
            "gpu",
            "lora_r",
            "full_ft",
            "packing",
            "include_piece_list",
            "include_legal_moves",
            "reward_mode",
            "epsilon",
            "epsilon_high",
            "beta",
            "loss_type",
            "filter_zero_variance",
        ):
            assert flag in fn_args["main"], f"Missing CLI param {flag!r} in modal_train.main"
        assert "positions" in fn_args["main"]
        assert "max_positions" in fn_args["main"]


class TestModalTrainRouting:
    @staticmethod
    def _load_modal_train():
        pytest.importorskip("modal")
        return _load_script_module("modal_train_test", "modal_train.py")

    def test_h100_rejects_non_sft_tasks(self):
        mt = self._load_modal_train()
        for bad_task in ("grpo", "smoke-pipeline", "eval", "static-eval"):
            with pytest.raises(ValueError, match="h100"):
                mt.main(task=bad_task, gpu="h100", track="v1")

    def test_h100_rejects_modern_tracks(self):
        mt = self._load_modal_train()
        for modern_track in ("v1.5", "v2", "coach", "student"):
            with pytest.raises(ValueError, match="modern-stack"):
                mt.main(task="sft", gpu="h100", track=modern_track)

    def test_h100_routes_legacy_sft(self, monkeypatch: pytest.MonkeyPatch):
        mt = self._load_modal_train()
        calls: list[dict] = []

        class _FakeRemote:
            def remote(self, **kwargs):
                calls.append(kwargs)
                return "/vol/checkpoints/sft_h100/final"

        monkeypatch.setattr(mt, "run_sft_h100", _FakeRemote())
        mt.main(task="sft", gpu="h100", track="v1", full_ft=True)
        assert len(calls) == 1
        assert calls[0]["full_ft"] is True
        assert calls[0]["merge_and_save"] is False

    def test_a100_rejects_modern_tracks_and_unsupported_tasks(self):
        mt = self._load_modal_train()
        for bad_task in ("eval", "static-eval", "coach-eval", "export-gguf"):
            with pytest.raises(ValueError, match="a100"):
                mt.main(task=bad_task, gpu="a100", track="v1")
        for task in ("sft", "grpo", "smoke-pipeline"):
            with pytest.raises(ValueError, match="modern-stack"):
                mt.main(task=task, gpu="a100", track="v1.5")

    def test_smoke_pipeline_passes_full_ft_false_and_preserves_returned_ckpt(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mt = self._load_modal_train()
        sft_calls: list[dict] = []
        grpo_calls: list[dict] = []
        eval_calls: list[dict] = []

        class _FakeSft:
            def remote(self, **kwargs):
                sft_calls.append(kwargs)
                return "/vol/checkpoints/sft_smoke/final"

        class _FakeGrpo:
            def remote(self, **kwargs):
                grpo_calls.append(kwargs)
                return "/vol/checkpoints/smoke_grpo_stage0/stage0_final_merged"

        class _FakeEval:
            def remote(self, **kwargs):
                eval_calls.append(kwargs)

        monkeypatch.setattr(mt, "run_sft_a100", _FakeSft())
        monkeypatch.setattr(mt, "run_grpo_a100", _FakeGrpo())
        monkeypatch.setattr(mt, "run_eval", _FakeEval())

        mt.main(task="smoke-pipeline", gpu="a100", track="v1")
        assert len(sft_calls) == 1
        assert sft_calls[0]["full_ft"] is False
        assert sft_calls[0]["merge_and_save"] is True
        assert len(grpo_calls) == 1
        assert grpo_calls[0]["model"] == "/vol/checkpoints/sft_smoke/final"
        assert len(eval_calls) == 1

    def test_exec_static_eval_fast_fails_when_suite_lacks_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mt = self._load_modal_train()
        ran: list[list[str]] = []
        monkeypatch.setattr(mt, "_ensure_installed", lambda: None)
        monkeypatch.setattr(mt, "_run", lambda cmd: ran.append(cmd))
        monkeypatch.setattr(mt, "_suite_supports_flag", lambda _flag: False)

        with pytest.raises(ValueError, match="--include-piece-list"):
            mt._exec_static_eval(
                model="Qwen/Qwen3-4B",
                benchmark="data_generation/eval_benchmark.jsonl",
                decoding_mode="strict",
                include_ascii=False,
                enable_thinking=True,
                max_positions=10,
                skip_probes=True,
                output_name="out.json",
                include_piece_list=True,
            )
        assert ran == []

        with pytest.raises(ValueError, match="--include-legal-moves"):
            mt._exec_static_eval(
                model="Qwen/Qwen3-4B",
                benchmark="data_generation/eval_benchmark.jsonl",
                decoding_mode="strict",
                include_ascii=False,
                enable_thinking=True,
                max_positions=10,
                skip_probes=True,
                output_name="out.json",
                include_legal_moves=True,
            )
        assert ran == []

    def test_exec_static_eval_forwards_flags_when_suite_supports_them(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mt = self._load_modal_train()
        ran: list[list[str]] = []
        committed: list[bool] = []
        monkeypatch.setattr(mt, "_ensure_installed", lambda: None)
        monkeypatch.setattr(mt, "_run", lambda cmd: ran.append(cmd))
        monkeypatch.setattr(mt.volume, "commit", lambda: committed.append(True))
        monkeypatch.setattr(mt, "_suite_supports_flag", lambda _flag: True)

        out = mt._exec_static_eval(
            model="Qwen/Qwen3-4B",
            benchmark="data_generation/eval_benchmark.jsonl",
            decoding_mode="strict",
            include_ascii=True,
            enable_thinking=True,
            max_positions=10,
            skip_probes=True,
            output_name="out.json",
            include_piece_list=True,
            include_legal_moves=True,
        )
        assert out == "/vol/results/out.json"
        assert committed == [True]
        assert len(ran) == 1
        assert "--include-piece-list" in ran[0]
        assert "--include-legal-moves" in ran[0]

    def test_exec_static_eval_forwards_phase2_flags_and_omits_empty_benchmark(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mt = self._load_modal_train()
        ran: list[list[str]] = []
        committed: list[bool] = []
        monkeypatch.setattr(mt, "_ensure_installed", lambda: None)
        monkeypatch.setattr(mt, "_run", lambda cmd: ran.append(cmd))
        monkeypatch.setattr(mt.volume, "commit", lambda: committed.append(True))
        monkeypatch.setattr(mt, "_suite_supports_flag", lambda _flag: True)

        out = mt._exec_static_eval(
            model="/vol/checkpoints/sft_e1a/final_merged",
            benchmark="",
            decoding_mode="strict",
            include_ascii=False,
            enable_thinking=True,
            max_positions=0,
            skip_probes=True,
            output_name="/vol/results/phase2/e1a_puzzles_chessqa.json",
            include_piece_list=True,
            include_legal_moves=False,
            batch_size=16,
            puzzles="/vol/data/phase2/test_puzzles_900.jsonl",
            chessqa="/vol/data/phase2/test_chessqa_1000.jsonl",
            english_forgetting="/vol/data/phase2/test_english_100.jsonl",
            depth=12,
            coaching_jsonl="/vol/data/phase2/test_coach_300.jsonl",
        )
        assert out == "/vol/results/phase2/e1a_puzzles_chessqa.json"
        assert committed == [True]
        assert len(ran) == 1
        cmd = ran[0]
        assert "--benchmark" not in cmd
        assert cmd[cmd.index("--output") + 1] == "/vol/results/phase2/e1a_puzzles_chessqa.json"
        assert cmd[cmd.index("--batch-size") + 1] == "16"
        assert cmd[cmd.index("--puzzles") + 1] == "/vol/data/phase2/test_puzzles_900.jsonl"
        assert cmd[cmd.index("--chessqa") + 1] == "/vol/data/phase2/test_chessqa_1000.jsonl"
        assert (
            cmd[cmd.index("--english-forgetting") + 1] == "/vol/data/phase2/test_english_100.jsonl"
        )
        assert cmd[cmd.index("--depth") + 1] == "12"
        assert cmd[cmd.index("--coaching-jsonl") + 1] == "/vol/data/phase2/test_coach_300.jsonl"
        assert "--include-piece-list" in cmd

    def test_exec_eval_forwards_phase2_flags(self, monkeypatch: pytest.MonkeyPatch):
        mt = self._load_modal_train()
        ran: list[list[str]] = []
        committed: list[bool] = []
        monkeypatch.setattr(mt, "_ensure_installed", lambda: None)
        monkeypatch.setattr(mt, "_run", lambda cmd: ran.append(cmd))
        monkeypatch.setattr(mt.volume, "commit", lambda: committed.append(True))

        mt._exec_eval(
            model="/vol/checkpoints/sft_e1a/final_merged",
            games=40,
            stockfish_level=1,
            random_opponent=False,
            decoding_mode="constrained",
            include_piece_list=True,
            batch_size=16,
            stockfish_elo=1320,
            manifest_output="/vol/results/phase2/e1a_sf1320.json",
            seed=123,
        )
        assert committed == [True]
        assert len(ran) == 2  # evaluate cmd + cp results
        cmd = ran[0]
        assert cmd[cmd.index("--batch-size") + 1] == "16"
        assert cmd[cmd.index("--stockfish-elo") + 1] == "1320"
        assert cmd[cmd.index("--manifest-output") + 1] == "/vol/results/phase2/e1a_sf1320.json"
        assert cmd[cmd.index("--seed") + 1] == "123"
        assert "--include-piece-list" in cmd

    def test_main_forwards_phase2_eval_and_static_eval_flags(self, monkeypatch: pytest.MonkeyPatch):
        mt = self._load_modal_train()
        static_calls: list[dict] = []
        eval_calls: list[dict] = []

        class _FakeStatic:
            def remote(self, **kwargs):
                static_calls.append(kwargs)
                return "/vol/results/phase2/out.json"

        class _FakeEval:
            def remote(self, **kwargs):
                eval_calls.append(kwargs)

        monkeypatch.setattr(mt, "run_static_eval", _FakeStatic())
        monkeypatch.setattr(mt, "run_eval", _FakeEval())

        mt.main(
            task="static-eval",
            track="v1",
            benchmark="",
            batch_size=16,
            puzzles="/vol/data/phase2/test_puzzles_900.jsonl",
            chessqa="/vol/data/phase2/test_chessqa_1000.jsonl",
            english_forgetting="builtin",
            depth=10,
            coaching_jsonl="/vol/data/phase2/test_coach_300.jsonl",
            output="/vol/results/phase2/out.json",
            include_piece_list=True,
        )
        assert len(static_calls) == 1
        sc = static_calls[0]
        assert sc["benchmark"] == ""
        assert sc["batch_size"] == 16
        assert sc["puzzles"] == "/vol/data/phase2/test_puzzles_900.jsonl"
        assert sc["chessqa"] == "/vol/data/phase2/test_chessqa_1000.jsonl"
        assert sc["english_forgetting"] == "builtin"
        assert sc["depth"] == 10
        assert sc["coaching_jsonl"] == "/vol/data/phase2/test_coach_300.jsonl"
        assert sc["output_name"] == "/vol/results/phase2/out.json"

        mt.main(
            task="eval",
            track="v1",
            games=40,
            batch_size=16,
            stockfish_elo=1320,
            manifest_output="/vol/results/phase2/match.json",
            seed=7,
            include_piece_list=True,
        )
        assert len(eval_calls) == 1
        ec = eval_calls[0]
        assert ec["batch_size"] == 16
        assert ec["stockfish_elo"] == 1320
        assert ec["manifest_output"] == "/vol/results/phase2/match.json"
        assert ec["seed"] == 7
        assert ec["include_piece_list"] is True

    def test_resolve_track_model_prioritizes_explicit_model(self):
        mt = self._load_modal_train()
        ckpt = "/vol/checkpoints/sft_e1a_piece_list/final_merged"
        assert mt.resolve_track_model(model=ckpt, track="v1") == ckpt
        assert mt.resolve_track_model(model="", track="v1") == "Qwen/Qwen3-4B"
        assert mt.resolve_track_model(model="", track="") == "Qwen/Qwen3-4B"


# ---------------------------------------------------------------------------
# upload_hf tests
# ---------------------------------------------------------------------------


upload_hf = _load_script_module("upload_hf", "upload_hf.py")


class TestUploadHF:
    def test_dry_run_validates_and_never_prints_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        artifact_dir = tmp_path / "ckpt"
        artifact_dir.mkdir()
        (artifact_dir / "config.json").write_text("{}", encoding="utf-8")

        secret_val = "hf_super_secret_test_token_12345"
        monkeypatch.setenv("HF_TOKEN", secret_val)

        rc = upload_hf.main(
            [
                "--local-dir",
                str(artifact_dir),
                "--repo-id",
                "winsonly/chessfm-test",
                "--dry-run",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "winsonly/chessfm-test" in out
        assert secret_val not in out

    def test_missing_local_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Local path does not exist"):
            upload_hf.upload_to_hub(
                local_dir=tmp_path / "nonexistent",
                repo_id="winsonly/chessfm-test",
                dry_run=True,
            )

    def test_missing_hf_token_exits_cleanly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        artifact_dir = tmp_path / "ckpt"
        artifact_dir.mkdir()
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(SystemExit, match="HF_TOKEN"):
            upload_hf.upload_to_hub(
                local_dir=artifact_dir,
                repo_id="winsonly/chessfm-test",
                dry_run=False,
            )
