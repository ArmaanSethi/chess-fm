"""Unit tests for scripts/export_onnx_webgpu.py and Layer B additions in scripts/modal_train.py."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_onnx = _load_script_module("export_onnx_webgpu", "export_onnx_webgpu.py")


class TestExportOnnxHelpers:
    def test_valid_quants_and_filenames(self):
        assert export_onnx.VALID_QUANTS == ("q4f16", "q4", "fp16", "fp32")
        assert export_onnx.target_onnx_filename("q4f16") == "model_q4f16.onnx"
        assert export_onnx.target_onnx_filename("q4") == "model_q4.onnx"
        assert export_onnx.target_onnx_filename("fp16") == "model_fp16.onnx"
        assert export_onnx.target_onnx_filename("fp32") == "model.onnx"
        with pytest.raises(ValueError, match="Unsupported quant"):
            export_onnx.target_onnx_filename("q8_0")

    def test_source_export_dtype(self):
        assert export_onnx.source_export_dtype("q4f16") == "fp16"
        assert export_onnx.source_export_dtype("fp16") == "fp16"
        assert export_onnx.source_export_dtype("q4") == "fp32"
        assert export_onnx.source_export_dtype("fp32") == "fp32"
        with pytest.raises(ValueError, match="Unsupported quant"):
            export_onnx.source_export_dtype("int8")

    def test_build_optimum_export_cmd(self):
        cmd = export_onnx.build_optimum_export_cmd(
            "Qwen/Qwen3-0.6B",
            "/tmp/onnx_out",
            dtype="fp16",
            opset=17,
            trust_remote_code=True,
        )
        assert cmd[:5] == ["optimum-cli", "export", "onnx", "--model", "Qwen/Qwen3-0.6B"]
        assert "--task" in cmd and "text-generation-with-past" in cmd
        assert "--dtype" in cmd and "fp16" in cmd
        assert "--opset" in cmd and "17" in cmd
        assert "--trust-remote-code" in cmd
        assert cmd[-1] == "/tmp/onnx_out"

        with pytest.raises(ValueError, match="Unsupported export dtype"):
            export_onnx.build_optimum_export_cmd("Qwen/Qwen3-0.6B", "/tmp/out", dtype="bf16")

    def test_build_export_manifest(self):
        manifest = export_onnx.build_export_manifest(
            model="Qwen/Qwen3-0.6B",
            quant="q4f16",
            dtype="fp16",
            opset=17,
            onnx_file="onnx/model_q4f16.onnx",
            size_mb=412.3456,
            ort_verified=True,
            input_names_count=58,
            output_names_count=57,
            timestamp_utc="2026-11-23T00:00:00Z",
            git_sha="a" * 40,
        )
        assert manifest == {
            "model": "Qwen/Qwen3-0.6B",
            "quant": "q4f16",
            "dtype": "fp16",
            "opset": 17,
            "onnx_file": "onnx/model_q4f16.onnx",
            "size_mb": 412.35,
            "ort_verified": True,
            "timestamp_utc": "2026-11-23T00:00:00Z",
            "git_sha": "a" * 40,
            "input_names_count": 58,
            "output_names_count": 57,
        }


class TestExportOnnxCLI:
    def test_dry_run_writes_zero_files_and_returns_zero(self, tmp_path: Path, capsys):
        out_dir = tmp_path / "webgpu_export"
        rc = export_onnx.main(
            [
                "--model",
                "Qwen/Qwen3-0.6B",
                "--output",
                str(out_dir),
                "--quant",
                "q4f16",
                "--opset",
                "17",
                "--upload-hf",
                "winsonly/chessfm-coach-0.6b-onnx",
                "--dry-run",
            ]
        )
        assert rc == 0
        assert not out_dir.exists()
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "optimum-cli export onnx" in out
        assert "model_q4f16.onnx" in out
        assert "MatMulNBitsQuantizer" in out
        assert "winsonly/chessfm-coach-0.6b-onnx" in out

    def test_invalid_quant_cli_exits_nonzero(self, tmp_path: Path):
        with pytest.raises(SystemExit) as exc_info:
            export_onnx.main(
                [
                    "--model",
                    "Qwen/Qwen3-0.6B",
                    "--output",
                    str(tmp_path / "out"),
                    "--quant",
                    "invalid_quant",
                    "--dry-run",
                ]
            )
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("bad_opset", ["0", "-5"])
    def test_non_positive_opset_cli_exits_nonzero(self, tmp_path: Path, bad_opset: str):
        with pytest.raises(SystemExit) as exc_info:
            export_onnx.main(
                [
                    "--model",
                    "Qwen/Qwen3-0.6B",
                    "--output",
                    str(tmp_path / "out"),
                    "--opset",
                    bad_opset,
                    "--dry-run",
                ]
            )
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("bad_block_size", ["0", "-16"])
    def test_non_positive_block_size_cli_exits_nonzero(self, tmp_path: Path, bad_block_size: str):
        with pytest.raises(SystemExit) as exc_info:
            export_onnx.main(
                [
                    "--model",
                    "Qwen/Qwen3-0.6B",
                    "--output",
                    str(tmp_path / "out"),
                    "--block-size",
                    bad_block_size,
                    "--dry-run",
                ]
            )
        assert exc_info.value.code != 0

    def test_build_optimum_export_cmd_rejects_non_positive_opset(self):
        with pytest.raises(ValueError, match="opset must be a positive integer"):
            export_onnx.build_optimum_export_cmd("Qwen/Qwen3-0.6B", "/tmp/out", opset=0)


class TestModalTrainLayerBAST:
    def test_layer_b_tasks_tracks_and_images_present(self):
        source = (SCRIPTS_DIR / "modal_train.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="scripts/modal_train.py")

        fn_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        for expected_fn in ("run_sft_student_l4", "run_export_onnx", "run_coach_benchmark_l4"):
            assert expected_fn in fn_names, f"Missing Layer B function {expected_fn}"

        assign_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "onnx_image" in assign_names

        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for literal in (
            "sft-student",
            "export-onnx",
            "coach-benchmark",
            "student-qwen3",
            "Qwen/Qwen3-0.6B",
            "scripts/export_onnx_webgpu.py",
            "--workers",
            "--gpu-memory-utilization",
            "bfloat16",
        ):
            assert literal in string_literals, (
                f"Missing Layer B literal {literal!r} in modal_train.py"
            )
