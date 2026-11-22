"""Unit tests for training run manifests, telemetry logging, and anomaly detection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from chessfm.training.manifest import (
    MANIFEST_FILENAME,
    _get_git_sha,
    append_telemetry_step,
    detect_training_anomalies,
    load_telemetry,
    sha256_file,
    write_run_manifest,
)


class TestSha256File:
    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(f) == hashlib.sha256(b"").hexdigest()

    def test_streaming_multi_chunk_file(self, tmp_path: Path):
        payload = b"chessfm-sft-record\n" * 10000  # > 64 KiB to exercise chunk loop
        f = tmp_path / "dataset.jsonl"
        f.write_bytes(payload)
        assert sha256_file(f) == hashlib.sha256(payload).hexdigest()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            sha256_file(tmp_path / "does_not_exist.jsonl")


class TestGetGitSha:
    def test_returns_sha_in_git_repo(self):
        sha = _get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) >= 7
        assert sha != "unknown"

    def test_returns_unknown_on_nonzero_exit(self, monkeypatch):
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], returncode=128, stdout="", stderr="fatal")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _get_git_sha() == "unknown"

    def test_returns_unknown_on_os_error(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise OSError("git not installed")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _get_git_sha() == "unknown"


class TestWriteRunManifest:
    def test_writes_complete_manifest_with_dataset(self, tmp_path: Path):
        dataset = tmp_path / "train.jsonl"
        dataset.write_text('{"fen": "startpos"}\n', encoding="utf-8")
        out_dir = tmp_path / "nested" / "run_001"

        manifest_path = write_run_manifest(
            out_dir,
            model_id="Qwen/Qwen3-4B",
            stage="grpo_stage1",
            seed=42,
            hyperparameters={"lr": 5e-6, "beta": 0.04, "batch_size": 8},
            dataset_path=dataset,
            gpu_type="L4",
            wall_clock_seconds=123.45,
            metrics={"legal_rate": 0.99, "reward_mean": 1.42},
        )

        assert manifest_path == out_dir / MANIFEST_FILENAME
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert isinstance(data["git_sha"], str) and data["git_sha"]
        # Validate ISO-8601 timestamp
        parsed_ts = datetime.fromisoformat(data["timestamp"])
        assert parsed_ts.tzinfo is not None
        assert data["model_id"] == "Qwen/Qwen3-4B"
        assert data["stage"] == "grpo_stage1"
        assert data["seed"] == 42
        assert data["dataset_sha256"] == sha256_file(dataset)
        assert data["hyperparameters"] == {"lr": 5e-6, "beta": 0.04, "batch_size": 8}
        assert data["gpu_type"] == "L4"
        assert data["wall_clock_seconds"] == 123.45
        assert data["metrics"] == {"legal_rate": 0.99, "reward_mean": 1.42}

    def test_defaults_without_optional_fields(self, tmp_path: Path):
        out_dir = tmp_path / "run_defaults"
        manifest_path = write_run_manifest(
            out_dir,
            model_id="unsloth/gemma-4-E4B-it",
            stage="sft",
            seed=7,
            hyperparameters={"epochs": 3},
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["dataset_sha256"] is None
        assert data["gpu_type"] == "cpu"
        assert data["wall_clock_seconds"] is None
        assert data["metrics"] == {}


class TestTelemetryLog:
    def test_load_missing_returns_empty_list(self, tmp_path: Path):
        assert load_telemetry(tmp_path / "missing.jsonl") == []

    def test_append_and_load_round_trip(self, tmp_path: Path):
        log_path = tmp_path / "logs" / "telemetry.jsonl"
        append_telemetry_step(
            log_path,
            10,
            {
                "loss": 1.25,
                "legal_rate": 0.94,
                "format_adherence": 0.98,
                "reward_mean": 0.85,
                "kl": 0.03,
                "entropy": 0.42,
                "mean_completion_len": 160,
            },
        )
        append_telemetry_step(
            log_path,
            20,
            {
                "loss": 0.95,
                "legal_rate": 0.97,
                "format_adherence": 1.0,
                "reward_mean": 1.15,
                "kl": 0.05,
                "entropy": 0.38,
                "mean_completion_len": 175,
            },
        )

        records = load_telemetry(log_path)
        assert len(records) == 2
        assert records[0]["step"] == 10
        assert records[0]["loss"] == 1.25
        assert records[0]["kl"] == 0.03
        assert "timestamp" in records[0]
        assert records[1]["step"] == 20
        assert records[1]["mean_completion_len"] == 175

    def test_ignores_blank_lines_and_protects_step_field(self, tmp_path: Path):
        log_path = tmp_path / "telemetry.jsonl"
        append_telemetry_step(
            log_path,
            5,
            {"step": 999, "timestamp": "fake", "loss": 0.5},
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n   \n")

        records = load_telemetry(log_path)
        assert len(records) == 1
        assert records[0]["step"] == 5
        assert records[0]["timestamp"] != "fake"
        assert records[0]["loss"] == 0.5


class TestDetectTrainingAnomalies:
    def test_empty_records_is_healthy(self):
        report = detect_training_anomalies([])
        assert report["healthy"] is True
        assert report["num_steps"] == 0
        assert report["anomalies"] == {
            "kl_explosion": False,
            "entropy_collapse": False,
            "completion_length_hacking": False,
        }
        assert report["warnings"] == []
        assert report["details"]["max_observed_kl"] is None
        assert report["details"]["min_observed_entropy"] is None
        assert report["details"]["max_observed_completion_len"] is None

    def test_healthy_records_and_exact_boundaries(self):
        records = [
            {"step": 1, "kl": 0.1, "entropy": 0.3, "mean_completion_len": 200},
            {"step": 2, "kl": 0.5, "entropy": 0.05, "mean_completion_len": 480},
        ]
        report = detect_training_anomalies(records)
        assert report["healthy"] is True
        assert report["warnings"] == []
        assert report["details"]["max_observed_kl"] == 0.5
        assert report["details"]["min_observed_entropy"] == 0.05
        assert report["details"]["max_observed_completion_len"] == 480.0

    def test_detects_kl_explosion(self):
        records = [
            {"step": 10, "kl": 0.2},
            {"step": 20, "kl": 0.75},
            {"step": 30, "kl": 1.10},
        ]
        report = detect_training_anomalies(records, max_kl=0.5)
        assert report["healthy"] is False
        assert report["anomalies"]["kl_explosion"] is True
        assert report["anomalies"]["entropy_collapse"] is False
        assert report["anomalies"]["completion_length_hacking"] is False
        assert len(report["warnings"]) == 1
        assert "KL explosion" in report["warnings"][0]
        assert "step 20" in report["warnings"][0]
        assert len(report["details"]["kl_violations"]) == 2
        assert report["details"]["max_observed_kl"] == 1.10

    def test_detects_entropy_collapse(self):
        records = [
            {"step": 10, "entropy": 0.25},
            {"step": 20, "entropy": 0.01},
        ]
        report = detect_training_anomalies(records, min_entropy=0.05)
        assert report["healthy"] is False
        assert report["anomalies"]["entropy_collapse"] is True
        assert len(report["warnings"]) == 1
        assert "Entropy collapse" in report["warnings"][0]
        assert "step 20" in report["warnings"][0]
        assert report["details"]["min_observed_entropy"] == 0.01

    def test_detects_completion_length_reward_hacking(self):
        records = [
            {"step": 10, "mean_completion_len": 250},
            {"step": 20, "mean_completion_len": 512},
        ]
        report = detect_training_anomalies(records, max_completion_len=480)
        assert report["healthy"] is False
        assert report["anomalies"]["completion_length_hacking"] is True
        assert len(report["warnings"]) == 1
        assert "Completion-length reward hacking" in report["warnings"][0]
        assert "step 20" in report["warnings"][0]
        assert report["details"]["max_observed_completion_len"] == 512.0

    def test_detects_all_anomalies_simultaneously(self):
        records = [
            {"step": 50, "kl": 0.9, "entropy": 0.02, "mean_completion_len": 600},
        ]
        report = detect_training_anomalies(records)
        assert report["healthy"] is False
        assert all(report["anomalies"].values())
        assert len(report["warnings"]) == 3

    def test_non_finite_values_flagged(self):
        records = [
            {
                "step": 1,
                "kl": float("nan"),
                "entropy": float("-inf"),
                "mean_completion_len": float("inf"),
            },
        ]
        report = detect_training_anomalies(records)
        assert report["healthy"] is False
        assert all(report["anomalies"].values())
        assert all("non-finite" in w for w in report["warnings"])

    def test_nested_metrics_and_non_numeric_ignored(self):
        records = [
            {
                "step": 1,
                "metrics": {"kl": 0.8, "entropy": True, "mean_completion_len": "bad"},
            },
            {"loss": 1.1},  # SFT-only step without RL fields or explicit step
        ]
        report = detect_training_anomalies(records)
        assert report["healthy"] is False
        assert report["anomalies"]["kl_explosion"] is True
        assert report["anomalies"]["entropy_collapse"] is False
        assert report["anomalies"]["completion_length_hacking"] is False

    def test_end_to_end_telemetry_to_anomaly_detection(self, tmp_path: Path):
        log_path = tmp_path / "grpo_telemetry.jsonl"
        append_telemetry_step(log_path, 1, {"loss": 1.0, "kl": 0.05, "entropy": 0.4})
        append_telemetry_step(
            log_path, 2, {"loss": 0.8, "kl": 0.65, "entropy": 0.03, "mean_completion_len": 500}
        )
        records = load_telemetry(log_path)
        report = detect_training_anomalies(records)
        assert report["healthy"] is False
        assert report["num_steps"] == 2
        assert report["anomalies"]["kl_explosion"] is True
        assert report["anomalies"]["entropy_collapse"] is True
        assert report["anomalies"]["completion_length_hacking"] is True
