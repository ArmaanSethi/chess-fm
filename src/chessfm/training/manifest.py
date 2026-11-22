"""Training run manifests, JSONL step telemetry, and anomaly detection.

Pure-stdlib utilities for recording reproducible SFT/GRPO training provenance
(``run_manifest.json``), streaming per-step telemetry to JSONL, and flagging
common RL failure modes (KL explosion, entropy collapse, completion-length
reward hacking).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"
TELEMETRY_FIELDS: tuple[str, ...] = (
    "loss",
    "legal_rate",
    "format_adherence",
    "reward_mean",
    "kl",
    "entropy",
    "mean_completion_len",
)
_CHUNK_SIZE = 65536


def _get_git_sha(cwd: Path | None = None) -> str:
    """Return the current git commit SHA, or ``"unknown"`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 hex digest of a dataset or artifact file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_dir: Path,
    *,
    model_id: str,
    stage: str,
    seed: int,
    hyperparameters: dict[str, Any],
    dataset_path: Path | None = None,
    gpu_type: str = "cpu",
    wall_clock_seconds: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_manifest.json`` inside ``output_dir`` and return its path.

    Records git SHA, UTC timestamp, model/stage/seed metadata, streaming
    dataset SHA-256 (when ``dataset_path`` is provided), hyperparameters,
    hardware type, wall-clock duration, and summary metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_sha256 = sha256_file(dataset_path) if dataset_path is not None else None
    manifest: dict[str, Any] = {
        "git_sha": _get_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "stage": stage,
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "hyperparameters": dict(hyperparameters),
        "gpu_type": gpu_type,
        "wall_clock_seconds": wall_clock_seconds,
        "metrics": dict(metrics) if metrics is not None else {},
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def append_telemetry_step(log_path: Path, step: int, metrics: dict[str, Any]) -> None:
    """Append a single step telemetry record to a JSONL file.

    Intended for per-step SFT/GRPO metrics such as ``loss``, ``legal_rate``,
    ``format_adherence``, ``reward_mean``, ``kl``, ``entropy``, and
    ``mean_completion_len``.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "step": int(step),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in metrics.items() if k not in ("step", "timestamp")},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_telemetry(log_path: Path) -> list[dict[str, Any]]:
    """Load JSONL step telemetry records from ``log_path``.

    Returns an empty list if ``log_path`` does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _extract_metric(record: dict[str, Any], key: str) -> float | None:
    """Extract a numeric metric from a flat or nested telemetry record."""
    if key in record:
        raw = record[key]
    elif isinstance(record.get("metrics"), dict):
        raw = record["metrics"].get(key)
    else:
        return None
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def detect_training_anomalies(
    records: list[dict[str, Any]],
    *,
    max_kl: float = 0.5,
    min_entropy: float = 0.05,
    max_completion_len: int = 480,
) -> dict[str, Any]:
    """Inspect telemetry records for RL/SFT training pathologies.

    Checks for:
    - **KL explosion**: ``kl > max_kl`` (or non-finite KL).
    - **Entropy collapse**: ``entropy < min_entropy`` (or non-finite entropy).
    - **Completion-length reward hacking**: ``mean_completion_len > max_completion_len``
      (or non-finite completion length).

    Returns a structured diagnostic report with ``healthy``, ``num_steps``,
    ``anomalies`` flags, human-readable ``warnings``, and per-anomaly ``details``.
    """
    kl_violations: list[dict[str, Any]] = []
    entropy_violations: list[dict[str, Any]] = []
    completion_len_violations: list[dict[str, Any]] = []

    max_observed_kl: float | None = None
    min_observed_entropy: float | None = None
    max_observed_completion_len: float | None = None

    for idx, record in enumerate(records):
        step = record.get("step", idx)

        kl = _extract_metric(record, "kl")
        if kl is not None:
            if not math.isfinite(kl):
                kl_violations.append({"step": step, "kl": kl})
            else:
                max_observed_kl = kl if max_observed_kl is None else max(max_observed_kl, kl)
                if kl > max_kl:
                    kl_violations.append({"step": step, "kl": kl})

        entropy = _extract_metric(record, "entropy")
        if entropy is not None:
            if not math.isfinite(entropy):
                entropy_violations.append({"step": step, "entropy": entropy})
            else:
                min_observed_entropy = (
                    entropy if min_observed_entropy is None else min(min_observed_entropy, entropy)
                )
                if entropy < min_entropy:
                    entropy_violations.append({"step": step, "entropy": entropy})

        comp_len = _extract_metric(record, "mean_completion_len")
        if comp_len is not None:
            if not math.isfinite(comp_len):
                completion_len_violations.append({"step": step, "mean_completion_len": comp_len})
            else:
                max_observed_completion_len = (
                    comp_len
                    if max_observed_completion_len is None
                    else max(max_observed_completion_len, comp_len)
                )
                if comp_len > max_completion_len:
                    completion_len_violations.append(
                        {"step": step, "mean_completion_len": comp_len}
                    )

    warnings: list[str] = []
    if kl_violations:
        first_step = kl_violations[0]["step"]
        peak_str = f"{max_observed_kl:.4f}" if max_observed_kl is not None else "non-finite"
        warnings.append(
            f"KL explosion detected on {len(kl_violations)} step(s) "
            f"(first at step {first_step}, max observed KL={peak_str} > threshold {max_kl}). "
            "Consider increasing the KL penalty (beta) or lowering the learning rate."
        )
    if entropy_violations:
        first_step = entropy_violations[0]["step"]
        min_str = (
            f"{min_observed_entropy:.4f}" if min_observed_entropy is not None else "non-finite"
        )
        warnings.append(
            f"Entropy collapse detected on {len(entropy_violations)} step(s) "
            f"(first at step {first_step}, min observed entropy={min_str} < threshold {min_entropy}). "
            "Consider lowering the learning rate or reviewing reward shaping."
        )
    if completion_len_violations:
        first_step = completion_len_violations[0]["step"]
        peak_str = (
            f"{max_observed_completion_len:.1f}"
            if max_observed_completion_len is not None
            else "non-finite"
        )
        warnings.append(
            f"Completion-length reward hacking detected on {len(completion_len_violations)} step(s) "
            f"(first at step {first_step}, max observed mean_completion_len={peak_str} > "
            f"threshold {max_completion_len}). "
            "Consider adding a length penalty or tightening max completion tokens."
        )

    return {
        "healthy": len(warnings) == 0,
        "num_steps": len(records),
        "anomalies": {
            "kl_explosion": bool(kl_violations),
            "entropy_collapse": bool(entropy_violations),
            "completion_length_hacking": bool(completion_len_violations),
        },
        "warnings": warnings,
        "details": {
            "kl_violations": kl_violations,
            "entropy_violations": entropy_violations,
            "completion_len_violations": completion_len_violations,
            "max_observed_kl": max_observed_kl,
            "min_observed_entropy": min_observed_entropy,
            "max_observed_completion_len": max_observed_completion_len,
        },
    }
