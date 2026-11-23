#!/usr/bin/env python3
"""Export a Hugging Face causal LM checkpoint to Transformers.js v3 WebGPU ONNX format.

Produces a Transformers.js-compatible directory layout:
    <output_dir>/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json      # (if present)
    ├── vocab.json / merges.txt      # (if present)
    ├── chat_template.jinja          # (if present)
    ├── onnx/
    │   └── model_q4f16.onnx         # (or model_q4.onnx / model_fp16.onnx / model.onnx)
    └── export_manifest.json         # Provenance & ONNX Runtime verification summary

Usage:
    # Preview commands and target artifacts without writing files:
    python scripts/export_onnx_webgpu.py \\
        --model Qwen/Qwen3-0.6B \\
        --output exports/qwen3-0.6b-webgpu \\
        --quant q4f16 --dry-run

    # Export, INT4-quantize, verify with ONNX Runtime, and upload to HF Hub:
    python scripts/export_onnx_webgpu.py \\
        --model /vol/checkpoints/sft_student/final_merged \\
        --output /vol/exports/chessfm-coach-0.6b-onnx \\
        --quant q4f16 \\
        --upload-hf winsonly/chessfm-coach-0.6b-onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_QUANTS: tuple[str, ...] = ("q4f16", "q4", "fp16", "fp32")

TOKENIZER_AND_CONFIG_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

# Single-file protobuf limit is 2 GiB; use external data above 1.8 GiB.
_EXTERNAL_DATA_THRESHOLD_BYTES = int(1.8 * 1024 * 1024 * 1024)


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string (YYYY-MM-DDTHH:MM:SSZ)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head_sha() -> str:
    """Return the 40-char git HEAD SHA, or ``'unknown'`` outside a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if len(out) == 40:
            return out
    except Exception:
        pass
    return "unknown"


def _normalize_quant(quant: str) -> str:
    key = quant.strip().lower()
    if key not in VALID_QUANTS:
        raise ValueError(f"Unsupported quant {quant!r}. Choose from: {', '.join(VALID_QUANTS)}")
    return key


def target_onnx_filename(quant: str) -> str:
    """Return the Transformers.js v3 ONNX filename for ``quant``."""
    key = _normalize_quant(quant)
    mapping = {
        "q4f16": "model_q4f16.onnx",
        "q4": "model_q4.onnx",
        "fp16": "model_fp16.onnx",
        "fp32": "model.onnx",
    }
    return mapping[key]


def source_export_dtype(quant: str) -> str:
    """Return the ``optimum-cli`` source export dtype (``fp16`` or ``fp32``) for ``quant``."""
    key = _normalize_quant(quant)
    return "fp16" if key in ("q4f16", "fp16") else "fp32"


def build_optimum_export_cmd(
    model: str,
    output_dir: str | Path,
    *,
    dtype: str = "fp16",
    opset: int = 17,
    trust_remote_code: bool = False,
) -> list[str]:
    """Build the ``optimum-cli export onnx`` command for causal LM export with KV cache."""
    if dtype not in ("fp16", "fp32"):
        raise ValueError(f"Unsupported export dtype {dtype!r}. Choose 'fp16' or 'fp32'.")
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(model),
        "--task",
        "text-generation-with-past",
        "--dtype",
        dtype,
        "--opset",
        str(opset),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(str(output_dir))
    return cmd


def build_export_manifest(
    *,
    model: str,
    quant: str,
    dtype: str,
    opset: int,
    onnx_file: str,
    size_mb: float,
    ort_verified: bool,
    input_names_count: int = 0,
    output_names_count: int = 0,
    timestamp_utc: str | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Construct a structured provenance manifest for an ONNX WebGPU export."""
    return {
        "model": str(model),
        "quant": _normalize_quant(quant),
        "dtype": dtype,
        "opset": int(opset),
        "onnx_file": str(onnx_file),
        "size_mb": round(float(size_mb), 2),
        "ort_verified": bool(ort_verified),
        "timestamp_utc": timestamp_utc or _utc_timestamp(),
        "git_sha": git_sha or _git_head_sha(),
        "input_names_count": int(input_names_count),
        "output_names_count": int(output_names_count),
    }


def _infer_dummy_input(inp: Any, np_mod: Any) -> Any:
    """Build a 1-token dummy numpy tensor matching an ONNX Runtime ``NodeArg``."""
    name = inp.name
    onnx_type = str(inp.type or "")
    if "int64" in onnx_type:
        np_dtype = np_mod.int64
    elif "int32" in onnx_type:
        np_dtype = np_mod.int32
    elif "float16" in onnx_type:
        np_dtype = np_mod.float16
    elif "bool" in onnx_type:
        np_dtype = np_mod.bool_
    else:
        np_dtype = np_mod.float32

    if name == "input_ids":
        return np_mod.array([[1]], dtype=np_dtype)
    if name == "attention_mask":
        return np_mod.array([[1]], dtype=np_dtype)
    if name == "position_ids":
        return np_mod.array([[0]], dtype=np_dtype)

    # Past KV-cache / recurrent state tensors: resolve symbolic dimensions cleanly.
    shape: list[int] = []
    for dim in inp.shape or []:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        elif isinstance(dim, str) and "past" in dim.lower():
            shape.append(0)
        elif isinstance(dim, str) and "batch" in dim.lower():
            shape.append(1)
        elif "past_key_values" in name and len(shape) == 2:
            shape.append(0)
        else:
            shape.append(1)
    return np_mod.zeros(shape, dtype=np_dtype)


def verify_onnx_model(onnx_path: Path) -> dict[str, Any]:
    """Load an exported ONNX model in ONNX Runtime and run a 1-token forward check.

    Verifies that the model loads cleanly on ``CPUExecutionProvider``, exposes
    valid input/output tensors, and produces finite (non-NaN / non-Inf) logits.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs_meta = sess.get_outputs()

    feeds = {inp.name: _infer_dummy_input(inp, np) for inp in inputs}
    outputs = sess.run(None, feeds)
    if not outputs:
        raise RuntimeError(f"ONNX model {onnx_path} produced no outputs on dummy forward pass.")

    logits_idx = 0
    for idx, meta in enumerate(outputs_meta):
        if meta.name == "logits":
            logits_idx = idx
            break
    logits = np.asarray(outputs[logits_idx], dtype=np.float32)
    if logits.size == 0 or not np.isfinite(logits).all():
        raise RuntimeError(f"ONNX model {onnx_path} produced non-finite or empty logits.")

    return {
        "ort_verified": True,
        "input_names_count": len(inputs),
        "output_names_count": len(outputs_meta),
        "logits_shape": list(logits.shape),
    }


def _copy_tokenizer_and_configs(src_dirs: list[Path], dst_dir: Path) -> list[str]:
    """Copy tokenizer and config files from ``src_dirs`` into ``dst_dir``."""
    copied: list[str] = []
    for filename in TOKENIZER_AND_CONFIG_FILES:
        dst_file = dst_dir / filename
        if dst_file.exists():
            continue
        for src_dir in src_dirs:
            src_file = src_dir / filename
            if src_file.is_file():
                shutil.copy2(src_file, dst_file)
                copied.append(filename)
                break
    return copied


def _quantize_4bit(raw_onnx_path: Path, target_onnx_path: Path, *, block_size: int = 32) -> None:
    """Quantize MatMul weights in ``raw_onnx_path`` to 4-bit via ``MatMulNBitsQuantizer``."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(
        model=str(raw_onnx_path),
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
    )
    quantizer.process()
    use_external = raw_onnx_path.stat().st_size > _EXTERNAL_DATA_THRESHOLD_BYTES
    quantizer.model.save_model_to_file(
        str(target_onnx_path),
        use_external_data_format=use_external,
    )


def _total_onnx_size_mb(onnx_path: Path) -> float:
    """Return combined size in MB of ``onnx_path`` plus optional external data file."""
    total_bytes = onnx_path.stat().st_size if onnx_path.is_file() else 0
    for suffix in (f"{onnx_path.name}_data", f"{onnx_path.name}.data"):
        ext = onnx_path.parent / suffix
        if ext.is_file():
            total_bytes += ext.stat().st_size
    return total_bytes / (1024 * 1024)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a causal LM checkpoint to Transformers.js v3 WebGPU ONNX format."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local merged checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for Transformers.js ONNX artifacts and manifest",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="q4f16",
        choices=list(VALID_QUANTS),
        help="Target ONNX quantization format (default: q4f16)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version for optimum export (default: 17)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for 4-bit MatMulNBitsQuantizer (default: 32)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to optimum-cli export",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime 1-token dummy forward verification",
    )
    parser.add_argument(
        "--upload-hf",
        type=str,
        default="",
        help="Optional Hugging Face Hub repo_id to upload the exported directory after verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print planned commands/artifacts without writing any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quant = _normalize_quant(args.quant)
    dtype = source_export_dtype(quant)
    onnx_filename = target_onnx_filename(quant)
    target_onnx_path = args.output / "onnx" / onnx_filename
    manifest_path = args.output / "export_manifest.json"

    planned_cmd = build_optimum_export_cmd(
        args.model,
        "<tmp_export_dir>",
        dtype=dtype,
        opset=args.opset,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dry_run:
        print(f"[dry-run] Model:          {args.model}")
        print(f"[dry-run] Output dir:     {args.output}")
        print(f"[dry-run] Quant / dtype:  {quant} (export dtype={dtype}, opset={args.opset})")
        print(f"[dry-run] Target ONNX:    {target_onnx_path}")
        print(f"[dry-run] Manifest:       {manifest_path}")
        print(f"[dry-run] Export command: {' '.join(planned_cmd)}")
        if quant in ("q4f16", "q4"):
            print(f"[dry-run] Quantization:   MatMulNBitsQuantizer(block_size={args.block_size})")
        print(f"[dry-run] ORT verify:     {'skip' if args.skip_verify else 'enabled'}")
        if args.upload_hf:
            print(f"[dry-run] HF upload:      {args.upload_hf}")
        return 0

    with tempfile.TemporaryDirectory(prefix="chessfm_onnx_export_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        export_cmd = build_optimum_export_cmd(
            args.model,
            tmp_dir,
            dtype=dtype,
            opset=args.opset,
            trust_remote_code=args.trust_remote_code,
        )
        print(f">>> {' '.join(export_cmd)}", flush=True)
        proc = subprocess.run(export_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"optimum-cli ONNX export failed for {args.model!r} (exit {proc.returncode}). "
                f"This often indicates an unsupported architecture (e.g. hybrid DeltaNet ops).\n"
                f"--- exporter output ---\n{err_tail}"
            )

        raw_onnx = tmp_dir / "model.onnx"
        if not raw_onnx.is_file():
            onnx_candidates = sorted(tmp_dir.glob("*.onnx"))
            if not onnx_candidates:
                raise FileNotFoundError(f"No .onnx file produced by optimum-cli in {tmp_dir}")
            raw_onnx = onnx_candidates[0]

        onnx_out_dir = args.output / "onnx"
        onnx_out_dir.mkdir(parents=True, exist_ok=True)

        if quant in ("q4f16", "q4"):
            print(
                f"Quantizing {raw_onnx.name} -> {target_onnx_path} "
                f"(quant={quant}, block_size={args.block_size})...",
                flush=True,
            )
            _quantize_4bit(raw_onnx, target_onnx_path, block_size=args.block_size)
        else:
            shutil.copy2(raw_onnx, target_onnx_path)
            raw_data = raw_onnx.parent / f"{raw_onnx.name}_data"
            if raw_data.is_file():
                shutil.copy2(raw_data, onnx_out_dir / f"{target_onnx_path.name}_data")

        src_dirs = [tmp_dir]
        local_model_dir = Path(args.model)
        if local_model_dir.is_dir():
            src_dirs.append(local_model_dir)
        copied_files = _copy_tokenizer_and_configs(src_dirs, args.output)
        print(f"Copied tokenizer/config files: {', '.join(copied_files) or '(none)'}")

    verify_info: dict[str, Any] = {
        "ort_verified": False,
        "input_names_count": 0,
        "output_names_count": 0,
    }
    if not args.skip_verify:
        print(f"Verifying {target_onnx_path} with ONNX Runtime...", flush=True)
        verify_info = verify_onnx_model(target_onnx_path)
        print(
            f"ONNX Runtime check PASSED: {verify_info['input_names_count']} inputs, "
            f"{verify_info['output_names_count']} outputs, logits={verify_info['logits_shape']}"
        )

    size_mb = _total_onnx_size_mb(target_onnx_path)
    manifest = build_export_manifest(
        model=args.model,
        quant=quant,
        dtype=dtype,
        opset=args.opset,
        onnx_file=f"onnx/{onnx_filename}",
        size_mb=size_mb,
        ort_verified=verify_info["ort_verified"],
        input_names_count=verify_info["input_names_count"],
        output_names_count=verify_info["output_names_count"],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote export manifest to {manifest_path} ({size_mb:.1f} MB)")

    if args.upload_hf:
        upload_script = Path(__file__).resolve().parent / "upload_hf.py"
        upload_cmd = [
            sys.executable,
            str(upload_script),
            "--local-dir",
            str(args.output),
            "--repo-id",
            args.upload_hf,
            "--commit-message",
            f"Upload {quant} ONNX WebGPU export for {Path(args.model).name}",
        ]
        print(f">>> {' '.join(upload_cmd)}", flush=True)
        subprocess.run(upload_cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
