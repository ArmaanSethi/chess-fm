#!/usr/bin/env python3
"""Export a merged ChessFM / ChessCoach checkpoint to GGUF and generate an Ollama Modelfile.

Usage:
    python scripts/export_gguf.py \\
        --model checkpoints/sft_coach/final_merged \\
        --output exports/chesscoach-gemma4 \\
        --quant q4_k_m \\
        --family gemma4

    # Preview the generated Modelfile and conversion command without running llama.cpp:
    python scripts/export_gguf.py \\
        --model checkpoints/sft_coach/final_merged \\
        --output exports/chesscoach-gemma4 \\
        --family gemma4 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHESS_COACH_SYSTEM_PROMPT = (
    "You are ChessFM Coach, an engine-grounded chess tutor and analyst. "
    "Reason carefully inside <think>...</think> tags using the exact FEN position, "
    "verify that every candidate move is legal, and then give a clear, actionable "
    "UCI move or coaching explanation tailored to the student's level."
)

FAMILY_STOP_TOKENS: dict[str, tuple[str, ...]] = {
    "gemma4": ("<turn|>", "<|turn>", "<eos>"),
    "qwen3": ("<|im_end|>", "<|im_start|>", "<|endoftext|>"),
    "qwen25": ("<|im_end|>", "<|im_start|>", "<|endoftext|>"),
}

_FAMILY_ALIASES: dict[str, str] = {
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "qwen3": "qwen3",
    "qwen-3": "qwen3",
    "qwen25": "qwen25",
    "qwen2.5": "qwen25",
    "qwen-2.5": "qwen25",
}


def normalize_model_family(model_family: str) -> str:
    """Normalize a model family identifier to one of ``gemma4``, ``qwen3``, ``qwen25``."""
    key = model_family.strip().lower()
    if key not in _FAMILY_ALIASES:
        valid = ", ".join(sorted(FAMILY_STOP_TOKENS))
        raise ValueError(f"Unsupported model_family {model_family!r}. Choose from: {valid}")
    return _FAMILY_ALIASES[key]


def build_ollama_modelfile(
    gguf_path: str,
    model_family: str = "gemma4",
    temperature: float = 0.2,
) -> str:
    """Generate a ready-to-run Ollama ``Modelfile`` for a ChessFM GGUF export.

    Configures the ChessFM Coach system prompt, sampling temperature, and
    family-specific chat template stop tokens for ``gemma4``, ``qwen3``, or
    ``qwen25``.
    """
    family = normalize_model_family(model_family)
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")

    stop_lines = "\n".join(f'PARAMETER stop "{tok}"' for tok in FAMILY_STOP_TOKENS[family])
    return (
        f"FROM {gguf_path}\n"
        f"\n"
        f'SYSTEM """{CHESS_COACH_SYSTEM_PROMPT}"""\n'
        f"\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER num_ctx 4096\n"
        f"{stop_lines}\n"
    )


def find_llama_cpp_converter(llama_cpp_dir: str | None = None) -> Path | None:
    """Locate ``llama.cpp``'s ``convert_hf_to_gguf.py`` script if installed locally."""
    candidates: list[Path] = []
    if llama_cpp_dir:
        candidates.append(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "convert_hf_to_gguf.py")

    which_path = shutil.which("convert_hf_to_gguf.py")
    if which_path:
        candidates.append(Path(which_path))

    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a merged ChessFM checkpoint to GGUF and generate an Ollama Modelfile."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the merged Hugging Face checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for GGUF weights and Ollama Modelfile",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        help="GGUF quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="gemma4",
        choices=["gemma4", "qwen3", "qwen25"],
        help="Model family for Ollama stop tokens (default: gemma4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Default Ollama inference temperature (default: 0.2)",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=None,
        help="Optional path to a local llama.cpp checkout containing convert_hf_to_gguf.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Modelfile and conversion plan without writing files or running llama.cpp",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    family = normalize_model_family(args.family)
    quant = args.quant.lower()

    model_slug = Path(args.model).name or f"chessfm-{family}"
    gguf_filename = f"{model_slug}-{quant}.gguf"
    gguf_path = args.output / gguf_filename
    modelfile_path = args.output / "Modelfile"

    modelfile_content = build_ollama_modelfile(
        gguf_path=f"./{gguf_filename}",
        model_family=family,
        temperature=args.temperature,
    )

    converter = find_llama_cpp_converter(args.llama_cpp_dir)

    if args.dry_run:
        print(f"[dry-run] Output directory: {args.output}")
        print(f"[dry-run] Target GGUF:      {gguf_path}")
        print(f"[dry-run] Target Modelfile: {modelfile_path}")
        print(f"[dry-run] llama.cpp found:  {converter or 'no'}")
        print("\n--- Modelfile preview ---")
        print(modelfile_content, end="")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"Wrote Ollama Modelfile to {modelfile_path}")

    if converter is None:
        print(
            "llama.cpp convert_hf_to_gguf.py not found locally. To finish GGUF export, run:\n"
            f"  git clone https://github.com/ggml-org/llama.cpp && python llama.cpp/convert_hf_to_gguf.py {args.model} --outfile {gguf_path} --outtype {quant}\n"
            f"  ollama create chessfm-{family} -f {modelfile_path}"
        )
        return 0

    outtype = quant if quant in {"f32", "f16", "bf16", "q8_0"} else "f16"
    raw_gguf = gguf_path if outtype == quant else args.output / f"{model_slug}-f16.gguf"
    convert_cmd = [
        sys.executable,
        str(converter),
        args.model,
        "--outfile",
        str(raw_gguf),
        "--outtype",
        outtype,
    ]
    print(f">>> {' '.join(convert_cmd)}", flush=True)
    subprocess.run(convert_cmd, check=True)

    if raw_gguf != gguf_path:
        quantizer = shutil.which("llama-quantize")
        if quantizer:
            quant_cmd = [quantizer, str(raw_gguf), str(gguf_path), quant.upper()]
            print(f">>> {' '.join(quant_cmd)}", flush=True)
            subprocess.run(quant_cmd, check=True)
        else:
            print(
                f"Converted f16 GGUF to {raw_gguf}. Quantize to {quant} with:\n"
                f"  llama-quantize {raw_gguf} {gguf_path} {quant.upper()}\n"
                f"  ollama create chessfm-{family} -f {modelfile_path}"
            )
            return 0

    print(
        f"Export complete! Register in Ollama with:\n  ollama create chessfm-{family} -f {modelfile_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
