#!/usr/bin/env python3
"""Upload ChessFM checkpoints, datasets, or eval manifests to Hugging Face Hub.

Reads ``HF_TOKEN`` strictly from the environment and never prints, echoes, or
logs credential values.

Usage:
    python scripts/upload_hf.py \\
        --local-dir checkpoints/sft_qwen3_4b/final_merged \\
        --repo-id winsonly/chessfm-qwen3-4b-sft \\
        --commit-message "Upload Qwen3-4B completion-masked SFT checkpoint"

    python scripts/upload_hf.py --local-dir results --repo-id winsonly/chessfm-evals --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def upload_to_hub(
    *,
    local_dir: Path,
    repo_id: str,
    repo_type: str = "model",
    path_in_repo: str = "",
    commit_message: str = "Upload ChessFM artifacts",
    private: bool = False,
    dry_run: bool = False,
) -> dict[str, str | bool]:
    """Upload a local directory or file to Hugging Face Hub.

    Returns a structured summary dictionary (never containing any token).
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")
    if repo_type not in ("model", "dataset", "space"):
        raise ValueError(
            f"Unsupported repo_type {repo_type!r}; expected 'model', 'dataset', or 'space'."
        )

    summary: dict[str, str | bool] = {
        "local_dir": str(local_path),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
        "private": private,
        "dry_run": dry_run,
    }

    if dry_run:
        print(
            f"[dry-run] Would upload {local_path} -> "
            f"hf://{repo_type}s/{repo_id}/{path_in_repo} ({commit_message})"
        )
        return summary

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN environment variable is required to upload to Hugging Face Hub.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    if local_path.is_file():
        target_in_repo = path_in_repo or local_path.name
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=target_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    else:
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=path_in_repo or None,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )

    print(f"Uploaded {local_path} to Hugging Face Hub ({repo_type}: {repo_id}).")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload ChessFM artifacts to Hugging Face Hub")
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local directory or file to upload",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g. winsonly/chessfm-qwen3-4b-sft)",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Hugging Face repository type",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Optional target subfolder or path inside the Hub repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload ChessFM artifacts",
        help="Commit message for the Hub upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repository as private if it does not exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print planned upload without contacting Hugging Face Hub",
    )
    args = parser.parse_args(argv)

    upload_to_hub(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        private=args.private,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
