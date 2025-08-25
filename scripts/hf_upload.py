#!/usr/bin/env python3
"""
Upload local evaluation data to Hugging Face Hub, excluding diskann_rpj_wiki.

Defaults:
- repo_id: LEANN-RAG/leann-rag-evaluation-data (dataset)
- folder_path: benchmarks/data
- ignore_patterns: diskann_rpj_wiki/** and .cache/**

Requires authentication via `huggingface-cli login` or HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
import os

try:
    from huggingface_hub import HfApi
except Exception as e:
    raise SystemExit(
        "huggingface_hub is required. Install with: pip install huggingface_hub hf_transfer"
    ) from e


def _enable_transfer_accel_if_available() -> None:
    """Best-effort enabling of accelerated transfers across hub versions.

    Tries the public util if present; otherwise, falls back to env flag when
    hf_transfer is installed. Silently no-ops if unavailable.
    """
    try:
        # Newer huggingface_hub exposes this under utils
        from huggingface_hub.utils import hf_hub_enable_hf_transfer  # type: ignore

        hf_hub_enable_hf_transfer()
        return
    except Exception:
        pass

    try:
        # If hf_transfer is installed, set env flag recognized by the hub
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except Exception:
        # Acceleration not available; proceed without it
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload local data to HF, excluding diskann_rpj_wiki")
    p.add_argument(
        "--repo-id",
        default="LEANN-RAG/leann-rag-evaluation-data",
        help="Target dataset repo id (namespace/name)",
    )
    p.add_argument(
        "--folder-path",
        default="benchmarks/data",
        help="Local folder to upload (default: benchmarks/data)",
    )
    p.add_argument(
        "--ignore",
        default=["diskann_rpj_wiki/**", ".cache/**"],
        nargs="+",
        help="Glob patterns to ignore (space-separated)",
    )
    p.add_argument(
        "--allow",
        default=["**"],
        nargs="+",
        help="Glob patterns to allow (space-separated). Defaults to everything.",
    )
    p.add_argument(
        "--message",
        default="sync local data (exclude diskann_rpj_wiki)",
        help="Commit message",
    )
    p.add_argument(
        "--no-transfer-accel",
        action="store_true",
        help="Disable hf_transfer accelerated uploads",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_transfer_accel:
        _enable_transfer_accel_if_available()

    if not os.path.isdir(args.folder_path):
        raise SystemExit(f"Folder not found: {args.folder_path}")

    print("Uploading to Hugging Face Hub:")
    print(f"  repo_id:        {args.repo_id}")
    print("  repo_type:      dataset")
    print(f"  folder_path:    {args.folder_path}")
    print(f"  allow_patterns: {args.allow}")
    print(f"  ignore_patterns:{args.ignore}")

    api = HfApi()

    # Perform upload. This skips unchanged files by content hash.
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder_path,
        path_in_repo=".",
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        commit_message=args.message,
    )

    print("Upload completed (unchanged files were skipped by the Hub).")


if __name__ == "__main__":
    main()
