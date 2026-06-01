#!/usr/bin/env python3
"""
Upload a local file or folder to a HuggingFace dataset repo.

Usage:
    python hf_upload.py --from path/to/dir  --to username/dataset
    python hf_upload.py --from path/to/file --to username/dataset --path-in-repo data/
    python hf_upload.py --from path/to/dir  --to username/dataset --dry-run

    python hf_upload.py --from outputs/ --to nguyenthanhdo/attribscope --path-in-repo nscc --dry-run
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi

LARGE_THRESHOLD = -1
NUM_WORKERS     = 8


def collect_files(src: Path, remote_dir: str) -> list[tuple[Path, str]]:
    """Return (local_path, repo_path) pairs for src (file or folder)."""
    if src.is_file():
        return [(src, f"{remote_dir}/{src.name}".lstrip("/"))]
    return [
        (local, f"{remote_dir}/{local.relative_to(src)}".lstrip("/"))
        for local in sorted(src.rglob("*"))
        if local.is_file()
    ]


def main():
    parser = argparse.ArgumentParser(description="Upload a file or folder to a HuggingFace dataset.")
    parser.add_argument("--from", dest="src", required=True,
                        help="Local file or folder to upload")
    parser.add_argument("--to", dest="repo", required=True,
                        help="HuggingFace dataset repo (username/dataset_name)")
    parser.add_argument("--path-in-repo", default=None,
                        help="Remote directory in the repo (default: src name)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    parser.add_argument("--private", action="store_true",
                        help="Create the dataset as private if it doesn't exist")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (defaults to $HF_TOKEN)")
    args = parser.parse_args()

    src = Path(args.src)
    assert src.exists(), f"Source path does not exist: {src}"

    remote_dir = src.name
    if args.path_in_repo is not None:
        remote_dir = f"{args.path_in_repo}/{src.name}"

    files = collect_files(src, remote_dir)
    assert files, f"No files found under {src}"

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Source : {src.resolve()}")
    print(f"{prefix}Target : hf://datasets/{args.repo}/{remote_dir}/")
    print(f"{prefix}Files  : {len(files)}\n")
    for _, repo_path in files:
        print(f"  {'(would upload) ' if args.dry_run else ''}{repo_path}")

    if args.dry_run:
        print(f"\n{prefix}Source : {src.resolve()}")
        print(f"{prefix}Target : hf://datasets/{args.repo}/{remote_dir}/")
        print(f"{prefix}Files  : {len(files)}\n")
        print("Dry run complete. Remove --dry-run to upload.")
        return

    assert args.token, "No HF token found. Set $HF_TOKEN or pass --token."
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"\nUploading {len(files)} file(s) to {args.repo} ...")
    if src.is_file():
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=files[0][1],
            repo_id=args.repo,
            repo_type="dataset",
        )
    elif LARGE_THRESHOLD > 0 and len(files) > LARGE_THRESHOLD:
        print(f"({len(files)} files exceeds threshold of {LARGE_THRESHOLD}, using upload_large_folder)")
        print("HfApi.upload_large_folder() does not support 'path_in_repo' argument")
        api.upload_large_folder(
            folder_path=str(src),
            # path_in_repo=remote_dir,
            repo_id=args.repo,
            repo_type="dataset",
            num_workers=NUM_WORKERS,
        )
    else:
        api.upload_folder(
            folder_path=str(src),
            path_in_repo=remote_dir,
            repo_id=args.repo,
            repo_type="dataset",
        )
    print("Done.")


if __name__ == "__main__":
    main()