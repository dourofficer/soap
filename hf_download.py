#!/usr/bin/env python3
"""
Download a file or folder from a HuggingFace dataset repo.

Usage:
    python hf_download.py --from username/dataset --path-in-repo data/ --to local/dir
    python hf_download.py --from username/dataset --path-in-repo data/file.csv --to .
    python hf_download.py --from username/dataset --path-in-repo nscc --to outputs/ --dry-run
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a file or folder from a HuggingFace dataset.")
    parser.add_argument("--from", dest="repo", required=True,
                        help="HuggingFace dataset repo (username/dataset_name)")
    parser.add_argument("--path-in-repo", required=True,
                        help="Remote file or folder path in the repo")
    parser.add_argument("--to", dest="dst", default=".",
                        help="Local destination directory (default: current dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (defaults to $HF_TOKEN)")
    args = parser.parse_args()

    dst = Path(args.dst)
    prefix = "[DRY RUN] " if args.dry_run else ""

    api = HfApi(token=args.token)

    # List all matching files in the repo
    all_files = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    remote = args.path_in_repo.rstrip("/")
    files = [f for f in all_files if f == remote or f.startswith(remote + "/")]
    assert files, f"No files found at '{remote}' in {args.repo}"

    print(f"{prefix}Source : hf://datasets/{args.repo}/{remote}")
    print(f"{prefix}Target : {dst.resolve()}")
    print(f"{prefix}Files  : {len(files)}\n")
    for f in files:
        print(f"  {'(would download) ' if args.dry_run else ''}{f}")

    if args.dry_run:
        print("\nDry run complete. Remove --dry-run to download.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(files)} file(s) from {args.repo} ...")

    for repo_path in files:
        # Preserve relative structure under the requested path
        rel = Path(repo_path).relative_to(remote) if repo_path != remote else Path(repo_path).name
        local_path = dst / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=args.repo,
            filename=repo_path,
            repo_type="dataset",
            local_dir=str(dst),
            token=args.token,
        )

    print("Done.")


if __name__ == "__main__":
    main()