from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish versioned StyleMatch model and index artifacts.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--index-repo", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--publish", action="store_true", help="Actually upload; otherwise print a dry-run plan.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model_dir, args.index_dir, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    plan = {
        "version": args.version,
        "model": {"local": str(args.model_dir), "repo": args.model_repo, "repo_type": "model"},
        "index": {"local": str(args.index_dir), "repo": args.index_repo, "repo_type": "dataset"},
        "manifest": str(args.manifest),
        "publish": args.publish,
    }
    print(json.dumps(plan, indent=2))
    if not args.publish:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.model_repo, repo_type="model", exist_ok=True, private=True)
    api.create_repo(args.index_repo, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.model_repo,
        repo_type="model",
        revision="main",
        commit_message=f"StyleMatch model {args.version}",
    )
    api.upload_folder(
        folder_path=args.index_dir,
        repo_id=args.index_repo,
        repo_type="dataset",
        revision="main",
        commit_message=f"StyleMatch index {args.version}",
    )
    api.upload_file(
        path_or_fileobj=args.manifest,
        path_in_repo="artifact_manifest.json",
        repo_id=args.index_repo,
        repo_type="dataset",
        commit_message=f"StyleMatch manifest {args.version}",
    )


if __name__ == "__main__":
    main()
