from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an immutable deployment index with a Hub model reference.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--release-gates", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gates = json.loads(args.release_gates.read_text(encoding="utf-8"))
    if not gates.get("private_beta_ready"):
        raise ValueError("Deployment bundle refused: private-beta gates have not passed")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {args.output_dir}")
    shutil.copytree(args.index_dir, args.output_dir)
    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "model_name": args.model_repo,
        "artifact_version": args.artifact_version,
        "release_status": "private_beta_ready",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(args.release_gates, args.output_dir / "release_gates.json")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "model_name": args.model_repo,
        "artifact_version": args.artifact_version,
    }, indent=2))


if __name__ == "__main__":
    main()
