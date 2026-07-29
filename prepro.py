"""Deterministic preprocessing for the NETtalk reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nettalk.data import build_repro_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NETtalk reproducible preprocessing artifacts.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current working directory).",
    )
    args = parser.parse_args()

    manifest = build_repro_artifacts(args.repo_root)
    repro_dir = args.repo_root.resolve() / "data" / "processed" / "repro"
    manifest_path = repro_dir / "manifest.json"

    print("Preprocessing complete.")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest["counts"], indent=2))
    print("Window sizes prepared:", ", ".join(sorted(manifest["window_sizes"].keys())))


if __name__ == "__main__":
    main()
