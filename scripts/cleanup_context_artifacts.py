"""Preview or apply retention cleanup for disk-backed context artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.memory.artifacts import ContextArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("data/context_artifacts"))
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument(
        "--apply", action="store_true",
        help="delete expired artifacts; without this flag the command is a dry-run",
    )
    args = parser.parse_args()
    result = ContextArtifactStore(args.artifact_dir).cleanup_expired(
        max_age_days=args.max_age_days, apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
