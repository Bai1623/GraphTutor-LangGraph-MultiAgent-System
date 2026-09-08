"""Validate every file-backed golden evaluation dataset without calling an LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.golden_dataset import load_golden_suite


def validate_directory(golden_dir: Path) -> int:
    paths = sorted(golden_dir.glob("*.yaml"))
    if not paths:
        print(f"No golden datasets found in {golden_dir}", file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        try:
            suite = load_golden_suite(path)
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"FAIL {path.name}: {exc}")
            continue
        case_count = len(suite.get("cases", []))
        print(f"PASS {path.name}: suite={suite['suite']} cases={case_count}")

    print(f"Validated {len(paths)} dataset(s): {len(paths) - failures} passed, {failures} failed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Golden Dataset schemas")
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_GOLDEN_DIR,
        help="Directory containing Golden Dataset YAML files",
    )
    args = parser.parse_args()
    return validate_directory(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
