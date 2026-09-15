"""Small reproducibility helpers shared by offline evaluation commands."""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    """Seed Python's built-in RNG used by deterministic evaluation fixtures."""
    random.seed(seed)


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_hashes(project_root: Path) -> dict[str, str]:
    config_dir = project_root / "config"
    hashes: dict[str, str] = {}
    if not config_dir.is_dir():
        return hashes
    for path in sorted(p for p in config_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.relative_to(project_root).as_posix()] = digest
    return hashes


def build_snapshot(project_root: Path, seed: int) -> dict[str, Any]:
    """Return a JSON-safe experiment snapshot without exposing environment secrets."""
    return {
        "seed": seed,
        "git_commit": _git_commit(project_root),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "config_sha256": _config_hashes(project_root),
    }
