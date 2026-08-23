"""Check whether the local machine can develop and run GraphTutor."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
Status = Literal["PASS", "WARN", "FAIL"]
CommandVersion = Callable[[str], str | None]


@dataclass(frozen=True)
class CheckResult:
    """One environment check with a redacted, user-facing detail."""

    name: str
    status: Status
    detail: str


def _command_version(command: str) -> str | None:
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip() or None


def _major_version(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(?:^|\s)v?(\d+)(?:\.\d+)", value)
    return int(match.group(1)) if match else None


def _effective_environment(
    project_root: Path,
    environ: Mapping[str, str] | None,
) -> dict[str, str | None]:
    values: dict[str, str | None] = dict(dotenv_values(project_root / ".env"))
    values.update(os.environ if environ is None else environ)
    return values


def collect_checks(
    *,
    project_root: Path = PROJECT_ROOT,
    environ: Mapping[str, str] | None = None,
    command_version: CommandVersion = _command_version,
) -> list[CheckResult]:
    """Collect local prerequisites without exposing environment values."""
    root = project_root.resolve()
    checks: list[CheckResult] = []

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    python_ok = sys.version_info >= (3, 11)
    checks.append(CheckResult(
        "Python",
        "PASS" if python_ok else "FAIL",
        f"{python_version} (required >=3.11)",
    ))

    node_version = command_version("node")
    node_major = _major_version(node_version)
    node_ok = node_major is not None and node_major >= 18
    checks.append(CheckResult(
        "Node.js",
        "PASS" if node_ok else "FAIL",
        f"{node_version} (recommended 20)" if node_version else "not found",
    ))

    required_files = (
        ("Python lockfile", root / "uv.lock"),
        ("Frontend lockfile", root / "frontend" / "package-lock.json"),
    )
    for name, path in required_files:
        checks.append(CheckResult(
            name,
            "PASS" if path.is_file() else "FAIL",
            str(path.relative_to(root)) if path.is_file() else "missing",
        ))

    subject_dirs = [root / "data" / subject for subject in ("math", "chinese", "english")]
    data_ok = all(path.is_dir() and any(item.is_file() for item in path.iterdir()) for path in subject_dirs)
    checks.append(CheckResult(
        "Knowledge data",
        "PASS" if data_ok else "FAIL",
        "math/chinese/english present" if data_ok else "one or more subject directories are empty",
    ))

    effective_env = _effective_environment(root, environ)
    key_names = ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "AUTH_SECRET")
    configured_keys = [name for name in key_names if effective_env.get(name)]
    missing_keys = [name for name in key_names if not effective_env.get(name)]
    detail = "configured: " + ", ".join(configured_keys) if configured_keys else "none configured"
    if missing_keys:
        detail += "; missing: " + ", ".join(missing_keys)
    checks.append(CheckResult(
        "Live credentials",
        "WARN" if missing_keys else "PASS",
        detail,
    ))

    index_candidates = (
        root / "chroma_store" / "gaokao_docs.pkl",
        root / "chroma_store" / "chroma.sqlite3",
    )
    index_path = next((path for path in index_candidates if path.is_file()), None)
    checks.append(CheckResult(
        "Vector index",
        "PASS" if index_path else "WARN",
        str(index_path.relative_to(root)) if index_path else "not built",
    ))

    return checks


def render_checks(checks: list[CheckResult]) -> str:
    lines = [f"[{check.status}] {check.name}: {check.detail}" for check in checks]
    counts = {status: sum(check.status == status for check in checks) for status in ("PASS", "WARN", "FAIL")}
    lines.append(
        "Summary: {PASS} passed, {WARN} warnings, {FAIL} failures".format(**counts)
    )
    return "\n".join(lines)


def exit_code(checks: list[CheckResult]) -> int:
    return 1 if any(check.status == "FAIL" for check in checks) else 0


def main() -> int:
    checks = collect_checks()
    print(render_checks(checks))
    return exit_code(checks)


if __name__ == "__main__":
    raise SystemExit(main())
