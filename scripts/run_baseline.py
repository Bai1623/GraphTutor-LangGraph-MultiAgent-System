"""Run the reproducible offline engineering baseline and write reports."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "baseline"
Status = Literal["PASS", "FAIL"]
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class BaselineStep:
    """One independently executable baseline command."""

    name: str
    command: tuple[str, ...]
    cwd: Path
    env: tuple[tuple[str, str], ...] = ()
    timeout_seconds: int = 600


@dataclass(frozen=True)
class StepResult:
    """Serializable result of one baseline command."""

    name: str
    command: str
    cwd: str
    status: Status
    exit_code: int
    duration_seconds: float
    output: str


def default_steps(project_root: Path = PROJECT_ROOT) -> list[BaselineStep]:
    root = project_root.resolve()
    frontend = root / "frontend"
    return [
        BaselineStep(
            "Environment doctor",
            (sys.executable, "scripts/project_doctor.py"),
            root,
        ),
        BaselineStep(
            "Backend tests and coverage",
            (
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--ignore=tests/test_integration.py",
                "--cov",
                "--cov-report=term-missing",
                "-q",
            ),
            root,
            env=(("OTEL_TRACING_ENABLED", "false"),),
        ),
        BaselineStep("Ruff", ("ruff", "check", "."), root),
        BaselineStep("Mypy", ("mypy",), root),
        BaselineStep("Frontend lint", ("npm", "run", "lint"), frontend),
        BaselineStep("Frontend typecheck", ("npm", "run", "typecheck"), frontend),
        BaselineStep(
            "Frontend build",
            ("npm", "run", "build"),
            frontend,
            env=(("NEXT_TELEMETRY_DISABLED", "1"),),
        ),
    ]


def _combined_output(result: subprocess.CompletedProcess[str], limit: int = 8000) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output[-limit:]


def run_step(step: BaselineStep, *, runner: Runner = subprocess.run) -> StepResult:
    started = time.monotonic()
    command = shlex.join(step.command)
    environment = {**os.environ, **dict(step.env)}
    try:
        completed = runner(
            list(step.command),
            cwd=step.cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=step.timeout_seconds,
        )
        exit_code = completed.returncode
        output = _combined_output(completed)
    except FileNotFoundError as exc:
        exit_code = 127
        output = str(exc)
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        output = f"timed out after {step.timeout_seconds}s: {exc}"
    return StepResult(
        name=step.name,
        command=command,
        cwd=str(step.cwd),
        status="PASS" if exit_code == 0 else "FAIL",
        exit_code=exit_code,
        duration_seconds=round(time.monotonic() - started, 2),
        output=output,
    )


def build_report(
    results: list[StepResult],
    *,
    git_commit: str,
    generated_at: str,
) -> dict:
    warnings = []
    for result in results:
        warnings.extend(
            line.removeprefix("[WARN] ")
            for line in result.output.splitlines()
            if line.startswith("[WARN] ")
        )
    passed = sum(result.status == "PASS" for result in results)
    return {
        "generated_at": generated_at,
        "git_commit": git_commit,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "warnings": warnings,
        "steps": [asdict(result) for result in results],
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# GraphTutor Baseline Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Passed: {summary['passed']}/{summary['total']}",
        f"- Failed: {summary['failed']}",
        "",
        "## Steps",
        "",
        "| Step | Status | Seconds | Command |",
        "|---|---:|---:|---|",
    ]
    for step in report["steps"]:
        command = step["command"].replace("|", "\\|")
        lines.append(
            f"| {step['name']} | {step['status']} | {step['duration_seconds']} | `{command}` |"
        )

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Command Output", ""])
    for step in report["steps"]:
        lines.extend([
            f"### {step['name']}",
            "",
            "```text",
            step["output"] or "(no output)",
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: dict,
    *,
    output_dir: Path,
    stamp: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"baseline_{stamp}.json"
    markdown_path = output_dir / f"baseline_{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def snapshot_file(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def restore_file(path: Path, snapshot: bytes | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
    elif not path.is_file() or path.read_bytes() != snapshot:
        path.write_bytes(snapshot)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def run_baseline(
    *,
    project_root: Path = PROJECT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    next_env_path = project_root / "frontend" / "next-env.d.ts"
    next_env_snapshot = snapshot_file(next_env_path)
    results = []
    try:
        for step in default_steps(project_root):
            print(f"Running {step.name} ...", flush=True)
            result = run_step(step)
            results.append(result)
            print(f"  {result.status} ({result.duration_seconds}s)", flush=True)
    finally:
        restore_file(next_env_path, next_env_snapshot)

    now = datetime.now(UTC)
    report = build_report(
        results,
        git_commit=_git_commit(project_root),
        generated_at=now.isoformat(),
    )
    json_path, markdown_path = write_report(
        report,
        output_dir=output_dir,
        stamp=now.strftime("%Y%m%d_%H%M%S"),
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 1 if report["summary"]["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline engineering baseline.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    raise SystemExit(run_baseline(output_dir=args.output))


if __name__ == "__main__":
    main()
