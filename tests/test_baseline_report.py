"""Tests for the reproducible project baseline report runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import run_baseline


def test_default_steps_cover_backend_and_frontend(tmp_path: Path) -> None:
    (tmp_path / "frontend").mkdir()

    steps = run_baseline.default_steps(tmp_path)

    assert [step.name for step in steps] == [
        "Environment doctor",
        "Backend tests and coverage",
        "Ruff",
        "Mypy",
        "Frontend lint",
        "Frontend typecheck",
        "Frontend build",
    ]
    assert steps[0].cwd == tmp_path
    assert steps[-1].cwd == tmp_path / "frontend"
    assert dict(steps[1].env)["OTEL_TRACING_ENABLED"] == "false"
    assert dict(steps[-1].env)["NEXT_TELEMETRY_DISABLED"] == "1"


def test_run_step_records_failure_and_output(tmp_path: Path) -> None:
    step = run_baseline.BaselineStep("Example", ("tool", "check"), tmp_path)

    def fake_runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["tool", "check"],
            returncode=2,
            stdout="failure detail\n",
            stderr="",
        )

    result = run_baseline.run_step(step, runner=fake_runner)

    assert result.status == "FAIL"
    assert result.exit_code == 2
    assert result.command == "tool check"
    assert result.output == "failure detail"


def test_build_report_collects_warnings_and_summary(tmp_path: Path) -> None:
    results = [
        run_baseline.StepResult(
            name="Doctor",
            command="doctor",
            cwd=str(tmp_path),
            status="PASS",
            exit_code=0,
            duration_seconds=0.1,
            output="[WARN] Vector index: not built",
        ),
        run_baseline.StepResult(
            name="Tests",
            command="pytest",
            cwd=str(tmp_path),
            status="FAIL",
            exit_code=1,
            duration_seconds=1.2,
            output="one failed",
        ),
    ]

    report = run_baseline.build_report(
        results,
        git_commit="abc123",
        generated_at="2026-08-23T12:00:00+00:00",
    )

    assert report["summary"] == {"total": 2, "passed": 1, "failed": 1}
    assert report["warnings"] == ["Vector index: not built"]
    assert report["git_commit"] == "abc123"
    assert "| Tests | FAIL | 1.2 | `pytest` |" in run_baseline.render_markdown(report)


def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "generated_at": "2026-08-23T12:00:00+00:00",
        "git_commit": "abc123",
        "summary": {"total": 1, "passed": 1, "failed": 0},
        "warnings": [],
        "steps": [],
    }

    json_path, markdown_path = run_baseline.write_report(
        report,
        output_dir=tmp_path,
        stamp="20260823_200000",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["git_commit"] == "abc123"
    assert markdown_path.read_text(encoding="utf-8").startswith("# GraphTutor Baseline Report")


def test_restore_generated_file_returns_it_to_original_content(tmp_path: Path) -> None:
    path = tmp_path / "next-env.d.ts"
    path.write_text("before", encoding="utf-8")
    snapshot = run_baseline.snapshot_file(path)
    path.write_text("after", encoding="utf-8")

    run_baseline.restore_file(path, snapshot)

    assert path.read_text(encoding="utf-8") == "before"
