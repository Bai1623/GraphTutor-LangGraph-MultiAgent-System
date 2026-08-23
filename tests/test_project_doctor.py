"""Tests for the local development environment doctor."""

from __future__ import annotations

from pathlib import Path

from scripts import project_doctor


def _prepare_project(root: Path) -> None:
    (root / "frontend").mkdir()
    (root / "data" / "math").mkdir(parents=True)
    (root / "data" / "chinese").mkdir()
    (root / "data" / "english").mkdir()
    (root / "uv.lock").write_text("", encoding="utf-8")
    (root / "frontend" / "package-lock.json").write_text("{}", encoding="utf-8")
    for subject in ("math", "chinese", "english"):
        (root / "data" / subject / "sample.txt").write_text("sample", encoding="utf-8")


def test_collect_checks_redacts_environment_values(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    secret = "must-not-appear"

    checks = project_doctor.collect_checks(
        project_root=tmp_path,
        environ={"DEEPSEEK_API_KEY": secret},
        command_version=lambda command: "v20.20.2" if command == "node" else None,
    )
    rendered = project_doctor.render_checks(checks)

    assert "DEEPSEEK_API_KEY" in rendered
    assert secret not in rendered
    assert "SILICONFLOW_API_KEY" in rendered
    assert "AUTH_SECRET" in rendered
    assert any(check.status == "WARN" for check in checks)
    assert not any(check.status == "FAIL" for check in checks)


def test_collect_checks_reports_hard_toolchain_failures(tmp_path: Path) -> None:
    checks = project_doctor.collect_checks(
        project_root=tmp_path,
        environ={},
        command_version=lambda _command: None,
    )

    failures = {check.name for check in checks if check.status == "FAIL"}

    assert "Node.js" in failures
    assert "Python lockfile" in failures
    assert "Frontend lockfile" in failures
    assert "Knowledge data" in failures


def test_collect_checks_detects_simple_vector_index(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    index_path = tmp_path / "chroma_store" / "gaokao_docs.pkl"
    index_path.parent.mkdir()
    index_path.write_bytes(b"index")

    checks = project_doctor.collect_checks(
        project_root=tmp_path,
        environ={
            "DEEPSEEK_API_KEY": "configured",
            "SILICONFLOW_API_KEY": "configured",
            "AUTH_SECRET": "configured",
        },
        command_version=lambda command: "v20.20.2" if command == "node" else None,
    )

    index_check = next(check for check in checks if check.name == "Vector index")
    assert index_check.status == "PASS"
    assert index_check.detail == "chroma_store/gaokao_docs.pkl"


def test_exit_code_only_fails_for_hard_failures() -> None:
    assert project_doctor.exit_code([
        project_doctor.CheckResult("a", "PASS", "ok"),
        project_doctor.CheckResult("b", "WARN", "missing"),
    ]) == 0
    assert project_doctor.exit_code([
        project_doctor.CheckResult("a", "FAIL", "missing"),
    ]) == 1
