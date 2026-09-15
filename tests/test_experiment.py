"""Tests for evaluation reproducibility metadata."""

from pathlib import Path

from src.evaluation.experiment import build_snapshot, seed_everything


def test_snapshot_hashes_config_without_reading_env(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text("temperature: 0\n", encoding="utf-8")

    snapshot = build_snapshot(tmp_path, 123)

    assert snapshot["seed"] == 123
    assert snapshot["config_sha256"]["config/settings.yaml"]
    assert "env" not in snapshot


def test_seed_everything_is_repeatable() -> None:
    import random

    seed_everything(7)
    first = [random.random() for _ in range(3)]
    seed_everything(7)
    second = [random.random() for _ in range(3)]

    assert first == second
