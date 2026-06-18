"""Encoding guardrails for human-readable Chinese harness files."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common mojibake markers produced when UTF-8 Chinese text is decoded with a
# legacy Windows code page. Keep these as escapes so this test cannot match
# itself if the checked paths are expanded later.
MOJIBAKE_MARKERS = {
    "\u951b": "锛",
    "\u9225": "鈥",
    "\u9a9e": "骞",
    "\u59ab": "妫",
    "\u7ef1": "绱",
    "\ufffd": "replacement character",
}

CHECKED_PATHS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "README_en.md",
    PROJECT_ROOT / "eval" / "golden",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "config" / "prompts",
]

CHECKED_GLOBS = [
    PROJECT_ROOT / "data" / "eval*.md",
    PROJECT_ROOT / "data" / "eval*.json",
]


def _iter_checked_files() -> list[Path]:
    files: list[Path] = []

    for path in CHECKED_PATHS:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".md", ".py", ".txt", ".xml", ".yaml", ".yml"}
            )

    for pattern in CHECKED_GLOBS:
        files.extend(pattern.parent.glob(pattern.name))

    return sorted(set(files))


def test_harness_text_files_do_not_contain_mojibake() -> None:
    violations: list[str] = []

    for path in _iter_checked_files():
        rel_path = path.relative_to(PROJECT_ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            violations.append(f"{rel_path}: not valid UTF-8 ({exc})")
            continue

        for marker, label in MOJIBAKE_MARKERS.items():
            index = text.find(marker)
            if index == -1:
                continue
            start = max(0, index - 20)
            end = min(len(text), index + 20)
            snippet = text[start:end].replace("\n", "\\n")
            violations.append(f"{rel_path}: found {label!r} near {snippet!r}")

    assert not violations, "Mojibake detected:\n" + "\n".join(violations)
