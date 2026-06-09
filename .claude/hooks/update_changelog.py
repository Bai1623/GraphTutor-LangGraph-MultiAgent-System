"""Hook script: 每次 git commit 后自动追加记录到 CHANGES.md"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHANGES_FILE = PROJECT_DIR / "CHANGES.md"


def main():
    # 获取最新提交信息
    try:
        hash_ = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        ).stdout.strip()
        date = subprocess.run(
            ["git", "log", "-1", "--format=%as"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        ).stdout.strip()
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, cwd=PROJECT_DIR,
        ).stdout.strip()
    except Exception:
        sys.exit(1)

    if not hash_ or not msg:
        sys.exit(1)

    # 跳过 hook 自身的提交
    if "更新CHANGES.md" in msg or "CHANGES.md" in msg:
        sys.exit(0)

    # 检查是否是语义缓存/PDF导出等已记录的提交（避免重复）
    new_line = f"| {date} | `{hash_}` | {msg} |\n"

    try:
        lines = CHANGES_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError:
        sys.exit(1)

    # 插入到第3行（第1行标题，第2行空行，第3行开始是内容）
    insert_at = 2
    # 检查是否已存在相同 hash 的行（防重复）
    already_exists = any(f"`{hash_}`" in line for line in lines)
    if already_exists:
        sys.exit(0)

    lines.insert(insert_at, new_line)
    CHANGES_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"[Hook] CHANGES.md updated: {hash_} {msg}")


if __name__ == "__main__":
    main()
