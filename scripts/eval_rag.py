# -*- coding: utf-8 -*-
"""Backward-compatible entry point for the RAG retrieval eval.

The golden cases now live in ``eval/golden/rag_retrieval.yaml`` and the
canonical harness is ``scripts/run_eval.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import load_suite, render_markdown, run_rag_suite


def main() -> int:
    suite = load_suite("rag")
    result = run_rag_suite(suite)

    report_path = PROJECT_ROOT / "data" / "eval_report.md"
    json_path = PROJECT_ROOT / "data" / "eval_result.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完整报告已保存: {report_path}")
    print(f"JSON 结果已保存: {json_path}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
