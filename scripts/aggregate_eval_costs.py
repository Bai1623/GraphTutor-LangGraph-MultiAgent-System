"""Aggregate cost and latency signals across evaluation JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "artifacts" / "eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
NUMERIC_FIELDS = (
    "total_tokens",
    "wall_time_ms",
    "tool_rounds",
    "retry_count",
    "adv_round",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Return grouped totals, means, P95s, and fallback rate for report data."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        suite = report.get("suite")
        if suite:
            groups.setdefault(str(suite), []).append(report)

    suites: dict[str, Any] = {}
    for suite, items in sorted(groups.items()):
        fields: dict[str, Any] = {}
        for field in NUMERIC_FIELDS:
            values = [
                number
                for item in items
                if (number := _number(item.get("metrics", {}).get("cost_latency", {}).get(field)))
                is not None
            ]
            fields[field] = {
                "total": round(sum(values), 2),
                "average": round(sum(values) / len(values), 2) if values else 0.0,
                "p95": _percentile(values, 0.95),
            }
        fallback_count = sum(
            bool(item.get("metrics", {}).get("cost_latency", {}).get("fallback_used"))
            for item in items
        )
        suites[suite] = {
            "reports": len(items),
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_count / len(items), 3) if items else 0.0,
            "cost_latency": fields,
        }
    return {"reports": len(reports), "suites": suites}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Evaluation Cost Summary", "", f"- Reports: `{summary['reports']}`", ""]
    for suite, data in summary["suites"].items():
        lines.extend([
            f"## {suite}",
            "",
            f"- Reports: `{data['reports']}`",
            f"- Fallback rate: `{data['fallback_rate']}`",
            "",
            "| Signal | Total | Average | P95 |",
            "| --- | ---: | ---: | ---: |",
        ])
        for field, values in data["cost_latency"].items():
            lines.append(f"| `{field}` | {values['total']} | {values['average']} | {values['p95']} |")
        lines.append("")
    return "\n".join(lines)


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read report '{path}': {exc}") from exc
        if isinstance(value, dict) and value.get("suite"):
            reports.append(value)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluation cost and latency reports")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR, help="Report directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    paths = sorted(input_dir.glob("*.json"))
    try:
        summary = aggregate_reports(_load_reports(paths))
    except ValueError as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"cost_summary_{stamp}.json"
    markdown_path = output_dir / f"cost_summary_{stamp}.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
