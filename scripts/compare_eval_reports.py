"""Compare two JSON evaluation reports produced by ``run_eval.py``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _numeric_metrics(value: Any, prefix: str = "metrics") -> dict[str, float]:
    """Flatten numeric metric leaves while ignoring booleans and metadata."""
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        flattened: dict[str, float] = {}
        for key, child in value.items():
            flattened.update(_numeric_metrics(child, f"{prefix}.{key}"))
        return flattened
    return {}


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read report '{path}': {exc}") from exc
    if not isinstance(report, dict) or not report.get("suite"):
        raise ValueError(f"Report '{path}' must be a JSON object with a suite")
    return report


def compare_reports(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe comparison and reject incomparable datasets."""
    if baseline.get("suite") != variant.get("suite"):
        raise ValueError("Baseline and variant must use the same suite")
    baseline_version = baseline.get("dataset", {}).get("version")
    variant_version = variant.get("dataset", {}).get("version")
    if baseline_version != variant_version:
        raise ValueError("Baseline and variant must use the same dataset version")

    baseline_metrics = _numeric_metrics(baseline.get("metrics", {}))
    variant_metrics = _numeric_metrics(variant.get("metrics", {}))
    rows = []
    for name in sorted(set(baseline_metrics) | set(variant_metrics)):
        base = baseline_metrics.get(name)
        current = variant_metrics.get(name)
        rows.append({
            "metric": name,
            "baseline": base,
            "variant": current,
            "delta": round(current - base, 6) if base is not None and current is not None else None,
        })
    return {
        "suite": baseline["suite"],
        "dataset_version": baseline_version or "unknown",
        "baseline": {
            "path": baseline.get("golden_path"),
            "git_commit": baseline.get("experiment", {}).get("git_commit"),
            "seed": baseline.get("experiment", {}).get("seed"),
        },
        "variant": {
            "path": variant.get("golden_path"),
            "git_commit": variant.get("experiment", {}).get("git_commit"),
            "seed": variant.get("experiment", {}).get("seed"),
        },
        "metrics": rows,
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# Experiment Comparison: {comparison['suite']}",
        "",
        f"- Dataset version: `{comparison['dataset_version']}`",
        f"- Baseline commit: `{comparison['baseline'].get('git_commit') or 'unknown'}`",
        f"- Variant commit: `{comparison['variant'].get('git_commit') or 'unknown'}`",
        f"- Baseline seed: `{comparison['baseline'].get('seed') or 'unknown'}`",
        f"- Variant seed: `{comparison['variant'].get('seed') or 'unknown'}`",
        "",
        "| Metric | Baseline | Variant | Delta (variant - baseline) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison["metrics"]:
        values = [row.get("baseline"), row.get("variant"), row.get("delta")]
        formatted = ["n/a" if value is None else str(value) for value in values]
        lines.append(f"| `{row['metric']}` | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two offline evaluation reports")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        comparison = compare_reports(load_report(args.baseline), load_report(args.variant))
    except ValueError as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{comparison['suite']}_comparison_{stamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
