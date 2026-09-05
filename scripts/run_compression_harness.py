"""Run the context compression harness.

Default mode is deterministic/offline: each golden case provides the expected
compressed episode, so the harness can evaluate metrics without calling an LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.golden_dataset import load_golden_suite
from src.memory.artifacts import ContextArtifactStore
from src.memory.compression_harness import (
    episode_from_case,
    evaluate_compression_result,
    messages_from_case,
    result_from_static_episode,
)
from src.memory.compressor import compress_conversation


DEFAULT_SUITE = PROJECT_ROOT / "eval" / "golden" / "compression.yaml"


def _load_suite(path: Path) -> dict[str, Any]:
    return load_golden_suite(path)


def _write_case_artifacts(case: dict[str, Any], store: ContextArtifactStore) -> None:
    for artifact in case.get("artifacts") or []:
        artifact_id = str(artifact["artifact_id"])
        created_at = datetime.now().isoformat()
        day_dir = store.root / created_at[:10]
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{artifact_id}.json"
        path.write_text(
            json.dumps(
                {
                    "artifact_id": artifact_id,
                    "kind": artifact.get("kind", ""),
                    "created_at": created_at,
                    "metadata": artifact.get("metadata") or {},
                    "payload": artifact.get("payload"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


async def _run_case(
    case: dict[str, Any],
    *,
    thresholds: dict[str, float],
    store: ContextArtifactStore,
    use_llm: bool,
) -> dict[str, Any]:
    _write_case_artifacts(case, store)
    before_messages = messages_from_case(case.get("messages") or [])
    recent_count = int(case.get("recent_message_count", 1))
    recent_messages = before_messages[-recent_count:] if recent_count > 0 else []

    if use_llm:
        result = await compress_conversation(
            before_messages,
            recent_turns=max(1, recent_count),
            soft_limit_tokens=1,
        )
    else:
        result = result_from_static_episode(
            before_messages=before_messages,
            episode=episode_from_case(case.get("expected_episode")),
            recent_messages=recent_messages,
        )

    metrics = evaluate_compression_result(
        case_id=str(case["id"]),
        before_messages=before_messages,
        result=result,
        expected_constraints=list(case.get("expected_constraints") or []),
        answer_terms=list(case.get("answer_terms") or []),
        expected_artifact_ids=list(case.get("expected_artifact_ids") or []),
        thresholds=thresholds,
        store=store,
    )
    return {
        **metrics.to_dict(),
        "expected_artifact_ids": list(case.get("expected_artifact_ids") or []),
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    avg = {}
    for key in (
        "token_reduction",
        "constraint_retention",
        "answer_consistency",
        "artifact_recoverability",
    ):
        avg[key] = round(sum(float(item[key]) for item in results) / max(total, 1), 4)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / max(total, 1), 4),
        "average": avg,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Context Compression Harness Report",
        "",
        f"- Suite: `{report['suite']}`",
        f"- Dataset: `{report.get('dataset', {}).get('dataset_name', 'unknown')}`",
        f"- Dataset version: `{report.get('dataset', {}).get('version', 'unknown')}`",
        f"- Mode: `{report['mode']}`",
        f"- Total: {report['summary']['total']}",
        f"- Passed: {report['summary']['passed']}",
        f"- Failed: {report['summary']['failed']}",
        f"- Pass rate: {report['summary']['pass_rate']}",
        "",
        "## Averages",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["summary"]["average"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Pass | Token reduction | Constraint retention | Answer consistency | Artifact recoverability |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["cases"]:
        lines.append(
            "| {case_id} | {passed} | {token_reduction} | {constraint_retention} | "
            "{answer_consistency} | {artifact_recoverability} |".format(**item)
        )
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite)
    suite = _load_suite(suite_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ContextArtifactStore(Path(args.artifact_dir))
    thresholds = dict(suite.get("thresholds") or {})
    results = [
        await _run_case(
            case,
            thresholds=thresholds,
            store=store,
            use_llm=args.use_llm,
        )
        for case in suite.get("cases") or []
    ]
    report = {
        "suite": suite.get("suite", suite_path.stem),
        "dataset": suite.get("metadata", {}),
        "mode": "live_llm" if args.use_llm else "offline_static_episode",
        "thresholds": thresholds,
        "summary": _summary(results),
        "cases": results,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"compression_{stamp}.json"
    md_path = output_dir / f"compression_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["summary"]["failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context compression harness.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "artifacts" / "eval"))
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "artifacts" / "compression_harness_artifacts"),
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Call the real compressor LLM instead of using expected_episode fixtures.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
