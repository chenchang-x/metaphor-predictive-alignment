#!/usr/bin/env python3
"""Score the frozen Qwen3.5-9B RQ2 controls and formal MUNCH items."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_csv, render_json, write_text_atomic
from qwen_alignment.direct_judgement import (
    CONTROL_COUNT,
    CONTROL_ORDER_COLUMNS,
    CONTROL_RESULT_COLUMNS,
    JUDGEMENT_CONTEXTS,
    MANIFEST_SCHEMA,
    ORDER_COLUMNS,
    SUMMARY_SCHEMA,
    TARGET_COLUMNS,
    TRIPLE_COLUMNS,
    DirectJudgementError,
    aggregate_formal_targets,
    aggregate_order_balanced,
    attach_formal_order_metadata,
    build_control_outputs,
    build_formal_triples,
    concise_runtime_identity,
    prompt_item_from_analysis_row,
    prompt_item_from_control,
    score_prompt_items,
    validate_formal_rows,
)
from qwen_alignment.model_runtime import DEFAULT_MODEL_ID, DEFAULT_REVISION, resolve_runtime


CONTROL_INPUT_SCHEMA = "munch-qwen3.5-9b-direct-controls/v1"


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DirectJudgementError(f"cannot read {path}") from exc
    if not rows:
        raise DirectJudgementError(f"CSV has no rows: {path}")
    return rows


def read_controls(path: Path) -> list[Mapping[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectJudgementError(f"cannot read controls: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CONTROL_INPUT_SCHEMA:
        raise DirectJudgementError("control file schema mismatch")
    controls = payload.get("controls")
    if not isinstance(controls, list) or len(controls) != CONTROL_COUNT:
        raise DirectJudgementError(f"control file must contain {CONTROL_COUNT} examples")
    identifiers = [str(row.get("control_id", "")) for row in controls if isinstance(row, dict)]
    if len(identifiers) != CONTROL_COUNT or len(set(identifiers)) != CONTROL_COUNT:
        raise DirectJudgementError("control IDs are missing or duplicated")
    return controls


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"munch-qwen3p5-9b-direct-{timestamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("controls", "formal"),
        default="controls",
        help="formal always runs the 12 controls first with the same loaded runtime",
    )
    parser.add_argument(
        "--analysis-items",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "analysis_items.csv",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=PROJECT_ROOT / "data" / "controls" / "qwen_instruct_direct_controls.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "huggingface",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _progress(label: str):
    def report(completed: int, total: int) -> None:
        if completed == 1 or completed == total or completed % 50 == 0:
            print(f"{label}={completed}/{total}", flush=True)

    return report


def run(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Any]]:
    controls_path = args.controls.resolve()
    control_rows = read_controls(controls_path)
    control_items = [prompt_item_from_control(row) for row in control_rows]
    analysis_path = args.analysis_items.resolve()
    formal_rows: list[dict[str, str]] = []
    if args.scope == "formal":
        formal_rows = read_csv(analysis_path)
        validate_formal_rows(formal_rows)

    runtime = resolve_runtime(
        args.cache_dir.resolve(),
        args.device,
        allow_download=bool(args.allow_download),
    )
    run_id = args.run_id or default_run_id()
    if not run_id.startswith("munch-qwen3p5-9b-"):
        raise DirectJudgementError("run_id does not use the Qwen3.5-9B namespace")

    control_order_raw: list[dict[str, object]] = []
    for judgement_context in JUDGEMENT_CONTEXTS:
        control_order_raw.extend(
            score_prompt_items(
                control_items,
                runtime,
                judgement_context=judgement_context,
                progress=_progress(f"controls_{judgement_context}_scored"),
            )
        )
    control_balanced = aggregate_order_balanced(control_order_raw)
    control_order, control_results, control_gate = build_control_outputs(
        run_id,
        control_items,
        control_order_raw,
        control_balanced,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "control_order": output_dir / "direct_controls_order.csv",
        "control_results": output_dir / "direct_controls.csv",
        "summary": output_dir / "direct_judgement_summary.json",
        "manifest": output_dir / "direct_judgement_manifest.json",
    }
    outputs: dict[str, dict[str, object]] = {
        "control_order": {"file": paths["control_order"].name, "rows": len(control_order)},
        "control_results": {
            "file": paths["control_results"].name,
            "rows": len(control_results),
        },
    }
    formal_counts: dict[str, int] | None = None
    if args.scope == "formal":
        formal_items = [prompt_item_from_analysis_row(row) for row in formal_rows]
        formal_order_raw: list[dict[str, object]] = []
        for judgement_context in JUDGEMENT_CONTEXTS:
            formal_order_raw.extend(
                score_prompt_items(
                    formal_items,
                    runtime,
                    judgement_context=judgement_context,
                    progress=_progress(
                        f"formal_{judgement_context}_triples_scored"
                    ),
                )
            )
        formal_balanced = aggregate_order_balanced(formal_order_raw)
        formal_order = attach_formal_order_metadata(
            run_id, formal_rows, formal_order_raw
        )
        formal_triples = build_formal_triples(
            run_id, formal_rows, formal_balanced
        )
        formal_targets = aggregate_formal_targets(run_id, formal_triples)
        paths.update(
            {
                "order": output_dir / "direct_judgement_order.csv",
                "triple": output_dir / "direct_judgement_triple.csv",
                "target": output_dir / "direct_judgement_target.csv",
            }
        )
        write_text_atomic(paths["order"], render_csv(formal_order, ORDER_COLUMNS))
        write_text_atomic(paths["triple"], render_csv(formal_triples, TRIPLE_COLUMNS))
        write_text_atomic(paths["target"], render_csv(formal_targets, TARGET_COLUMNS))
        outputs.update(
            {
                "order": {"file": paths["order"].name, "rows": len(formal_order)},
                "triple": {"file": paths["triple"].name, "rows": len(formal_triples)},
                "target": {"file": paths["target"].name, "rows": len(formal_targets)},
            }
        )
        formal_counts = {
            "order_rows": len(formal_order),
            "response_score_count": 2 * len(formal_order),
            "triple_rows": len(formal_triples),
            "target_rows": len(formal_targets),
            "source_sid_clusters": len({row["source_sid"] for row in formal_targets}),
            "judgement_contexts": len(JUDGEMENT_CONTEXTS),
        }

    write_text_atomic(
        paths["control_order"], render_csv(control_order, CONTROL_ORDER_COLUMNS)
    )
    write_text_atomic(
        paths["control_results"], render_csv(control_results, CONTROL_RESULT_COLUMNS)
    )
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "scope": args.scope,
        "run_id": run_id,
        "control_gate": control_gate,
        "formal_counts": formal_counts,
        "estimand": {
            "triple": (
                "P_context,t and P_word,t are separately order-balanced; "
                "C_t = P_context,t - P_word,t"
            ),
            "target": (
                "P_context,i and P_word,i are equal triple means within "
                "sentence_id; C_i = P_context,i - P_word,i"
            ),
        },
    }
    write_text_atomic(paths["summary"], render_json(summary))
    outputs["summary"] = {"file": paths["summary"].name, "rows": 1}

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": {
            "identifier": DEFAULT_MODEL_ID,
            "revision": DEFAULT_REVISION,
            "interface": (
                "official chat template; matched context/word inputs; one user "
                "message; teacher-forced A/B; enable_thinking=False"
            ),
        },
        "runtime": concise_runtime_identity(runtime),
        "inputs": {
            "controls": {"file": _project_path(controls_path), "rows": len(control_rows)},
            "analysis_items": (
                {"file": _project_path(analysis_path), "rows": len(formal_rows)}
                if args.scope == "formal"
                else None
            ),
        },
        "outputs": outputs,
    }
    write_text_atomic(paths["manifest"], render_json(manifest))
    return paths, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths, summary = run(args)
    except (DirectJudgementError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"status={summary['status']}")
    gate = summary["control_gate"]
    print(
        "controls_positive="
        f"{gate['positive_count']}/{gate['control_count']}"
    )
    if summary["formal_counts"] is not None:
        counts = summary["formal_counts"]
        print(
            "formal_rows="
            f"{counts['order_rows']}/{counts['triple_rows']}/{counts['target_rows']}"
        )
    print(f"manifest={_project_path(paths['manifest'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
