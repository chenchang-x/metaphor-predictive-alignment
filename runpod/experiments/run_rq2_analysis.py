#!/usr/bin/env python3
"""Join matched explicit scores with Delta_i and fit frozen RQ2 M0/M1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_csv, render_json, write_text_atomic
from qwen_alignment.direct_judgement import MANIFEST_SCHEMA as DIRECT_MANIFEST_SCHEMA
from qwen_alignment.model_runtime import DEFAULT_MODEL_ID, DEFAULT_REVISION
from qwen_alignment.rq2_analysis import (
    MANIFEST_SCHEMA,
    MERGED_COLUMNS,
    RESULTS_SCHEMA,
    RQ2AnalysisError,
    fit_rq2_models,
    normalise_direct_target_rows,
    normalise_distance_rows,
    strict_item_merge,
)


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
        raise RQ2AnalysisError(f"cannot read {path}") from exc
    if not rows:
        raise RQ2AnalysisError(f"CSV has no rows: {path}")
    return rows


def read_direct_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RQ2AnalysisError(f"cannot read direct manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise RQ2AnalysisError("direct manifest is not a JSON object")
    if manifest.get("schema") != DIRECT_MANIFEST_SCHEMA or manifest.get("status") != "complete":
        raise RQ2AnalysisError("direct manifest is not a completed Qwen3.5-9B run")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise RQ2AnalysisError("direct manifest has no model identity")
    if model.get("identifier") != DEFAULT_MODEL_ID or model.get("revision") != DEFAULT_REVISION:
        raise RQ2AnalysisError("direct scores do not use the frozen Qwen3.5-9B revision")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or "target" not in outputs:
        raise RQ2AnalysisError("direct manifest does not describe formal target output")
    return manifest


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"munch-qwen3p5-9b-rq2-{timestamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-target", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--sentence-distances", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Any]]:
    direct_path = args.direct_target.resolve()
    direct_manifest_path = args.direct_manifest.resolve()
    distance_path = args.sentence_distances.resolve()

    # Matched explicit scores are loaded before Delta_i is read or joined.
    direct_manifest = read_direct_manifest(direct_manifest_path)
    direct_rows = normalise_direct_target_rows(read_csv(direct_path))
    distance_rows = normalise_distance_rows(read_csv(distance_path))
    merged_rows = strict_item_merge(distance_rows, direct_rows, formal=True)
    models = fit_rq2_models(merged_rows, formal=True)

    direct_run_ids = {str(row["run_id"]) for row in direct_rows}
    if len(direct_run_ids) != 1 or "" in direct_run_ids:
        raise RQ2AnalysisError("direct target rows do not share one run_id")
    direct_run_id = next(iter(direct_run_ids))
    if direct_run_id != direct_manifest.get("run_id"):
        raise RQ2AnalysisError("direct target rows and direct manifest use different run IDs")

    run_id = args.run_id or default_run_id()
    if not run_id.startswith("munch-qwen3p5-9b-"):
        raise RQ2AnalysisError("run_id does not use the Qwen3.5-9B namespace")
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "pass",
        "run_id": run_id,
        "direct_run_id": direct_run_id,
        "research_question": (
            "Does contextual direct apt preference covary with downstream "
            "alignment, and does its context-linked component add information "
            "beyond target-only preference?"
        ),
        "models": models,
        "interpretation": (
            "beta_total is the total contextual association. gamma_C is the "
            "context-linked association conditional on P_word. Neither is causal "
            "or a pure metaphor-interpretation effect."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "merged": output_dir / "rq2_items.csv",
        "results": output_dir / "rq2_results.json",
        "manifest": output_dir / "rq2_manifest.json",
    }
    write_text_atomic(paths["merged"], render_csv(merged_rows, MERGED_COLUMNS))
    write_text_atomic(paths["results"], render_json(results))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": {
            "identifier": DEFAULT_MODEL_ID,
            "revision": DEFAULT_REVISION,
        },
        "runtime": direct_manifest.get("runtime"),
        "inputs": {
            "direct_target": {"file": _project_path(direct_path), "rows": len(direct_rows)},
            "sentence_distances": {
                "file": _project_path(distance_path),
                "rows": len(distance_rows),
            },
        },
        "outputs": {
            "items": {"file": paths["merged"].name, "rows": len(merged_rows)},
            "results": {"file": paths["results"].name, "rows": 1},
        },
    }
    write_text_atomic(paths["manifest"], render_json(manifest))
    return paths, results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths, results = run(args)
    except (RQ2AnalysisError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    models = results["models"]
    m0 = models["M0_total_association"]
    m1 = models["M1_context_decomposition"]
    beta_total = m0["coefficients"]["beta_total"]
    beta_interval = beta_total["confidence_interval_95"]
    gamma_c = m1["coefficients"]["gamma_C"]
    gamma_interval = gamma_c["confidence_interval_95"]
    print("status=pass")
    print(f"N={models['n_observations']}")
    print(f"G={models['n_clusters']}")
    print(f"degrees_of_freedom={models['degrees_of_freedom']}")
    print(f"beta_total={beta_total['estimate']:.17g}")
    print(
        "beta_total_cr1_standard_error="
        f"{beta_total['cr1_standard_error']:.17g}"
    )
    print(
        "beta_total_confidence_interval_95="
        f"[{beta_interval['lower']:.17g}, {beta_interval['upper']:.17g}]"
    )
    print(f"beta_total_two_sided_p={beta_total['test']['p_value']:.17g}")
    print(f"gamma_C={gamma_c['estimate']:.17g}")
    print(f"gamma_C_cr1_standard_error={gamma_c['cr1_standard_error']:.17g}")
    print(
        "gamma_C_confidence_interval_95="
        f"[{gamma_interval['lower']:.17g}, {gamma_interval['upper']:.17g}]"
    )
    print(f"gamma_C_two_sided_p={gamma_c['test']['p_value']:.17g}")
    print(f"M0_r_squared={m0['r_squared']:.17g}")
    print(f"M1_r_squared={m1['r_squared']:.17g}")
    print(f"manifest={_project_path(paths['manifest'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
