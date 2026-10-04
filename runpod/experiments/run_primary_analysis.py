#!/usr/bin/env python3
"""Run the frozen target-item primary analysis with source-clustered CR1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_json, write_text_atomic
from qwen_alignment.cluster_regression import (
    student_t_critical_two_sided_95,
    student_t_survival,
)
from qwen_alignment.distance_contracts import SENTENCE_COLUMNS, SENTENCE_SCHEMA

EXPECTED_ITEMS = 595
EXPECTED_CLUSTERS = 553
EXPECTED_DF = 552
EXPECTED_SEEDS = [11, 23, 37]
RESULTS_SCHEMA = "munch-qwen3.5-9b-primary-analysis-results/v1"
IDENTITY_SCHEMA = "munch-qwen3.5-9b-primary-analysis-identity/v1"


class PrimaryAnalysisError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise PrimaryAnalysisError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryAnalysisError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"top-level JSON is not an object: {path}")
    return value


def parse_int(value: str, label: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise PrimaryAnalysisError(f"invalid integer {label}={value!r}") from exc
    require(str(result) == value, f"non-canonical integer {label}={value!r}")
    return result


def parse_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise PrimaryAnalysisError(f"invalid number {label}={value!r}") from exc
    require(math.isfinite(result), f"non-finite number {label}")
    return result


def read_effects(path: Path) -> tuple[list[float], list[float], list[float], list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(tuple(reader.fieldnames or ()) == SENTENCE_COLUMNS, "sentence-distance header mismatch")
            rows = list(reader)
    except OSError as exc:
        raise PrimaryAnalysisError(f"cannot read {path}") from exc
    require(len(rows) == EXPECTED_ITEMS, f"expected {EXPECTED_ITEMS} item rows")
    sentence_ids: list[int] = []
    clusters: list[str] = []
    m_a_values: list[float] = []
    m_i_values: list[float] = []
    effects: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        label = f"row {line_number}"
        require(row["schema"] == SENTENCE_SCHEMA, f"schema mismatch at {label}")
        sentence_id = parse_int(row["sentence_id"], f"{label} sentence_id")
        source_sid = row["source_sid"]
        require(bool(source_sid), f"empty source_sid at {label}")
        require(json.loads(row["seeds"]) == EXPECTED_SEEDS, f"seed mismatch at {label}")
        triple_count = parse_int(row["triple_count"], f"{label} triple_count")
        require(triple_count >= 1, f"empty target item at {label}")
        triple_ids = json.loads(row["triple_ids"])
        i0s = json.loads(row["i0s"])
        require(
            isinstance(triple_ids, list)
            and isinstance(i0s, list)
            and len(triple_ids) == len(i0s) == triple_count,
            f"triple provenance mismatch at {label}",
        )
        m_a = parse_float(row["m_a_distance_triple_mean"], f"{label} M-A")
        m_i = parse_float(row["m_i_distance_triple_mean"], f"{label} M-I")
        effect = parse_float(row["mi_minus_ma_triple_mean"], f"{label} effect")
        require(0.0 <= m_a <= 2.0 and 0.0 <= m_i <= 2.0, f"distance outside [0,2] at {label}")
        require(math.isclose(effect, m_i - m_a, abs_tol=1e-15, rel_tol=0.0), f"effect arithmetic mismatch at {label}")
        sentence_ids.append(sentence_id)
        clusters.append(source_sid)
        m_a_values.append(m_a)
        m_i_values.append(m_i)
        effects.append(effect)
    require(sentence_ids == sorted(sentence_ids), "sentence_id rows are not sorted")
    require(len(set(sentence_ids)) == EXPECTED_ITEMS, "sentence_id rows are duplicated")
    require(len(set(clusters)) == EXPECTED_CLUSTERS, f"expected {EXPECTED_CLUSTERS} source clusters")
    return m_a_values, m_i_values, effects, clusters


def describe(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    n = len(ordered)
    mean = math.fsum(ordered) / n
    variance = math.fsum((value - mean) ** 2 for value in ordered) / (n - 1)
    return {
        "n": n,
        "mean": mean,
        "sample_standard_deviation": math.sqrt(variance),
        "minimum": ordered[0],
        "median": ordered[n // 2],
        "maximum": ordered[-1],
    }


def cr1_intercept_only(effects: Sequence[float], clusters: Sequence[str]) -> dict[str, object]:
    n = len(effects)
    estimate = math.fsum(effects) / n
    residuals: dict[str, list[float]] = defaultdict(list)
    for effect, cluster in zip(effects, clusters, strict=True):
        residuals[cluster].append(effect - estimate)
    g = len(residuals)
    variance = (
        g / (g - 1)
        * math.fsum(math.fsum(values) ** 2 for values in residuals.values())
        / (n * n)
    )
    require(variance > 0.0 and math.isfinite(variance), "CR1 variance is invalid")
    standard_error = math.sqrt(variance)
    t_statistic = estimate / standard_error
    degrees_of_freedom = g - 1
    require(degrees_of_freedom == EXPECTED_DF, "unexpected degrees of freedom")
    critical = student_t_critical_two_sided_95(degrees_of_freedom)
    margin = critical * standard_error
    return {
        "estimate": estimate,
        "cr1_variance": variance,
        "cr1_standard_error": standard_error,
        "confidence_interval_95_two_sided": {
            "lower": estimate - margin,
            "upper": estimate + margin,
            "critical_value": critical,
            "reference_distribution": "Student t",
            "degrees_of_freedom": degrees_of_freedom,
        },
        "directional_test": {
            "null_hypothesis": "theta <= 0",
            "alternative_hypothesis": "theta > 0",
            "t_statistic": t_statistic,
            "p_value_one_sided_upper": student_t_survival(t_statistic, degrees_of_freedom),
            "degrees_of_freedom": degrees_of_freedom,
        },
        "n_target_items": n,
        "n_source_clusters": g,
        "source_cluster_averaging": False,
    }


def verify_distance_summary(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    require(summary.get("status") == "pass", "distance pipeline did not pass")
    counts = summary.get("counts")
    require(isinstance(counts, dict), "distance counts missing")
    require(counts.get("sentence_rows") == EXPECTED_ITEMS, "distance summary item count mismatch")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-distances", type=Path, required=True)
    parser.add_argument("--distance-summary", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sentence_path = args.sentence_distances.resolve()
        summary_path = args.distance_summary.resolve()
        run_manifest_path = args.run_manifest.resolve()
        output_dir = args.output_dir.resolve()
        summary = verify_distance_summary(summary_path)
        run_manifest = load_json(run_manifest_path)
        require(str(run_manifest.get("schema", "")).startswith("munch-qwen3.5-9b-"), "run manifest schema mismatch")
        require(isinstance(run_manifest.get("run_id"), str), "run_id missing")
        m_a, m_i, effects, clusters = read_effects(sentence_path)
        inference = cr1_intercept_only(effects, clusters)
        signs = Counter("positive" if x > 0 else "negative" if x < 0 else "zero" for x in effects)
        result = {
            "schema": RESULTS_SCHEMA,
            "status": "pass",
            "run_id": run_manifest["run_id"],
            "scope": "frozen primary continuation analysis only",
            "estimand": "equal target-item mean of distance(M,I)-distance(M,A)",
            "descriptive": {
                "m_a_distance": describe(m_a),
                "m_i_distance": describe(m_i),
                "mi_minus_ma": describe(effects),
                "effect_sign_counts": dict(signs),
            },
            "primary_effect": inference,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "primary_analysis_results.json"
        identity_path = output_dir / "primary_analysis_identity.json"
        write_text_atomic(results_path, render_json(result))
        identity = {
            "schema": IDENTITY_SCHEMA,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_manifest["run_id"],
            "inputs": {
                "sentence_distances": {"path": str(sentence_path)},
                "distance_summary": {"path": str(summary_path)},
                "run_manifest": {"path": str(run_manifest_path)},
            },
            "verified_distance_summary_schema": summary.get("schema"),
            "outputs": {"results": {"path": str(results_path)}},
        }
        write_text_atomic(identity_path, render_json(identity))
    except (PrimaryAnalysisError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    effect = inference
    test = effect["directional_test"]
    print(f"estimate={effect['estimate']:.17g}")
    print(f"cr1_standard_error={effect['cr1_standard_error']:.17g}")
    print(f"one_sided_p_value={test['p_value_one_sided_upper']:.17g}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
