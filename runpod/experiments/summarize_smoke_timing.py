#!/usr/bin/env python3
"""Extrapolate a transparent end-to-end formal runtime from smoke timings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "munch-qwen3.5-9b-end-to-end-runtime-plan/v3"
AUXILIARY_BENCHMARK_SCHEMA = "munch-qwen3.5-9b-auxiliary-stage-benchmark/v2"
GIBIBYTE = 1024**3
MINIMUM_FREE_HEADROOM_BYTES = 2 * GIBIBYTE
MAXIMUM_RESERVED_PERCENT = 92
REQUIRED_TIMINGS = (
    "reference_validation",
    "distance_smoke",
    "surprisal_smoke",
    "rq2_controls",
)


class TimingPlanError(RuntimeError):
    pass


def dynamic_peak_reserved_limit(total_memory_bytes: int) -> int:
    """Reserve both an absolute and proportional margin outside PyTorch.

    ``max_memory_reserved`` does not include every CUDA/driver allocation.
    The admissible peak therefore leaves at least 2 GiB and at least 8% of the
    measured device total free.  Integer arithmetic keeps the gate exactly
    reproducible across Python runtimes.
    """

    if type(total_memory_bytes) is not int:
        raise TimingPlanError("GPU total memory must be an integer byte count")
    if total_memory_bytes <= MINIMUM_FREE_HEADROOM_BYTES:
        raise TimingPlanError("GPU total memory is too small for the safety margin")
    absolute_limit = total_memory_bytes - MINIMUM_FREE_HEADROOM_BYTES
    proportional_limit = (
        total_memory_bytes * MAXIMUM_RESERVED_PERCENT // 100
    )
    threshold = min(absolute_limit, proportional_limit)
    if threshold <= 0 or threshold >= total_memory_bytes:
        raise TimingPlanError("dynamic GPU memory threshold is invalid")
    return threshold


def read_timings(path: Path) -> dict[str, dict[str, float | str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise TimingPlanError(f"cannot read stage timings: {path}") from exc
    observed: dict[str, dict[str, float | str]] = {}
    for row in rows:
        stage = str(row.get("stage", ""))
        try:
            elapsed = float(row.get("elapsed_seconds", ""))
        except (TypeError, ValueError) as exc:
            raise TimingPlanError(f"invalid elapsed time for stage {stage!r}") from exc
        if not stage or stage in observed or not math.isfinite(elapsed) or elapsed < 0:
            raise TimingPlanError(f"invalid or duplicate timing stage {stage!r}")
        observed[stage] = {
            "elapsed_seconds": elapsed,
            "status": str(row.get("status", "")),
        }
    missing = [stage for stage in REQUIRED_TIMINGS if stage not in observed]
    if missing:
        raise TimingPlanError("missing stage timing(s): " + ", ".join(missing))
    return observed


def read_generation_benchmark(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimingPlanError(f"cannot read generation benchmark: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("timing"), dict):
        raise TimingPlanError("generation benchmark has no timing object")
    return value


def read_auxiliary_benchmark(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimingPlanError(f"cannot read auxiliary benchmark: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("timing"), dict):
        raise TimingPlanError("auxiliary benchmark has no timing object")
    if value.get("schema") != AUXILIARY_BENCHMARK_SCHEMA:
        raise TimingPlanError("auxiliary benchmark does not use the matched-RQ2 schema")
    workload = value.get("formal_workload")
    if not isinstance(workload, Mapping) or workload.get(
        "rq2_input_conditions_per_triple"
    ) != 2:
        raise TimingPlanError("auxiliary benchmark does not cover both RQ2 inputs")
    return value


def _positive_number(mapping: Mapping[str, object], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimingPlanError(f"generation benchmark has invalid {key}") from exc
    if not math.isfinite(value) or value < 0:
        raise TimingPlanError(f"generation benchmark has invalid {key}")
    return value


def extrapolate_loaded_stage(
    *,
    name: str,
    observed_seconds: float,
    observed_units: int,
    formal_units: int,
    model_load_proxy_seconds: float,
    safety_factor: float,
) -> dict[str, object]:
    if observed_units < 1 or formal_units < 1:
        raise TimingPlanError(f"{name} unit counts must be positive")
    compute_observed = max(observed_seconds - model_load_proxy_seconds, 0.0)
    compute_per_unit = compute_observed / observed_units
    point_compute = compute_per_unit * formal_units
    point = model_load_proxy_seconds + point_compute

    upper_compute = compute_per_unit * formal_units * safety_factor
    upper = model_load_proxy_seconds + upper_compute
    return {
        "stage": name,
        "observed": {
            "units": observed_units,
            "wall_seconds_including_model_load": observed_seconds,
        },
        "formal_units": formal_units,
        "model_load_proxy_seconds_added_once": model_load_proxy_seconds,
        "point": {
            "load_subtracted_smoke_compute_seconds": compute_observed,
            "compute_seconds_per_unit": compute_per_unit,
            "compute_seconds": point_compute,
            "total_seconds": point,
            "load_subtraction_clipped_at_zero": (
                observed_seconds < model_load_proxy_seconds
            ),
        },
        "planning_upper": {
            "compute_seconds": upper_compute,
            "total_seconds": upper,
            "load_treatment": (
                "subtract one load proxy before scaling smoke compute; multiply "
                "compute by formal/smoke units and the safety factor; then add "
                "one formal load proxy"
            ),
        },
    }


def extrapolate_compute_stage(
    *,
    name: str,
    observed_compute_seconds: float,
    observed_units: int,
    formal_units: int,
    model_load_seconds: float,
    safety_factor: float,
    fixed_point_compute_seconds: float = 0.0,
    fixed_upper_compute_seconds: float = 0.0,
) -> dict[str, object]:
    if observed_units < 1 or formal_units < 1:
        raise TimingPlanError(f"{name} unit counts must be positive")
    if min(
        observed_compute_seconds,
        model_load_seconds,
        fixed_point_compute_seconds,
        fixed_upper_compute_seconds,
    ) < 0:
        raise TimingPlanError(f"{name} timing values must be non-negative")
    seconds_per_unit = observed_compute_seconds / observed_units
    scaled_compute = seconds_per_unit * formal_units
    point_compute = scaled_compute + fixed_point_compute_seconds
    upper_compute = (
        scaled_compute + fixed_upper_compute_seconds
    ) * safety_factor
    return {
        "stage": name,
        "source": "12 representative formal analysis_items selected by token length",
        "observed": {
            "units": observed_units,
            "compute_seconds_with_model_already_loaded": observed_compute_seconds,
            "seconds_per_unit": seconds_per_unit,
        },
        "formal_units": formal_units,
        "model_load_proxy_seconds_added_once": model_load_seconds,
        "point": {
            "scaled_munch_compute_seconds": scaled_compute,
            "fixed_compute_seconds": fixed_point_compute_seconds,
            "compute_seconds": point_compute,
            "total_seconds": model_load_seconds + point_compute,
        },
        "planning_upper": {
            "scaled_munch_compute_seconds": scaled_compute * safety_factor,
            "fixed_compute_seconds": fixed_upper_compute_seconds * safety_factor,
            "compute_seconds": upper_compute,
            "total_seconds": model_load_seconds + upper_compute,
            "rule": "representative compute x formal/sample units x safety factor, plus one model load",
        },
    }


def build_plan(
    timings: Mapping[str, Mapping[str, float | str]],
    benchmark: Mapping[str, object],
    auxiliary_benchmark: Mapping[str, object],
    *,
    smoke_items: int,
    formal_items: int,
    rq2_controls: int,
    safety_factor: float,
    cpu_postprocess_allowance_seconds: float,
) -> dict[str, object]:
    if smoke_items < 1 or formal_items < 1 or rq2_controls < 1:
        raise TimingPlanError("item and control counts must be positive")
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise TimingPlanError("safety factor must be at least 1")
    if (
        not math.isfinite(cpu_postprocess_allowance_seconds)
        or cpu_postprocess_allowance_seconds < 0
    ):
        raise TimingPlanError("CPU postprocessing allowance must be non-negative")
    raw_generation_timing = benchmark.get("timing")
    assert isinstance(raw_generation_timing, Mapping)
    generation_load = _positive_number(raw_generation_timing, "model_load_seconds")
    continuation_point = _positive_number(
        raw_generation_timing, "point_estimate_seconds"
    )
    continuation_upper = _positive_number(
        raw_generation_timing, "planning_upper_seconds"
    )
    if continuation_upper < continuation_point:
        raise TimingPlanError("generation planning upper is below its point estimate")
    raw_memory_probe = benchmark.get("gpu_memory_probe")
    if not isinstance(raw_memory_probe, Mapping):
        raise TimingPlanError("generation benchmark has no GPU memory probe")
    raw_runtime_identity = benchmark.get("runtime_identity")
    if not isinstance(raw_runtime_identity, Mapping):
        raise TimingPlanError("generation benchmark has no runtime identity")
    raw_gpu_identity = raw_runtime_identity.get("gpu")
    raw_software_identity = raw_runtime_identity.get("software")
    if not isinstance(raw_gpu_identity, Mapping):
        raise TimingPlanError("generation benchmark has no GPU identity")
    gpu_name = raw_gpu_identity.get("name")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise TimingPlanError("generation benchmark has no GPU name")
    total_memory_raw = raw_gpu_identity.get("total_memory_bytes")
    if type(total_memory_raw) is not int:
        raise TimingPlanError(
            "generation benchmark GPU identity has no integer total_memory_bytes"
        )
    total_memory_bytes = total_memory_raw
    threshold_bytes = dynamic_peak_reserved_limit(total_memory_bytes)
    if not isinstance(raw_software_identity, Mapping):
        raw_software_identity = {}
    try:
        peak_reserved_bytes = int(raw_memory_probe["peak_reserved_bytes"])
        peak_allocated_bytes = int(raw_memory_probe["peak_allocated_bytes"])
        allocated_after_bytes = int(
            raw_memory_probe["allocated_bytes_after_generation"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TimingPlanError("generation benchmark has invalid GPU memory facts") from exc
    if min(peak_reserved_bytes, peak_allocated_bytes, allocated_after_bytes) < 0:
        raise TimingPlanError("generation benchmark has negative GPU memory facts")
    if peak_reserved_bytes > total_memory_bytes:
        raise TimingPlanError("peak reserved memory exceeds reported GPU total memory")
    if peak_allocated_bytes > peak_reserved_bytes:
        raise TimingPlanError("peak allocated memory exceeds peak reserved memory")
    if allocated_after_bytes > peak_allocated_bytes:
        raise TimingPlanError(
            "post-generation allocated memory exceeds peak allocated memory"
        )
    gpu_gate = {
        "status": (
            "pass" if peak_reserved_bytes <= threshold_bytes else "switch_required"
        ),
        "decision_metric": "gpu_memory_probe.peak_reserved_bytes",
        "threshold_policy": (
            "min(total_memory_bytes - 2 GiB, floor(0.92 * total_memory_bytes))"
        ),
        "minimum_free_headroom_bytes": MINIMUM_FREE_HEADROOM_BYTES,
        "minimum_free_headroom_gib": 2.0,
        "maximum_reserved_fraction": MAXIMUM_RESERVED_PERCENT / 100,
        "threshold_bytes": threshold_bytes,
        "threshold_gib": threshold_bytes / GIBIBYTE,
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gib": total_memory_bytes / GIBIBYTE,
        "headroom_at_threshold_bytes": total_memory_bytes - threshold_bytes,
        "headroom_at_threshold_gib": (
            total_memory_bytes - threshold_bytes
        ) / GIBIBYTE,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_gib": peak_reserved_bytes / GIBIBYTE,
        "peak_reserved_fraction_of_total": (
            peak_reserved_bytes / total_memory_bytes
        ),
        "peak_allocated_bytes": peak_allocated_bytes,
        "allocated_bytes_after_generation": allocated_after_bytes,
        "runtime_identity": {
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": total_memory_bytes,
            "torch": raw_software_identity.get("torch"),
            "cuda_runtime": raw_software_identity.get("cuda_runtime"),
        },
        "probe": dict(raw_memory_probe),
        "action_if_exceeded": (
            "repeat bootstrap and smoke on a larger single GPU; prefer a 48 GiB "
            "L40S, RTX A6000, or A40 after a 32 GiB RTX 5090 fails; scientific "
            "parameters remain unchanged"
        ),
    }

    continuation = {
        "stage": "continuation",
        "source": "generation_speed_benchmark.json",
        "formal_units": formal_items,
        "model_load_proxy_seconds_added_once": generation_load,
        "point": {
            "compute_seconds": max(continuation_point - generation_load, 0.0),
            "total_seconds": continuation_point,
        },
        "planning_upper": {
            "total_seconds": continuation_upper,
            "rule": raw_generation_timing.get("planning_upper_rule"),
        },
    }

    def elapsed(stage: str) -> float:
        return float(timings[stage]["elapsed_seconds"])

    distance = extrapolate_loaded_stage(
        name="distance",
        observed_seconds=elapsed("distance_smoke"),
        observed_units=smoke_items,
        formal_units=formal_items,
        model_load_proxy_seconds=generation_load,
        safety_factor=safety_factor,
    )

    raw_auxiliary_timing = auxiliary_benchmark.get("timing")
    if not isinstance(raw_auxiliary_timing, Mapping):
        raise TimingPlanError("auxiliary benchmark has no timing object")
    auxiliary_load = _positive_number(
        raw_auxiliary_timing, "model_load_seconds"
    )

    def auxiliary_stage(name: str) -> tuple[int, float]:
        raw_stage = raw_auxiliary_timing.get(name)
        if not isinstance(raw_stage, Mapping):
            raise TimingPlanError(f"auxiliary benchmark has no {name} timing")
        try:
            items = int(raw_stage["items"])
            seconds = float(raw_stage["elapsed_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TimingPlanError(
                f"auxiliary benchmark has invalid {name} timing"
            ) from exc
        if items < 1 or not math.isfinite(seconds) or seconds < 0:
            raise TimingPlanError(f"auxiliary benchmark has invalid {name} timing")
        return items, seconds

    surprisal_items, surprisal_seconds = auxiliary_stage("surprisal")
    surprisal = extrapolate_compute_stage(
        name="surprisal",
        observed_compute_seconds=surprisal_seconds,
        observed_units=surprisal_items,
        formal_units=formal_items,
        model_load_seconds=auxiliary_load,
        safety_factor=safety_factor,
    )

    rq2_items, rq2_seconds = auxiliary_stage("rq2_direct_judgement")
    controls_wall_seconds = elapsed("rq2_controls")
    controls_point_compute = max(controls_wall_seconds - auxiliary_load, 0.0)
    rq2 = extrapolate_compute_stage(
        name="rq2_direct_judgement",
        observed_compute_seconds=rq2_seconds,
        observed_units=rq2_items,
        formal_units=formal_items,
        model_load_seconds=auxiliary_load,
        safety_factor=safety_factor,
        fixed_point_compute_seconds=controls_point_compute,
        fixed_upper_compute_seconds=controls_wall_seconds,
    )
    rq2["formal_control_gate"] = {
        "controls": rq2_controls,
        "smoke_wall_seconds_including_load": controls_wall_seconds,
        "point_compute_after_load_proxy_subtraction": controls_point_compute,
        "upper_uses_unsubtracted_control_wall_seconds": controls_wall_seconds,
        "controls_are_fixed_overhead_and_are_not_extrapolated_to_MUNCH_items": True,
    }
    reference_validation_seconds = elapsed("reference_validation")

    model_stages = (continuation, distance, surprisal, rq2)
    point = sum(float(stage["point"]["total_seconds"]) for stage in model_stages)
    upper = sum(
        float(stage["planning_upper"]["total_seconds"])
        for stage in model_stages
    )
    point += reference_validation_seconds + cpu_postprocess_allowance_seconds
    upper += (
        reference_validation_seconds * safety_factor
        + cpu_postprocess_allowance_seconds
    )
    return {
        "schema": SCHEMA,
        "scope": (
            "formal.sh after bootstrap and a successful smoke; through primary and "
            "RQ2 analysis outputs"
        ),
        "inputs": {
            "smoke_items": smoke_items,
            "formal_items": formal_items,
            "rq2_control_items": rq2_controls,
            "safety_factor": safety_factor,
            "model_load_proxy_seconds": {
                "continuation_and_distance": generation_load,
                "surprisal_and_rq2": auxiliary_load,
                "formal_total_for_four_loads": 2 * generation_load
                + 2 * auxiliary_load,
            },
            "stage_timings": {
                name: dict(value) for name, value in sorted(timings.items())
            },
            "auxiliary_selection": auxiliary_benchmark.get("selection"),
        },
        "method": {
            "model_load_invocations": 4,
            "point_rule": (
                "continuation uses its benchmark; distance subtracts one generation "
                "load proxy from smoke; surprisal and RQ2 scale compute-only timing "
                "from separate 12-item formal-input length quantiles; every GPU "
                "stage then adds one measured load proxy"
            ),
            "planning_upper_rule": (
                "continuation uses its slowest-block upper; distance subtracts one "
                "load proxy before scaling compute by 880/3 and the safety factor; "
                "surprisal and RQ2 scale their representative compute by the safety "
                "factor; RQ2 controls remain one fixed unsubtracted overhead; each "
                "stage adds one load proxy"
            ),
            "cpu_postprocessing_allowance_seconds": (
                cpu_postprocess_allowance_seconds
            ),
            "reference_statistics": (
                "built during smoke when absent and reused by formal; validation "
                "time is included, reference build time is excluded"
            ),
        },
        "gpu_memory_gate": gpu_gate,
        "stages": {
            "continuation": continuation,
            "distance": distance,
            "surprisal": surprisal,
            "rq2_direct_judgement": rq2,
            "reference_validation": {
                "point_seconds": reference_validation_seconds,
                "planning_upper_seconds": (
                    reference_validation_seconds * safety_factor
                ),
            },
            "cpu_primary_and_rq2_analysis": {
                "allowance_seconds": cpu_postprocess_allowance_seconds
            },
        },
        "total": {
            "point_seconds": point,
            "point_hours": point / 3600,
            "planning_upper_seconds": upper,
            "planning_upper_hours": upper / 3600,
        },
        "excluded": [
            "Pod provisioning and queue time",
            "bootstrap dependency installation",
            "model download",
            "reference-statistics build completed during smoke",
            "upload and result download",
        ],
    }


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--generation-benchmark", type=Path, required=True)
    parser.add_argument("--auxiliary-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-items", type=int, default=3)
    parser.add_argument("--formal-items", type=int, default=880)
    parser.add_argument("--rq2-controls", type=int, default=12)
    parser.add_argument("--safety-factor", type=float, default=1.20)
    parser.add_argument(
        "--cpu-postprocess-allowance-seconds", type=float, default=300.0
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        timings = read_timings(args.timings.resolve())
        benchmark = read_generation_benchmark(args.generation_benchmark.resolve())
        auxiliary_benchmark = read_auxiliary_benchmark(
            args.auxiliary_benchmark.resolve()
        )
        plan = build_plan(
            timings,
            benchmark,
            auxiliary_benchmark,
            smoke_items=args.smoke_items,
            formal_items=args.formal_items,
            rq2_controls=args.rq2_controls,
            safety_factor=args.safety_factor,
            cpu_postprocess_allowance_seconds=(
                args.cpu_postprocess_allowance_seconds
            ),
        )
        write_json_atomic(args.output.resolve(), plan)
    except (TimingPlanError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"formal_point_seconds={plan['total']['point_seconds']:.3f}")
    print(
        "formal_planning_upper_seconds="
        f"{plan['total']['planning_upper_seconds']:.3f}"
    )
    print(f"planning_summary={args.output.resolve()}")
    print(f"gpu_memory_gate={plan['gpu_memory_gate']['status']}")
    if plan["gpu_memory_gate"]["status"] != "pass":
        print(
            "ERROR: peak reserved CUDA memory exceeded the smoke gate; "
            "repeat smoke on a larger single GPU (prefer a 48 GiB fallback)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
