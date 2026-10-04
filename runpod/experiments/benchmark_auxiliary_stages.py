#!/usr/bin/env python3
"""Time representative formal-input surprisal and RQ2 scoring without saving scores."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_json, write_text_atomic
from qwen_alignment.continuation_input import ContinuationError, read_input_rows
from qwen_alignment.direct_judgement import (
    CANDIDATE_ORDERS,
    JUDGEMENT_CONTEXTS,
    DirectJudgementError,
    prepare_response_sequences,
    prompt_item_from_analysis_row,
    render_user_message,
    score_prompt_items,
)
from qwen_alignment.model_runtime import resolve_runtime
from qwen_alignment.surprisal import SurprisalError, prepare_rows, score_prepared


SCHEMA = "munch-qwen3.5-9b-auxiliary-stage-benchmark/v2"


class AuxiliaryBenchmarkError(RuntimeError):
    pass


def select_length_quantiles(
    profiles: Sequence[Mapping[str, Any]], item_count: int
) -> list[Mapping[str, Any]]:
    """Select deterministic length quantiles including shortest and exact longest."""
    if item_count < 2 or item_count > len(profiles):
        raise AuxiliaryBenchmarkError(
            f"item_count must be between 2 and {len(profiles)}"
        )
    ordered = sorted(
        profiles,
        key=lambda item: (
            int(item["token_length"]),
            int(item["i0"]),
            str(item["triple_id"]),
        ),
    )
    denominator = item_count - 1
    indices = [
        (rank * (len(ordered) - 1) + denominator // 2) // denominator
        for rank in range(item_count)
    ]
    if len(set(indices)) != item_count:
        raise AuxiliaryBenchmarkError("quantile selection repeated an item")
    selected = [ordered[index] for index in indices]
    if selected[-1] is not ordered[-1]:
        raise AuxiliaryBenchmarkError("quantile selection did not include the longest item")
    return selected


def direct_profiles(
    rows: Sequence[Mapping[str, str]], tokenizer: Any
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for row in rows:
        item = prompt_item_from_analysis_row(row)
        lengths: list[int] = []
        for judgement_context in JUDGEMENT_CONTEXTS:
            for order in CANDIDATE_ORDERS:
                rendered = render_user_message(
                    item,
                    order,
                    judgement_context=judgement_context,
                )
                prepared = prepare_response_sequences(
                    tokenizer, rendered["user_message"]
                )
                combined = prepared["combined_ids"]
                if not isinstance(combined, Mapping):
                    raise AuxiliaryBenchmarkError(
                        "RQ2 prepared sequences are malformed"
                    )
                lengths.extend(len(list(values)) for values in combined.values())
        profiles.append(
            {
                "triple_id": row["triple_id"],
                "i0": int(row["i0"]),
                "token_length": max(lengths),
                "payload": item,
            }
        )
    return profiles


def surprisal_profiles(
    rows: Sequence[Mapping[str, str]], tokenizer: Any
) -> list[dict[str, Any]]:
    prepared = prepare_rows(rows, tokenizer)
    profiles: list[dict[str, Any]] = []
    for item in prepared:
        row = item["row"]
        if not isinstance(row, Mapping):
            raise AuxiliaryBenchmarkError("surprisal prepared row is malformed")
        lengths = []
        for condition in ("A", "I"):
            identity = item[condition]
            if not isinstance(identity, Mapping):
                raise AuxiliaryBenchmarkError("surprisal identity is malformed")
            lengths.append(len(list(identity["prefix_token_ids"])))
        profiles.append(
            {
                "triple_id": str(row["triple_id"]),
                "i0": int(row["i0"]),
                "token_length": max(lengths),
                "payload": row,
            }
        )
    return profiles


def public_selection(
    profiles: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    lengths = [int(item["token_length"]) for item in profiles]
    return {
        "method": (
            "equally spaced ranks after stable sort by tokenizer length and i0; "
            "includes shortest and exact longest"
        ),
        "population_items": len(profiles),
        "benchmark_items": len(selected),
        "population_token_length_range": [min(lengths), max(lengths)],
        "triple_ids": [str(item["triple_id"]) for item in selected],
        "i0s": [int(item["i0"]) for item in selected],
        "token_lengths": [int(item["token_length"]) for item in selected],
        "longest_item": {
            "triple_id": str(selected[-1]["triple_id"]),
            "i0": int(selected[-1]["i0"]),
            "token_length": int(selected[-1]["token_length"]),
        },
    }


def _sync_and_clear(runtime: Mapping[str, Any]) -> None:
    torch = runtime["torch"]
    device = runtime["device"]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def benchmark(
    rows: Sequence[Mapping[str, str]],
    runtime: Mapping[str, Any],
    *,
    item_count: int,
    surprisal_batch_size: int,
    model_load_seconds: float,
) -> dict[str, object]:
    tokenizer = runtime["tokenizer"]
    rq2_profiles = direct_profiles(rows, tokenizer)
    selected_rq2 = select_length_quantiles(rq2_profiles, item_count)
    surprisal_profile_rows = surprisal_profiles(rows, tokenizer)
    selected_surprisal = select_length_quantiles(
        surprisal_profile_rows, item_count
    )

    _sync_and_clear(runtime)
    started = time.perf_counter()
    direct_records: list[dict[str, object]] = []
    for judgement_context in JUDGEMENT_CONTEXTS:
        direct_records.extend(
            score_prompt_items(
                [profile["payload"] for profile in selected_rq2],
                runtime,
                judgement_context=judgement_context,
            )
        )
    runtime["torch"].cuda.synchronize(runtime["device"])
    rq2_seconds = time.perf_counter() - started
    if len(direct_records) != 2 * len(JUDGEMENT_CONTEXTS) * item_count:
        raise AuxiliaryBenchmarkError("RQ2 benchmark returned an unexpected record count")
    del direct_records

    selected_surprisal_rows = [
        profile["payload"] for profile in selected_surprisal
    ]
    _sync_and_clear(runtime)
    started = time.perf_counter()
    prepared = prepare_rows(selected_surprisal_rows, tokenizer)
    surprisal_records = score_prepared(
        prepared, runtime, batch_size=surprisal_batch_size
    )
    runtime["torch"].cuda.synchronize(runtime["device"])
    surprisal_seconds = time.perf_counter() - started
    if len(surprisal_records) != item_count:
        raise AuxiliaryBenchmarkError(
            "surprisal benchmark returned an unexpected record count"
        )
    del prepared, surprisal_records

    return {
        "schema": SCHEMA,
        "purpose": (
            "pre-results compute timing on representative formal MUNCH inputs; "
            "all scores discarded"
        ),
        "runtime_identity": runtime["runtime_identity"],
        "formal_workload": {
            "munch_triples": len(rows),
            "rq2_input_conditions_per_triple": len(JUDGEMENT_CONTEXTS),
            "rq2_candidate_orders_per_triple": len(CANDIDATE_ORDERS),
            "surprisal_conditions_per_triple": 2,
            "surprisal_batch_size": surprisal_batch_size,
        },
        "selection": {
            "rq2_direct_judgement": public_selection(
                rq2_profiles, selected_rq2
            ),
            "surprisal": public_selection(
                surprisal_profile_rows, selected_surprisal
            ),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "rq2_direct_judgement": {
                "items": item_count,
                "elapsed_seconds": rq2_seconds,
                "seconds_per_item": rq2_seconds / item_count,
            },
            "surprisal": {
                "items": item_count,
                "elapsed_seconds": surprisal_seconds,
                "seconds_per_item": surprisal_seconds / item_count,
            },
        },
        "scores_persisted": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "analysis_items.csv",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--items", type=int, default=12)
    parser.add_argument("--surprisal-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = read_input_rows(args.input.resolve())
        started = time.perf_counter()
        runtime = resolve_runtime(
            args.cache_dir.resolve(), args.device, allow_download=False
        )
        runtime["torch"].cuda.synchronize(runtime["device"])
        model_load_seconds = time.perf_counter() - started
        summary = benchmark(
            rows,
            runtime,
            item_count=args.items,
            surprisal_batch_size=args.surprisal_batch_size,
            model_load_seconds=model_load_seconds,
        )
        write_text_atomic(args.output.resolve(), render_json(summary))
    except (
        AuxiliaryBenchmarkError,
        ContinuationError,
        DirectJudgementError,
        SurprisalError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"rq2_benchmark_seconds={summary['timing']['rq2_direct_judgement']['elapsed_seconds']:.3f}")
    print(f"surprisal_benchmark_seconds={summary['timing']['surprisal']['elapsed_seconds']:.3f}")
    print("scores_persisted=false")
    print(f"benchmark_summary={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
