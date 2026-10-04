#!/usr/bin/env python3
"""Compute traceable M-A and M-I continuation-set distances.

This stage entry point coordinates validation, representation, distance, and
aggregation. Reusable algorithms live in ``src/qwen_alignment``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment import representation as REPRESENTATION
from qwen_alignment.aggregation import aggregate_sentences, aggregate_triples
from qwen_alignment.artifacts import (
    render_csv as _render_csv,
    write_text_atomic as _write_text_atomic,
)
from qwen_alignment.continuation_input import ContinuationError, read_input_rows
from qwen_alignment.continuation_records import read_jsonl, validate_records
from qwen_alignment.contracts import (
    CONDITIONS,
    CONTINUATION_OUTPUT_JSONL,
    H_QWEN_TOKENS,
    SAMPLES_PER_CONDITION as SAMPLES_PER_SET,
    SEEDS,
)
from qwen_alignment.distance_contracts import (
    DETAIL_COLUMNS,
    DETAIL_SCHEMA,
    REFERENCE_STATS_SCHEMA,
    SENTENCE_COLUMNS,
    SENTENCE_SCHEMA,
    SUMMARY_SCHEMA,
    TRIPLE_COLUMNS,
    TRIPLE_SCHEMA,
    DistanceError,
)
from qwen_alignment.distance_inputs import (
    encode_standardised_groups,
    group_continuation_records,
    load_reference_statistics,
)
from qwen_alignment.distance_metrics import (
    compute_triple_seed_distances,
    cosine_distance_matrix,
    mean_pairwise_set_distance,
    standardised_cosine_distance_matrix,
)
from qwen_alignment.model_runtime import resolve_runtime


COMPARISONS = (("M", "A"), ("M", "I"))


def run_pipeline(
    *,
    input_path: Path,
    items_path: Path,
    stats_path: Path,
    output_dir: Path,
    output_prefix: str,
    cache_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    records = read_jsonl(input_path)
    input_rows = read_input_rows(items_path)
    validate_records(records, input_rows)
    grouped = group_continuation_records(records)
    reference_mean, reference_std, reference_count = load_reference_statistics(stats_path)
    runtime = resolve_runtime(cache_dir, device, allow_download=False)
    standardised_groups = encode_standardised_groups(
        grouped,
        model=runtime["model"],
        device=runtime["device"],
        reference_mean=reference_mean,
        reference_std=reference_std,
        torch=runtime["torch"],
        batch_size=batch_size,
    )
    detail_rows = compute_triple_seed_distances(
        grouped, standardised_groups, torch=runtime["torch"]
    )
    triple_rows = aggregate_triples(detail_rows)
    sentence_rows = aggregate_sentences(triple_rows)

    paths = {
        "triple_seed": output_dir / f"{output_prefix}_triple_seed_distances.csv",
        "triple": output_dir / f"{output_prefix}_triple_distances.csv",
        "sentence": output_dir / f"{output_prefix}_sentence_distances.csv",
        "summary": output_dir / f"{output_prefix}_distance_summary.json",
    }
    _write_text_atomic(paths["triple_seed"], _render_csv(detail_rows, DETAIL_COLUMNS))
    _write_text_atomic(paths["triple"], _render_csv(triple_rows, TRIPLE_COLUMNS))
    _write_text_atomic(paths["sentence"], _render_csv(sentence_rows, SENTENCE_COLUMNS))

    summary: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "input": {
            "continuations_file": input_path.name,
            "items_file": items_path.name,
            "records": len(records),
        },
        "reference": {
            "file": stats_path.name,
            "count": reference_count,
            "shared_for_conditions": list(CONDITIONS),
        },
        "configuration": {
            "encoding": "continuation token IDs only; prefix excluded",
            "h_qwen_tokens": H_QWEN_TOKENS,
            "representation": "hidden_states[-1] mean over 5 token positions",
            "distance": "cosine after shared dimension-wise IAS standardisation",
            "set_distance": "mean of all 32 x 32 cross-set distances",
            "comparisons": ["M-A", "M-I"],
            "aggregation_order": [
                "triple_x_seed",
                "mean_seeds_within_triple",
                "mean_triples_within_sentence_id",
            ],
            "seeds": list(SEEDS),
        },
        "counts": {
            "triple_seed_rows": len(detail_rows),
            "triple_rows": len(triple_rows),
            "sentence_rows": len(sentence_rows),
        },
        "traceability": {
            "triple_seed_keys": ["triple_id", "i0", "seed"],
            "triple_summary_provenance": ["triple_id", "i0", "seeds"],
            "sentence_summary_provenance": [
                "sentence_id", "triple_ids", "i0s", "seeds",
            ],
        },
        "outputs": {
            "triple_seed": {"file": paths["triple_seed"].name},
            "triple": {"file": paths["triple"].name},
            "sentence": {"file": paths["sentence"].name},
        },
        "status": "pass",
    }
    _write_text_atomic(
        paths["summary"],
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "smoke" / CONTINUATION_OUTPUT_JSONL,
        help="Frozen continuation JSONL",
    )
    parser.add_argument(
        "--items", type=Path,
        default=PROJECT_ROOT / "data" / "smoke" / "munch_smoke_subset.csv",
        help="Matching analysis-item CSV used to validate all record identities",
    )
    parser.add_argument(
        "--stats", type=Path,
        default=PROJECT_ROOT / "data" / "reference" / "ias_naturalstories_qwen3_5_9b_h5_final_stats.json",
        help="Shared IAS reference mean and standard deviation",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "smoke",
        help="Directory for distance and aggregation outputs",
    )
    parser.add_argument("--output-prefix", default="qwen3p5_9b_smoke")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=PROJECT_ROOT / ".cache" / "huggingface",
        help="Local Hugging Face cache populated by Step 0",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_pipeline(
            input_path=args.input.resolve(),
            items_path=args.items.resolve(),
            stats_path=args.stats.resolve(),
            output_dir=args.output_dir.resolve(),
            output_prefix=args.output_prefix,
            cache_dir=args.cache_dir.resolve(),
            device=args.device,
            batch_size=args.batch_size,
        )
    except (
        ContinuationError,
        DistanceError,
        REPRESENTATION.RepresentationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Distance and aggregation pipeline completed")
    print(f"triple_seed_rows={summary['counts']['triple_seed_rows']}")
    print(f"triple_rows={summary['counts']['triple_rows']}")
    print(f"sentence_rows={summary['counts']['sentence_rows']}")
    print("comparisons=M-A,M-I")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
