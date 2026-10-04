#!/usr/bin/env python3
"""Run the exact shared-q3 local-surprisal baseline under Qwen3.5-9B."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_csv, render_json, write_text_atomic
from qwen_alignment.cluster_regression import fit_ols_cr1, student_t_survival
from qwen_alignment.continuation_input import ContinuationError, read_input_rows
from qwen_alignment.model_runtime import MODEL_ID, MODEL_REVISION, resolve_runtime
from qwen_alignment.surprisal import SCHEMA, SurprisalError, prepare_rows, score_prepared

TRIPLE_COLUMNS = (
    "schema", "triple_id", "i0", "sentence_id", "source_sid",
    "q3_token_count", "apt_surprisal_nats", "inapt_surprisal_nats",
    "inapt_minus_apt_nats", "direction",
)
SENTENCE_COLUMNS = (
    "schema", "sentence_id", "source_sid", "triple_ids", "i0s",
    "triple_count", "inapt_minus_apt_nats_triple_mean",
)
TRIPLE_SCHEMA = "munch-qwen3.5-9b-shared-q3-surprisal-triple/v1"
SENTENCE_SCHEMA = "munch-qwen3.5-9b-shared-q3-surprisal-target-item/v1"
SUMMARY_SCHEMA = "munch-qwen3.5-9b-shared-q3-surprisal-summary/v1"


def aggregate(records: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    triple_rows: list[dict[str, object]] = []
    by_sentence: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        apt = record["apt"]
        inapt = record["inapt"]
        assert isinstance(apt, Mapping) and isinstance(inapt, Mapping)
        triple_rows.append(
            {
                "schema": TRIPLE_SCHEMA,
                "triple_id": record["triple_id"],
                "i0": record["i0"],
                "sentence_id": record["sentence_id"],
                "source_sid": record["source_sid"],
                "q3_token_count": record["q3_token_count"],
                "apt_surprisal_nats": apt["surprisal_nats"],
                "inapt_surprisal_nats": inapt["surprisal_nats"],
                "inapt_minus_apt_nats": record["inapt_minus_apt_nats"],
                "direction": record["direction"],
            }
        )
        by_sentence[int(record["sentence_id"])].append(record)
    sentence_rows: list[dict[str, object]] = []
    for sentence_id, rows in sorted(by_sentence.items()):
        source_sids = {str(row["source_sid"]) for row in rows}
        if len(source_sids) != 1:
            raise SurprisalError(f"sentence_id={sentence_id} spans multiple source_sid values")
        ordered = sorted(rows, key=lambda row: int(row["i0"]))
        sentence_rows.append(
            {
                "schema": SENTENCE_SCHEMA,
                "sentence_id": sentence_id,
                "source_sid": next(iter(source_sids)),
                "triple_ids": json.dumps([row["triple_id"] for row in ordered], separators=(",", ":")),
                "i0s": json.dumps([row["i0"] for row in ordered], separators=(",", ":")),
                "triple_count": len(ordered),
                "inapt_minus_apt_nats_triple_mean": math.fsum(float(row["inapt_minus_apt_nats"]) for row in ordered) / len(ordered),
            }
        )
    return triple_rows, sentence_rows


def formal_inference(sentence_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(sentence_rows) != 595:
        raise SurprisalError("formal baseline requires exactly 595 target-item rows")
    clusters = [str(row["source_sid"]) for row in sentence_rows]
    if len(set(clusters)) != 553:
        raise SurprisalError("formal baseline requires exactly 553 source clusters")
    values = [float(row["inapt_minus_apt_nats_triple_mean"]) for row in sentence_rows]
    regression = fit_ols_cr1(values, {}, clusters)
    coefficient = regression["coefficients"]["intercept"]
    t_statistic = float(coefficient["test"]["t_statistic"])
    degrees_of_freedom = int(regression["degrees_of_freedom"])
    coefficient["directional_test"] = {
        "null_hypothesis": "mean <= 0",
        "alternative_hypothesis": "mean > 0",
        "t_statistic": t_statistic,
        "p_value_one_sided_upper": student_t_survival(t_statistic, degrees_of_freedom),
        "degrees_of_freedom": degrees_of_freedom,
    }
    coefficient["sign_counts"] = dict(
        Counter("positive" if value > 0 else "negative" if value < 0 else "zero" for value in values)
    )
    return regression


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_path = args.input.resolve()
        output_dir = args.output_dir.resolve()
        rows = read_input_rows(input_path)
        if args.scope == "formal" and len(rows) != 880:
            raise SurprisalError("formal baseline requires exactly 880 retained triples")
        runtime = resolve_runtime(args.cache_dir.resolve(), args.device, allow_download=False)
        # prepare_rows completes the all-row boundary gate before score_prepared
        # is allowed to execute its first model forward pass.
        prepared = prepare_rows(rows, runtime["tokenizer"])
        records = score_prepared(prepared, runtime, batch_size=args.batch_size)
        triple_rows, sentence_rows = aggregate(records)
        inference = formal_inference(sentence_rows) if args.scope == "formal" else None
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "shared_q3_surprisal_records.jsonl"
        triple_path = output_dir / "shared_q3_surprisal_triples.csv"
        sentence_path = output_dir / "shared_q3_surprisal_target_items.csv"
        summary_path = output_dir / "shared_q3_surprisal_summary.json"
        write_text_atomic(raw_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in records))
        write_text_atomic(triple_path, render_csv(triple_rows, TRIPLE_COLUMNS))
        write_text_atomic(sentence_path, render_csv(sentence_rows, SENTENCE_COLUMNS))
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "pass",
            "scope": args.scope,
            "input": {"file": input_path.name, "triple_rows": len(rows)},
            "model": {"identifier": MODEL_ID, "revision": MODEL_REVISION, "runtime": runtime.get("runtime_identity", {})},
            "configuration": {
                "measure": "summed exact shared-q3 surprisal in natural-log nats",
                "contrast": "surprisal(q3|I)-surprisal(q3|A)",
                "add_special_tokens": False,
                "boundary_gate_passed_before_scoring": True,
                "seeds": None,
            },
            "counts": {"triple_rows": len(triple_rows), "target_item_rows": len(sentence_rows), "source_clusters": len({row['source_sid'] for row in sentence_rows})},
            "inference": inference,
            "outputs": {
                "records": {"file": raw_path.name},
                "triples": {"file": triple_path.name},
                "target_items": {"file": sentence_path.name},
            },
        }
        write_text_atomic(summary_path, render_json(summary))
    except (ContinuationError, SurprisalError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"triple_rows={len(triple_rows)}")
    print(f"target_item_rows={len(sentence_rows)}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
