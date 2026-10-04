"""Validation and canonical serialization for Qwen continuation records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import render_jsonl
from .continuation_input import ContinuationError
from .contracts import (
    CONDITIONS,
    CONTINUATION_SCHEMA,
    H_QWEN_TOKENS,
    Q_ORTHOGRAPHIC_WORDS,
    SAMPLES_PER_CONDITION,
    SEEDS,
)


def validate_record(
    record: Mapping[str, object],
    input_row: Mapping[str, str],
    *,
    allowed_seeds: Sequence[int] = SEEDS,
    samples_per_condition: int = SAMPLES_PER_CONDITION,
    label: str = "output record",
) -> dict[str, object]:
    """Validate one durable item x condition x seed generation group."""
    required = ("schema", "seed", "i0", "condition", "continuations")
    missing = [field for field in required if field not in record]
    if missing:
        raise ContinuationError(f"{label} is missing field {missing[0]!r}")

    seed = record["seed"]
    i0 = record["i0"]
    condition = record["condition"]
    continuations = record["continuations"]
    expected_i0 = int(input_row["i0"])
    if record["schema"] != CONTINUATION_SCHEMA:
        raise ContinuationError(f"{label} schema mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed_seeds:
        raise ContinuationError(f"{label} has invalid seed")
    if isinstance(i0, bool) or not isinstance(i0, int) or i0 != expected_i0:
        raise ContinuationError(f"{label} has invalid i0")
    if not isinstance(condition, str) or condition not in CONDITIONS:
        raise ContinuationError(f"{label} has invalid condition")

    lower = condition.lower()
    expected_metadata: dict[str, object] = {
        "triple_id": input_row["triple_id"],
        "sentence_id": int(input_row["sentence_id"]),
        "source_sid": input_row["source_sid"],
        "target": input_row[f"{lower}_word"],
        "q_orthographic_words": Q_ORTHOGRAPHIC_WORDS,
        "q3_right_context": input_row["q3_right_context"],
        "prefix_text": input_row[f"{lower}_prefix_q3"],
        "h_qwen_tokens": H_QWEN_TOKENS,
    }
    for field, expected in expected_metadata.items():
        if record.get(field) != expected:
            raise ContinuationError(f"{label} field {field!r} mismatch")

    prefix_ids = record.get("prefix_token_ids")
    prefix_count = record.get("prefix_token_count")
    if (
        not isinstance(prefix_ids, list)
        or not prefix_ids
        or not all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in prefix_ids
        )
        or prefix_count != len(prefix_ids)
    ):
        raise ContinuationError(f"{label} has invalid prefix token IDs")

    if not isinstance(continuations, list) or len(continuations) != samples_per_condition:
        raise ContinuationError(
            f"{label} does not contain {samples_per_condition} continuations"
        )
    lengths: set[int] = set()
    for sample_index, continuation in enumerate(continuations):
        if not isinstance(continuation, dict):
            raise ContinuationError(f"{label} sample {sample_index} is not an object")
        token_ids = continuation.get("token_ids")
        tokens = continuation.get("tokens")
        if continuation.get("sample_index") != sample_index:
            raise ContinuationError(f"{label} sample index mismatch")
        if not isinstance(token_ids, list) or not all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in token_ids
        ):
            raise ContinuationError(f"{label} sample {sample_index} has invalid token IDs")
        if not isinstance(tokens, list) or not all(isinstance(value, str) for value in tokens):
            raise ContinuationError(
                f"{label} sample {sample_index} has invalid token strings"
            )
        if not isinstance(continuation.get("text"), str):
            raise ContinuationError(
                f"{label} sample {sample_index} has invalid decoded text"
            )
        if len(token_ids) != H_QWEN_TOKENS or len(tokens) != H_QWEN_TOKENS:
            raise ContinuationError(
                f"{label} sample {sample_index} does not have h={H_QWEN_TOKENS} tokens"
            )
        lengths.add(len(token_ids))
    return {
        "key": (input_row["triple_id"], condition, seed),
        "continuations": len(continuations),
        "continuation_token_lengths": sorted(lengths),
    }


def validate_records(
    records: Sequence[Mapping[str, object]],
    input_rows: Sequence[Mapping[str, str]],
    *,
    seeds: Sequence[int] = SEEDS,
    samples_per_condition: int = SAMPLES_PER_CONDITION,
) -> dict[str, object]:
    expected_order = [
        (seed, int(row["i0"]), condition)
        for seed in seeds
        for row in input_rows
        for condition in CONDITIONS
    ]
    observed_order: list[tuple[int, int, str]] = []
    condition_counts: Counter[str] = Counter()
    seed_counts: Counter[int] = Counter()
    continuation_count = 0
    lengths: set[int] = set()
    rows_by_i0 = {int(row["i0"]): row for row in input_rows}
    if len(rows_by_i0) != len(input_rows):
        raise ContinuationError("input rows contain duplicate i0 values")

    for record_number, record in enumerate(records, start=1):
        i0 = record.get("i0")
        if isinstance(i0, bool) or not isinstance(i0, int) or i0 not in rows_by_i0:
            raise ContinuationError(f"output record {record_number} has invalid i0")
        result = validate_record(
            record,
            rows_by_i0[i0],
            allowed_seeds=seeds,
            samples_per_condition=samples_per_condition,
            label=f"output record {record_number}",
        )
        seed = int(record["seed"])
        condition = str(record["condition"])
        group_continuations = record["continuations"]
        assert isinstance(group_continuations, list)
        lengths.update(result["continuation_token_lengths"])
        continuation_count += int(result["continuations"])
        observed_order.append((seed, i0, condition))
        condition_counts[condition] += len(group_continuations)
        seed_counts[seed] += len(group_continuations)

    if observed_order != expected_order:
        raise ContinuationError("output group keys or stable ordering do not match input")
    expected_count = (
        len(input_rows) * len(CONDITIONS) * len(seeds) * samples_per_condition
    )
    if continuation_count != expected_count:
        raise ContinuationError(
            f"expected {expected_count} continuations, found {continuation_count}"
        )
    return {
        "groups": len(records),
        "continuations": continuation_count,
        "conditions_run": list(CONDITIONS),
        "seeds_run": list(seeds),
        "continuation_token_lengths": sorted(lengths),
        "continuations_by_condition": {
            condition: condition_counts[condition] for condition in CONDITIONS
        },
        "continuations_by_seed": {str(seed): seed_counts[seed] for seed in seeds},
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    payload = Path(path).read_bytes()
    if payload.startswith(bytes((0xEF, 0xBB, 0xBF))):
        raise ContinuationError(f"{Path(path).name} must not contain a UTF-8 BOM")
    if bytes((13, 10)) in payload:
        raise ContinuationError(f"{Path(path).name} must use LF line endings")
    if not payload.endswith(b"\n"):
        raise ContinuationError(f"{Path(path).name} must end with one newline")
    records: list[dict[str, object]] = []
    try:
        text = payload.decode("utf-8")
        for line_number, line in enumerate(text[:-1].split("\n"), start=1):
            if not line:
                raise ContinuationError(
                    f"{Path(path).name} line {line_number} is empty"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContinuationError(
                    f"{Path(path).name} line {line_number} is not a JSON object"
                )
            records.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"could not read {Path(path).name}: {exc}") from exc
    return records


__all__ = ["read_jsonl", "render_jsonl", "validate_record", "validate_records"]
