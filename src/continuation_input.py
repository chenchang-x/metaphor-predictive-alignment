"""Validation contract for the M/A/I rows consumed by Qwen experiments."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import CONDITIONS, CONTINUATION_INPUT_COLUMNS, Q_ORTHOGRAPHIC_WORDS


class ContinuationError(RuntimeError):
    """Raised when a continuation input row violates the frozen data design."""


ORTHOGRAPHIC_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


def canonical_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContinuationError(
            f"invalid integer {field}={value!r} at CSV row {row_number}"
        ) from exc
    if str(parsed) != value:
        raise ContinuationError(
            f"non-canonical integer {field}={value!r} at CSV row {row_number}"
        )
    return parsed


def orthographic_words(text: str) -> tuple[str, ...]:
    """Return whitespace tokens containing at least one Unicode letter/digit."""

    return tuple(
        match.group(0)
        for match in ORTHOGRAPHIC_TOKEN_RE.finditer(text)
        if any(character.isalnum() for character in match.group(0))
    )


def read_input_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ContinuationError(f"input CSV not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CONTINUATION_INPUT_COLUMNS:
                raise ContinuationError(
                    "input schema mismatch: expected "
                    f"{list(CONTINUATION_INPUT_COLUMNS)}, found {reader.fieldnames}"
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ContinuationError(f"input CSV is not valid UTF-8: {path.name}") from exc
    if not rows:
        raise ContinuationError("input CSV contains no data rows")
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ContinuationError(f"malformed CSV fields at row {row_number}")
    validate_input_rows(rows)
    return rows


def validate_input_rows(rows: Sequence[Mapping[str, str]]) -> None:
    seen_i0: set[int] = set()
    previous_i0: int | None = None
    for row_number, row in enumerate(rows, start=2):
        missing = [column for column in CONTINUATION_INPUT_COLUMNS if column not in row]
        if missing:
            raise ContinuationError(
                f"missing required fields at row {row_number}: {missing}"
            )
        i0 = canonical_int(row["i0"], "i0", row_number)
        canonical_int(row["sentence_id"], "sentence_id", row_number)
        if i0 in seen_i0:
            raise ContinuationError(f"duplicate i0={i0}")
        if previous_i0 is not None and i0 <= previous_i0:
            raise ContinuationError("input rows are not in strictly increasing i0 order")
        seen_i0.add(i0)
        previous_i0 = i0
        if row["triple_id"] != f"munch-judgement-{i0}":
            raise ContinuationError(f"i0={i0} triple_id does not match")

        q3_words = orthographic_words(row["q3_right_context"])
        expected_words = tuple(
            row[f"right_context_word_{index}"] for index in range(1, 4)
        )
        if len(q3_words) != Q_ORTHOGRAPHIC_WORDS or q3_words != expected_words:
            raise ContinuationError(
                f"i0={i0} q=3 mismatch: stored={expected_words!r}, "
                f"recomputed={q3_words!r}"
            )

        critical_start = canonical_int(row["critical_start"], "critical_start", row_number)
        left_contexts: set[str] = set()
        right_contexts: set[str] = set()
        targets: set[str] = set()
        for condition in CONDITIONS:
            lower = condition.lower()
            sentence = row[f"{lower}_sentence"]
            target = row[f"{lower}_word"]
            critical_end = canonical_int(
                row[f"{lower}_critical_end"], f"{lower}_critical_end", row_number
            )
            if not (0 <= critical_start < critical_end <= len(sentence)):
                raise ContinuationError(f"i0={i0} {condition} target offsets are invalid")
            if sentence[critical_start:critical_end] != target:
                raise ContinuationError(f"i0={i0} {condition} target offset mismatch")
            right_context = sentence[critical_end:]
            if not right_context.startswith(row["q3_right_context"]):
                raise ContinuationError(
                    f"i0={i0} {condition} sentence does not contain stored q=3 context"
                )
            expected_prefix = sentence[:critical_end] + row["q3_right_context"]
            if row[f"{lower}_prefix_q3"] != expected_prefix:
                raise ContinuationError(f"i0={i0} {condition} q=3 prefix mismatch")
            left_contexts.add(sentence[:critical_start])
            right_contexts.add(right_context)
            targets.add(target)
        if len(left_contexts) != 1 or len(right_contexts) != 1:
            raise ContinuationError(f"i0={i0} M/A/I non-target context differs")
        if len(targets) != len(CONDITIONS):
            raise ContinuationError(f"i0={i0} M/A/I targets are not distinct")


__all__ = [
    "ContinuationError",
    "canonical_int",
    "orthographic_words",
    "read_input_rows",
    "validate_input_rows",
]
