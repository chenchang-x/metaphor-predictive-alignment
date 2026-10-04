"""Pinned MUNCH source identity and strict CSV boundary helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .artifacts import sha256_file


SOURCE_REPOSITORY = "https://github.com/xiaoyuisrain/metaphor-understanding-challenge.git"
SOURCE_COMMIT = "5b78a540705661f49fda693e504250d4a1fc4516"
SOURCE_FILES = {
    "correct_answers/for_judgement.csv": "6d3a69c9efeaf570e25ad981fa3e967563495c42429d6111877c1e2eb577d99a",
    "correct_answers/for_generation.csv": "7816b422516608d015eb69926571eb7647899e9eee4bc65dbbf46aeb2f64ca7f",
}
JUDGEMENT_COLUMNS = (
    "i0", "s0_idx", "s0", "s1", "s1_label", "s2", "s2_label",
)
GENERATION_COLUMNS = (
    "i0", "idx", "s0", "novelty", "sid", "genre", "human_ans",
)
ANALYSIS_COLUMNS = (
    "triple_id", "i0", "sentence_id", "source_sid", "genre", "novelty",
    "m_sentence", "a_sentence", "i_sentence", "m_word", "a_word", "i_word",
    "critical_start", "m_critical_end", "a_critical_end", "i_critical_end",
    "right_context_word_1", "right_context_word_2", "right_context_word_3",
    "q3_right_context", "m_prefix_q3", "a_prefix_q3", "i_prefix_q3",
)


class SourceIntegrityError(RuntimeError):
    """Raised when the pinned upstream checkout violates its small contract."""


def validate_source_checkout(source_root: Path) -> dict[str, str]:
    """Validate the small source manifest and the two required raw CSVs."""

    if not source_root.is_dir():
        raise SourceIntegrityError(f"MUNCH source directory not found: {source_root}")
    manifest_path = source_root / "source_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntegrityError(f"Invalid MUNCH source manifest: {manifest_path}") from exc
    if manifest.get("repository") != SOURCE_REPOSITORY:
        raise SourceIntegrityError("MUNCH source repository mismatch")
    if manifest.get("upstream_commit") != SOURCE_COMMIT:
        raise SourceIntegrityError(
            "MUNCH source commit mismatch"
        )
    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict):
        raise SourceIntegrityError("MUNCH source manifest has no files map")
    actual_hashes: dict[str, str] = {}
    for relative_name, expected_hash in SOURCE_FILES.items():
        recorded = recorded_files.get(relative_name)
        if not isinstance(recorded, dict) or recorded.get("sha256") != expected_hash:
            raise SourceIntegrityError(
                f"MUNCH source manifest mismatch for {relative_name}"
            )
        path = source_root / Path(relative_name)
        if not path.is_file():
            raise SourceIntegrityError(f"Required MUNCH source file not found: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise SourceIntegrityError(
                f"SHA-256 mismatch for {relative_name}: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        actual_hashes[relative_name] = actual_hash
    return actual_hashes


def read_csv_exact(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise SourceIntegrityError(
                    f"Schema mismatch for {path.name}: expected {list(expected_columns)}, "
                    f"found {reader.fieldnames}"
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise SourceIntegrityError(f"{path.name} is not valid UTF-8") from exc
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise SourceIntegrityError(f"Extra CSV fields in {path.name}, row {row_number}")
        if any(value is None for value in row.values()):
            raise SourceIntegrityError(f"Missing CSV field in {path.name}, row {row_number}")
    return rows


def parse_int(value: str, field: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SourceIntegrityError(
            f"Invalid integer {field}={value!r} at CSV row {row_number}"
        ) from exc
    if str(parsed) != value:
        raise SourceIntegrityError(
            f"Non-canonical integer {field}={value!r} at CSV row {row_number}"
        )
    return parsed


def ensure_unique_int_field(
    rows: Sequence[Mapping[str, str]], field: str, source_name: str
) -> dict[int, Mapping[str, str]]:
    indexed: dict[int, Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        key = parse_int(row[field], field, row_number)
        if key in indexed:
            raise SourceIntegrityError(f"Duplicate {field}={key} in {source_name}")
        indexed[key] = row
    return indexed


def write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "ANALYSIS_COLUMNS",
    "GENERATION_COLUMNS",
    "JUDGEMENT_COLUMNS",
    "SOURCE_COMMIT",
    "SOURCE_FILES",
    "SOURCE_REPOSITORY",
    "SourceIntegrityError",
    "ensure_unique_int_field",
    "parse_int",
    "read_csv_exact",
    "validate_source_checkout",
    "write_csv",
]
