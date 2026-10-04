#!/usr/bin/env python3
"""Build the Qwen project's MUNCH M/A/I analysis table from raw sources.

The raw MUNCH files are read-only input.  This script validates the pinned
source, applies the protocol rules in a fixed order, and rewrites all processed
artifacts from scratch.  Final N is observed, never supplied as a target.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import sha256_file
from qwen_alignment.munch_source import (
    ANALYSIS_COLUMNS,
    GENERATION_COLUMNS,
    JUDGEMENT_COLUMNS,
    SOURCE_COMMIT,
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    SourceIntegrityError,
    ensure_unique_int_field,
    parse_int,
    read_csv_exact,
    validate_source_checkout,
    write_csv,
)

EXCLUSION_COLUMNS = (
    "i0",
    "s0_idx",
    "sentence_id",
    "source_sid",
    "reason_code",
    "reason_detail",
    "duplicate_of_i0",
)

REASON_LABEL = "LABEL_NOT_ONE_APT_ONE_INAPT"
REASON_MARKUP = "MALFORMED_TARGET_MARKUP"
REASON_CONTEXT = "UNRESOLVED_NON_TARGET_CONTEXT_MISMATCH"
REASON_RIGHT_CONTEXT = "INSUFFICIENT_SHARED_RIGHT_CONTEXT"
REASON_DUPLICATE = "EXACT_DUPLICATE_RECORD"

ORTHOGRAPHIC_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


class TargetMarkupError(ValueError):
    """Raised when one sentence does not contain one usable marked target."""


@dataclass(frozen=True)
class MarkedText:
    left: str
    word: str
    right: str

    @property
    def sentence(self) -> str:
        return self.left + self.word + self.right

    @property
    def critical_start(self) -> int:
        return len(self.left)

    @property
    def critical_end(self) -> int:
        return len(self.left) + len(self.word)


def parse_marked_text(text: str) -> MarkedText:
    if text.count("<b>") != 1 or text.count("</b>") != 1:
        raise TargetMarkupError("expected exactly one literal <b> and one literal </b>")

    open_at = text.find("<b>")
    close_at = text.find("</b>")
    if close_at < open_at + len("<b>"):
        raise TargetMarkupError("target tags are reversed, overlapping, or empty")

    left = text[:open_at]
    word = text[open_at + len("<b>") : close_at]
    right = text[close_at + len("</b>") :]
    if not word:
        raise TargetMarkupError("marked target is empty")
    marked = MarkedText(left=left, word=word, right=right)
    if not marked.sentence:
        raise TargetMarkupError("sentence is empty after removing target tags")
    return marked


def first_q_orthographic_segment(right_context: str, q: int = 3) -> tuple[str, tuple[str, ...]]:
    if q < 1:
        raise ValueError("q must be positive")

    qualifying = [
        match
        for match in ORTHOGRAPHIC_TOKEN_RE.finditer(right_context)
        if any(character.isalnum() for character in match.group(0))
    ]
    if len(qualifying) < q:
        raise ValueError(f"found {len(qualifying)} orthographic words; require at least {q}")

    chosen = qualifying[:q]
    segment = right_context[: chosen[-1].end()]
    return segment, tuple(match.group(0) for match in chosen)


def _exclusion(
    row: Mapping[str, str],
    metadata: Mapping[str, str],
    reason_code: str,
    reason_detail: str,
    duplicate_of_i0: str = "",
) -> dict[str, object]:
    sentence_id = parse_int(row["s0_idx"], "s0_idx", -1)
    return {
        "i0": parse_int(row["i0"], "i0", -1),
        "s0_idx": sentence_id,
        "sentence_id": sentence_id,
        "source_sid": metadata["sid"],
        "reason_code": reason_code,
        "reason_detail": reason_detail,
        "duplicate_of_i0": duplicate_of_i0,
    }


def build_records(
    judgement_rows: Sequence[Mapping[str, str]],
    generation_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    judgement_by_i0 = ensure_unique_int_field(
        judgement_rows, "i0", "for_judgement.csv"
    )
    generation_by_i0 = ensure_unique_int_field(
        generation_rows, "i0", "for_generation.csv"
    )
    del generation_by_i0  # Validation only; joins use generation.idx.
    generation_by_idx = ensure_unique_int_field(
        generation_rows, "idx", "for_generation.csv"
    )

    items: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    seen_exact: dict[tuple[object, ...], int] = {}
    label_eligible_sentence_ids: set[int] = set()
    stage_counts: Counter[str] = Counter()
    right_word_count_distribution: Counter[str] = Counter()

    ordered_rows = [judgement_by_i0[key] for key in sorted(judgement_by_i0)]
    stage_counts["raw_judgement_triples"] = len(ordered_rows)

    for row_number, row in enumerate(ordered_rows, start=2):
        i0 = parse_int(row["i0"], "i0", row_number)
        sentence_id = parse_int(row["s0_idx"], "s0_idx", row_number)
        metadata = generation_by_idx.get(sentence_id)
        if metadata is None:
            raise SourceIntegrityError(
                f"No for_generation.csv row with idx={sentence_id} for judgement i0={i0}"
            )
        if row["s0"] != metadata["s0"]:
            raise SourceIntegrityError(
                f"Joined source sentence mismatch for judgement i0={i0}, s0_idx={sentence_id}"
            )
        for required_field in ("sid", "genre", "novelty"):
            if metadata[required_field] == "":
                raise SourceIntegrityError(
                    f"Empty generation metadata {required_field} for idx={sentence_id}"
                )

        labels = sorted((row["s1_label"], row["s2_label"]))
        if labels != ["apt", "inapt"]:
            exclusions.append(
                _exclusion(
                    row,
                    metadata,
                    REASON_LABEL,
                    f"s1_label={row['s1_label']};s2_label={row['s2_label']}",
                )
            )
            continue

        stage_counts["after_one_apt_one_inapt"] += 1
        label_eligible_sentence_ids.add(sentence_id)
        apt_text = row["s1"] if row["s1_label"] == "apt" else row["s2"]
        inapt_text = row["s1"] if row["s1_label"] == "inapt" else row["s2"]

        parsed: dict[str, MarkedText] = {}
        markup_failure = None
        for condition, text in (("M", row["s0"]), ("A", apt_text), ("I", inapt_text)):
            try:
                parsed[condition] = parse_marked_text(text)
            except TargetMarkupError as exc:
                markup_failure = f"condition={condition};error={exc}"
                break
        if markup_failure is not None:
            exclusions.append(
                _exclusion(row, metadata, REASON_MARKUP, markup_failure)
            )
            continue

        stage_counts["after_target_markup"] += 1
        m_text, a_text, i_text = parsed["M"], parsed["A"], parsed["I"]
        if not (
            m_text.left == a_text.left == i_text.left
            and m_text.right == a_text.right == i_text.right
        ):
            mismatch_parts = []
            if not (m_text.left == a_text.left == i_text.left):
                mismatch_parts.append("left_context")
            if not (m_text.right == a_text.right == i_text.right):
                mismatch_parts.append("right_context")
            exclusions.append(
                _exclusion(
                    row,
                    metadata,
                    REASON_CONTEXT,
                    "mismatch=" + "+".join(mismatch_parts),
                )
            )
            continue

        stage_counts["after_identical_non_target_context"] += 1
        observed_right_words = sum(
            1
            for match in ORTHOGRAPHIC_TOKEN_RE.finditer(m_text.right)
            if any(character.isalnum() for character in match.group(0))
        )
        right_word_count_distribution[
            str(observed_right_words) if observed_right_words < 3 else "3_or_more"
        ] += 1
        try:
            q3_segment, q3_words = first_q_orthographic_segment(m_text.right, q=3)
        except ValueError:
            exclusions.append(
                _exclusion(
                    row,
                    metadata,
                    REASON_RIGHT_CONTEXT,
                    f"observed_orthographic_words={observed_right_words};required=3",
                )
            )
            continue

        stage_counts["after_at_least_3_right_words"] += 1
        exact_key = (sentence_id, row["s0"], apt_text, inapt_text)
        if exact_key in seen_exact:
            duplicate_of = seen_exact[exact_key]
            exclusions.append(
                _exclusion(
                    row,
                    metadata,
                    REASON_DUPLICATE,
                    f"same_constructed_triple_as_i0={duplicate_of}",
                    duplicate_of_i0=str(duplicate_of),
                )
            )
            continue
        seen_exact[exact_key] = i0

        critical_start = m_text.critical_start
        items.append(
            {
                "triple_id": f"munch-judgement-{i0}",
                "i0": i0,
                "sentence_id": sentence_id,
                "source_sid": metadata["sid"],
                "genre": metadata["genre"],
                "novelty": metadata["novelty"],
                "m_sentence": m_text.sentence,
                "a_sentence": a_text.sentence,
                "i_sentence": i_text.sentence,
                "m_word": m_text.word,
                "a_word": a_text.word,
                "i_word": i_text.word,
                "critical_start": critical_start,
                "m_critical_end": m_text.critical_end,
                "a_critical_end": a_text.critical_end,
                "i_critical_end": i_text.critical_end,
                "right_context_word_1": q3_words[0],
                "right_context_word_2": q3_words[1],
                "right_context_word_3": q3_words[2],
                "q3_right_context": q3_segment,
                "m_prefix_q3": m_text.left + m_text.word + q3_segment,
                "a_prefix_q3": a_text.left + a_text.word + q3_segment,
                "i_prefix_q3": i_text.left + i_text.word + q3_segment,
            }
        )

    stage_counts["after_exact_duplicate_removal"] = len(items)
    stage_counts["final_analysis_triples"] = len(items)
    stage_counts["final_sentence_ids"] = len({item["sentence_id"] for item in items})
    stage_counts["one_apt_one_inapt_sentence_ids"] = len(label_eligible_sentence_ids)

    exclusions.sort(key=lambda record: int(record["i0"]))
    reason_counts = Counter(str(record["reason_code"]) for record in exclusions)
    audit = {
        "stage_counts": dict(stage_counts),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "right_word_count_distribution_after_context_match": dict(
            sorted(right_word_count_distribution.items())
        ),
    }
    validate_records(items, exclusions, len(ordered_rows))
    return items, exclusions, audit


def validate_records(
    items: Sequence[Mapping[str, object]],
    exclusions: Sequence[Mapping[str, object]],
    raw_count: int,
) -> None:
    if len(items) + len(exclusions) != raw_count:
        raise AssertionError("Every raw judgement row must be retained or excluded exactly once")
    if len({item["i0"] for item in items}) != len(items):
        raise AssertionError("Retained i0 values are not unique")
    if len({item["triple_id"] for item in items}) != len(items):
        raise AssertionError("Retained triple_id values are not unique")
    if [int(item["i0"]) for item in items] != sorted(int(item["i0"]) for item in items):
        raise AssertionError("Retained records are not sorted by integer i0")

    for item in items:
        for column in ANALYSIS_COLUMNS:
            if item[column] == "" or item[column] is None:
                raise AssertionError(f"Empty required field {column} for i0={item['i0']}")
        start = int(item["critical_start"])
        for condition in ("m", "a", "i"):
            sentence = str(item[f"{condition}_sentence"])
            word = str(item[f"{condition}_word"])
            end = int(item[f"{condition}_critical_end"])
            if sentence[start:end] != word:
                raise AssertionError(
                    f"Critical offset mismatch for i0={item['i0']}, condition={condition.upper()}"
                )
            expected_prefix = sentence[:end] + str(item["q3_right_context"])
            if str(item[f"{condition}_prefix_q3"]) != expected_prefix:
                raise AssertionError(
                    f"q=3 prefix mismatch for i0={item['i0']}, condition={condition.upper()}"
                )
        q3_segment, q3_words = first_q_orthographic_segment(
            str(item["q3_right_context"]), q=3
        )
        if q3_segment != item["q3_right_context"] or q3_words != tuple(
            str(item[f"right_context_word_{index}"]) for index in (1, 2, 3)
        ):
            raise AssertionError(f"q=3 right-context audit mismatch for i0={item['i0']}")


def build_outputs(source_root: Path, output_dir: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    source_hashes = validate_source_checkout(source_root)

    judgement_rows = read_csv_exact(
        source_root / "correct_answers" / "for_judgement.csv", JUDGEMENT_COLUMNS
    )
    generation_rows = read_csv_exact(
        source_root / "correct_answers" / "for_generation.csv", GENERATION_COLUMNS
    )
    items, exclusions, audit = build_records(judgement_rows, generation_rows)

    analysis_path = output_dir / "analysis_items.csv"
    exclusions_path = output_dir / "exclusions.csv"
    summary_path = output_dir / "preprocessing_summary.json"
    write_csv(analysis_path, items, ANALYSIS_COLUMNS)
    write_csv(exclusions_path, exclusions, EXCLUSION_COLUMNS)

    summary: dict[str, object] = {
        "protocol": {
            "name": "Qwen3.5-9B Metaphor Predictive Alignment",
            "date": "2026-09-08",
            "q_orthographic_words": 3,
            "final_n_forced": False,
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "files_sha256": source_hashes,
        },
        "filter_order": [
            "exactly one apt and one inapt label",
            "one valid marked target in M, A, and I",
            "identical non-target left and right context",
            "at least three shared right-context orthographic words",
            "remove exact duplicate constructed triples, retaining the lowest i0",
        ],
        **audit,
        "outputs": {
            "analysis_items.csv": {
                "rows": len(items),
                "sentence_ids": len({item["sentence_id"] for item in items}),
                "sha256": sha256_file(analysis_path),
            },
            "exclusions.csv": {
                "rows": len(exclusions),
            },
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "vendor" / "metaphor-understanding-challenge",
        help="Path to the project-local pinned MUNCH raw files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory rewritten with deterministic processed artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_outputs(args.source_root, args.output_dir)
    except (SourceIntegrityError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    stage_counts = summary["stage_counts"]
    reason_counts = summary["exclusion_reason_counts"]
    output = summary["outputs"]["analysis_items.csv"]
    print("MUNCH preprocessing completed")
    for key, value in stage_counts.items():
        print(f"{key}={value}")
    for key, value in reason_counts.items():
        print(f"excluded_{key}={value}")
    print(f"analysis_items_sha256={output['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
