#!/usr/bin/env python3
"""Rebuild manuscript screening/check tables without changing frozen data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
RUN = Path("results/formal/20260908T224807Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(output_dir: Path) -> dict:
    builder = runpy.run_path(str(ROOT / "runpod/experiments/build_analysis_items.py"))
    source = ROOT / "vendor/metaphor-understanding-challenge"
    source_hashes = builder["validate_source_checkout"](source)
    raw = read_csv(source / "correct_answers/for_judgement.csv")
    generation = read_csv(source / "correct_answers/for_generation.csv")
    items, exclusions, audit = builder["build_records"](raw, generation)
    frozen_path = ROOT / "data/processed/analysis_items.csv"
    frozen = read_csv(frozen_path)
    rebuilt = [{key: str(value) for key, value in row.items()} for row in items]
    if rebuilt != frozen:
        raise ValueError("Reconstructed rows do not exactly match frozen analysis_items.csv")
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()

    counts = audit["stage_counts"]
    stages = [
        ("Raw judgement records", "raw_judgement_triples", ""),
        ("Exactly one apt and one inapt label", "after_one_apt_one_inapt", "LABEL_NOT_ONE_APT_ONE_INAPT"),
        ("Valid target markup in all three conditions", "after_target_markup", "MALFORMED_TARGET_MARKUP"),
        ("Identical non-target context", "after_identical_non_target_context", "UNRESOLVED_NON_TARGET_CONTEXT_MISMATCH"),
        ("At least three shared post-target words", "after_at_least_3_right_words", "INSUFFICIENT_SHARED_RIGHT_CONTEXT"),
        ("Remove exact duplicate records", "after_exact_duplicate_removal", "EXACT_DUPLICATE_RECORD"),
    ]
    screening = []
    previous = len(raw)
    for number, (label, key, reason) in enumerate(stages):
        retained = counts[key]
        screening.append({
            "stage": number,
            "rule": label,
            "retained_records": retained,
            "excluded_at_stage": previous - retained,
            "reason_code": reason,
            "provenance": "Reconstructed from project-local pinned raw files using build_records",
        })
        previous = retained

    checks: list[dict] = []

    def check(name: str, observed: object, expected: object, source_file: str,
              detail: str = "", passed: bool | None = None) -> None:
        checks.append({
            "check": name,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if (observed == expected if passed is None else passed) else "FAIL",
            "source": source_file,
            "detail": detail,
        })

    check("Pinned raw source hashes", "verified", "verified",
          "vendor/metaphor-understanding-challenge/source_manifest.json",
          "Validated by this project's validate_source_checkout before reconstruction")
    check("Reconstructed retained rows match frozen table", rebuilt == frozen, True,
          "data/processed/analysis_items.csv", "Every row, field, value and row order compared")
    for label, observed, expected in [
        ("Retained three-condition records", len(frozen), 880),
        ("Target items", len({row["sentence_id"] for row in frozen}), 595),
        ("Source-sentence clusters", len({row["source_sid"] for row in frozen}), 553),
        ("Excluded raw records", len(exclusions), 612),
        ("Every raw record retained or excluded once", len(items) + len(exclusions), len(raw)),
    ]:
        check(label, observed, expected, "data/processed/analysis_items.csv; runpod/experiments/build_analysis_items.py")

    direct_path = RUN / "direct/direct_judgement_order.csv"
    direct = read_csv(ROOT / direct_path)
    by_triple = {row["triple_id"]: row for row in frozen}
    contextual = [row for row in direct if row["judgement_context"] == "context"]
    word = [row for row in direct if row["judgement_context"] == "word"]
    context_matches = sum(
        row["task_input"].replace("[TARGET: " + row["m_word"] + "]", row["m_word"])
        == by_triple[row["triple_id"]]["m_prefix_q3"] for row in contextual
    )
    word_matches = sum(row["task_input"] == "[TARGET: " + row["m_word"] + "]" for row in word)
    check("RQ2 input-order rows", len(direct), 3520, direct_path.as_posix())
    check("RQ2 unique triple-input-order keys", len({
        (row["triple_id"], row["judgement_context"], row["candidate_order"]) for row in direct
    }), 3520, direct_path.as_posix())
    check("Context input recovers RQ1 metaphor prefix", context_matches, 1760,
          direct_path.as_posix(), "Remove target markup; compare exact characters with m_prefix_q3, not full m_sentence")
    check("Word-only input contains only marked target", word_matches, 1760,
          direct_path.as_posix())
    score_count = sum(math.isfinite(float(row[key])) for row in direct
                      for key in ("log_probability_a_nats", "log_probability_b_nats"))
    check("Finite complete-response scores", score_count, 7040, direct_path.as_posix())

    controls_path = RUN / "direct/direct_controls.csv"
    controls = read_csv(ROOT / controls_path)
    context_controls = [row for row in controls if row["judgement_context"] == "context"]
    positive_controls = sum(float(row["p_t_nats"]) > 0 for row in context_controls)
    direct_summary_path = RUN / "direct/direct_judgement_summary.json"
    gate = read_json(ROOT / direct_summary_path)["control_gate"]
    check("Unambiguous contextual control items", len(context_controls), 12, controls_path.as_posix())
    check("Positive contextual control items", positive_controls, 12, controls_path.as_posix(),
          "Observed 12/12; frozen passing threshold is fraction >= 0.60")
    fraction = positive_controls / len(context_controls)
    check("Frozen contextual control gate", fraction, ">= " + str(gate["minimum_positive_fraction"]),
          direct_summary_path.as_posix(), "Word-only inputs have no correctness gate; this is an interface check",
          passed=fraction >= gate["minimum_positive_fraction"] and gate["passed"])

    completion_path = RUN / "continuations/completion_summary.json"
    completion = read_json(ROOT / completion_path)
    validation = completion["validation"]
    check("Formal run input hash matches frozen table", frozen_hash,
          completion["identity"]["dataset"]["input_sha256"], completion_path.as_posix())
    for label, key, expected in [
        ("Completed continuation groups", "observed_groups", 7920),
        ("Stored continuations", "observed_continuations", 253440),
        ("Missing continuation groups", "missing_groups", 0),
        ("Duplicate continuation groups", "duplicate_groups", 0),
        ("Unexpected continuation groups", "unexpected_groups", 0),
    ]:
        check(label, validation[key], expected, completion_path.as_posix(),
              "Recorded formal completion audit; this script does not rerun generation")

    for suffix, expected in [("triple_seed", 2640), ("triple", 880), ("sentence", 595)]:
        path = RUN / f"distances/qwen3p5_9b_formal_{suffix}_distances.csv"
        check(f"Distance table rows: {suffix}", len(read_csv(ROOT / path)), expected, path.as_posix())
    rq2_path = RUN / "rq2/rq2_items.csv"
    rq2 = read_csv(ROOT / rq2_path)
    check("Merged RQ2 target rows", len(rq2), 595, rq2_path.as_posix())
    check("Merged RQ2 target IDs match material IDs", {row["sentence_id"] for row in rq2}
          == {row["sentence_id"] for row in frozen}, True, rq2_path.as_posix())
    check("Context increment arithmetic", sum(math.isclose(
        float(row["c_i_nats"]), float(row["p_context_i_nats"]) - float(row["p_word_i_nats"]),
        rel_tol=0, abs_tol=1e-12) for row in rq2), 595, rq2_path.as_posix(),
        "C = S_context - S_word; absolute tolerance 1e-12 nats")

    token_path = Path("environment/tokenization_audit_summary.json")
    tokens = read_json(ROOT / token_path)
    for key, outcome in tokens["checks"].items():
        check("Tokenizer boundary audit: " + key, outcome["passed"], outcome["cases"],
              token_path.as_posix(), "Recorded tokenizer audit; not recomputed by this script",
              passed=outcome["passed"] == outcome["cases"] and outcome["failed"] == 0)
    failures = [row for row in checks if row["status"] != "PASS"]
    if failures:
        raise ValueError("Audit failed: " + json.dumps(failures))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "screening_stages.csv", screening,
              ("stage", "rule", "retained_records", "excluded_at_stage", "reason_code", "provenance"))
    write_csv(output_dir / "screening_exclusions.csv", exclusions, builder["EXCLUSION_COLUMNS"])
    write_csv(output_dir / "measurement_checks.csv", checks,
              ("check", "expected", "observed", "status", "source", "detail"))
    notes = f"""# Screening and Measurement Provenance

Formal experiment: `Qwen/Qwen3.5-9B`, `{RUN.as_posix()}`. These tables use this project's materials and saved results.

## Screening records

The project archive did not contain separate `data/processed/exclusions.csv` or `preprocessing_summary.json` files. The screening records were reconstructed with `build_records` from `runpod/experiments/build_analysis_items.py`, using the two fixed source CSVs in `vendor/metaphor-understanding-challenge/correct_answers/`.

`validate_source_checkout` verifies source hashes before reconstruction. Every field, row and row position in the reconstructed 880 retained records matches the frozen `data/processed/analysis_items.csv`.
Frozen-table SHA-256: `{frozen_hash}`.

`screening_stages.csv` records the ordered sequence 1492 → 1072 → 1072 → 1046 → 880 → 880. `screening_exclusions.csv` retains all 612 exclusions and their reasons: 420 label failures, 26 non-target context mismatches and 166 cases with fewer than three shared post-target words. Target-markup and deduplication stages each excluded 0 records.

These are reproducible reconstructions rather than independently saved historical screening files. Frozen materials and formal results are unchanged.

## Measurement checks

`measurement_checks.csv` contains {len(checks)} checks. CSV counts, score finiteness, exact input matching and context-increment arithmetic are recomputed. Continuation completion counts and tokenizer-boundary checks are taken from their formal-run records, as identified in `detail`. Model generation and tokenisation are not rerun.

All 12 unambiguous contextual controls have positive scores; the prespecified threshold is at least 60%. Word-only controls have no correctness threshold. Passing verifies the scoring interface against its rules, rather than establishing the validity of every formal item.

The contextual task input comprises the complete left context, marked target and three shared post-target orthographic words. It is not the complete source sentence. Removing target markers from all 1760 contextual order records exactly recovers the RQ1 `m_prefix_q3`; all 1760 word-only order records contain only the marked target word. For {sum(row['m_sentence'] != row['m_prefix_q3'] for row in frozen)} of the 880 materials, the original sentence extends beyond the actual input prefix.

## Reproduction

Run `python runpod/experiments/build_screening_audit.py` from the repository root. The script writes these four audit files to `figures/source_data/`.

Source-file SHA-256 values:

```json
{json.dumps(source_hashes, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    (output_dir / "audit_notes.md").write_text(notes, encoding="utf-8", newline="\n")
    return {"screening_exclusions": len(exclusions), "measurement_checks": len(checks),
            "all_checks_pass": True, "frozen_rows_match": True,
            "output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "figures/source_data")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
