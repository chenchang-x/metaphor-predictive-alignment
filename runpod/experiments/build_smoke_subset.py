#!/usr/bin/env python3
"""Select the fixed three-row engineering smoke fixture."""

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

from qwen_alignment import munch_source as SOURCE


SMOKE_I0 = (0, 1287, 1491)
EXPECTED_ANALYSIS_ROWS = 880
EXPECTED_ANALYSIS_SHA256 = (
    "e9282952a95f6a0b33a229c7bb4f9ab71496a2c7fafbf60af502917ef3a72f93"
)
OUTPUT_CSV = "munch_smoke_subset.csv"
OUTPUT_SUMMARY = "munch_smoke_subset_summary.json"


class SmokeValidationError(RuntimeError):
    """Raised when the frozen analysis table cannot supply the smoke rows."""


def build_outputs(analysis_path: Path, output_dir: Path) -> dict[str, object]:
    """Copy fixed identifiers from the already validated analysis table."""

    analysis_path = analysis_path.resolve()
    if not analysis_path.is_file():
        raise SmokeValidationError(f"analysis table not found: {analysis_path}")
    analysis_hash = SOURCE.sha256_file(analysis_path)
    if analysis_hash != EXPECTED_ANALYSIS_SHA256:
        raise SmokeValidationError(
            "analysis table does not match the frozen preprocessing output"
        )

    rows = SOURCE.read_csv_exact(analysis_path, SOURCE.ANALYSIS_COLUMNS)
    if len(rows) != EXPECTED_ANALYSIS_ROWS:
        raise SmokeValidationError(
            f"expected {EXPECTED_ANALYSIS_ROWS} analysis rows, found {len(rows)}"
        )

    by_i0: dict[int, dict[str, str]] = {}
    for row in rows:
        i0 = int(row["i0"])
        if i0 in by_i0:
            raise SmokeValidationError(f"duplicate i0={i0}")
        by_i0[i0] = row
    try:
        selected = [by_i0[i0] for i0 in SMOKE_I0]
    except KeyError as exc:
        raise SmokeValidationError(f"selected i0={exc.args[0]} is absent") from exc

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    SOURCE.write_csv(output_dir / OUTPUT_CSV, selected, SOURCE.ANALYSIS_COLUMNS)
    summary: dict[str, object] = {
        "purpose": "engineering smoke fixture; not for confirmatory inference",
        "selection": {"method": "fixed i0", "i0": list(SMOKE_I0)},
        "input": {
            "file": analysis_path.name,
            "rows": len(rows),
            "sha256": analysis_hash,
        },
        "output": {"file": OUTPUT_CSV, "rows": len(selected)},
    }
    (output_dir / OUTPUT_SUMMARY).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-items",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "analysis_items.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "smoke",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_outputs(args.analysis_items, args.output_dir)
    except (SmokeValidationError, SOURCE.SourceIntegrityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"smoke_triples={summary['output']['rows']}")
    print(f"selected_i0={','.join(str(value) for value in SMOKE_I0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
