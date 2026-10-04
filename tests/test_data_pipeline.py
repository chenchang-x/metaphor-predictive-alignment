from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

from qwen_alignment.artifacts import sha256_file
from qwen_alignment.munch_source import (
    GENERATION_COLUMNS,
    JUDGEMENT_COLUMNS,
    read_csv_exact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "vendor" / "metaphor-understanding-challenge"
EXPECTED_ANALYSIS_SHA256 = (
    "e9282952a95f6a0b33a229c7bb4f9ab71496a2c7fafbf60af502917ef3a72f93"
)


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DataPipelineTests(unittest.TestCase):
    def test_raw_preprocessing_independently_reproduces_frozen_table(self) -> None:
        build = load_script("build_analysis_items")
        judgement = read_csv_exact(
            SOURCE_ROOT / "correct_answers" / "for_judgement.csv",
            JUDGEMENT_COLUMNS,
        )
        generation = read_csv_exact(
            SOURCE_ROOT / "correct_answers" / "for_generation.csv",
            GENERATION_COLUMNS,
        )
        items, exclusions, audit = build.build_records(judgement, generation)
        self.assertEqual(len(judgement), 1_492)
        self.assertEqual(len(items), 880)
        self.assertEqual(len(exclusions), 612)
        self.assertEqual(len({row["sentence_id"] for row in items}), 595)
        self.assertEqual(len({row["source_sid"] for row in items}), 553)
        self.assertEqual(audit["stage_counts"]["after_at_least_3_right_words"], 880)

    def test_written_analysis_and_smoke_are_new_project_outputs(self) -> None:
        analysis_path = PROJECT_ROOT / "data" / "processed" / "analysis_items.csv"
        smoke_path = PROJECT_ROOT / "data" / "smoke" / "munch_smoke_subset.csv"
        self.assertEqual(sha256_file(analysis_path), EXPECTED_ANALYSIS_SHA256)
        with smoke_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["i0"] for row in rows], ["0", "1287", "1491"])


if __name__ == "__main__":
    unittest.main()
