from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_auxiliary_stages.py"
SPEC = importlib.util.spec_from_file_location("benchmark_auxiliary_stages", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
auxiliary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auxiliary)


class AuxiliarySelectionTests(unittest.TestCase):
    def test_length_quantiles_are_deterministic_and_include_exact_longest(self) -> None:
        profiles = [
            {
                "triple_id": f"item-{index}",
                "i0": index,
                "token_length": (index * 7) % 19,
            }
            for index in range(30)
        ]
        selected = auxiliary.select_length_quantiles(profiles, 12)
        repeated = auxiliary.select_length_quantiles(list(reversed(profiles)), 12)
        self.assertEqual(
            [row["triple_id"] for row in selected],
            [row["triple_id"] for row in repeated],
        )
        expected_longest = sorted(
            profiles,
            key=lambda row: (
                row["token_length"],
                row["i0"],
                row["triple_id"],
            ),
        )[-1]
        self.assertEqual(selected[-1], expected_longest)
        self.assertEqual(len({row["triple_id"] for row in selected}), 12)

    def test_public_selection_contains_no_scores_or_payloads(self) -> None:
        profiles = [
            {
                "triple_id": f"item-{index}",
                "i0": index,
                "token_length": index + 5,
                "payload": {"score": 123},
            }
            for index in range(12)
        ]
        selected = auxiliary.select_length_quantiles(profiles, 12)
        public = auxiliary.public_selection(profiles, selected)
        self.assertNotIn("payload", repr(public))
        self.assertNotIn("score", repr(public))
        self.assertEqual(public["longest_item"]["token_length"], 16)


if __name__ == "__main__":
    unittest.main()
