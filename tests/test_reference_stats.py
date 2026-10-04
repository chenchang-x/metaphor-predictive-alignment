from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_reference_stats.py"
SPEC = importlib.util.spec_from_file_location("build_reference_stats", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTorch:
    @staticmethod
    def isfinite(value):
        return np.isfinite(value)


class ReferenceStatisticsTests(unittest.TestCase):
    def test_valid_statistics_preserve_expected_false_provenance_flag(self):
        mean = np.zeros(MODULE.representation.HIDDEN_SIZE, dtype=np.float64)
        std = np.ones(MODULE.representation.HIDDEN_SIZE, dtype=np.float64)

        checks = MODULE.validate_statistics(mean, std, FakeTorch)

        self.assertTrue(checks["mean_finite"])
        self.assertTrue(checks["std_finite"])
        self.assertTrue(checks["std_strictly_positive"])
        self.assertFalse(checks["experimental_data_used"])

    def test_nonpositive_standard_deviation_still_fails(self):
        mean = np.zeros(MODULE.representation.HIDDEN_SIZE, dtype=np.float64)
        std = np.ones(MODULE.representation.HIDDEN_SIZE, dtype=np.float64)
        std[0] = 0.0

        with self.assertRaises(MODULE.ReferenceBuildError):
            MODULE.validate_statistics(mean, std, FakeTorch)


if __name__ == "__main__":
    unittest.main()
