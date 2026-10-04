from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.direct_judgement import TARGET_SCHEMA
from qwen_alignment.distance_contracts import SENTENCE_SCHEMA
from qwen_alignment.rq2_analysis import (
    RQ2AnalysisError,
    fit_rq2_models,
    normalise_direct_target_rows,
    normalise_distance_rows,
    strict_item_merge,
)


def formal_normalised_rows():
    distances = []
    direct = []
    next_i0 = 0
    for sentence_id in range(1, 596):
        triple_count = 2 if sentence_id <= 285 else 1
        i0_values = tuple(range(next_i0, next_i0 + triple_count))
        next_i0 += triple_count
        triple_ids = tuple(f"munch-judgement-{i0}" for i0 in i0_values)
        source_sid = f"source-{((sentence_id - 1) % 553) + 1}"
        p_context_i = (sentence_id - 298.0) / 100.0
        p_word_i = ((sentence_id % 7) - 3.0) / 10.0
        c_i = p_context_i - p_word_i
        noise = ((sentence_id % 11) - 5) / 1000.0
        delta_i = 0.2 + 0.5 * p_context_i + noise
        shared = {
            "sentence_id": sentence_id,
            "source_sid": source_sid,
            "triple_count": triple_count,
            "triple_ids": triple_ids,
            "i0_values": i0_values,
        }
        distances.append({**shared, "delta_i": delta_i})
        direct.append(
            {
                **shared,
                "p_context_i_nats": p_context_i,
                "p_word_i_nats": p_word_i,
                "c_i_nats": c_i,
                "run_id": "munch-qwen3p5-9b-test",
            }
        )
    assert next_i0 == 880
    return distances, direct


class RQ2AnalysisTests(unittest.TestCase):
    def test_raw_rows_normalise_to_the_same_item_identity(self) -> None:
        distance_raw = [
            {
                "schema": SENTENCE_SCHEMA,
                "sentence_id": "7",
                "source_sid": "source-7",
                "triple_count": "2",
                "triple_ids": '["munch-judgement-1","munch-judgement-2"]',
                "i0s": "[1,2]",
                "m_a_distance_triple_mean": "0.4",
                "m_i_distance_triple_mean": "0.7",
                "mi_minus_ma_triple_mean": "0.3",
            }
        ]
        direct_raw = [
            {
                "schema": TARGET_SCHEMA,
                "run_id": "munch-qwen3p5-9b-test",
                "sentence_id": "7",
                "source_sid": "source-7",
                "triple_count": "2",
                "triple_ids": '["munch-judgement-1","munch-judgement-2"]',
                "i0_values": "[1,2]",
                "p_context_i_nats": "1.25",
                "p_word_i_nats": "0.40",
                "c_i_nats": "0.85",
            }
        ]
        distance = normalise_distance_rows(distance_raw)
        direct = normalise_direct_target_rows(direct_raw)
        merged = strict_item_merge(distance, direct, formal=False)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["sentence_id"], 7)
        self.assertAlmostEqual(merged[0]["delta_i"], 0.3)
        self.assertAlmostEqual(merged[0]["p_context_i_nats"], 1.25)
        self.assertAlmostEqual(merged[0]["p_word_i_nats"], 0.40)
        self.assertAlmostEqual(merged[0]["c_i_nats"], 0.85)
        self.assertEqual(json.loads(merged[0]["i0_values"]), [1, 2])

    def test_merge_rejects_changed_triple_membership(self) -> None:
        distance = [
            {
                "sentence_id": 1,
                "source_sid": "source-1",
                "triple_count": 1,
                "triple_ids": ("munch-judgement-1",),
                "i0_values": (1,),
                "delta_i": 0.2,
            }
        ]
        direct = [
            {
                "sentence_id": 1,
                "source_sid": "source-1",
                "triple_count": 1,
                "triple_ids": ("munch-judgement-2",),
                "i0_values": (2,),
                "p_context_i_nats": 0.4,
                "p_word_i_nats": 0.1,
                "c_i_nats": 0.3,
                "run_id": "munch-qwen3p5-9b-test",
            }
        ]
        with self.assertRaisesRegex(RQ2AnalysisError, "triple_ids differs"):
            strict_item_merge(distance, direct, formal=False)

    def test_formal_ols_cr1_keeps_595_items_and_553_clusters(self) -> None:
        distance, direct = formal_normalised_rows()
        merged = strict_item_merge(distance, direct, formal=True)
        models = fit_rq2_models(merged, formal=True)
        m0 = models["M0_total_association"]
        m1 = models["M1_context_decomposition"]
        self.assertEqual(
            m0["formula"],
            "Delta_i = beta_0 + beta_total P_context_i + epsilon_i",
        )
        self.assertEqual(
            m1["formula"],
            "Delta_i = gamma_0 + gamma_C C_i + gamma_W P_word_i + epsilon_i",
        )
        self.assertEqual(models["n_observations"], 595)
        self.assertEqual(models["n_clusters"], 553)
        self.assertEqual(models["degrees_of_freedom"], 552)
        self.assertEqual(sum(row["triple_count"] for row in merged), 880)

        x = [float(row["p_context_i_nats"]) for row in merged]
        y = [float(row["delta_i"]) for row in merged]
        x_mean = math.fsum(x) / len(x)
        y_mean = math.fsum(y) / len(y)
        expected_beta_1 = math.fsum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        ) / math.fsum((x_value - x_mean) ** 2 for x_value in x)
        expected_beta_0 = y_mean - expected_beta_1 * x_mean
        self.assertAlmostEqual(
            m0["coefficients"]["beta_0"]["estimate"], expected_beta_0, places=13
        )
        self.assertAlmostEqual(
            m0["coefficients"]["beta_total"]["estimate"],
            expected_beta_1,
            places=13,
        )
        self.assertEqual(
            m0["coefficients"]["beta_total"]["test"]["p_value_sidedness"],
            "two-sided",
        )
        self.assertTrue(
            math.isfinite(
                m0["coefficients"]["beta_total"]["cr1_standard_error"]
            )
        )
        self.assertTrue(math.isfinite(m0["r_squared"]))
        self.assertTrue(math.isfinite(m1["r_squared"]))
        self.assertTrue(
            math.isfinite(m1["coefficients"]["gamma_C"]["cr1_standard_error"])
        )
        self.assertEqual(
            set(m1["coefficients"]),
            {"gamma_0", "gamma_C", "gamma_W"},
        )
        self.assertAlmostEqual(
            m1["coefficients"]["gamma_C"]["estimate"],
            0.5,
            places=3,
        )

    def test_formal_merge_does_not_silently_drop_an_item(self) -> None:
        distance, direct = formal_normalised_rows()
        direct.pop()
        with self.assertRaisesRegex(RQ2AnalysisError, "sentence_id sets differ"):
            strict_item_merge(distance, direct, formal=True)


if __name__ == "__main__":
    unittest.main()
