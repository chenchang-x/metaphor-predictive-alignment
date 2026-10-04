from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "summarize_smoke_timing.py"
SPEC = importlib.util.spec_from_file_location("summarize_smoke_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)
GIB = 1024**3


class SmokeTimingPlanTests(unittest.TestCase):
    def test_end_to_end_plan_includes_four_loads_and_all_gpu_stages(self) -> None:
        timings = {
            "reference_validation": {"elapsed_seconds": 2.0, "status": "pass"},
            "distance_smoke": {"elapsed_seconds": 16.0, "status": "pass"},
            "surprisal_smoke": {"elapsed_seconds": 13.0, "status": "pass"},
            "rq2_controls": {"elapsed_seconds": 14.0, "status": "pass"},
        }
        benchmark = {
            "runtime_identity": {
                "gpu": {
                    "name": "NVIDIA GeForce RTX 5090",
                    "total_memory_bytes": 32 * GIB,
                },
                "software": {"torch": "2.8.0", "cuda_runtime": "12.8"},
            },
            "timing": {
                "model_load_seconds": 10.0,
                "point_estimate_seconds": 110.0,
                "planning_upper_seconds": 140.0,
                "planning_upper_rule": "fixture slowest block rule",
            },
            "gpu_memory_probe": {
                "peak_reserved_bytes": 18 * GIB,
                "peak_allocated_bytes": 17 * GIB,
                "allocated_bytes_after_generation": 16 * GIB,
                "num_return_sequences": 32,
                "prefix_token_count": 101,
            },
        }
        auxiliary = {
            "selection": {
                "rq2_direct_judgement": {"benchmark_items": 12},
                "surprisal": {"benchmark_items": 12},
            },
            "timing": {
                "model_load_seconds": 9.0,
                "rq2_direct_judgement": {
                    "items": 12,
                    "elapsed_seconds": 6.0,
                },
                "surprisal": {"items": 12, "elapsed_seconds": 3.0},
            },
        }
        plan = timing.build_plan(
            timings,
            benchmark,
            auxiliary,
            smoke_items=3,
            formal_items=880,
            rq2_controls=12,
            safety_factor=1.2,
            cpu_postprocess_allowance_seconds=300.0,
        )
        self.assertEqual(plan["method"]["model_load_invocations"], 4)
        self.assertEqual(
            set(plan["stages"]),
            {
                "continuation",
                "distance",
                "surprisal",
                "rq2_direct_judgement",
                "reference_validation",
                "cpu_primary_and_rq2_analysis",
            },
        )
        self.assertAlmostEqual(
            plan["stages"]["distance"]["point"]["total_seconds"],
            10.0 + (16.0 - 10.0) / 3 * 880,
        )
        self.assertAlmostEqual(
            plan["stages"]["distance"]["planning_upper"]["total_seconds"],
            10.0 + (16.0 - 10.0) / 3 * 880 * 1.2,
        )
        self.assertAlmostEqual(
            plan["stages"]["rq2_direct_judgement"]["point"]["total_seconds"],
            9.0 + 6.0 / 12 * 880 + (14.0 - 9.0),
        )
        self.assertAlmostEqual(
            plan["stages"]["surprisal"]["point"]["total_seconds"],
            9.0 + 3.0 / 12 * 880,
        )
        self.assertTrue(
            plan["stages"]["rq2_direct_judgement"]["formal_control_gate"]
            ["controls_are_fixed_overhead_and_are_not_extrapolated_to_MUNCH_items"]
        )
        self.assertGreater(
            plan["total"]["planning_upper_seconds"],
            plan["total"]["point_seconds"],
        )
        self.assertIn("model download", plan["excluded"])
        self.assertEqual(plan["gpu_memory_gate"]["status"], "pass")
        self.assertEqual(
            plan["gpu_memory_gate"]["decision_metric"],
            "gpu_memory_probe.peak_reserved_bytes",
        )
        self.assertEqual(
            plan["gpu_memory_gate"]["runtime_identity"],
            {
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "gpu_total_memory_bytes": 32 * GIB,
                "torch": "2.8.0",
                "cuda_runtime": "12.8",
            },
        )
        self.assertEqual(
            plan["gpu_memory_gate"]["threshold_bytes"],
            32 * GIB * 92 // 100,
        )
        self.assertEqual(
            plan["gpu_memory_gate"]["threshold_policy"],
            "min(total_memory_bytes - 2 GiB, floor(0.92 * total_memory_bytes))",
        )

    def test_load_subtraction_is_reported_when_clipped(self) -> None:
        stage = timing.extrapolate_loaded_stage(
            name="surprisal",
            observed_seconds=8.0,
            observed_units=3,
            formal_units=880,
            model_load_proxy_seconds=10.0,
            safety_factor=1.2,
        )
        self.assertTrue(stage["point"]["load_subtraction_clipped_at_zero"])
        self.assertEqual(stage["point"]["compute_seconds"], 0.0)
        self.assertEqual(
            stage["planning_upper"]["total_seconds"],
            stage["point"]["total_seconds"],
        )

    def test_peak_reserved_above_dynamic_32_gib_limit_requires_switch(self) -> None:
        timings = {
            "reference_validation": {"elapsed_seconds": 1.0, "status": "pass"},
            "distance_smoke": {"elapsed_seconds": 12.0, "status": "pass"},
            "surprisal_smoke": {"elapsed_seconds": 12.0, "status": "pass"},
            "rq2_controls": {"elapsed_seconds": 12.0, "status": "pass"},
        }
        benchmark = {
            "runtime_identity": {
                "gpu": {
                    "name": "NVIDIA GeForce RTX 5090",
                    "total_memory_bytes": 32 * GIB,
                },
                "software": {"torch": "2.8.0", "cuda_runtime": "12.8"},
            },
            "timing": {
                "model_load_seconds": 10.0,
                "point_estimate_seconds": 100.0,
                "planning_upper_seconds": 120.0,
            },
            "gpu_memory_probe": {
                "peak_reserved_bytes": 30 * GIB,
                "peak_allocated_bytes": 29 * GIB,
                "allocated_bytes_after_generation": 16 * GIB,
            },
        }
        auxiliary = {
            "timing": {
                "model_load_seconds": 10.0,
                "rq2_direct_judgement": {
                    "items": 12,
                    "elapsed_seconds": 4.0,
                },
                "surprisal": {"items": 12, "elapsed_seconds": 2.0},
            }
        }
        plan = timing.build_plan(
            timings,
            benchmark,
            auxiliary,
            smoke_items=3,
            formal_items=880,
            rq2_controls=12,
            safety_factor=1.2,
            cpu_postprocess_allowance_seconds=300.0,
        )
        self.assertEqual(plan["gpu_memory_gate"]["status"], "switch_required")

    def test_dynamic_limit_leaves_larger_of_two_gib_or_eight_percent(self) -> None:
        # 24 GiB: the absolute 2 GiB reserve is larger than 8% (1.92 GiB).
        self.assertEqual(timing.dynamic_peak_reserved_limit(24 * GIB), 22 * GIB)
        # 48 GiB: the proportional 8% reserve (3.84 GiB) is larger.
        self.assertEqual(
            timing.dynamic_peak_reserved_limit(48 * GIB),
            48 * GIB * 92 // 100,
        )

    def test_missing_total_memory_fails_closed(self) -> None:
        timings = {
            "reference_validation": {"elapsed_seconds": 1.0, "status": "pass"},
            "distance_smoke": {"elapsed_seconds": 12.0, "status": "pass"},
            "surprisal_smoke": {"elapsed_seconds": 12.0, "status": "pass"},
            "rq2_controls": {"elapsed_seconds": 12.0, "status": "pass"},
        }
        benchmark = {
            "runtime_identity": {
                "gpu": {"name": "NVIDIA GeForce RTX 5090"},
                "software": {"torch": "2.8.0", "cuda_runtime": "12.8"},
            },
            "timing": {
                "model_load_seconds": 10.0,
                "point_estimate_seconds": 100.0,
                "planning_upper_seconds": 120.0,
            },
            "gpu_memory_probe": {
                "peak_reserved_bytes": 18 * GIB,
                "peak_allocated_bytes": 17 * GIB,
                "allocated_bytes_after_generation": 16 * GIB,
            },
        }
        auxiliary = {
            "timing": {
                "model_load_seconds": 10.0,
                "rq2_direct_judgement": {
                    "items": 12,
                    "elapsed_seconds": 4.0,
                },
                "surprisal": {"items": 12, "elapsed_seconds": 2.0},
            }
        }
        with self.assertRaisesRegex(timing.TimingPlanError, "total_memory_bytes"):
            timing.build_plan(
                timings,
                benchmark,
                auxiliary,
                smoke_items=3,
                formal_items=880,
                rq2_controls=12,
                safety_factor=1.2,
                cpu_postprocess_allowance_seconds=300.0,
            )


if __name__ == "__main__":
    unittest.main()
