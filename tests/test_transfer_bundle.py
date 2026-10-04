from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_transfer_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_transfer_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class ProjectSandbox:
    def __enter__(self) -> Path:
        self.path = PROJECT_ROOT / "runs" / f"test-transfer-{uuid.uuid4().hex}"
        self.path.mkdir(parents=True)
        return self.path

    def __exit__(self, *_args: object) -> None:
        shutil.rmtree(self.path)


def write(root: Path, relative: Path | str, text: str = "fixture\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def complete_fixture(root: Path) -> None:
    for relative in (*bundle.STATIC_FILES, *bundle.INPUT_FILES):
        write(root, relative)
    write(root, "src/qwen_alignment/core.py")
    write(root, "scripts/stage.py")
    write(root, "tests/test_stage.py")
    write(root, "runpod/stage.sh", "#!/usr/bin/env bash\n")


class TransferBundleTests(unittest.TestCase):
    def test_missing_required_input_stops_bundle(self) -> None:
        with ProjectSandbox() as root:
            for relative in bundle.STATIC_FILES:
                write(root, relative)
            for directory in bundle.CODE_ALLOWLIST:
                (root / directory).mkdir(parents=True)
            with self.assertRaisesRegex(bundle.BundleError, "analysis_items.csv"):
                bundle.collect_files(root)

    def test_archive_contains_only_positive_allowlist(self) -> None:
        with ProjectSandbox() as root:
            complete_fixture(root)
            write(root, "runs/formal/continuations.jsonl", "generated\n")
            write(root, "data/reference/generated_stats.json", "{}\n")
            write(root, "data/smoke/generated_continuations.jsonl", "generated\n")
            write(root, "environment/runtime.json", "{}\n")
            write(root, ".cache/huggingface/model.bin", "weights\n")
            write(root, "researchwrite/notes.md", "notes\n")
            write(
                root,
                "proposals/rq2_validity_extension/src/qwen_alignment/proposed.py",
                "# pending author review\n",
            )
            write(root, "transfer/previous.tar.gz", "archive\n")
            write(root, "scripts/bytecode.pyc", "compiled\n")

            output = root / "transfer" / "bundle.tar.gz"
            selected, digest = bundle.build_bundle(root, output)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, bundle.sha256_file(output))

            prefix = f"{bundle.PROJECT_NAME}/"
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            expected = {prefix + path.as_posix() for path in selected}
            self.assertEqual(names, expected)
            self.assertIn(prefix + "protocol.md", names)
            self.assertIn(prefix + "protocol_en.md", names)
            self.assertFalse(any("runs/" in name for name in names))
            self.assertFalse(any(".cache/" in name for name in names))
            self.assertFalse(any(name.endswith(".jsonl") for name in names))
            self.assertFalse(any("proposals/" in name for name in names))
            self.assertNotIn(prefix + "data/reference/generated_stats.json", names)

    def test_shell_entrypoints_keep_all_state_under_persistent_root(self) -> None:
        for name in ("bootstrap.sh", "smoke.sh", "formal.sh"):
            text = (PROJECT_ROOT / "runpod" / name).read_text(encoding="utf-8")
            self.assertIn(
                'PROJECT_ROOT="/workspace/qwen3p5-9b-metaphor-predictive-alignment"',
                text,
            )
            self.assertNotIn("/root/", text)
            self.assertNotIn("qwen7b-metaphor-predictive-alignment", text)
        for name in ("smoke.sh", "formal.sh"):
            text = (PROJECT_ROOT / "runpod" / name).read_text(encoding="utf-8")
            self.assertIn("HF_HUB_OFFLINE=1", text)
            self.assertIn("TRANSFORMERS_OFFLINE=1", text)
            self.assertNotIn("--allow-download", text)
        bootstrap = (PROJECT_ROOT / "runpod" / "bootstrap.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(bootstrap.count("--allow-download"), 1)

        smoke = (PROJECT_ROOT / "runpod" / "smoke.sh").read_text(encoding="utf-8")
        for stage in (
            "build_reference_stats.py",
            "validate_reference_stats.py",
            "generate_continuations.py",
            "benchmark_auxiliary_stages.py",
            "summarize_smoke_timing.py",
            "compute_distances.py",
            "run_surprisal.py",
            "run_direct_judgement.py",
        ):
            self.assertIn(stage, smoke)
        self.assertIn("--benchmark", smoke)
        self.assertIn('--input "$FORMAL_INPUT"', smoke)
        self.assertIn("--auxiliary-benchmark", smoke)

        formal = (PROJECT_ROOT / "runpod" / "formal.sh").read_text(encoding="utf-8")
        for stage in (
            "generate_continuations.py",
            "compute_distances.py",
            "run_primary_analysis.py",
            "run_surprisal.py",
            "run_direct_judgement.py",
            "run_rq2_analysis.py",
        ):
            self.assertIn(stage, formal)
        self.assertIn("GENERATION_RESUME=(--resume)", formal)
        self.assertIn("latest_end_to_end_planning.json", formal)
        self.assertIn(
            'get("runtime_identity", {}).get("gpu_name")', formal
        )
        self.assertIn("sys.exit(1) if not isinstance(name, str)", formal)
        self.assertNotIn("raise SystemExit(1) if not isinstance(name, str)", formal)
        self.assertIn(
            "nvidia-smi --query-gpu=name --format=csv,noheader", formal
        )
        self.assertIn('"$CURRENT_GPU_NAME" != "$SMOKE_GPU_NAME"', formal)

    def test_formal_dynamic_gpu_memory_gate_behavior(self) -> None:
        formal = (PROJECT_ROOT / "runpod" / "formal.sh").read_text(encoding="utf-8")
        match = re.search(
            r'''if ! "\$PYTHON" -c '([^']+)' "\$PLANNING_GATE" "\$PLANNING_SCHEMA"; then''',
            formal,
        )
        self.assertIsNotNone(match, "formal.sh has no executable planning-gate check")
        assert match is not None
        gate_program = match.group(1)
        schema = "munch-qwen3.5-9b-end-to-end-runtime-plan/v3"

        valid = {
            "schema": schema,
            "gpu_memory_gate": {
                "status": "pass",
                "total_memory_bytes": 48 * 1024**3,
                "runtime_identity": {
                    "gpu_name": "fixture GPU",
                    "gpu_total_memory_bytes": 48 * 1024**3,
                },
            },
        }
        invalid = {
            "failed status": {**valid, "gpu_memory_gate": {**valid["gpu_memory_gate"], "status": "fail"}},
            "stale schema": {**valid, "schema": "munch-qwen3.5-9b-end-to-end-runtime-plan/v2"},
            "boolean total memory": {
                **valid,
                "gpu_memory_gate": {
                    **valid["gpu_memory_gate"],
                    "total_memory_bytes": True,
                    "runtime_identity": {
                        **valid["gpu_memory_gate"]["runtime_identity"],
                        "gpu_total_memory_bytes": True,
                    },
                },
            },
            "identity memory mismatch": {
                **valid,
                "gpu_memory_gate": {
                    **valid["gpu_memory_gate"],
                    "runtime_identity": {
                        **valid["gpu_memory_gate"]["runtime_identity"],
                        "gpu_total_memory_bytes": 24 * 1024**3,
                    },
                },
            },
        }

        with ProjectSandbox() as root:
            planning = root / "planning.json"

            def run_gate(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
                planning.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.run(
                    [sys.executable, "-c", gate_program, str(planning), schema],
                    check=False,
                    capture_output=True,
                    text=True,
                )

            self.assertEqual(run_gate(valid).returncode, 0)
            for label, payload in invalid.items():
                with self.subTest(label=label):
                    self.assertNotEqual(run_gate(payload).returncode, 0)


if __name__ == "__main__":
    unittest.main()
