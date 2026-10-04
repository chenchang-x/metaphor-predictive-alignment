from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.continuation_input import ContinuationError
from qwen_alignment.continuation_records import read_jsonl, validate_records
from qwen_alignment.contracts import (
    CONTINUATION_SCHEMA,
    H_QWEN_TOKENS,
    SAMPLES_PER_CONDITION,
)
from qwen_alignment.production_run import (
    CHECKPOINT_NAME,
    RUN_ID_PREFIX,
    RUN_SCHEMA,
    CheckpointStore,
    expected_groups,
    finalize_run,
    initialise_or_resume_manifest,
    run_checkpointed_generation,
)


@contextmanager
def project_tempdir():
    path = PROJECT_ROOT / "runs" / f"test-generation-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class FakeStateTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeCudaRng:
    def __init__(self):
        self.state = [0, 0, 0, 0]

    def manual_seed_all(self, seed):
        self.state = [seed % 256, (seed >> 8) % 256, 17, 91]

    def get_rng_state(self, _device):
        return FakeStateTensor(self.state)

    def set_rng_state(self, state, *, device):
        assert device == "cuda:0"
        self.state = state.tolist()

    def draw(self):
        value = (self.state[0] * 73 + self.state[2] * 19 + 41) % 251
        self.state = [value, self.state[0], (self.state[2] + 7) % 256, self.state[3]]
        return value


class FakeTorchRng:
    uint8 = "uint8"

    def __init__(self):
        self.cpu_state = [0, 0, 0, 0]
        self.cuda = FakeCudaRng()

    def manual_seed(self, seed):
        self.cpu_state = [seed % 256, (seed >> 8) % 256, 29, 113]

    def get_rng_state(self):
        return FakeStateTensor(self.cpu_state)

    def set_rng_state(self, state):
        self.cpu_state = state.tolist()

    @staticmethod
    def tensor(values, dtype):
        assert dtype == "uint8"
        return FakeStateTensor(values)


def fixture_row() -> dict[str, str]:
    return {
        "triple_id": "qwen-triple-0001",
        "i0": "0",
        "sentence_id": "1",
        "source_sid": "source-1",
        "m_word": "sparked",
        "a_word": "caused",
        "i_word": "extinguished",
        "q3_right_context": "a lively debate",
        "m_prefix_q3": "The claim sparked a lively debate",
        "a_prefix_q3": "The claim caused a lively debate",
        "i_prefix_q3": "The claim extinguished a lively debate",
    }


def fake_record_builder(torch, row, condition, seed):
    lower = condition.lower()
    continuations = []
    for sample_index in range(SAMPLES_PER_CONDITION):
        token_ids = [torch.cuda.draw() for _ in range(H_QWEN_TOKENS)]
        continuations.append(
            {
                "sample_index": sample_index,
                "token_ids": token_ids,
                "tokens": [f"tok-{value}" for value in token_ids],
                "text": "|".join(str(value) for value in token_ids),
            }
        )
    return {
        "schema": CONTINUATION_SCHEMA,
        "triple_id": row["triple_id"],
        "i0": int(row["i0"]),
        "sentence_id": int(row["sentence_id"]),
        "source_sid": row["source_sid"],
        "condition": condition,
        "target": row[f"{lower}_word"],
        "q_orthographic_words": 3,
        "q3_right_context": row["q3_right_context"],
        "prefix_text": row[f"{lower}_prefix_q3"],
        "prefix_token_ids": [1, 2, 3],
        "prefix_token_count": 3,
        "seed": seed,
        "h_qwen_tokens": 5,
        "continuations": continuations,
    }


class ManifestIsolationTests(unittest.TestCase):
    def test_new_run_uses_qwen_schema_and_filesystem_safe_id(self) -> None:
        with project_tempdir() as run_dir:
            runtime = {
                "model": {
                    "identifier": "Qwen/Qwen3.5-9B",
                    "requested_revision": "fixed",
                    "resolved_commit": "fixed",
                },
                "load": {
                    "dtype": "bfloat16",
                    "attention_implementation": "eager",
                    "single_gpu": True,
                    "quantization": None,
                },
                "gpu": {"name": "RTX 4090", "uuid": "GPU-a"},
                "software": {
                    "torch": "2.8.0",
                    "transformers": "5.16.1",
                    "cuda_runtime": "12.8",
                },
            }
            identity = {
                "project_namespace": "qwen3.5-9b",
                "runtime": runtime,
            }
            manifest = initialise_or_resume_manifest(run_dir, identity, resume=False)
            self.assertEqual(manifest["schema"], RUN_SCHEMA)
            self.assertTrue(str(manifest["run_id"]).startswith(RUN_ID_PREFIX))
            same_gpu_new_instance = {
                **runtime,
                "gpu": {"name": "RTX 4090", "uuid": "GPU-b"},
            }
            self.assertEqual(
                initialise_or_resume_manifest(
                    run_dir,
                    {**identity, "runtime": same_gpu_new_instance},
                    resume=True,
                ),
                manifest,
            )
            different_gpu = {
                **runtime,
                "gpu": {"name": "RTX 5090", "uuid": "GPU-c"},
            }
            with self.assertRaisesRegex(ContinuationError, "identity mismatch"):
                initialise_or_resume_manifest(
                    run_dir,
                    {**identity, "runtime": different_gpu},
                    resume=True,
                )
            with self.assertRaisesRegex(ContinuationError, "identity mismatch"):
                initialise_or_resume_manifest(
                    run_dir,
                    {"project_namespace": "different"},
                    resume=True,
                )


class DurableFakeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [fixture_row()]

    @staticmethod
    def runtime():
        torch = FakeTorchRng()
        return torch, {"torch": torch, "device": "cuda:0"}

    def test_interrupted_resume_equals_uninterrupted_rng_chain(self) -> None:
        with project_tempdir() as root:
            resumed_path = root / "resumed.sqlite3"
            full_path = root / "full.sqlite3"

            torch, runtime = self.runtime()
            store = CheckpointStore(resumed_path, "qwen-run-resume")
            partial = run_checkpointed_generation(
                store=store,
                rows=self.rows,
                runtime=runtime,
                record_builder=lambda row, condition, seed: fake_record_builder(
                    torch, row, condition, seed
                ),
                stop_after_new_groups=4,
            )
            self.assertFalse(partial["complete"])
            store.close()

            torch, runtime = self.runtime()
            resumed_store = CheckpointStore(resumed_path, "qwen-run-resume")
            complete = run_checkpointed_generation(
                store=resumed_store,
                rows=self.rows,
                runtime=runtime,
                record_builder=lambda row, condition, seed: fake_record_builder(
                    torch, row, condition, seed
                ),
            )
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["skipped_completed"], 4)
            resumed_records = {
                ordinal: saved.record_json
                for ordinal, saved in resumed_store.load().items()
            }

            torch, runtime = self.runtime()
            full_store = CheckpointStore(full_path, "qwen-run-resume")
            run_checkpointed_generation(
                store=full_store,
                rows=self.rows,
                runtime=runtime,
                record_builder=lambda row, condition, seed: fake_record_builder(
                    torch, row, condition, seed
                ),
            )
            full_records = {
                ordinal: saved.record_json for ordinal, saved in full_store.load().items()
            }
            self.assertEqual(resumed_records, full_records)

            first_group = expected_groups(self.rows)[0]
            first_saved = resumed_store.load()[0]
            with self.assertRaisesRegex(ContinuationError, "duplicate generation refused"):
                resumed_store.insert(
                    first_group,
                    json.loads(first_saved.record_json),
                    first_saved.rng_before_json,
                    first_saved.rng_after_json,
                )
            resumed_store.close()
            full_store.close()

    def test_complete_checkpoint_finalizes_canonical_jsonl(self) -> None:
        with project_tempdir() as run_dir:
            torch, runtime = self.runtime()
            store = CheckpointStore(run_dir / CHECKPOINT_NAME, "qwen-run-final")
            invocation = run_checkpointed_generation(
                store=store,
                rows=self.rows,
                runtime=runtime,
                record_builder=lambda row, condition, seed: fake_record_builder(
                    torch, row, condition, seed
                ),
            )
            summary = finalize_run(
                run_dir=run_dir,
                manifest={"run_id": "qwen-run-final", "identity": {"fixture": True}},
                store=store,
                rows=self.rows,
                invocation=invocation,
            )
            self.assertEqual(summary["validation"]["expected_groups"], 9)
            self.assertEqual(summary["validation"]["observed_groups"], 9)
            self.assertEqual(summary["validation"]["expected_continuations"], 288)
            records = read_jsonl(run_dir / "continuations.jsonl")
            self.assertEqual({record["run_id"] for record in records}, {"qwen-run-final"})
            validate_records(records, self.rows)
            store.close()


if __name__ == "__main__":
    unittest.main()
