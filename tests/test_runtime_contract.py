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

from qwen_alignment import model_runtime


@contextmanager
def project_tempdir():
    path = PROJECT_ROOT / "runs" / f"test-runtime-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class FakeDevice:
    def __init__(self, value: str):
        if ":" in value:
            self.type, raw_index = value.split(":", 1)
            self.index = int(raw_index)
        else:
            self.type = value
            self.index = None

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FakeCuda:
    def __init__(self, available: bool = True, count: int = 1):
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def is_bf16_supported(self) -> bool:
        return True


class FakeTorch:
    def __init__(self, available: bool = True, count: int = 1):
        self.cuda = FakeCuda(available, count)

    @staticmethod
    def device(value: str) -> FakeDevice:
        return FakeDevice(value)


class FakeParameter:
    dtype = "torch.bfloat16"
    device = "cuda:0"

    @staticmethod
    def numel() -> int:
        return model_runtime.EXPECTED_PARAMETER_COUNT


def pinned_text_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "dtype": "bfloat16",
    }


def pinned_config() -> dict[str, object]:
    return {
        "architectures": [model_runtime.EXPECTED_REPOSITORY_ARCHITECTURE],
        "model_type": "qwen3_5",
        "text_config": pinned_text_config(),
    }


class Qwen3_5ForCausalLM:
    training = False

    def __init__(self) -> None:
        self.config = {
            **pinned_text_config(),
            "_attn_implementation": model_runtime.ATTENTION_IMPLEMENTATION,
        }

    @staticmethod
    def parameters():
        return iter((FakeParameter(),))


class ArchitectureContractTests(unittest.TestCase):
    def test_exact_qwen35_text_contract_is_accepted(self) -> None:
        self.assertEqual(model_runtime.DEFAULT_MODEL_ID, "Qwen/Qwen3.5-9B")
        self.assertEqual(
            model_runtime.DEFAULT_REVISION,
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        )
        self.assertEqual(
            model_runtime.EXPECTED_REPOSITORY_ARCHITECTURE,
            "Qwen3_5ForConditionalGeneration",
        )
        self.assertEqual(
            model_runtime.EXPECTED_ARCHITECTURE, "Qwen3_5ForCausalLM"
        )
        self.assertEqual(model_runtime.EXPECTED_PARAMETER_COUNT, 8_953_803_264)
        observed = model_runtime.validate_architecture_config(pinned_config())
        self.assertEqual(observed["model_type"], "qwen3_5_text")
        self.assertEqual(observed["num_hidden_layers"], 32)
        self.assertEqual(observed["hidden_size"], 4096)
        self.assertEqual(observed["vocab_size"], 248320)
        self.assertEqual(observed["max_position_embeddings"], 262144)
        self.assertEqual(observed["num_attention_heads"], 16)
        self.assertEqual(observed["num_key_value_heads"], 4)

        loaded_observed = model_runtime.validate_architecture_config(
            pinned_text_config()
        )
        self.assertEqual(loaded_observed, observed)

    def test_quantised_or_wrong_architecture_is_rejected(self) -> None:
        config = pinned_config()
        config["text_config"]["hidden_size"] = 3584
        config["quantization_config"] = {"load_in_4bit": True}
        with self.assertRaisesRegex(RuntimeError, "hidden_size.*quantization"):
            model_runtime.validate_architecture_config(config)

    def test_composite_config_without_text_config_is_rejected(self) -> None:
        config = pinned_config()
        del config["text_config"]
        with self.assertRaisesRegex(RuntimeError, "model_type.*num_hidden_layers"):
            model_runtime.validate_architecture_config(config)

    def test_loaded_text_only_causal_runtime_is_accepted(self) -> None:
        tokenizer = type("Tokenizer", (), {"chat_template": "{{ messages }}"})()
        observed = model_runtime.validate_loaded_runtime(
            Qwen3_5ForCausalLM(), tokenizer, FakeDevice("cuda:0"), FakeTorch()
        )
        self.assertEqual(observed["architecture"], "Qwen3_5ForCausalLM")
        self.assertEqual(observed["model_type"], "qwen3_5_text")
        self.assertEqual(observed["parameter_count"], 8_953_803_264)
        self.assertEqual(observed["parameter_dtypes"], ["bfloat16"])

    def test_only_visible_cuda_zero_is_accepted(self) -> None:
        self.assertEqual(
            str(model_runtime.select_device("cuda", FakeTorch())), "cuda:0"
        )
        with self.assertRaisesRegex(RuntimeError, "requires one explicit CUDA"):
            model_runtime.select_device("cpu", FakeTorch())
        with self.assertRaisesRegex(RuntimeError, "cuda:0"):
            model_runtime.select_device("cuda:1", FakeTorch(count=2))
        with self.assertRaisesRegex(RuntimeError, "is_available"):
            model_runtime.select_device("cuda:0", FakeTorch(available=False))

    def test_conversational_chat_template_is_required_and_preserved(self) -> None:
        tokenizer = type("Tokenizer", (), {"chat_template": "{{ messages }}"})()
        model_runtime.validate_chat_template(tokenizer)
        self.assertEqual(tokenizer.chat_template, "{{ messages }}")
        tokenizer.chat_template = None
        with self.assertRaisesRegex(RuntimeError, "no chat template"):
            model_runtime.validate_chat_template(tokenizer)


class SnapshotContractTests(unittest.TestCase):
    def test_exact_text_only_artifact_names_are_frozen(self) -> None:
        self.assertEqual(
            model_runtime.MODEL_ARTIFACT_FILES,
            (
                "config.json",
                "chat_template.jinja",
                "merges.txt",
                "model.safetensors-00001-of-00004.safetensors",
                "model.safetensors-00002-of-00004.safetensors",
                "model.safetensors-00003-of-00004.safetensors",
                "model.safetensors-00004-of-00004.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
        )
        self.assertEqual(
            model_runtime.MODEL_SHARDS,
            model_runtime.MODEL_ARTIFACT_FILES[3:7],
        )

    def test_snapshot_resolution_is_offline_unless_explicitly_allowed(self) -> None:
        with project_tempdir() as directory:
            snapshot = directory / model_runtime.DEFAULT_REVISION
            snapshot.mkdir()
            calls: list[dict[str, object]] = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)

            resolved = model_runtime.resolve_snapshot(
                directory, snapshot_download=fake_download
            )
            self.assertEqual(resolved, snapshot)
            self.assertTrue(calls[-1]["local_files_only"])
            model_runtime.resolve_snapshot(
                directory,
                allow_download=True,
                snapshot_download=fake_download,
            )
            self.assertFalse(calls[-1]["local_files_only"])
            self.assertEqual(calls[-1]["revision"], model_runtime.DEFAULT_REVISION)
            self.assertEqual(
                calls[-1]["allow_patterns"], model_runtime.MODEL_ARTIFACT_FILES
            )

    def test_every_runtime_artifact_exists_and_shard_index_is_checked(self) -> None:
        with project_tempdir() as directory:
            snapshot = directory
            for name in model_runtime.MODEL_ARTIFACT_FILES:
                path = snapshot / name
                path.write_bytes(b"fixture")
            (snapshot / "config.json").write_text(
                json.dumps(pinned_config()), encoding="utf-8"
            )
            weight_map = {
                f"weight_{index}": shard
                for index, shard in enumerate(model_runtime.MODEL_SHARDS)
            }
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            manifest = model_runtime.verify_artifacts(snapshot)
            self.assertEqual(tuple(manifest), model_runtime.MODEL_ARTIFACT_FILES)
            for details in manifest.values():
                self.assertEqual(set(details), {"bytes"})
                self.assertGreater(int(details["bytes"]), 0)


if __name__ == "__main__":
    unittest.main()
