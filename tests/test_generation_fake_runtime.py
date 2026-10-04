from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
MODULE_PATH = PROJECT_ROOT / "scripts" / "generate_continuations.py"
SPEC = importlib.util.spec_from_file_location("qwen_generate_continuations", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generation
SPEC.loader.exec_module(generation)


class FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, index):
        return FakeVector(self.rows[index])

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [list(row) for row in self.rows]


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class FakeCuda:
    def __init__(self):
        self.seeds = []
        self.reset_calls = []
        self.empty_cache_calls = 0
        self.synchronize_calls = []

    def manual_seed_all(self, seed):
        self.seeds.append(seed)

    def empty_cache(self):
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    @staticmethod
    def max_memory_reserved(_device):
        return 18 * 1024**3

    @staticmethod
    def max_memory_allocated(_device):
        return 17 * 1024**3

    @staticmethod
    def memory_reserved(_device):
        return 18 * 1024**3

    @staticmethod
    def memory_allocated(_device):
        return 16 * 1024**3


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.seeds = []

    def inference_mode(self):
        return NullContext()

    def manual_seed(self, seed):
        self.seeds.append(seed)


class FakeTokenizer:
    pad_token_id = 248044
    chat_template = "frozen-instruct-template"

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        values = [10 + index for index, _ in enumerate(text.split(), start=1)]
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": FakeMatrix([values]),
                "attention_mask": FakeMatrix([[1] * len(values)]),
            }
        return {"input_ids": values}

    @staticmethod
    def convert_ids_to_tokens(token_ids):
        return [f"tok-{token_id}" for token_id in token_ids]

    @staticmethod
    def decode(token_ids, **_kwargs):
        return "|".join(str(token_id) for token_id in token_ids)

    @staticmethod
    def apply_chat_template(*_args, **_kwargs):
        raise AssertionError("RQ1 must not call the preserved template")


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=262144)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs["input_ids"].rows[0]
        # Sampling EOS as content must not shorten the fixed horizon.
        continuation = [248044, 20, 21, 22, 23]
        return FakeMatrix(
            [prefix + continuation for _ in range(kwargs["num_return_sequences"])]
        )


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


def frozen_config(tokenizer):
    return generation.fixed_generation_config(
        tokenizer,
        generation_config_class=lambda **kwargs: SimpleNamespace(**kwargs),
    )


class FrozenGenerationTests(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.runtime = {
            "torch": self.torch,
            "tokenizer": self.tokenizer,
            "model": self.model,
            "device": "cuda:0",
            "runtime_identity": {"fake_runtime": True},
        }

    def test_config_is_untruncated_sampling_and_disables_eos_stopping(self) -> None:
        config = frozen_config(self.tokenizer)
        self.assertTrue(config.do_sample)
        self.assertEqual(config.min_new_tokens, 5)
        self.assertEqual(config.max_new_tokens, 5)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_k, 0)
        self.assertEqual(config.top_p, 1.0)
        self.assertIsNone(config.eos_token_id)
        self.assertIsNone(config.forced_eos_token_id)

    def test_fake_runtime_generates_exact_raw_qwen_h5_records(self) -> None:
        record = generation.generate_record(
            fixture_row(),
            "M",
            11,
            self.runtime,
            generation_config=frozen_config(self.tokenizer),
        )
        self.assertEqual(record["schema"], generation.SCHEMA)
        self.assertIn("qwen3.5-9b", record["schema"])
        self.assertEqual(record["h_qwen_tokens"], 5)
        self.assertEqual(
            [key for key in record if key.startswith("h_")], ["h_qwen_tokens"]
        )
        self.assertEqual(len(record["continuations"]), 32)
        self.assertEqual(record["continuations"][0]["token_ids"], [248044, 20, 21, 22, 23])
        self.assertFalse(self.tokenizer.calls[0][1]["add_special_tokens"])
        self.assertEqual(self.model.calls[0]["num_return_sequences"], 32)

    def test_template_is_preserved_but_rq1_does_not_use_it(self) -> None:
        before = self.tokenizer.chat_template
        generation.generate_record(
            fixture_row(),
            "M",
            11,
            self.runtime,
            generation_config=frozen_config(self.tokenizer),
        )
        self.assertEqual(self.tokenizer.chat_template, before)

    def test_changed_sampling_is_rejected(self) -> None:
        config = frozen_config(self.tokenizer)
        config.top_k = 50
        with self.assertRaisesRegex(generation.ContinuationError, "top_k=50"):
            generation.generate_record(
                fixture_row(), "M", 11, self.runtime, generation_config=config
            )

    def test_runtime_wrapper_never_allows_download(self) -> None:
        sentinel = {"runtime": "ok"}
        with patch.object(
            generation.ENVIRONMENT, "resolve_runtime", return_value=sentinel
        ) as mocked:
            self.assertIs(
                generation.resolve_runtime(Path("cache"), "cuda:0"), sentinel
            )
        mocked.assert_called_once_with(
            Path("cache"), "cuda:0", allow_download=False
        )

    def test_benchmark_memory_probe_uses_exact_longest_prefix_and_32_returns(self) -> None:
        rows = []
        for index in range(3):
            row = fixture_row()
            row["triple_id"] = f"qwen-triple-{index}"
            row["i0"] = str(index)
            row["sentence_id"] = str(index + 1)
            row["source_sid"] = f"source-{index}"
            rows.append(row)
        rows[2]["i_prefix_q3"] = "one two three four five six seven eight nine"

        summary = generation.benchmark_generation(
            rows,
            self.runtime,
            item_count=3,
            block_count=1,
            full_item_count=3,
            model_load_seconds=2.0,
            generation_config=frozen_config(self.tokenizer),
        )
        probe = summary["gpu_memory_probe"]
        self.assertEqual(probe["triple_id"], "qwen-triple-2")
        self.assertEqual(probe["condition"], "I")
        self.assertEqual(probe["prefix_token_count"], 9)
        self.assertEqual(probe["num_return_sequences"], 32)
        self.assertEqual(probe["peak_reserved_bytes"], 18 * 1024**3)
        self.assertEqual(probe["peak_allocated_bytes"], 17 * 1024**3)
        self.assertEqual(self.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(self.model.calls[-1]["num_return_sequences"], 32)


if __name__ == "__main__":
    unittest.main()
