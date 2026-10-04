from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = PROJECT_ROOT / "scripts" / "tokenization_audit.py"
SPEC = importlib.util.spec_from_file_location("tokenization_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class FakeTokenizer:
    chat_template = "fake official instruct template"

    def __init__(self) -> None:
        self.encode_special_flags: list[bool] = []
        self.chat_calls: list[tuple[bool, bool, bool]] = []

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) + 100 for character in text]

    def __call__(self, text: str, *, add_special_tokens: bool, **_kwargs):
        self.encode_special_flags.append(add_special_tokens)
        return {"input_ids": self._encode(text)}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id - 100) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        self.chat_calls.append(
            (tokenize, add_generation_prompt, enable_thinking)
        )
        text = f"<user>{messages[0]['content']}</user><assistant>"
        return self._encode(text) if tokenize else text


class DifferentSuffixTokenizer(FakeTokenizer):
    ALIAS_SPACE = 900_001

    def __call__(self, text: str, *, add_special_tokens: bool, **kwargs):
        encoded = super().__call__(
            text, add_special_tokens=add_special_tokens, **kwargs
        )["input_ids"]
        prefix = "The claim extinguished"
        if text == prefix + " a lively debate":
            return {
                "input_ids": self._encode(prefix)
                + [self.ALIAS_SPACE]
                + self._encode("a lively debate")
            }
        return {"input_ids": encoded}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(
            " " if token_id == self.ALIAS_SPACE else chr(token_id - 100)
            for token_id in token_ids
        )


class BrokenChatBoundaryTokenizer(FakeTokenizer):
    def __call__(self, text: str, *, add_special_tokens: bool, **kwargs):
        ids = super().__call__(
            text, add_special_tokens=add_special_tokens, **kwargs
        )["input_ids"]
        if text.endswith("<assistant>B"):
            ids = [777_777] + ids
        return {"input_ids": ids}


def analysis_row(i0: int = 7) -> dict[str, str]:
    left = "The claim "
    q3 = " a lively debate"
    tail = " followed."
    words = {"m": "sparked", "a": "caused", "i": "extinguished"}
    row = {
        "triple_id": f"munch-judgement-{i0}",
        "i0": str(i0),
        "sentence_id": str(100 + i0),
        "source_sid": f"story-{i0}",
        "genre": "TEST",
        "novelty": "0.5",
        "critical_start": str(len(left)),
        "right_context_word_1": "a",
        "right_context_word_2": "lively",
        "right_context_word_3": "debate",
        "q3_right_context": q3,
    }
    for condition, word in words.items():
        row[f"{condition}_sentence"] = left + word + q3 + tail
        row[f"{condition}_word"] = word
        row[f"{condition}_critical_end"] = str(len(left + word))
        row[f"{condition}_prefix_q3"] = left + word + q3
    return row


class TokenizationAuditTests(unittest.TestCase):
    def test_all_three_interfaces_pass_and_use_raw_no_special_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        report = audit.audit_rows([analysis_row()], tokenizer)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["coverage"]["rows_received"], 1)
        self.assertEqual(report["coverage"]["rows_audited"], 1)
        self.assertEqual(report["coverage"]["rows_removed"], 0)
        self.assertEqual(
            report["checks"]["rq1_raw_prefixes"],
            {"cases": 3, "passed": 3, "failed": 0},
        )
        self.assertEqual(
            report["checks"]["surprisal_context_q3_boundaries"],
            {"cases": 2, "passed": 2, "failed": 0},
        )
        self.assertEqual(
            report["checks"]["surprisal_shared_a_i_q3_suffix"],
            {"cases": 1, "passed": 1, "failed": 0},
        )
        self.assertEqual(
            report["checks"]["rq2_chat_a_b_append_boundaries"],
            {"cases": 4, "passed": 4, "failed": 0},
        )
        self.assertTrue(tokenizer.encode_special_flags)
        self.assertFalse(any(tokenizer.encode_special_flags))
        self.assertEqual(
            tokenizer.chat_calls,
            [
                (True, True, False),
                (False, True, False),
                (True, True, False),
                (False, True, False),
                (True, True, False),
                (False, True, False),
                (True, True, False),
                (False, True, False),
            ],
        )

    def test_same_q3_text_with_different_a_i_suffix_ids_is_reported(self) -> None:
        report = audit.audit_rows([analysis_row()], DifferentSuffixTokenizer())

        self.assertEqual(report["status"], "fail")
        self.assertEqual(
            report["checks"]["surprisal_shared_a_i_q3_suffix"]["failed"], 1
        )
        self.assertEqual(
            report["failures"]["by_code"], {"A_I_Q3_SUFFIX_IDS_DIFFER": 1}
        )
        self.assertEqual(report["failures"]["rows"], [7])
        self.assertEqual(report["coverage"]["rows_removed"], 0)

    def test_rq2_append_failure_is_retained_while_all_rows_are_audited(self) -> None:
        rows = [analysis_row(7), analysis_row(8)]
        report = audit.audit_rows(rows, BrokenChatBoundaryTokenizer())

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["coverage"]["rows_received"], 2)
        self.assertEqual(report["coverage"]["rows_audited"], 2)
        self.assertEqual(report["coverage"]["rows_removed"], 0)
        self.assertTrue(report["coverage"]["failed_rows_retained"])
        self.assertEqual(
            report["checks"]["rq2_chat_a_b_append_boundaries"],
            {"cases": 8, "passed": 0, "failed": 8},
        )
        self.assertEqual(report["failures"]["rows"], [7, 8])
        self.assertEqual(
            report["failures"]["by_code"],
            {"RQ2_A_B_APPEND_BOUNDARY_FAILED": 8},
        )

    def test_report_adds_no_hash_chain(self) -> None:
        report = audit.audit_rows([analysis_row()], FakeTokenizer())

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key)
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        self.assertFalse(any("hash" in key.lower() for key in keys(report)))


if __name__ == "__main__":
    unittest.main()
