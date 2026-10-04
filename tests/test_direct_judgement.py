from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.direct_judgement import (
    CANDIDATE_ORDERS,
    CONTROL_COUNT,
    JUDGEMENT_CONTEXTS,
    DirectJudgementError,
    PromptItem,
    aggregate_formal_targets,
    aggregate_order_balanced,
    build_control_outputs,
    build_formal_triples,
    marked_prefix,
    prepare_response_sequences,
    prompt_item_from_control,
    render_user_message,
    score_prompt_item,
)


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool, bool, bool]] = []

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [ord(character) + 10 for character in text]

    @staticmethod
    def decode_ids(ids: list[int]) -> str:
        return "".join(chr(token_id - 10) for token_id in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        self.calls.append(
            (messages, tokenize, add_generation_prompt, enable_thinking)
        )
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        text = f"<user>{messages[0]['content']}</user><assistant>"
        return self._encode(text) if tokenize else text

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": self._encode(text)}


def semantic_fake_runtime(tokenizer: FakeChatTokenizer, apt_candidate: str):
    def score(combined_ids, prompt_count, response_ids):
        prompt = tokenizer.decode_ids(combined_ids[:prompt_count])
        response = tokenizer.decode_ids(response_ids)
        apt_response = "A" if f"Option A: {apt_candidate}\n" in prompt else "B"
        return -0.1 if response == apt_response else -2.0

    return {"tokenizer": tokenizer, "sequence_log_probability": score}


class DirectJudgementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = PromptItem(
            item_id="example",
            left_context="After the insult, Daniel ",
            m_word="exploded",
            q3_right_context=" in furious protest",
            apt_candidate="became extremely angry",
            inapt_candidate="burst into physical pieces",
            expected_unmarked_prefix=(
                "After the insult, Daniel exploded in furious protest"
            ),
        )

    def test_exact_user_message_and_marker_round_trip(self) -> None:
        self.assertEqual(
            marked_prefix(self.item),
            "After the insult, Daniel [TARGET: exploded] in furious protest",
        )
        rendered = render_user_message(self.item, "AI")
        self.assertEqual(
            rendered["user_message"],
            'Input:\nAfter the insult, Daniel [TARGET: exploded] in furious protest\n\n'
            "Which option better matches the meaning of the marked target in the input?\n\n"
            "Option A: became extremely angry\n"
            "Option B: burst into physical pieces\n\n"
            'Answer with exactly "A" or "B".',
        )
        word_rendered = render_user_message(
            self.item,
            "AI",
            judgement_context="word",
        )
        self.assertEqual(word_rendered["task_input"], "[TARGET: exploded]")
        self.assertEqual(
            rendered["user_message"].replace(rendered["task_input"], "<INPUT>"),
            word_rendered["user_message"].replace(
                word_rendered["task_input"], "<INPUT>"
            ),
        )

    def test_fake_runtime_scores_complete_labels_in_ai_then_ia_order(self) -> None:
        tokenizer = FakeChatTokenizer()
        runtime = semantic_fake_runtime(tokenizer, self.item.apt_candidate)
        records = score_prompt_item(self.item, runtime)
        self.assertEqual(
            [record["candidate_order"] for record in records],
            list(CANDIDATE_ORDERS),
        )
        self.assertTrue(all(record["apt_minus_inapt_nats"] > 0 for record in records))
        balanced = aggregate_order_balanced(records)
        self.assertEqual(len(balanced), 1)
        self.assertAlmostEqual(balanced[0]["p_t_nats"], 1.9)
        self.assertTrue(
            all(
                len(messages) == 1 and messages[0]["role"] == "user"
                for messages, _, _, enable_thinking in tokenizer.calls
                if enable_thinking is False
            )
        )
        self.assertTrue(all(call[3] is False for call in tokenizer.calls))

    def test_response_must_preserve_prompt_token_prefix(self) -> None:
        class BoundaryChangingTokenizer(FakeChatTokenizer):
            def __call__(self, text: str, *, add_special_tokens: bool):
                ids = super().__call__(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
                if text.endswith("A"):
                    ids[-2] += 1
                return {"input_ids": ids}

        tokenizer = BoundaryChangingTokenizer()
        message = render_user_message(self.item, "AI")["user_message"]
        with self.assertRaisesRegex(DirectJudgementError, "complete prefix"):
            prepare_response_sequences(tokenizer, message)

    def test_project_controls_are_twelve_clear_three_word_windows(self) -> None:
        payload = json.loads(
            (
                PROJECT_ROOT
                / "data"
                / "controls"
                / "qwen_instruct_direct_controls.json"
            ).read_text(encoding="utf-8")
        )
        items = [prompt_item_from_control(row) for row in payload["controls"]]
        self.assertEqual(len(items), CONTROL_COUNT)
        self.assertEqual(len({item.item_id for item in items}), CONTROL_COUNT)

    def test_control_gate_uses_order_balanced_positive_fraction(self) -> None:
        items = [
            PromptItem(
                item_id=f"control-{index:02d}",
                left_context="They ",
                m_word="rose",
                q3_right_context=" above the problem",
                apt_candidate="succeeded",
                inapt_candidate="moved upward physically",
                expected_unmarked_prefix="They rose above the problem",
            )
            for index in range(1, CONTROL_COUNT + 1)
        ]
        raw = []
        for index, item in enumerate(items):
            score = 1.0 if index < 8 else -1.0
            for judgement_context in JUDGEMENT_CONTEXTS:
                for candidate_order in CANDIDATE_ORDERS:
                    raw.append(
                        {
                            "item_id": item.item_id,
                            "judgement_context": judgement_context,
                            "candidate_order": candidate_order,
                            "m_word": item.m_word,
                            "option_a": item.apt_candidate,
                            "option_b": item.inapt_candidate,
                            "apt_response": "A",
                            "inapt_response": "B",
                            "task_input": (
                                marked_prefix(item)
                                if judgement_context == "context"
                                else f"[TARGET: {item.m_word}]"
                            ),
                            "user_message": "fixture",
                            "prompt_token_count": 4,
                            "response_a_token_ids": [1],
                            "response_b_token_ids": [2],
                            "log_probability_a_nats": -0.1,
                            "log_probability_b_nats": -1.1,
                            "apt_minus_inapt_nats": score,
                        }
                    )
        balanced = aggregate_order_balanced(raw)
        _, results, gate = build_control_outputs(
            "munch-qwen3p5-9b-test", items, raw, balanced
        )
        self.assertEqual(len(results), 2 * CONTROL_COUNT)
        self.assertEqual(gate["positive_count"], 8)
        self.assertEqual(gate["positive_gate_context"], "context")
        self.assertTrue(gate["word_only_has_no_correctness_gate"])
        self.assertTrue(gate["passed"])

    def test_p_i_is_computed_after_p_t_with_equal_triple_weights(self) -> None:
        triple_rows = []
        i0 = 0
        for sentence_id in range(1, 596):
            repetitions = 2 if sentence_id <= 285 else 1
            for repeat in range(repetitions):
                triple_rows.append(
                    {
                        "triple_id": f"munch-judgement-{i0}",
                        "i0": i0,
                        "sentence_id": sentence_id,
                        "source_sid": f"source-{((sentence_id - 1) % 553) + 1}",
                        "p_context_t_nats": float(sentence_id + repeat),
                        "p_word_t_nats": float(sentence_id + repeat) / 2.0,
                        "c_t_nats": float(sentence_id + repeat) / 2.0,
                    }
                )
                i0 += 1
        targets = aggregate_formal_targets(
            "munch-qwen3p5-9b-test", triple_rows
        )
        self.assertEqual(len(triple_rows), 880)
        self.assertEqual(len(targets), 595)
        self.assertEqual(targets[0]["p_context_i_nats"], 1.5)
        self.assertEqual(targets[0]["p_word_i_nats"], 0.75)
        self.assertEqual(targets[0]["c_i_nats"], 0.75)
        self.assertEqual(targets[300]["p_context_i_nats"], 301.0)

    def test_context_and_word_scores_form_one_contrast_per_item(self) -> None:
        tokenizer = FakeChatTokenizer()
        runtime = semantic_fake_runtime(tokenizer, self.item.apt_candidate)
        raw = []
        for judgement_context in JUDGEMENT_CONTEXTS:
            raw.extend(
                score_prompt_item(
                    self.item,
                    runtime,
                    judgement_context=judgement_context,
                )
            )
        balanced = aggregate_order_balanced(raw)
        self.assertEqual(
            [row["judgement_context"] for row in balanced],
            list(JUDGEMENT_CONTEXTS),
        )
        self.assertEqual(len(raw), 4)

    def test_formal_triples_require_both_matched_conditions(self) -> None:
        rows = []
        balanced = []
        for i0 in range(880):
            triple_id = f"munch-judgement-{i0}"
            rows.append(
                {
                    "triple_id": triple_id,
                    "i0": str(i0),
                    "sentence_id": str((i0 % 595) + 1),
                    "source_sid": f"source-{(i0 % 553) + 1}",
                    "genre": "fixture",
                }
            )
            for judgement_context, score in (("context", 1.5), ("word", 0.5)):
                balanced.append(
                    {
                        "item_id": triple_id,
                        "judgement_context": judgement_context,
                        "ai_apt_minus_inapt_nats": score,
                        "ia_apt_minus_inapt_nats": score,
                        "p_t_nats": score,
                    }
                )
        triples = build_formal_triples(
            "munch-qwen3p5-9b-test",
            rows,
            balanced,
        )
        self.assertEqual(len(triples), 880)
        self.assertEqual(triples[0]["p_context_t_nats"], 1.5)
        self.assertEqual(triples[0]["p_word_t_nats"], 0.5)
        self.assertEqual(triples[0]["c_t_nats"], 1.0)

        with self.assertRaisesRegex(
            DirectJudgementError,
            "one balanced context and word score",
        ):
            build_formal_triples(
                "munch-qwen3p5-9b-test",
                rows,
                balanced[:-1],
            )


if __name__ == "__main__":
    unittest.main()
