"""Matched contextual and word-only direct judgements for frozen RQ2.

The model receives one user message through its official chat template.  The
complete assistant labels ``A`` and ``B`` are scored by teacher forcing; this
module never generates text, samples, or uses a random seed.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .continuation_input import orthographic_words
from .model_runtime import DEFAULT_MODEL_ID, DEFAULT_REVISION, EXPECTED_MAX_POSITIONS


FORMAL_TRIPLES = 880
FORMAL_ORDER_ROWS = 3_520
FORMAL_TARGETS = 595
FORMAL_SOURCE_SIDS = 553

CANDIDATE_ORDERS = ("AI", "IA")
JUDGEMENT_CONTEXTS = ("context", "word")
ASSISTANT_RESPONSES = ("A", "B")
CONTROL_COUNT = 12
CONTROL_MINIMUM_POSITIVE_FRACTION = 0.60

USER_MESSAGE_TEMPLATE = (
    "Input:\n{task_input}\n\n"
    "Which option better matches the meaning of the marked target in the input?\n\n"
    "Option A: {candidate_a}\n"
    "Option B: {candidate_b}\n\n"
    'Answer with exactly "A" or "B".'
)

ORDER_SCHEMA = "munch-qwen3.5-9b-matched-direct-order/v2"
TRIPLE_SCHEMA = "munch-qwen3.5-9b-matched-direct-triple/v2"
TARGET_SCHEMA = "munch-qwen3.5-9b-matched-direct-target/v2"
CONTROL_ORDER_SCHEMA = "munch-qwen3.5-9b-matched-direct-control-order/v2"
CONTROL_RESULT_SCHEMA = "munch-qwen3.5-9b-matched-direct-control-result/v2"
SUMMARY_SCHEMA = "munch-qwen3.5-9b-matched-direct-summary/v2"
MANIFEST_SCHEMA = "munch-qwen3.5-9b-matched-direct-manifest/v2"

ORDER_COLUMNS = (
    "schema",
    "run_id",
    "triple_id",
    "i0",
    "sentence_id",
    "source_sid",
    "genre",
    "judgement_context",
    "candidate_order",
    "m_word",
    "option_a",
    "option_b",
    "apt_response",
    "inapt_response",
    "task_input",
    "user_message",
    "prompt_token_count",
    "response_a_token_ids",
    "response_b_token_ids",
    "log_probability_a_nats",
    "log_probability_b_nats",
    "apt_minus_inapt_nats",
)

TRIPLE_COLUMNS = (
    "schema",
    "run_id",
    "triple_id",
    "i0",
    "sentence_id",
    "source_sid",
    "genre",
    "context_ai_apt_minus_inapt_nats",
    "context_ia_apt_minus_inapt_nats",
    "p_context_t_nats",
    "word_ai_apt_minus_inapt_nats",
    "word_ia_apt_minus_inapt_nats",
    "p_word_t_nats",
    "c_t_nats",
)

TARGET_COLUMNS = (
    "schema",
    "run_id",
    "sentence_id",
    "source_sid",
    "triple_count",
    "triple_ids",
    "i0_values",
    "p_context_i_nats",
    "p_word_i_nats",
    "c_i_nats",
)

CONTROL_ORDER_COLUMNS = (
    "schema",
    "run_id",
    "control_id",
    "judgement_context",
    "candidate_order",
    "m_word",
    "option_a",
    "option_b",
    "apt_response",
    "inapt_response",
    "task_input",
    "user_message",
    "prompt_token_count",
    "response_a_token_ids",
    "response_b_token_ids",
    "log_probability_a_nats",
    "log_probability_b_nats",
    "apt_minus_inapt_nats",
)

CONTROL_RESULT_COLUMNS = (
    "schema",
    "run_id",
    "control_id",
    "judgement_context",
    "ai_apt_minus_inapt_nats",
    "ia_apt_minus_inapt_nats",
    "p_t_nats",
    "positive",
    "used_for_positive_gate",
)


class DirectJudgementError(RuntimeError):
    """Raised when the frozen prompt, token boundary, or hierarchy is invalid."""


@dataclass(frozen=True)
class PromptItem:
    item_id: str
    left_context: str
    m_word: str
    q3_right_context: str
    apt_candidate: str
    inapt_candidate: str
    expected_unmarked_prefix: str


def _strict_int(value: object, label: str) -> int:
    text = str(value)
    try:
        parsed = int(text)
    except ValueError as exc:
        raise DirectJudgementError(f"{label} is not an integer: {value!r}") from exc
    if str(parsed) != text:
        raise DirectJudgementError(f"{label} is not canonical: {value!r}")
    return parsed


def _one_line_nonempty(value: str, label: str) -> None:
    if not value:
        raise DirectJudgementError(f"{label} is empty")
    if "\r" in value or "\n" in value:
        raise DirectJudgementError(f"{label} contains a line break")


def validate_prompt_item(item: PromptItem) -> None:
    _one_line_nonempty(item.item_id, "item_id")
    for label, value in (
        ("m_word", item.m_word),
        ("q3_right_context", item.q3_right_context),
        ("apt_candidate", item.apt_candidate),
        ("inapt_candidate", item.inapt_candidate),
    ):
        _one_line_nonempty(value, f"{item.item_id} {label}")
    if '"' in item.m_word:
        raise DirectJudgementError(f"{item.item_id} m_word contains a double quote")
    if item.apt_candidate == item.inapt_candidate:
        raise DirectJudgementError(f"{item.item_id} candidates are identical")
    expected = item.left_context + item.m_word + item.q3_right_context
    if expected != item.expected_unmarked_prefix:
        raise DirectJudgementError(f"{item.item_id} unmarked prefix does not match")


def prompt_item_from_analysis_row(row: Mapping[str, str]) -> PromptItem:
    """Recover the exact left context, M target, and q3 continuation."""
    i0 = _strict_int(row["i0"], "i0")
    start = _strict_int(row["critical_start"], f"i0={i0} critical_start")
    end = _strict_int(row["m_critical_end"], f"i0={i0} m_critical_end")
    sentence = row["m_sentence"]
    if not (0 <= start < end <= len(sentence)):
        raise DirectJudgementError(f"i0={i0} has invalid M target offsets")
    if sentence[start:end] != row["m_word"]:
        raise DirectJudgementError(f"i0={i0} target offsets do not recover m_word")
    item = PromptItem(
        item_id=row["triple_id"],
        left_context=sentence[:start],
        m_word=row["m_word"],
        q3_right_context=row["q3_right_context"],
        apt_candidate=row["a_word"],
        inapt_candidate=row["i_word"],
        expected_unmarked_prefix=row["m_prefix_q3"],
    )
    validate_prompt_item(item)
    if len(orthographic_words(item.q3_right_context)) != 3:
        raise DirectJudgementError(f"i0={i0} q3_right_context must contain three words")
    return item


def prompt_item_from_control(row: Mapping[str, object]) -> PromptItem:
    required = (
        "control_id",
        "left_context",
        "m_word",
        "q3_right_context",
        "apt_candidate",
        "inapt_candidate",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise DirectJudgementError(f"control is missing fields: {missing}")
    left = str(row["left_context"])
    m_word = str(row["m_word"])
    q3 = str(row["q3_right_context"])
    item = PromptItem(
        item_id=str(row["control_id"]),
        left_context=left,
        m_word=m_word,
        q3_right_context=q3,
        apt_candidate=str(row["apt_candidate"]),
        inapt_candidate=str(row["inapt_candidate"]),
        expected_unmarked_prefix=left + m_word + q3,
    )
    validate_prompt_item(item)
    if len(orthographic_words(q3)) != 3:
        raise DirectJudgementError(
            f"control {item.item_id} q3_right_context must contain three words"
        )
    return item


def marked_prefix(item: PromptItem) -> str:
    """Insert the target marker and verify its removal restores the RQ1 prefix."""
    marker = f"[TARGET: {item.m_word}]"
    marked = item.left_context + marker + item.q3_right_context
    marker_start = len(item.left_context)
    marker_end = marker_start + len(marker)
    recovered = marked[:marker_start] + item.m_word + marked[marker_end:]
    if recovered != item.expected_unmarked_prefix:
        raise DirectJudgementError(f"{item.item_id} target-marker round trip failed")
    return marked


def render_user_message(
    item: PromptItem,
    candidate_order: str,
    *,
    judgement_context: str = "context",
) -> dict[str, str]:
    """Render one matched prompt and align response labels by semantic role.

    The task skeleton is identical across conditions.  Only ``task_input``
    changes: the RQ1-visible metaphor prefix for ``context``, or the marked
    target expression alone for ``word``.
    """
    if candidate_order == "AI":
        option_a = item.apt_candidate
        option_b = item.inapt_candidate
        apt_response, inapt_response = "A", "B"
    elif candidate_order == "IA":
        option_a = item.inapt_candidate
        option_b = item.apt_candidate
        apt_response, inapt_response = "B", "A"
    else:
        raise DirectJudgementError(f"invalid candidate order: {candidate_order!r}")
    if judgement_context == "context":
        task_input = marked_prefix(item)
    elif judgement_context == "word":
        task_input = f"[TARGET: {item.m_word}]"
    else:
        raise DirectJudgementError(
            f"invalid judgement context: {judgement_context!r}"
        )
    message = USER_MESSAGE_TEMPLATE.format(
        task_input=task_input,
        candidate_a=option_a,
        candidate_b=option_b,
    )
    return {
        "judgement_context": judgement_context,
        "candidate_order": candidate_order,
        "m_word": item.m_word,
        "option_a": option_a,
        "option_b": option_b,
        "apt_response": apt_response,
        "inapt_response": inapt_response,
        "task_input": task_input,
        "user_message": message,
    }


def _token_ids(value: Any, label: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise DirectJudgementError(f"{label} unexpectedly contains a batch")
        value = value[0]
    if not isinstance(value, list) or not value:
        raise DirectJudgementError(f"{label} produced no token IDs")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in value):
        raise DirectJudgementError(f"{label} produced invalid token IDs")
    return list(value)


def prepare_response_sequences(
    tokenizer: Any, user_message: str
) -> dict[str, object]:
    """Apply the official template, then establish the A/B append boundaries."""
    messages = [{"role": "user", "content": user_message}]
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        "chat generation prompt",
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise DirectJudgementError("chat template did not render text")
    retokenized_prompt = _token_ids(
        tokenizer(prompt_text, add_special_tokens=False),
        "rendered chat generation prompt",
    )
    if retokenized_prompt != prompt_ids:
        raise DirectJudgementError(
            "tokenize=True and tokenize=False chat-template paths disagree"
        )

    combined: dict[str, list[int]] = {}
    response_ids: dict[str, list[int]] = {}
    for response in ASSISTANT_RESPONSES:
        full_ids = _token_ids(
            tokenizer(prompt_text + response, add_special_tokens=False),
            f"prompt plus response {response}",
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise DirectJudgementError(
                f"prompt is not a complete prefix after appending response {response}"
            )
        appended = full_ids[len(prompt_ids) :]
        if not appended:
            raise DirectJudgementError(f"response {response} adds no token")
        combined[response] = full_ids
        response_ids[response] = appended
    if len(response_ids["A"]) != len(response_ids["B"]):
        raise DirectJudgementError("responses A and B have different token counts")
    if response_ids["A"] == response_ids["B"]:
        raise DirectJudgementError("responses A and B have identical token IDs")
    return {
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "combined_ids": combined,
        "response_ids": response_ids,
    }


def _model_context_limit(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    return int(value if value is not None else EXPECTED_MAX_POSITIONS)


def teacher_forced_response_scores(
    prepared: Mapping[str, object], runtime: Mapping[str, Any]
) -> dict[str, float]:
    """Sum log probabilities over every token in each complete response label."""
    prompt_ids = list(prepared["prompt_ids"])
    combined = prepared["combined_ids"]
    response_ids = prepared["response_ids"]
    if not isinstance(combined, Mapping) or not isinstance(response_ids, Mapping):
        raise DirectJudgementError("prepared response sequences are malformed")
    response_length = len(response_ids["A"])
    if response_length < 1 or len(response_ids["B"]) != response_length:
        raise DirectJudgementError("prepared responses do not share one positive length")

    scorer = runtime.get("sequence_log_probability")
    if scorer is not None:
        if not callable(scorer):
            raise DirectJudgementError("sequence_log_probability is not callable")
        scores = {
            response: float(
                scorer(
                    list(combined[response]),
                    len(prompt_ids),
                    list(response_ids[response]),
                )
            )
            for response in ASSISTANT_RESPONSES
        }
    else:
        torch = runtime["torch"]
        model = runtime["model"]
        device = runtime["device"]
        batch_ids = [list(combined[response]) for response in ASSISTANT_RESPONSES]
        if any(len(ids) > _model_context_limit(model) for ids in batch_ids):
            raise DirectJudgementError("prompt plus response exceeds model context")
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            expected_prefix_shape = (2, len(batch_ids[0]))
            if tuple(logits.shape[:2]) != expected_prefix_shape:
                raise DirectJudgementError("model logits have unexpected dimensions")
            start = len(prompt_ids) - 1
            stop = start + response_length
            response_logits = logits[:, start:stop, :].float()
            targets = input_ids[:, len(prompt_ids) :]
            selected = torch.log_softmax(response_logits, dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            totals = selected.sum(dim=1).detach().cpu().tolist()
        scores = {response: float(totals[index]) for index, response in enumerate(ASSISTANT_RESPONSES)}
    if not all(math.isfinite(value) for value in scores.values()):
        raise DirectJudgementError("assistant response score is non-finite")
    return scores


def score_prompt_item(
    item: PromptItem,
    runtime: Mapping[str, Any],
    *,
    judgement_context: str = "context",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    tokenizer = runtime["tokenizer"]
    for candidate_order in CANDIDATE_ORDERS:
        rendered = render_user_message(
            item,
            candidate_order,
            judgement_context=judgement_context,
        )
        prepared = prepare_response_sequences(tokenizer, rendered["user_message"])
        scores = teacher_forced_response_scores(prepared, runtime)
        apt = rendered["apt_response"]
        inapt = rendered["inapt_response"]
        contrast = scores[apt] - scores[inapt]
        if not math.isfinite(contrast):
            raise DirectJudgementError(f"{item.item_id} has a non-finite contrast")
        records.append(
            {
                "item_id": item.item_id,
                **rendered,
                "prompt_token_count": len(prepared["prompt_ids"]),
                "response_a_token_ids": list(prepared["response_ids"]["A"]),
                "response_b_token_ids": list(prepared["response_ids"]["B"]),
                "log_probability_a_nats": scores["A"],
                "log_probability_b_nats": scores["B"],
                "apt_minus_inapt_nats": contrast,
            }
        )
    return records


def score_prompt_items(
    items: Sequence[PromptItem],
    runtime: Mapping[str, Any],
    *,
    judgement_context: str = "context",
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    if not items:
        raise DirectJudgementError("no prompt items were supplied")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        if item.item_id in seen:
            raise DirectJudgementError(f"duplicate item ID: {item.item_id}")
        seen.add(item.item_id)
        records.extend(
            score_prompt_item(
                item,
                runtime,
                judgement_context=judgement_context,
            )
        )
        if progress is not None:
            progress(index, len(items))
    return records


def aggregate_order_balanced(
    order_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compute P_t within each input condition after AI/IA averaging."""
    grouped: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    order_of_items: list[tuple[str, str]] = []
    for record in order_records:
        item_id = str(record["item_id"])
        judgement_context = str(record.get("judgement_context", ""))
        if judgement_context not in JUDGEMENT_CONTEXTS:
            raise DirectJudgementError(
                f"invalid judgement context: {judgement_context!r}"
            )
        key = (item_id, judgement_context)
        candidate_order = str(record["candidate_order"])
        if key not in grouped:
            order_of_items.append(key)
        if candidate_order in grouped[key]:
            raise DirectJudgementError(
                f"duplicate {candidate_order} record for {item_id}/{judgement_context}"
            )
        grouped[key][candidate_order] = record

    results: list[dict[str, object]] = []
    for item_id, judgement_context in order_of_items:
        records = grouped[(item_id, judgement_context)]
        if tuple(records) != CANDIDATE_ORDERS:
            raise DirectJudgementError(
                f"{item_id}/{judgement_context} does not have one AI then one IA row"
            )
        ai = float(records["AI"]["apt_minus_inapt_nats"])
        ia = float(records["IA"]["apt_minus_inapt_nats"])
        p_t = math.fsum((ai, ia)) / 2.0
        if not all(math.isfinite(value) for value in (ai, ia, p_t)):
            raise DirectJudgementError(f"{item_id} aggregation is non-finite")
        results.append(
            {
                "item_id": item_id,
                "judgement_context": judgement_context,
                "ai_apt_minus_inapt_nats": ai,
                "ia_apt_minus_inapt_nats": ia,
                "p_t_nats": p_t,
            }
        )
    return results


def validate_formal_rows(rows: Sequence[Mapping[str, str]]) -> None:
    counts = {
        "triples": len(rows),
        "targets": len({row["sentence_id"] for row in rows}),
        "source_sids": len({row["source_sid"] for row in rows}),
    }
    expected = {
        "triples": FORMAL_TRIPLES,
        "targets": FORMAL_TARGETS,
        "source_sids": FORMAL_SOURCE_SIDS,
    }
    if counts != expected:
        raise DirectJudgementError(
            f"formal hierarchy mismatch: expected {expected}, observed {counts}"
        )
    seen_triples: set[str] = set()
    sentence_sources: dict[str, set[str]] = defaultdict(set)
    previous_i0 = -1
    for row in rows:
        i0 = _strict_int(row["i0"], "i0")
        if i0 <= previous_i0:
            raise DirectJudgementError("formal rows are not in increasing i0 order")
        previous_i0 = i0
        triple_id = row["triple_id"]
        if triple_id != f"munch-judgement-{i0}":
            raise DirectJudgementError(f"triple_id/i0 mismatch for {triple_id}")
        if triple_id in seen_triples:
            raise DirectJudgementError(f"duplicate triple_id: {triple_id}")
        seen_triples.add(triple_id)
        sentence_sources[row["sentence_id"]].add(row["source_sid"])
    if any(len(values) != 1 for values in sentence_sources.values()):
        raise DirectJudgementError("a sentence_id maps to multiple source_sid values")


def attach_formal_order_metadata(
    run_id: str,
    rows: Sequence[Mapping[str, str]],
    order_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {row["triple_id"]: row for row in rows}
    output: list[dict[str, object]] = []
    for record in order_records:
        item_id = str(record["item_id"])
        row = by_id.get(item_id)
        if row is None:
            raise DirectJudgementError(f"unknown scored triple: {item_id}")
        output.append(
            {
                "schema": ORDER_SCHEMA,
                "run_id": run_id,
                "triple_id": item_id,
                "i0": _strict_int(row["i0"], "i0"),
                "sentence_id": _strict_int(row["sentence_id"], "sentence_id"),
                "source_sid": row["source_sid"],
                "genre": row["genre"],
                **{
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if key in {"response_a_token_ids", "response_b_token_ids"}
                        else value
                    )
                    for key, value in record.items()
                    if key != "item_id"
                },
            }
        )
    if len(output) != FORMAL_ORDER_ROWS:
        raise DirectJudgementError(
            f"formal order output has {len(output)} rows, expected {FORMAL_ORDER_ROWS}"
        )
    return output


def build_formal_triples(
    run_id: str,
    rows: Sequence[Mapping[str, str]],
    balanced_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {row["triple_id"]: row for row in rows}
    by_item_context: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for record in balanced_records:
        item_id = str(record["item_id"])
        judgement_context = str(record["judgement_context"])
        if judgement_context in by_item_context[item_id]:
            raise DirectJudgementError(
                f"duplicate balanced {judgement_context} record for {item_id}"
            )
        by_item_context[item_id][judgement_context] = record
    if set(by_item_context) != set(by_id):
        raise DirectJudgementError(
            "balanced matched scores do not cover the formal triple set exactly"
        )
    output: list[dict[str, object]] = []
    for row in rows:
        item_id = row["triple_id"]
        row = by_id.get(item_id)
        if row is None:
            raise DirectJudgementError(f"unknown balanced triple: {item_id}")
        contexts = by_item_context.get(item_id, {})
        if set(contexts) != set(JUDGEMENT_CONTEXTS):
            raise DirectJudgementError(
                f"{item_id} does not have one balanced context and word score"
            )
        context = contexts["context"]
        word = contexts["word"]
        p_context = float(context["p_t_nats"])
        p_word = float(word["p_t_nats"])
        c_t = p_context - p_word
        if not all(math.isfinite(value) for value in (p_context, p_word, c_t)):
            raise DirectJudgementError(f"{item_id} has a non-finite matched score")
        output.append(
            {
                "schema": TRIPLE_SCHEMA,
                "run_id": run_id,
                "triple_id": item_id,
                "i0": _strict_int(row["i0"], "i0"),
                "sentence_id": _strict_int(row["sentence_id"], "sentence_id"),
                "source_sid": row["source_sid"],
                "genre": row["genre"],
                "context_ai_apt_minus_inapt_nats": context[
                    "ai_apt_minus_inapt_nats"
                ],
                "context_ia_apt_minus_inapt_nats": context[
                    "ia_apt_minus_inapt_nats"
                ],
                "p_context_t_nats": p_context,
                "word_ai_apt_minus_inapt_nats": word[
                    "ai_apt_minus_inapt_nats"
                ],
                "word_ia_apt_minus_inapt_nats": word[
                    "ia_apt_minus_inapt_nats"
                ],
                "p_word_t_nats": p_word,
                "c_t_nats": c_t,
            }
        )
    if len(output) != FORMAL_TRIPLES:
        raise DirectJudgementError(
            f"formal triple output has {len(output)} rows, expected {FORMAL_TRIPLES}"
        )
    return output


def aggregate_formal_targets(
    run_id: str, triple_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Aggregate matched P_context, P_word, and C within sentence_id."""
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in triple_rows:
        grouped[int(row["sentence_id"])].append(row)
    output: list[dict[str, object]] = []
    for sentence_id in sorted(grouped):
        rows = grouped[sentence_id]
        sources = {str(row["source_sid"]) for row in rows}
        if len(sources) != 1:
            raise DirectJudgementError(
                f"sentence_id={sentence_id} maps to multiple source_sid values"
            )
        p_context_i = math.fsum(
            float(row["p_context_t_nats"]) for row in rows
        ) / len(rows)
        p_word_i = math.fsum(float(row["p_word_t_nats"]) for row in rows) / len(rows)
        c_i = math.fsum(float(row["c_t_nats"]) for row in rows) / len(rows)
        if not all(math.isfinite(value) for value in (p_context_i, p_word_i, c_i)):
            raise DirectJudgementError(
                f"sentence_id={sentence_id} has a non-finite matched score"
            )
        if not math.isclose(
            c_i,
            p_context_i - p_word_i,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DirectJudgementError(
                f"sentence_id={sentence_id} has inconsistent context contrast"
            )
        rows_by_i0 = sorted(rows, key=lambda row: int(row["i0"]))
        output.append(
            {
                "schema": TARGET_SCHEMA,
                "run_id": run_id,
                "sentence_id": sentence_id,
                "source_sid": next(iter(sources)),
                "triple_count": len(rows),
                "triple_ids": json.dumps(
                    [str(row["triple_id"]) for row in rows_by_i0],
                    separators=(",", ":"),
                ),
                "i0_values": json.dumps(
                    [int(row["i0"]) for row in rows_by_i0],
                    separators=(",", ":"),
                ),
                "p_context_i_nats": p_context_i,
                "p_word_i_nats": p_word_i,
                "c_i_nats": c_i,
            }
        )
    if len(output) != FORMAL_TARGETS:
        raise DirectJudgementError(
            f"formal target output has {len(output)} rows, expected {FORMAL_TARGETS}"
        )
    if len({row["source_sid"] for row in output}) != FORMAL_SOURCE_SIDS:
        raise DirectJudgementError(
            f"formal target output does not contain {FORMAL_SOURCE_SIDS} source clusters"
        )
    return output


def build_control_outputs(
    run_id: str,
    items: Sequence[PromptItem],
    order_records: Sequence[Mapping[str, object]],
    balanced_records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if len(items) != CONTROL_COUNT:
        raise DirectJudgementError(f"expected {CONTROL_COUNT} controls")
    order_output: list[dict[str, object]] = []
    for record in order_records:
        order_output.append(
            {
                "schema": CONTROL_ORDER_SCHEMA,
                "run_id": run_id,
                "control_id": record["item_id"],
                **{
                    key: (
                        json.dumps(value, separators=(",", ":"))
                        if key in {"response_a_token_ids", "response_b_token_ids"}
                        else value
                    )
                    for key, value in record.items()
                    if key != "item_id"
                },
            }
        )
    result_output: list[dict[str, object]] = []
    for record in balanced_records:
        p_t = float(record["p_t_nats"])
        result_output.append(
            {
                "schema": CONTROL_RESULT_SCHEMA,
                "run_id": run_id,
                "control_id": record["item_id"],
                "judgement_context": record["judgement_context"],
                "ai_apt_minus_inapt_nats": record["ai_apt_minus_inapt_nats"],
                "ia_apt_minus_inapt_nats": record["ia_apt_minus_inapt_nats"],
                "p_t_nats": p_t,
                "positive": p_t > 0.0,
                "used_for_positive_gate": (
                    record["judgement_context"] == "context"
                ),
            }
        )
    item_contexts: dict[str, set[str]] = defaultdict(set)
    for row in result_output:
        item_contexts[str(row["control_id"])].add(str(row["judgement_context"]))
    if len(item_contexts) != CONTROL_COUNT or any(
        contexts != set(JUDGEMENT_CONTEXTS) for contexts in item_contexts.values()
    ):
        raise DirectJudgementError(
            "each control must have one context and one word-only score"
        )
    contextual_results = [
        row for row in result_output if bool(row["used_for_positive_gate"])
    ]
    positive_count = sum(bool(row["positive"]) for row in contextual_results)
    positive_fraction = positive_count / CONTROL_COUNT
    passed = positive_fraction >= CONTROL_MINIMUM_POSITIVE_FRACTION
    gate = {
        "passed": passed,
        "control_count": CONTROL_COUNT,
        "order_row_count": len(order_output),
        "response_score_count": 2 * len(order_output),
        "result_row_count": len(result_output),
        "positive_count": positive_count,
        "positive_fraction": positive_fraction,
        "minimum_positive_fraction": CONTROL_MINIMUM_POSITIVE_FRACTION,
        "all_scores_finite": all(
            math.isfinite(float(row["p_t_nats"])) for row in result_output
        ),
        "positive_gate_context": "context",
        "word_only_has_no_correctness_gate": True,
    }
    if (
        len(order_output) != 2 * len(JUDGEMENT_CONTEXTS) * CONTROL_COUNT
        or len(result_output) != len(JUDGEMENT_CONTEXTS) * CONTROL_COUNT
    ):
        raise DirectJudgementError("control output counts are incomplete")
    if not gate["all_scores_finite"] or not passed:
        raise DirectJudgementError(
            f"direct-judgement control gate failed: {positive_count}/{CONTROL_COUNT} positive"
        )
    return order_output, result_output, gate


def concise_runtime_identity(runtime: Mapping[str, Any]) -> dict[str, object]:
    """Keep only the runtime facts needed to interpret or repeat the scores."""
    identity = runtime.get("runtime_identity")
    if not isinstance(identity, Mapping):
        return {
            "model_id": DEFAULT_MODEL_ID,
            "model_revision": DEFAULT_REVISION,
        }
    model = identity.get("model") if isinstance(identity.get("model"), Mapping) else {}
    load = identity.get("load") if isinstance(identity.get("load"), Mapping) else {}
    gpu = identity.get("gpu") if isinstance(identity.get("gpu"), Mapping) else {}
    software = (
        identity.get("software") if isinstance(identity.get("software"), Mapping) else {}
    )
    return {
        "model_id": model.get("identifier", DEFAULT_MODEL_ID),
        "model_revision": model.get("resolved_commit", DEFAULT_REVISION),
        "dtype": load.get("dtype"),
        "attention_implementation": load.get("attention_implementation"),
        "gpu": gpu.get("name"),
        "cuda_driver": gpu.get("driver"),
        "python": software.get("python"),
        "pytorch": software.get("torch"),
        "transformers": software.get("transformers"),
        "cuda_runtime": software.get("cuda_runtime"),
    }


__all__ = [
    "ASSISTANT_RESPONSES",
    "CANDIDATE_ORDERS",
    "CONTROL_COUNT",
    "CONTROL_MINIMUM_POSITIVE_FRACTION",
    "CONTROL_ORDER_COLUMNS",
    "CONTROL_RESULT_COLUMNS",
    "DirectJudgementError",
    "FORMAL_ORDER_ROWS",
    "FORMAL_SOURCE_SIDS",
    "FORMAL_TARGETS",
    "FORMAL_TRIPLES",
    "JUDGEMENT_CONTEXTS",
    "MANIFEST_SCHEMA",
    "ORDER_COLUMNS",
    "PromptItem",
    "SUMMARY_SCHEMA",
    "TARGET_COLUMNS",
    "TARGET_SCHEMA",
    "TRIPLE_COLUMNS",
    "USER_MESSAGE_TEMPLATE",
    "aggregate_formal_targets",
    "aggregate_order_balanced",
    "attach_formal_order_metadata",
    "build_control_outputs",
    "build_formal_triples",
    "concise_runtime_identity",
    "marked_prefix",
    "prepare_response_sequences",
    "prompt_item_from_analysis_row",
    "prompt_item_from_control",
    "render_user_message",
    "score_prompt_item",
    "score_prompt_items",
    "teacher_forced_response_scores",
    "validate_formal_rows",
]
