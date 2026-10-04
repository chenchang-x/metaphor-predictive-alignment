"""Exact-boundary summed-q3 surprisal for the frozen A/I baseline."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

SCHEMA = "munch-qwen3.5-9b-shared-q3-surprisal/v1"


class SurprisalError(RuntimeError):
    pass


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise SurprisalError("tokenizer unexpectedly returned multiple rows")
        values = values[0]
    return [int(value) for value in values]


def condition_identity(
    row: Mapping[str, str], condition: str, tokenizer: Any
) -> dict[str, object]:
    if condition not in ("A", "I"):
        raise SurprisalError(f"baseline condition must be A or I, found {condition!r}")
    lower = condition.lower()
    try:
        critical_end = int(row[f"{lower}_critical_end"])
    except (KeyError, ValueError) as exc:
        raise SurprisalError(f"i0={row.get('i0')} invalid {condition} critical_end") from exc
    sentence = row[f"{lower}_sentence"]
    context_text = sentence[:critical_end]
    q3_text = row["q3_right_context"]
    prefix_text = context_text + q3_text
    if prefix_text != row[f"{lower}_prefix_q3"]:
        raise SurprisalError(f"i0={row['i0']} {condition} prefix reconstruction failed")
    context_ids = _encode_ids(tokenizer, context_text)
    prefix_ids = _encode_ids(tokenizer, prefix_text)
    if not context_ids:
        raise SurprisalError(f"i0={row['i0']} {condition} context has no tokens")
    if prefix_ids[: len(context_ids)] != context_ids:
        raise SurprisalError(f"i0={row['i0']} {condition} q3 changed context tokenisation")
    q3_ids = prefix_ids[len(context_ids) :]
    if not q3_ids:
        raise SurprisalError(f"i0={row['i0']} {condition} q3 has no Qwen tokens")
    decoded = tokenizer.decode(
        q3_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != q3_text:
        raise SurprisalError(f"i0={row['i0']} {condition} q3 token round trip failed")
    return {
        "condition": condition,
        "context_text": context_text,
        "context_token_ids": context_ids,
        "prefix_text": prefix_text,
        "prefix_token_ids": prefix_ids,
        "q3_text": q3_text,
        "q3_token_ids": q3_ids,
        "q3_tokens": tokenizer.convert_ids_to_tokens(q3_ids),
    }


def prepare_rows(
    rows: Sequence[Mapping[str, str]], tokenizer: Any
) -> list[dict[str, object]]:
    """Validate every boundary before any model forward pass is allowed."""

    prepared: list[dict[str, object]] = []
    failures: list[str] = []
    for row in rows:
        try:
            apt = condition_identity(row, "A", tokenizer)
            inapt = condition_identity(row, "I", tokenizer)
            if apt["q3_token_ids"] != inapt["q3_token_ids"]:
                raise SurprisalError("A/I q3 token IDs differ")
            if apt["q3_tokens"] != inapt["q3_tokens"]:
                raise SurprisalError("A/I q3 token strings differ")
            prepared.append({"row": row, "A": apt, "I": inapt})
        except SurprisalError as exc:
            failures.append(f"i0={row.get('i0')}: {exc}")
    if failures:
        preview = "; ".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"; ... {len(failures) - 20} more"
        raise SurprisalError(
            f"tokenizer boundary gate failed for {len(failures)} rows: {preview}{suffix}"
        )
    return prepared


def score_prepared(
    prepared: Sequence[Mapping[str, object]],
    runtime: Mapping[str, Any],
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    if batch_size < 1:
        raise SurprisalError("batch_size must be positive")
    torch = runtime["torch"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = runtime["device"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise SurprisalError("Qwen tokenizer has no pad_token_id")
    max_positions = int(model.config.max_position_embeddings)
    jobs: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for item in prepared:
        for condition in ("A", "I"):
            identity = item[condition]
            assert isinstance(identity, Mapping)
            prefix_ids = identity["prefix_token_ids"]
            assert isinstance(prefix_ids, list)
            if len(prefix_ids) > max_positions:
                raise SurprisalError("baseline prefix exceeds model context window")
            jobs.append((item, identity))

    condition_scores: dict[tuple[str, str], dict[str, object]] = {}
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        max_length = max(len(identity["prefix_token_ids"]) for _, identity in batch)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for _, identity in batch:
            ids = list(identity["prefix_token_ids"])
            padding = max_length - len(ids)
            input_rows.append(ids + [int(pad_id)] * padding)
            mask_rows.append([1] * len(ids) + [0] * padding)
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits
        if tuple(logits.shape[:2]) != (len(batch), max_length):
            raise SurprisalError("model returned an unexpected logits shape")
        for batch_index, (item, identity) in enumerate(batch):
            row = item["row"]
            assert isinstance(row, Mapping)
            condition = str(identity["condition"])
            prefix_ids = list(identity["prefix_token_ids"])
            context_count = len(identity["context_token_ids"])
            q3_ids = list(identity["q3_token_ids"])
            token_rows: list[dict[str, object]] = []
            for q3_index, prefix_index in enumerate(range(context_count, len(prefix_ids))):
                token_id = prefix_ids[prefix_index]
                log_probability = torch.log_softmax(
                    logits[batch_index, prefix_index - 1].to(dtype=torch.float32), dim=-1
                )[token_id]
                value = -float(log_probability.detach().cpu().item())
                if not math.isfinite(value):
                    raise SurprisalError("model returned non-finite surprisal")
                token_rows.append(
                    {
                        "q3_token_index": q3_index,
                        "prefix_token_index": prefix_index,
                        "token_id": token_id,
                        "token": identity["q3_tokens"][q3_index],
                        "surprisal_nats": value,
                    }
                )
            if [value["token_id"] for value in token_rows] != q3_ids:
                raise SurprisalError("scored q3 token identity changed")
            condition_scores[(str(row["triple_id"]), condition)] = {
                **dict(identity),
                "token_surprisals": token_rows,
                "surprisal_nats": math.fsum(
                    float(value["surprisal_nats"]) for value in token_rows
                ),
            }

    records: list[dict[str, object]] = []
    for item in prepared:
        row = item["row"]
        assert isinstance(row, Mapping)
        triple_id = str(row["triple_id"])
        apt = condition_scores[(triple_id, "A")]
        inapt = condition_scores[(triple_id, "I")]
        contrast = float(inapt["surprisal_nats"]) - float(apt["surprisal_nats"])
        records.append(
            {
                "schema": SCHEMA,
                "triple_id": triple_id,
                "i0": int(row["i0"]),
                "sentence_id": int(row["sentence_id"]),
                "source_sid": str(row["source_sid"]),
                "q3_right_context": str(row["q3_right_context"]),
                "q3_token_ids": apt["q3_token_ids"],
                "q3_tokens": apt["q3_tokens"],
                "q3_token_count": len(apt["q3_token_ids"]),
                "apt": apt,
                "inapt": inapt,
                "inapt_minus_apt_nats": contrast,
                "direction": "apt_preferred" if contrast > 0 else "inapt_preferred" if contrast < 0 else "tie",
            }
        )
    return records
