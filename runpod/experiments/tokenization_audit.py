#!/usr/bin/env python3
"""Token-only boundary audit for the frozen Qwen3.5-9B design.

The audit accepts any already-loaded tokenizer through :func:`audit_rows`.
The CLI resolves the pinned Qwen tokenizer through ``model_runtime`` without
loading model weights onto a GPU.  A failed token boundary is reported; it
never removes or rewrites an analysis row.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_json, write_text_atomic
from qwen_alignment.continuation_input import canonical_int, read_input_rows
from qwen_alignment.direct_judgement import (
    ASSISTANT_RESPONSES,
    CANDIDATE_ORDERS,
    JUDGEMENT_CONTEXTS,
    prompt_item_from_analysis_row,
    render_user_message,
)
from qwen_alignment.model_runtime import DEFAULT_MODEL_ID, DEFAULT_REVISION


SCHEMA = "munch-qwen3.5-9b-tokenization-audit/v2"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_items.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "environment" / "tokenization_audit_summary.json"
CONDITIONS = ("M", "A", "I")
SURPRISAL_CONDITIONS = ("A", "I")


class TokenizationAuditError(RuntimeError):
    """Raised when one tokenizer operation cannot establish its boundary."""


def _token_ids(value: Any, label: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise TokenizationAuditError(f"{label} unexpectedly returned a batch")
        value = list(value[0])
    if not isinstance(value, list) or not value:
        raise TokenizationAuditError(f"{label} produced no token IDs")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in value):
        raise TokenizationAuditError(f"{label} produced invalid token IDs")
    return list(value)


def _encode_raw(tokenizer: Any, text: str, label: str) -> list[int]:
    """Use the RQ1 raw causal interface, never the chat template."""

    return _token_ids(
        tokenizer(text, add_special_tokens=False),
        label,
    )


def _decode_exact(tokenizer: Any, token_ids: Sequence[int], expected: str, label: str) -> None:
    decoded = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != expected:
        raise TokenizationAuditError(
            f"{label} did not round-trip exactly: expected {expected!r}, found {decoded!r}"
        )


def _audit_raw_prefix(row: Mapping[str, str], condition: str, tokenizer: Any) -> None:
    lower = condition.lower()
    text = row[f"{lower}_prefix_q3"]
    ids = _encode_raw(tokenizer, text, f"{condition} raw prefix")
    _decode_exact(tokenizer, ids, text, f"{condition} raw prefix")


def _audit_surprisal_condition(
    row: Mapping[str, str], condition: str, tokenizer: Any, row_number: int
) -> list[int]:
    lower = condition.lower()
    end = canonical_int(
        row[f"{lower}_critical_end"], f"{lower}_critical_end", row_number
    )
    context = row[f"{lower}_sentence"][:end]
    q3 = row["q3_right_context"]
    prefix = context + q3
    if prefix != row[f"{lower}_prefix_q3"]:
        raise TokenizationAuditError(f"{condition} stored q3 prefix differs from context+q3")

    context_ids = _encode_raw(tokenizer, context, f"{condition} surprisal context")
    prefix_ids = _encode_raw(tokenizer, prefix, f"{condition} surprisal context+q3")
    if prefix_ids[: len(context_ids)] != context_ids:
        raise TokenizationAuditError(
            f"{condition} q3 append changed the context token prefix"
        )
    suffix_ids = prefix_ids[len(context_ids) :]
    if not suffix_ids:
        raise TokenizationAuditError(f"{condition} q3 append produced no tokens")
    _decode_exact(tokenizer, suffix_ids, q3, f"{condition} q3 suffix")
    return suffix_ids


def _audit_chat_order(
    row: Mapping[str, str],
    candidate_order: str,
    judgement_context: str,
    tokenizer: Any,
) -> None:
    item = prompt_item_from_analysis_row(row)
    rendered = render_user_message(
        item,
        candidate_order,
        judgement_context=judgement_context,
    )
    expected_input = (
        item.left_context + f"[TARGET: {item.m_word}]" + item.q3_right_context
        if judgement_context == "context"
        else f"[TARGET: {item.m_word}]"
    )
    if rendered["task_input"] != expected_input:
        raise TokenizationAuditError(
            f"RQ2 {judgement_context} input does not match its frozen definition"
        )
    user_message = rendered["user_message"]
    messages = [{"role": "user", "content": user_message}]

    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        f"RQ2 {judgement_context}/{candidate_order} chat prompt",
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise TokenizationAuditError(
            f"RQ2 {judgement_context}/{candidate_order} chat template produced no prompt text"
        )
    if _encode_raw(
        tokenizer,
        prompt_text,
        f"RQ2 {judgement_context}/{candidate_order} rendered prompt",
    ) != prompt_ids:
        raise TokenizationAuditError(
            f"RQ2 {judgement_context}/{candidate_order} tokenize=True and rendered-text paths disagree"
        )

    response_suffixes: dict[str, list[int]] = {}
    for response in ASSISTANT_RESPONSES:
        full_ids = _encode_raw(
            tokenizer,
            prompt_text + response,
            f"RQ2 {judgement_context}/{candidate_order} prompt+{response}",
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise TokenizationAuditError(
                f"RQ2 {judgement_context}/{candidate_order} response {response} changed the prompt token prefix"
            )
        suffix = full_ids[len(prompt_ids) :]
        if not suffix:
            raise TokenizationAuditError(
                f"RQ2 {judgement_context}/{candidate_order} response {response} added no tokens"
            )
        _decode_exact(
            tokenizer,
            suffix,
            response,
            f"RQ2 {judgement_context}/{candidate_order} response {response}",
        )
        response_suffixes[response] = suffix

    if len(response_suffixes["A"]) != len(response_suffixes["B"]):
        raise TokenizationAuditError(
            f"RQ2 {judgement_context}/{candidate_order} A/B responses have different token counts"
        )
    if response_suffixes["A"] == response_suffixes["B"]:
        raise TokenizationAuditError(
            f"RQ2 {judgement_context}/{candidate_order} A/B responses have identical token IDs"
        )


def _issue(
    *,
    stage: str,
    row: Mapping[str, str],
    code: str,
    detail: str,
    condition: str = "",
    judgement_context: str = "",
    candidate_order: str = "",
) -> dict[str, object]:
    return {
        "stage": stage,
        "i0": int(row["i0"]),
        "triple_id": row["triple_id"],
        "condition": condition,
        "judgement_context": judgement_context,
        "candidate_order": candidate_order,
        "code": code,
        "detail": detail,
    }


def audit_rows(
    rows: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
) -> dict[str, object]:
    """Audit every supplied row and return a complete, non-filtering report.

    ``rows`` should already satisfy ``continuation_input``'s data contract.  A
    boundary failure is isolated to its case so all remaining rows are still
    audited.
    """

    if not rows:
        raise TokenizationAuditError("no analysis rows were supplied")

    issues: list[dict[str, object]] = []
    rq1_cases = rq1_passed = 0
    surprisal_condition_cases = surprisal_condition_passed = 0
    surprisal_shared_cases = surprisal_shared_passed = 0
    rq2_cases = rq2_passed = 0

    for row_number, row in enumerate(rows, start=2):
        for condition in CONDITIONS:
            rq1_cases += 1
            try:
                _audit_raw_prefix(row, condition, tokenizer)
            except Exception as exc:
                issues.append(
                    _issue(
                        stage="rq1_raw_prefix",
                        row=row,
                        condition=condition,
                        code="RAW_PREFIX_TOKENIZATION_FAILED",
                        detail=str(exc),
                    )
                )
            else:
                rq1_passed += 1

        q3_suffixes: dict[str, list[int]] = {}
        for condition in SURPRISAL_CONDITIONS:
            surprisal_condition_cases += 1
            try:
                q3_suffixes[condition] = _audit_surprisal_condition(
                    row, condition, tokenizer, row_number
                )
            except Exception as exc:
                issues.append(
                    _issue(
                        stage="surprisal_boundary",
                        row=row,
                        condition=condition,
                        code="Q3_APPEND_BOUNDARY_FAILED",
                        detail=str(exc),
                    )
                )
            else:
                surprisal_condition_passed += 1

        surprisal_shared_cases += 1
        if len(q3_suffixes) == len(SURPRISAL_CONDITIONS):
            if q3_suffixes["A"] == q3_suffixes["I"]:
                surprisal_shared_passed += 1
            else:
                issues.append(
                    _issue(
                        stage="surprisal_boundary",
                        row=row,
                        code="A_I_Q3_SUFFIX_IDS_DIFFER",
                        detail="A and I produced different token IDs for the same q3 text",
                    )
                )

        try:
            prompt_item_from_analysis_row(row)
        except Exception as exc:
            for judgement_context in JUDGEMENT_CONTEXTS:
                for candidate_order in CANDIDATE_ORDERS:
                    rq2_cases += 1
                    issues.append(
                        _issue(
                            stage="rq2_chat_append",
                            row=row,
                            judgement_context=judgement_context,
                            candidate_order=candidate_order,
                            code="RQ2_PROMPT_CONSTRUCTION_FAILED",
                            detail=str(exc),
                        )
                    )
        else:
            for judgement_context in JUDGEMENT_CONTEXTS:
                for candidate_order in CANDIDATE_ORDERS:
                    rq2_cases += 1
                    try:
                        _audit_chat_order(
                            row,
                            candidate_order,
                            judgement_context,
                            tokenizer,
                        )
                    except Exception as exc:
                        issues.append(
                            _issue(
                                stage="rq2_chat_append",
                                row=row,
                                judgement_context=judgement_context,
                                candidate_order=candidate_order,
                                code="RQ2_A_B_APPEND_BOUNDARY_FAILED",
                                detail=str(exc),
                            )
                        )
                    else:
                        rq2_passed += 1

    failed_i0 = sorted({int(issue["i0"]) for issue in issues})
    issue_counts = Counter(str(issue["code"]) for issue in issues)
    return {
        "schema": SCHEMA,
        "status": "pass" if not issues else "fail",
        "model": {"identifier": model_id, "revision": revision},
        "tokenizer_class": type(tokenizer).__name__,
        "interfaces": {
            "rq1": "raw causal prefixes; add_special_tokens=False; no chat template",
            "surprisal": "raw A/I target-ending context plus identical q3 text",
            "rq2": (
                "matched context/word inputs; official chat template; one user "
                "message; generation prompt; complete A/B responses"
            ),
        },
        "coverage": {
            "rows_received": len(rows),
            "rows_audited": len(rows),
            "rows_removed": 0,
            "failed_rows_retained": True,
            "sentence_ids": len({row["sentence_id"] for row in rows}),
            "source_sids": len({row["source_sid"] for row in rows}),
        },
        "checks": {
            "rq1_raw_prefixes": {
                "cases": rq1_cases,
                "passed": rq1_passed,
                "failed": rq1_cases - rq1_passed,
            },
            "surprisal_context_q3_boundaries": {
                "cases": surprisal_condition_cases,
                "passed": surprisal_condition_passed,
                "failed": surprisal_condition_cases - surprisal_condition_passed,
            },
            "surprisal_shared_a_i_q3_suffix": {
                "cases": surprisal_shared_cases,
                "passed": surprisal_shared_passed,
                "failed": surprisal_shared_cases - surprisal_shared_passed,
            },
            "rq2_chat_a_b_append_boundaries": {
                "cases": rq2_cases,
                "passed": rq2_passed,
                "failed": rq2_cases - rq2_passed,
            },
        },
        "failures": {
            "issue_count": len(issues),
            "rows": failed_i0,
            "by_code": dict(sorted(issue_counts.items())),
            "issues": issues,
        },
    }


def load_cli_tokenizer(cache_dir: Path, *, allow_download: bool) -> Any:
    """Resolve the pinned tokenizer using the Qwen runtime's snapshot policy."""

    from transformers import AutoTokenizer

    from qwen_alignment.model_runtime import resolve_snapshot, validate_chat_template

    snapshot_path = resolve_snapshot(cache_dir, allow_download=allow_download)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    validate_chat_template(tokenizer)
    return tokenizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit the pinned runtime resolver to populate a missing snapshot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = read_input_rows(args.input)
        tokenizer = load_cli_tokenizer(
            args.cache_dir.resolve(), allow_download=args.allow_download
        )
        report = audit_rows(rows, tokenizer)
        write_text_atomic(args.output.resolve(), render_json(report))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"tokenization_audit={report['status']}")
    print(f"rows_audited={report['coverage']['rows_audited']}")
    print(f"issues={report['failures']['issue_count']}")
    print(f"summary={args.output.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
