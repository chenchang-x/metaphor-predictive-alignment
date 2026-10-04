"""Frozen matched-input RQ2 merge and source-level CR1 inference."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Mapping, Sequence

from .cluster_regression import fit_ols_cr1
from .direct_judgement import (
    FORMAL_SOURCE_SIDS,
    FORMAL_TARGETS,
    FORMAL_TRIPLES,
    TARGET_SCHEMA,
)
from .distance_contracts import SENTENCE_SCHEMA


EXPECTED_DEGREES_OF_FREEDOM = 552
RESULTS_SCHEMA = "munch-qwen3.5-9b-matched-rq2-results/v2"
MANIFEST_SCHEMA = "munch-qwen3.5-9b-matched-rq2-manifest/v2"
MERGED_SCHEMA = "munch-qwen3.5-9b-matched-rq2-items/v2"

MERGED_COLUMNS = (
    "schema",
    "sentence_id",
    "source_sid",
    "triple_count",
    "triple_ids",
    "i0_values",
    "p_context_i_nats",
    "p_word_i_nats",
    "c_i_nats",
    "delta_i",
)


class RQ2AnalysisError(RuntimeError):
    """Raised when the item-level join or frozen regression is invalid."""


def _strict_int(value: object, label: str) -> int:
    text = str(value)
    try:
        parsed = int(text)
    except ValueError as exc:
        raise RQ2AnalysisError(f"{label} is not an integer: {value!r}") from exc
    if str(parsed) != text:
        raise RQ2AnalysisError(f"{label} is not canonical: {value!r}")
    return parsed


def _finite_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RQ2AnalysisError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise RQ2AnalysisError(f"{label} is not finite")
    return parsed


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise RQ2AnalysisError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise RQ2AnalysisError(f"{label} must be a non-empty list")
    if not all(isinstance(item, int) for item in parsed):
        raise RQ2AnalysisError(f"{label} must contain integers")
    return tuple(parsed)


def _string_list(value: object, label: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise RQ2AnalysisError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise RQ2AnalysisError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in parsed):
        raise RQ2AnalysisError(f"{label} must contain non-empty strings")
    return tuple(parsed)


def normalise_distance_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalised: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        label = f"distance row {row_number}"
        if row.get("schema") != SENTENCE_SCHEMA:
            raise RQ2AnalysisError(f"{label} has the wrong schema")
        sentence_id = _strict_int(row.get("sentence_id"), f"{label} sentence_id")
        source_sid = str(row.get("source_sid", ""))
        if not source_sid:
            raise RQ2AnalysisError(f"{label} has an empty source_sid")
        triple_count = _strict_int(row.get("triple_count"), f"{label} triple_count")
        triple_ids = _string_list(row.get("triple_ids"), f"{label} triple_ids")
        i0_values = _integer_list(row.get("i0s"), f"{label} i0s")
        if len(triple_ids) != triple_count or len(i0_values) != triple_count:
            raise RQ2AnalysisError(f"{label} triple membership is incomplete")
        m_a = _finite_float(
            row.get("m_a_distance_triple_mean"), f"{label} M-A distance"
        )
        m_i = _finite_float(
            row.get("m_i_distance_triple_mean"), f"{label} M-I distance"
        )
        delta_i = _finite_float(
            row.get("mi_minus_ma_triple_mean"), f"{label} Delta_i"
        )
        if not math.isclose(delta_i, m_i - m_a, rel_tol=0.0, abs_tol=1e-14):
            raise RQ2AnalysisError(f"{label} Delta_i arithmetic is inconsistent")
        normalised.append(
            {
                "sentence_id": sentence_id,
                "source_sid": source_sid,
                "triple_count": triple_count,
                "triple_ids": triple_ids,
                "i0_values": i0_values,
                "delta_i": delta_i,
            }
        )
    return normalised


def normalise_direct_target_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    normalised: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        label = f"direct row {row_number}"
        if row.get("schema") != TARGET_SCHEMA:
            raise RQ2AnalysisError(f"{label} has the wrong schema")
        sentence_id = _strict_int(row.get("sentence_id"), f"{label} sentence_id")
        source_sid = str(row.get("source_sid", ""))
        if not source_sid:
            raise RQ2AnalysisError(f"{label} has an empty source_sid")
        triple_count = _strict_int(row.get("triple_count"), f"{label} triple_count")
        triple_ids = _string_list(row.get("triple_ids"), f"{label} triple_ids")
        i0_values = _integer_list(row.get("i0_values"), f"{label} i0_values")
        if len(triple_ids) != triple_count or len(i0_values) != triple_count:
            raise RQ2AnalysisError(f"{label} triple membership is incomplete")
        normalised.append(
            {
                "sentence_id": sentence_id,
                "source_sid": source_sid,
                "triple_count": triple_count,
                "triple_ids": triple_ids,
                "i0_values": i0_values,
                "p_context_i_nats": _finite_float(
                    row.get("p_context_i_nats"), f"{label} P_context,i"
                ),
                "p_word_i_nats": _finite_float(
                    row.get("p_word_i_nats"), f"{label} P_word,i"
                ),
                "c_i_nats": _finite_float(row.get("c_i_nats"), f"{label} C_i"),
                "run_id": str(row.get("run_id", "")),
            }
        )
        if not math.isclose(
            float(normalised[-1]["c_i_nats"]),
            float(normalised[-1]["p_context_i_nats"])
            - float(normalised[-1]["p_word_i_nats"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RQ2AnalysisError(f"{label} C_i arithmetic is inconsistent")
    return normalised


def _unique_by_sentence(
    rows: Sequence[Mapping[str, object]], label: str
) -> dict[int, Mapping[str, object]]:
    indexed: dict[int, Mapping[str, object]] = {}
    for row in rows:
        sentence_id = int(row["sentence_id"])
        if sentence_id in indexed:
            raise RQ2AnalysisError(f"duplicate sentence_id={sentence_id} in {label}")
        indexed[sentence_id] = row
    return indexed


def strict_item_merge(
    distance_rows: Sequence[Mapping[str, object]],
    direct_rows: Sequence[Mapping[str, object]],
    *,
    formal: bool = True,
) -> list[dict[str, object]]:
    """Join Delta_i and matched explicit scores without changing item weights."""
    distances = _unique_by_sentence(distance_rows, "distance rows")
    direct = _unique_by_sentence(direct_rows, "direct rows")
    if set(distances) != set(direct):
        missing_direct = sorted(set(distances) - set(direct))[:5]
        missing_distance = sorted(set(direct) - set(distances))[:5]
        raise RQ2AnalysisError(
            "sentence_id sets differ: "
            f"missing direct={missing_direct}, missing distance={missing_distance}"
        )

    merged: list[dict[str, object]] = []
    for sentence_id in sorted(distances):
        distance = distances[sentence_id]
        judgement = direct[sentence_id]
        for field in ("source_sid", "triple_count", "triple_ids", "i0_values"):
            if distance[field] != judgement[field]:
                raise RQ2AnalysisError(
                    f"sentence_id={sentence_id} {field} differs between RQ1 and RQ2"
                )
        merged.append(
            {
                "schema": MERGED_SCHEMA,
                "sentence_id": sentence_id,
                "source_sid": distance["source_sid"],
                "triple_count": distance["triple_count"],
                "triple_ids": json.dumps(distance["triple_ids"], separators=(",", ":")),
                "i0_values": json.dumps(distance["i0_values"], separators=(",", ":")),
                "p_context_i_nats": judgement["p_context_i_nats"],
                "p_word_i_nats": judgement["p_word_i_nats"],
                "c_i_nats": judgement["c_i_nats"],
                "delta_i": distance["delta_i"],
            }
        )

    if formal:
        if len(merged) != FORMAL_TARGETS:
            raise RQ2AnalysisError(
                f"formal RQ2 requires {FORMAL_TARGETS} items, observed {len(merged)}"
            )
        clusters = {str(row["source_sid"]) for row in merged}
        if len(clusters) != FORMAL_SOURCE_SIDS:
            raise RQ2AnalysisError(
                f"formal RQ2 requires {FORMAL_SOURCE_SIDS} clusters, observed {len(clusters)}"
            )
        total_triples = sum(int(row["triple_count"]) for row in merged)
        if total_triples != FORMAL_TRIPLES:
            raise RQ2AnalysisError(
                f"formal RQ2 requires {FORMAL_TRIPLES} triple memberships, "
                f"observed {total_triples}"
            )
    return merged


def _label_two_sided_test(
    coefficient: Mapping[str, object],
    *,
    symbol: str,
) -> dict[str, object]:
    output = dict(coefficient)
    raw_test = output.get("test")
    if not isinstance(raw_test, Mapping):
        raise RQ2AnalysisError(f"{symbol} test is missing")
    output["test"] = {
        **raw_test,
        "null_hypothesis": f"{symbol} = 0",
        "alternative_hypothesis": f"{symbol} != 0",
    }
    return output


def fit_rq2_models(
    merged_rows: Sequence[Mapping[str, object]],
    *,
    formal: bool = True,
) -> dict[str, object]:
    """Fit the total-association M0 and matched context-decomposition M1."""
    if not merged_rows:
        raise RQ2AnalysisError("RQ2 has no merged items")
    response = [_finite_float(row["delta_i"], "Delta_i") for row in merged_rows]
    p_context = [
        _finite_float(row["p_context_i_nats"], "P_context,i")
        for row in merged_rows
    ]
    p_word = [
        _finite_float(row["p_word_i_nats"], "P_word,i")
        for row in merged_rows
    ]
    context_shift = [_finite_float(row["c_i_nats"], "C_i") for row in merged_rows]
    clusters = [str(row["source_sid"]) for row in merged_rows]
    m0 = fit_ols_cr1(response, {"P_context_i": p_context}, clusters)
    m1 = fit_ols_cr1(
        response,
        {"C_i": context_shift, "P_word_i": p_word},
        clusters,
    )
    if formal:
        expected = (FORMAL_TARGETS, FORMAL_SOURCE_SIDS, EXPECTED_DEGREES_OF_FREEDOM)
        for label, fitted in (("M0", m0), ("M1", m1)):
            observed = (
                fitted["n_observations"],
                fitted["n_clusters"],
                fitted["degrees_of_freedom"],
            )
            if observed != expected:
                raise RQ2AnalysisError(
                    f"formal {label} dimensions are {observed}, expected {expected}"
                )

    beta_0 = _label_two_sided_test(m0["coefficients"]["intercept"], symbol="beta_0")
    beta_total = _label_two_sided_test(
        m0["coefficients"]["P_context_i"], symbol="beta_total"
    )
    gamma_0 = _label_two_sided_test(m1["coefficients"]["intercept"], symbol="gamma_0")
    gamma_c = _label_two_sided_test(m1["coefficients"]["C_i"], symbol="gamma_C")
    gamma_w = _label_two_sided_test(
        m1["coefficients"]["P_word_i"], symbol="gamma_W"
    )
    p_context_signs = Counter(
        "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
        for value in p_context
    )
    context_shift_signs = Counter(
        "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
        for value in context_shift
    )
    return {
        "M0_total_association": {
            "formula": (
                "Delta_i = beta_0 + beta_total P_context_i + epsilon_i"
            ),
            "estimation": "ordinary least squares",
            "covariance": "one-way CR1 clustered by source_sid",
            "n_observations": m0["n_observations"],
            "n_clusters": m0["n_clusters"],
            "degrees_of_freedom": m0["degrees_of_freedom"],
            "coefficients": {"beta_0": beta_0, "beta_total": beta_total},
            "r_squared": m0["r_squared"],
            "cr1_finite_sample_correction": m0[
                "cr1_finite_sample_correction"
            ],
            "covariance_matrix": m0["covariance_matrix"],
        },
        "M1_context_decomposition": {
            "formula": (
                "Delta_i = gamma_0 + gamma_C C_i + gamma_W P_word_i + epsilon_i"
            ),
            "estimation": "ordinary least squares",
            "covariance": "one-way CR1 clustered by source_sid",
            "n_observations": m1["n_observations"],
            "n_clusters": m1["n_clusters"],
            "degrees_of_freedom": m1["degrees_of_freedom"],
            "coefficients": {
                "gamma_0": gamma_0,
                "gamma_C": gamma_c,
                "gamma_W": gamma_w,
            },
            "r_squared": m1["r_squared"],
            "cr1_finite_sample_correction": m1[
                "cr1_finite_sample_correction"
            ],
            "covariance_matrix": m1["covariance_matrix"],
        },
        "shared_sample": True,
        "n_observations": m0["n_observations"],
        "n_clusters": m0["n_clusters"],
        "degrees_of_freedom": m0["degrees_of_freedom"],
        "p_context_i_sign_counts": dict(p_context_signs),
        "c_i_sign_counts": dict(context_shift_signs),
        "interpretation_gate": (
            "M1 cannot rescue RQ2 if M0 does not support the total association"
        ),
    }


__all__ = [
    "EXPECTED_DEGREES_OF_FREEDOM",
    "MANIFEST_SCHEMA",
    "MERGED_COLUMNS",
    "MERGED_SCHEMA",
    "RESULTS_SCHEMA",
    "RQ2AnalysisError",
    "fit_rq2_models",
    "normalise_direct_target_rows",
    "normalise_distance_rows",
    "strict_item_merge",
]
