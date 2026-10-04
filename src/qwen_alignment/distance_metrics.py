"""Standardised cosine set distances for labelled M/A/I groups."""

from __future__ import annotations

import math
from typing import Any, Mapping

from . import representation
from .distance_contracts import DETAIL_SCHEMA, DistanceError


def cosine_distance_matrix(standardised_a: Any, standardised_b: Any, *, torch: Any) -> Any:
    a = torch.as_tensor(standardised_a)
    b = torch.as_tensor(standardised_b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or b.shape[0] < 1:
        raise DistanceError("both alternative sets must be non-empty 2-D matrices")
    if a.shape[1] != b.shape[1] or a.shape[1] < 1:
        raise DistanceError("alternative-set representation dimensions do not match")
    if not a.dtype.is_floating_point:
        a = a.to(dtype=torch.float64)
    if not b.dtype.is_floating_point:
        b = b.to(dtype=torch.float64)
    common_dtype = torch.promote_types(a.dtype, b.dtype)
    a, b = a.to(dtype=common_dtype), b.to(dtype=common_dtype)
    if not bool(torch.isfinite(a).all().item()) or not bool(torch.isfinite(b).all().item()):
        raise DistanceError("alternative-set representation contains a non-finite value")
    norms_a = torch.linalg.vector_norm(a, dim=1)
    norms_b = torch.linalg.vector_norm(b, dim=1)
    if bool((norms_a == 0).any().item()) or bool((norms_b == 0).any().item()):
        raise DistanceError("cosine distance is undefined for a zero vector")
    similarities = (a @ b.transpose(0, 1)) / (
        norms_a.unsqueeze(1) * norms_b.unsqueeze(0)
    )
    return (1.0 - similarities.clamp(min=-1.0, max=1.0)).clamp(min=0.0, max=2.0)


def standardised_cosine_distance_matrix(
    representations_a: Any,
    representations_b: Any,
    *,
    reference_mean: Any,
    reference_std: Any,
    torch: Any,
) -> Any:
    standardised_a = representation.standardise_representations(
        representations_a,
        reference_mean=reference_mean,
        reference_std=reference_std,
        torch=torch,
    )
    standardised_b = representation.standardise_representations(
        representations_b,
        reference_mean=reference_mean,
        reference_std=reference_std,
        torch=torch,
    )
    return cosine_distance_matrix(standardised_a, standardised_b, torch=torch)


def mean_pairwise_set_distance(standardised_a: Any, standardised_b: Any, *, torch: Any) -> float:
    distances = cosine_distance_matrix(standardised_a, standardised_b, torch=torch)
    value = float(distances.mean().item())
    if not math.isfinite(value) or not 0.0 <= value <= 2.0:
        raise DistanceError("mean pairwise set distance is outside [0, 2]")
    return value


def compute_triple_seed_distances(
    grouped: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    standardised_groups: Mapping[tuple[str, int, str], Any],
    *,
    torch: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered_keys = sorted(
        grouped, key=lambda value: (int(grouped[value]["M"]["i0"]), value[1])
    )
    for triple_id, seed in ordered_keys:
        m_record = grouped[(triple_id, seed)]["M"]
        m_vectors = standardised_groups[(triple_id, seed, "M")]
        a_vectors = standardised_groups[(triple_id, seed, "A")]
        i_vectors = standardised_groups[(triple_id, seed, "I")]
        sample_count = int(m_vectors.shape[0])
        if int(a_vectors.shape[0]) != sample_count or int(i_vectors.shape[0]) != sample_count:
            raise DistanceError(f"M/A/I sample counts differ for {triple_id}, seed={seed}")
        m_a_distance = mean_pairwise_set_distance(m_vectors, a_vectors, torch=torch)
        m_i_distance = mean_pairwise_set_distance(m_vectors, i_vectors, torch=torch)
        rows.append(
            {
                "schema": DETAIL_SCHEMA,
                "triple_id": triple_id,
                "i0": int(m_record["i0"]),
                "sentence_id": int(m_record["sentence_id"]),
                "source_sid": str(m_record["source_sid"]),
                "seed": seed,
                "m_condition": "M",
                "a_condition": "A",
                "i_condition": "I",
                "samples_per_set": sample_count,
                "pairwise_distances_per_comparison": sample_count * sample_count,
                "distance_metric": "cosine_after_shared_IAS_dimension_standardisation",
                "set_summary": "mean_all_cross_pairs",
                "m_a_distance": m_a_distance,
                "m_i_distance": m_i_distance,
                "mi_minus_ma": m_i_distance - m_a_distance,
            }
        )
    return rows

