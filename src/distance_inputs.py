"""Load, validate, group, and encode inputs for distance computation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import representation
from .contracts import CONDITIONS, H_QWEN_TOKENS, SAMPLES_PER_CONDITION, SEEDS
from .distance_contracts import DistanceError, REFERENCE_STATS_SCHEMA


def load_reference_statistics(path: Path) -> tuple[list[float], list[float], int]:
    if not path.is_file():
        raise DistanceError(f"reference statistics not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistanceError(f"invalid reference statistics JSON: {path}") from exc
    if payload.get("schema") != REFERENCE_STATS_SCHEMA:
        raise DistanceError("reference statistics schema mismatch")
    mean, std, count = payload.get("mean"), payload.get("std"), payload.get("count")
    if (
        not isinstance(mean, list) or not isinstance(std, list)
        or len(mean) != representation.HIDDEN_SIZE
        or len(std) != representation.HIDDEN_SIZE
    ):
        raise DistanceError("reference mean and std must each have 4096 values")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in mean):
        raise DistanceError("reference mean contains a non-finite value")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in std
    ):
        raise DistanceError("reference std must be finite and strictly positive")
    if not isinstance(count, int) or count < 2:
        raise DistanceError("reference statistics count must be an integer >= 2")
    if payload.get("mean_shape") != [representation.HIDDEN_SIZE]:
        raise DistanceError("reference mean_shape mismatch")
    if payload.get("std_shape") != [representation.HIDDEN_SIZE]:
        raise DistanceError("reference std_shape mismatch")
    if payload.get("std_ddof") != 1:
        raise DistanceError("reference standard deviation must use ddof=1")
    return mean, std, count


def group_continuation_records(
    records: Sequence[Mapping[str, object]],
    *,
    expected_seeds: Sequence[int] = SEEDS,
    samples_per_set: int = SAMPLES_PER_CONDITION,
) -> dict[tuple[str, int], dict[str, Mapping[str, object]]]:
    if not records:
        raise DistanceError("continuation input contains no records")
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise DistanceError("expected seeds must be non-empty and unique")
    if samples_per_set < 1:
        raise DistanceError("samples_per_set must be positive")
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    identities: dict[str, tuple[int, int, str]] = {}
    seeds_by_triple: dict[str, set[int]] = defaultdict(set)
    triple_by_i0: dict[int, str] = {}
    for record_number, record in enumerate(records, start=1):
        triple_id = record.get("triple_id")
        i0 = record.get("i0")
        sentence_id = record.get("sentence_id")
        source_sid = record.get("source_sid")
        seed = record.get("seed")
        condition = record.get("condition")
        continuations = record.get("continuations")
        if not isinstance(triple_id, str) or not triple_id:
            raise DistanceError(f"record {record_number} has invalid triple_id")
        if not isinstance(i0, int) or not isinstance(sentence_id, int):
            raise DistanceError(f"record {record_number} has invalid i0 or sentence_id")
        if triple_id != f"munch-judgement-{i0}":
            raise DistanceError(f"record {record_number} triple_id/i0 mismatch")
        if not isinstance(source_sid, str) or not source_sid:
            raise DistanceError(f"record {record_number} has invalid source_sid")
        if not isinstance(seed, int) or seed not in expected_seeds:
            raise DistanceError(f"record {record_number} has unexpected seed")
        if condition not in CONDITIONS:
            raise DistanceError(f"record {record_number} has invalid M/A/I condition")
        if not isinstance(continuations, list) or len(continuations) != samples_per_set:
            raise DistanceError(
                f"record {record_number} must contain {samples_per_set} continuations"
            )
        identity = (i0, sentence_id, source_sid)
        if triple_id in identities and identities[triple_id] != identity:
            raise DistanceError(f"metadata changed across records for {triple_id}")
        identities[triple_id] = identity
        if i0 in triple_by_i0 and triple_by_i0[i0] != triple_id:
            raise DistanceError(f"i0={i0} maps to multiple triple IDs")
        triple_by_i0[i0] = triple_id
        for expected_index, continuation in enumerate(continuations):
            if not isinstance(continuation, dict):
                raise DistanceError(
                    f"record {record_number} continuation {expected_index} is not an object"
                )
            token_ids = continuation.get("token_ids")
            if continuation.get("sample_index") != expected_index:
                raise DistanceError(
                    f"record {record_number} continuation sample order mismatch"
                )
            if (
                not isinstance(token_ids, list) or len(token_ids) != H_QWEN_TOKENS
                or not all(isinstance(value, int) and value >= 0 for value in token_ids)
            ):
                raise DistanceError(
                    f"record {record_number} continuation {expected_index} is not h=5"
                )
        key = (triple_id, seed)
        if condition in grouped[key]:
            raise DistanceError(f"duplicate {condition} record for {triple_id}, seed={seed}")
        grouped[key][condition] = record
        seeds_by_triple[triple_id].add(seed)
    expected_conditions = set(CONDITIONS)
    for (triple_id, seed), condition_records in grouped.items():
        if set(condition_records) != expected_conditions:
            raise DistanceError(
                f"{triple_id}, seed={seed} does not contain exactly M, A, and I"
            )
    expected_seed_set = set(expected_seeds)
    for triple_id, observed_seeds in seeds_by_triple.items():
        if observed_seeds != expected_seed_set:
            raise DistanceError(
                f"{triple_id} seeds mismatch: expected {sorted(expected_seed_set)}, "
                f"found {sorted(observed_seeds)}"
            )
    return dict(grouped)


def encode_standardised_groups(
    grouped: Mapping[tuple[str, int], Mapping[str, Mapping[str, object]]],
    *,
    model: Any,
    device: Any,
    reference_mean: Sequence[float],
    reference_std: Sequence[float],
    torch: Any,
    batch_size: int = 64,
) -> dict[tuple[str, int, str], Any]:
    if batch_size < 1:
        raise DistanceError("batch_size must be positive")
    flattened: list[tuple[tuple[str, int, str], list[int]]] = []
    for triple_id, seed in sorted(
        grouped, key=lambda value: (int(grouped[value]["M"]["i0"]), value[1])
    ):
        for condition in CONDITIONS:
            record = grouped[(triple_id, seed)][condition]
            for continuation in record["continuations"]:
                flattened.append(
                    ((triple_id, seed, condition), list(continuation["token_ids"]))
                )
    by_group: dict[tuple[str, int, str], list[Any]] = defaultdict(list)
    for start in range(0, len(flattened), batch_size):
        batch = flattened[start : start + batch_size]
        representations = representation.continuation_representations(
            model, [token_ids for _, token_ids in batch], device=device, torch=torch
        )
        standardised = representation.standardise_representations(
            representations,
            reference_mean=reference_mean,
            reference_std=reference_std,
            torch=torch,
        ).detach().cpu()
        for (key, _), vector in zip(batch, standardised):
            by_group[key].append(vector)
    return {key: torch.stack(vectors) for key, vectors in by_group.items()}
