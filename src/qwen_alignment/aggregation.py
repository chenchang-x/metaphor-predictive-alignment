"""Protocol-order seed, triple, and sentence aggregation with provenance."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Mapping, Sequence

from .contracts import SEEDS
from .distance_contracts import DistanceError, SENTENCE_SCHEMA, TRIPLE_SCHEMA


def _checked_common_identity(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str], label: str
) -> dict[str, object]:
    identity: dict[str, object] = {}
    for field in fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise DistanceError(f"{label} has inconsistent {field}")
        identity[field] = next(iter(values))
    return identity


def aggregate_triples(
    detail_rows: Sequence[Mapping[str, object]], *, expected_seeds: Sequence[int] = SEEDS
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(row["triple_id"])].append(row)
    output: list[dict[str, object]] = []
    for triple_id, rows in sorted(grouped.items(), key=lambda item: int(item[1][0]["i0"])):
        identity = _checked_common_identity(
            rows, ("i0", "sentence_id", "source_sid"), triple_id
        )
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != sorted(expected_seeds) or len(seeds) != len(set(seeds)):
            raise DistanceError(f"{triple_id} does not have exactly the expected seeds")
        m_a = math.fsum(float(row["m_a_distance"]) for row in rows) / len(rows)
        m_i = math.fsum(float(row["m_i_distance"]) for row in rows) / len(rows)
        output.append(
            {
                "schema": TRIPLE_SCHEMA,
                "triple_id": triple_id,
                **identity,
                "seeds": json.dumps(seeds, separators=(",", ":")),
                "seed_count": len(seeds),
                "m_a_distance_seed_mean": m_a,
                "m_i_distance_seed_mean": m_i,
                "mi_minus_ma_seed_mean": m_i - m_a,
            }
        )
    return output


def aggregate_sentences(
    triple_rows: Sequence[Mapping[str, object]], *, expected_seeds: Sequence[int] = SEEDS
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in triple_rows:
        grouped[int(row["sentence_id"])].append(row)
    output: list[dict[str, object]] = []
    for sentence_id, rows in sorted(grouped.items()):
        identity = _checked_common_identity(rows, ("source_sid",), f"sentence_id={sentence_id}")
        ordered = sorted(rows, key=lambda row: int(row["i0"]))
        triple_ids = [str(row["triple_id"]) for row in ordered]
        i0s = [int(row["i0"]) for row in ordered]
        if len(triple_ids) != len(set(triple_ids)):
            raise DistanceError(f"sentence_id={sentence_id} repeats a triple")
        expected_seed_json = json.dumps(sorted(expected_seeds), separators=(",", ":"))
        if any(row["seeds"] != expected_seed_json for row in ordered):
            raise DistanceError(f"sentence_id={sentence_id} has inconsistent seeds")
        m_a = math.fsum(float(row["m_a_distance_seed_mean"]) for row in ordered) / len(ordered)
        m_i = math.fsum(float(row["m_i_distance_seed_mean"]) for row in ordered) / len(ordered)
        output.append(
            {
                "schema": SENTENCE_SCHEMA,
                "sentence_id": sentence_id,
                **identity,
                "triple_ids": json.dumps(triple_ids, separators=(",", ":")),
                "i0s": json.dumps(i0s, separators=(",", ":")),
                "seeds": expected_seed_json,
                "triple_count": len(ordered),
                "m_a_distance_triple_mean": m_a,
                "m_i_distance_triple_mean": m_i,
                "mi_minus_ma_triple_mean": m_i - m_a,
            }
        )
    return output

