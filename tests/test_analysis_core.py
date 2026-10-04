from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_alignment.aggregation import aggregate_sentences, aggregate_triples
from qwen_alignment.cluster_regression import fit_ols_cr1
from qwen_alignment.distance_metrics import cosine_distance_matrix, mean_pairwise_set_distance
from qwen_alignment.representation import HIDDEN_SIZE, continuation_representations
from qwen_alignment.surprisal import SurprisalError, prepare_rows


class FakeModel:
    config = SimpleNamespace(
        vocab_size=200_000,
        num_hidden_layers=2,
    )

    def __call__(self, *, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
        assert output_hidden_states is True and use_cache is False and return_dict is True
        base = input_ids.to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)
        return SimpleNamespace(hidden_states=(base, base + 1, base + 2))


def test_continuation_representation_is_4096_dimensional_float32() -> None:
    vectors = continuation_representations(
        FakeModel(), [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        device=torch.device("cpu"), torch=torch,
    )
    assert tuple(vectors.shape) == (2, 4_096)
    assert vectors.dtype == torch.float32
    assert float(vectors[0, 0]) == pytest.approx(5.0)


def test_cosine_distance_and_set_mean() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    matrix = cosine_distance_matrix(a, b, torch=torch)
    assert matrix.tolist() == pytest.approx([[0.0, 2.0], [1.0, 1.0]])
    assert mean_pairwise_set_distance(a, b, torch=torch) == pytest.approx(1.0)


def _detail(triple: str, i0: int, sentence: int, source: str, seed: int, ma: float, mi: float):
    return {
        "triple_id": triple, "i0": i0, "sentence_id": sentence,
        "source_sid": source, "seed": seed, "m_a_distance": ma,
        "m_i_distance": mi,
    }


def test_aggregation_order_is_seed_then_triple_then_sentence() -> None:
    details = []
    for seed, offset in zip((11, 23, 37), (0.0, 0.1, 0.2), strict=True):
        details.append(_detail("munch-judgement-1", 1, 9, "s", seed, 0.2 + offset, 0.5 + offset))
        details.append(_detail("munch-judgement-2", 2, 9, "s", seed, 0.4 + offset, 0.6 + offset))
    triples = aggregate_triples(details)
    sentences = aggregate_sentences(triples)
    assert len(triples) == 2 and len(sentences) == 1
    assert sentences[0]["m_a_distance_triple_mean"] == pytest.approx(0.4)
    assert sentences[0]["m_i_distance_triple_mean"] == pytest.approx(0.65)
    assert sentences[0]["mi_minus_ma_triple_mean"] == pytest.approx(0.25)
    assert json.loads(sentences[0]["seeds"]) == [11, 23, 37]


def test_cr1_retains_rows_instead_of_cluster_averaging() -> None:
    response = [0.1, 0.2, -0.1, 0.4, 0.3]
    clusters = ["a", "a", "b", "c", "d"]
    result = fit_ols_cr1(response, {}, clusters)
    assert result["n_observations"] == 5
    assert result["n_clusters"] == 4
    assert result["source_cluster_averaging"] is False
    assert result["coefficients"]["intercept"]["estimate"] == pytest.approx(np.mean(response))


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, ids, **_kwargs):
        return "".join(chr(value) for value in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(value) for value in ids]


def _row() -> dict[str, str]:
    q3 = " one two three"
    return {
        "triple_id": "munch-judgement-1", "i0": "1",
        "sentence_id": "9", "source_sid": "s",
        "a_sentence": "A" + q3, "i_sentence": "I" + q3,
        "a_critical_end": "1", "i_critical_end": "1",
        "a_prefix_q3": "A" + q3, "i_prefix_q3": "I" + q3,
        "q3_right_context": q3,
    }


def test_surprisal_boundary_gate_runs_before_scoring() -> None:
    prepared = prepare_rows([_row()], CharacterTokenizer())
    assert prepared[0]["A"]["q3_token_ids"] == prepared[0]["I"]["q3_token_ids"]
    bad = _row()
    bad["i_prefix_q3"] += "x"
    with pytest.raises(SurprisalError, match="boundary gate failed"):
        prepare_rows([bad], CharacterTokenizer())
