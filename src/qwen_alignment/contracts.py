"""Frozen, Qwen-specific contracts shared by every experiment stage.

This module deliberately contains no compatibility aliases for any foreign
model project.  A record produced here therefore cannot be mistaken for an
artifact from another experiment.
"""

from __future__ import annotations


Q_ORTHOGRAPHIC_WORDS = 3
H_QWEN_TOKENS = 5
CONDITIONS = ("M", "A", "I")
SEEDS = (11, 23, 37)
SAMPLES_PER_CONDITION = 32

DO_SAMPLE = True
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0
EOS_STOPPING = False

CONTINUATION_FORMAT_NAME = "munch-qwen3.5-9b-continuations"
CONTINUATION_FORMAT_VERSION = 1
CONTINUATION_SCHEMA = f"{CONTINUATION_FORMAT_NAME}/v{CONTINUATION_FORMAT_VERSION}"
CONTINUATION_OUTPUT_JSONL = "qwen3_5_9b_smoke_continuations.jsonl"
CONTINUATION_OUTPUT_SUMMARY = "qwen3_5_9b_smoke_continuations_summary.json"

CONTINUATION_INPUT_COLUMNS = (
    "triple_id",
    "i0",
    "sentence_id",
    "source_sid",
    "genre",
    "novelty",
    "m_sentence",
    "a_sentence",
    "i_sentence",
    "m_word",
    "a_word",
    "i_word",
    "critical_start",
    "m_critical_end",
    "a_critical_end",
    "i_critical_end",
    "right_context_word_1",
    "right_context_word_2",
    "right_context_word_3",
    "q3_right_context",
    "m_prefix_q3",
    "a_prefix_q3",
    "i_prefix_q3",
)


__all__ = [
    "CONDITIONS",
    "CONTINUATION_FORMAT_NAME",
    "CONTINUATION_FORMAT_VERSION",
    "CONTINUATION_INPUT_COLUMNS",
    "CONTINUATION_OUTPUT_JSONL",
    "CONTINUATION_OUTPUT_SUMMARY",
    "CONTINUATION_SCHEMA",
    "DO_SAMPLE",
    "EOS_STOPPING",
    "H_QWEN_TOKENS",
    "Q_ORTHOGRAPHIC_WORDS",
    "SAMPLES_PER_CONDITION",
    "SEEDS",
    "TEMPERATURE",
    "TOP_K",
    "TOP_P",
]
