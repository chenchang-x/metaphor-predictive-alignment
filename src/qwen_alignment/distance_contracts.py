"""Schemas and public error type for distance and aggregation stages."""

DETAIL_SCHEMA = "munch-qwen3.5-9b-standardised-cosine-set-distances/v1"
TRIPLE_SCHEMA = "munch-qwen3.5-9b-seed-averaged-triple-distances/v1"
SENTENCE_SCHEMA = "munch-qwen3.5-9b-triple-averaged-sentence-distances/v1"
SUMMARY_SCHEMA = "munch-qwen3.5-9b-distance-pipeline-summary/v1"
REFERENCE_STATS_SCHEMA = "ias-naturalstories-qwen3.5-9b-h5-final-reference-stats/v1"

DETAIL_COLUMNS = (
    "schema", "triple_id", "i0", "sentence_id", "source_sid", "seed",
    "m_condition", "a_condition", "i_condition", "samples_per_set",
    "pairwise_distances_per_comparison", "distance_metric", "set_summary",
    "m_a_distance", "m_i_distance", "mi_minus_ma",
)
TRIPLE_COLUMNS = (
    "schema", "triple_id", "i0", "sentence_id", "source_sid", "seeds",
    "seed_count", "m_a_distance_seed_mean", "m_i_distance_seed_mean",
    "mi_minus_ma_seed_mean",
)
SENTENCE_COLUMNS = (
    "schema", "sentence_id", "source_sid", "triple_ids", "i0s", "seeds",
    "triple_count", "m_a_distance_triple_mean", "m_i_distance_triple_mean",
    "mi_minus_ma_triple_mean",
)


class DistanceError(RuntimeError):
    """Raised when distance inputs or aggregation identities are invalid."""
