#!/usr/bin/env python3
"""Independently validate Qwen Natural Stories reference artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import sha256_file
from qwen_alignment.distance_contracts import REFERENCE_STATS_SCHEMA
from qwen_alignment.model_runtime import MODEL_ID, MODEL_REVISION
from qwen_alignment.representation import HIDDEN_SIZE, H_QWEN_TOKENS

DEFAULT_STATS = (
    PROJECT_ROOT / "data" / "reference"
    / "ias_naturalstories_qwen3_5_9b_h5_final_stats.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "reference"
    / "ias_naturalstories_qwen3_5_9b_h5_final_manifest.json"
)


class ValidationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"top-level JSON is not an object: {path}")
    return value


def validate(stats_path: Path, manifest_path: Path) -> dict[str, object]:
    stats = load(stats_path)
    manifest = load(manifest_path)
    require(stats.get("schema") == REFERENCE_STATS_SCHEMA, "statistics schema mismatch")
    mean, std = stats.get("mean"), stats.get("std")
    require(isinstance(mean, list) and len(mean) == HIDDEN_SIZE, "mean shape mismatch")
    require(isinstance(std, list) and len(std) == HIDDEN_SIZE, "std shape mismatch")
    require(stats.get("mean_shape") == [HIDDEN_SIZE], "mean_shape metadata mismatch")
    require(stats.get("std_shape") == [HIDDEN_SIZE], "std_shape metadata mismatch")
    require(stats.get("std_ddof") == 1, "reference std must use ddof=1")
    require(isinstance(stats.get("count"), int) and stats["count"] >= 2, "bad count")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in mean), "non-finite mean")
    require(
        all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in std),
        "reference std is not finite and strictly positive",
    )
    require(manifest.get("status") == "pass", "manifest did not pass")
    require(manifest.get("blind_reference_build") is True, "build was not blind")
    require(manifest.get("experimental_m_a_i_data_used") is False, "experimental data leaked into reference")
    source = manifest.get("source")
    require(isinstance(source, dict), "source identity missing")
    require(source.get("qwen_h5_windows") == stats["count"], "window count mismatch")
    model = manifest.get("model")
    require(isinstance(model, dict), "model identity missing")
    require(model.get("identifier") == MODEL_ID, "model identifier mismatch")
    require(model.get("revision") == MODEL_REVISION, "model revision mismatch")
    configuration = manifest.get("configuration")
    require(isinstance(configuration, dict), "configuration missing")
    require(configuration.get("h_qwen_tokens") == H_QWEN_TOKENS, "horizon mismatch")
    require(configuration.get("cross_document_windows") is False, "cross-document windows enabled")
    require(configuration.get("add_special_tokens") is False, "special tokens enabled")
    require(configuration.get("conditions_sharing_statistics") == ["M", "A", "I"], "statistics are not shared")
    output = manifest.get("output")
    require(isinstance(output, dict), "output identity missing")
    observed_hash = sha256_file(stats_path)
    require(output.get("sha256") == observed_hash, "statistics hash mismatch")
    return {
        "status": "pass",
        "count": stats["count"],
        "hidden_size": HIDDEN_SIZE,
        "statistics_sha256": observed_hash,
        "std_min": min(float(v) for v in std),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate(args.stats.resolve(), args.manifest.resolve())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
