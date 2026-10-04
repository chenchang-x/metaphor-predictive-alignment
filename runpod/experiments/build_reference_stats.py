#!/usr/bin/env python3
"""Build the Qwen-specific shared h=5 reference statistics.

Only the project-local IAS Natural Stories raw text is read. Experimental M/A/I
data are deliberately outside this stage's input graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment import model_runtime, representation
from qwen_alignment.artifacts import render_json, sha256_file, write_text_atomic
from qwen_alignment.distance_contracts import REFERENCE_STATS_SCHEMA

DEFAULT_INPUT = PROJECT_ROOT / "vendor" / "ias-naturalstories" / "texts_400words.csv"
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT / "vendor" / "ias-naturalstories" / "source_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "ias_naturalstories_qwen3_5_9b_h5_final_stats.json"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "ias_naturalstories_qwen3_5_9b_h5_final_manifest.json"
)
EXPECTED_INPUT_SHA256 = (
    "5747a7de60b3b2d6ae5dbb1934b2e47c4ef17ce8eb86cd7473c75cf449db437b"
)
EXPECTED_SOURCE_REPOSITORY = "https://github.com/glnmario/ias-lang-comp.git"
EXPECTED_SOURCE_COMMIT = "b2b54eb3bb7decf46b7eff27e61d8b5306f04355"
EXPECTED_SOURCE_PATH = "data/corpora/naturalstories/texts_400words.csv"
EXPECTED_INPUT_IDS = tuple(str(index) for index in range(1, 11))
CONDITIONS = ("M", "A", "I")
STD_DDOF = 1
MANIFEST_SCHEMA = "ias-naturalstories-qwen3.5-9b-reference-freeze/v1"


class ReferenceBuildError(RuntimeError):
    """Raised when a source, runtime, or numerical invariant fails."""


def read_source_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceBuildError(f"invalid IAS source manifest: {path}") from exc
    if payload.get("repository") != EXPECTED_SOURCE_REPOSITORY:
        raise ReferenceBuildError("IAS source repository mismatch")
    if payload.get("upstream_commit") != EXPECTED_SOURCE_COMMIT:
        raise ReferenceBuildError("IAS source commit mismatch")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ReferenceBuildError("IAS source manifest has no files map")
    item = files.get("texts_400words.csv")
    if (
        not isinstance(item, dict)
        or item.get("source_path") != EXPECTED_SOURCE_PATH
        or item.get("sha256") != EXPECTED_INPUT_SHA256
    ):
        raise ReferenceBuildError("IAS source file identity mismatch")
    return payload


def read_reference_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ReferenceBuildError(f"IAS Natural Stories input not found: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise ReferenceBuildError(f"IAS Natural Stories hash mismatch: {actual_hash}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("id", "text"):
            raise ReferenceBuildError(
                f"Natural Stories schema mismatch: {reader.fieldnames}"
            )
        rows = list(reader)
    if tuple(row["id"] for row in rows) != EXPECTED_INPUT_IDS:
        raise ReferenceBuildError("Natural Stories IDs are not exactly 1..10")
    if any(not row["text"] for row in rows):
        raise ReferenceBuildError("Natural Stories contains an empty document")
    return rows


def token_windows(
    rows: Sequence[Mapping[str, str]], tokenizer: Any
) -> tuple[list[list[int]], list[dict[str, object]]]:
    """Create all within-document h=5 windows without assuming their count."""

    horizon = representation.H_QWEN_TOKENS
    windows: list[list[int]] = []
    documents: list[dict[str, object]] = []
    for row in rows:
        encoded = tokenizer(
            row["text"], add_special_tokens=False, return_attention_mask=False
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        if len(token_ids) < horizon:
            raise ReferenceBuildError(
                f"Natural Stories document {row['id']} has fewer than {horizon} tokens"
            )
        count = len(token_ids) - horizon + 1
        windows.extend(token_ids[start : start + horizon] for start in range(count))
        documents.append(
            {
                "id": row["id"],
                "qwen_tokens": len(token_ids),
                "h5_windows": count,
                "decoded_round_trip": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                == row["text"],
            }
        )
    if len(windows) < 2:
        raise ReferenceBuildError("reference corpus produced fewer than two windows")
    return windows, documents


def batches(values: Sequence[list[int]], size: int) -> Iterator[Sequence[list[int]]]:
    if size < 1:
        raise ReferenceBuildError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def compute_statistics(
    windows: Sequence[list[int]], runtime: Mapping[str, Any], *, batch_size: int
) -> tuple[Any, Any]:
    torch = runtime["torch"]
    chunks: list[Any] = []
    total_batches = (len(windows) + batch_size - 1) // batch_size
    for batch_number, batch in enumerate(batches(windows, batch_size), start=1):
        vectors = representation.continuation_representations(
            runtime["model"], batch, device=runtime["device"], torch=torch
        )
        chunks.append(vectors.detach().cpu())
        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            print(f"reference batches={batch_number}/{total_batches}", flush=True)
    matrix = torch.cat(chunks, dim=0)
    expected_shape = (len(windows), representation.HIDDEN_SIZE)
    if tuple(matrix.shape) != expected_shape:
        raise ReferenceBuildError(
            f"reference matrix shape {tuple(matrix.shape)} != {expected_shape}"
        )
    matrix64 = matrix.to(dtype=torch.float64)
    return matrix64.mean(dim=0), matrix64.std(dim=0, correction=STD_DDOF)


def validate_statistics(mean: Any, std: Any, torch: Any) -> dict[str, object]:
    expected = (representation.HIDDEN_SIZE,)
    if tuple(mean.shape) != expected or tuple(std.shape) != expected:
        raise ReferenceBuildError(f"reference vectors must have shape {expected}")
    checks = {
        "mean_finite": bool(torch.isfinite(mean).all().item()),
        "std_finite": bool(torch.isfinite(std).all().item()),
        "std_strictly_positive": bool((std > 0).all().item()),
        "shared_for_all_conditions": True,
        "experimental_data_used": False,
    }
    passed = (
        checks["mean_finite"]
        and checks["std_finite"]
        and checks["std_strictly_positive"]
        and checks["shared_for_all_conditions"]
        and checks["experimental_data_used"] is False
    )
    if not passed:
        raise ReferenceBuildError(f"reference numerical QC failed: {checks}")
    checks["std_min"] = float(std.min().item())
    checks["std_max"] = float(std.max().item())
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "huggingface"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        args.input = args.input.resolve()
        args.source_manifest = args.source_manifest.resolve()
        args.output = args.output.resolve()
        args.manifest = args.manifest.resolve()
        if args.output == args.manifest:
            raise ReferenceBuildError("statistics and manifest paths must differ")
        read_source_manifest(args.source_manifest)
        rows = read_reference_rows(args.input)
        runtime = model_runtime.resolve_runtime(
            args.cache_dir.resolve(), args.device, allow_download=False
        )
        windows, documents = token_windows(rows, runtime["tokenizer"])
        mean, std = compute_statistics(windows, runtime, batch_size=args.batch_size)
        qc = validate_statistics(mean, std, runtime["torch"])
        stats = {
            "schema": REFERENCE_STATS_SCHEMA,
            "count": len(windows),
            "dtype": "float64 JSON numbers computed from FP32 representations",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "mean_shape": [representation.HIDDEN_SIZE],
            "std_shape": [representation.HIDDEN_SIZE],
            "std_ddof": STD_DDOF,
        }
        write_text_atomic(args.output, render_json(stats))
        stats_hash = sha256_file(args.output)
        runtime_identity = runtime.get("runtime_identity")
        runtime_details = (
            dict(runtime_identity) if isinstance(runtime_identity, Mapping) else {}
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "pass",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "blind_reference_build": True,
            "experimental_m_a_i_data_used": False,
            "source": {
                "file": args.input.relative_to(PROJECT_ROOT).as_posix(),
                "repository": EXPECTED_SOURCE_REPOSITORY,
                "sha256": EXPECTED_INPUT_SHA256,
                "upstream_commit": EXPECTED_SOURCE_COMMIT,
                "upstream_path": EXPECTED_SOURCE_PATH,
                "documents": documents,
                "total_qwen_tokens": sum(int(row["qwen_tokens"]) for row in documents),
                "qwen_h5_windows": len(windows),
            },
            "configuration": {
                "h_qwen_tokens": representation.H_QWEN_TOKENS,
                "stride_qwen_tokens": 1,
                "cross_document_windows": False,
                "add_special_tokens": False,
                "representation": representation.REPRESENTATION_LAYER,
                "pooling": representation.POOLING,
                "batch_size": args.batch_size,
                "std_ddof": STD_DDOF,
                "conditions_sharing_statistics": list(CONDITIONS),
            },
            "model": {
                "identifier": model_runtime.MODEL_ID,
                "revision": model_runtime.MODEL_REVISION,
                "resolved_snapshot": runtime["snapshot_path"].name,
                "tokenizer_class": type(runtime["tokenizer"]).__name__,
                "model_class": type(runtime["model"]).__name__,
            },
            "runtime": runtime_details,
            "qc": qc,
            "output": {
                "file": args.output.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": stats_hash,
                "count": len(windows),
            },
        }
        write_text_atomic(args.manifest, render_json(manifest))
    except (ReferenceBuildError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"qwen_h5_windows={len(windows)}")
    print(f"statistics_sha256={stats_hash}")
    print("status=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
