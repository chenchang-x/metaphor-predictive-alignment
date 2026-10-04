"""Deterministic text rendering and small artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from typing import Mapping, Sequence


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def render_jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def render_csv(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8/LF text and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


__all__ = [
    "render_csv",
    "render_json",
    "render_jsonl",
    "sha256_file",
    "write_text_atomic",
]
