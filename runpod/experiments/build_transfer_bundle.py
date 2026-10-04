#!/usr/bin/env python3
"""Build the source-and-input-only archive uploaded to the RunPod volume."""

from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_NAME = "qwen3p5-9b-metaphor-predictive-alignment"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "transfer" / f"{PROJECT_NAME}.tar.gz"

# Only these individual project files and these code-only directories may enter
# the upload archive. Model caches, environments, results, transfer artifacts and
# research notes are outside the allowlist.
STATIC_FILES = (
    Path(".gitignore"),
    Path("README.md"),
    Path("docs/protocol.md"),
    Path("docs/protocol_en.md"),
    Path("pyproject.toml"),
    Path("requirements-local.txt"),
    Path("requirements-runpod.txt"),
)
CODE_ALLOWLIST = {
    Path("src/qwen_alignment"): frozenset({".py"}),
    Path("runpod/experiments"): frozenset({".py"}),
    Path("tests"): frozenset({".py"}),
    Path("runpod"): frozenset({".sh"}),
}
INPUT_FILES = (
    Path("data/processed/analysis_items.csv"),
    Path("data/smoke/munch_smoke_subset.csv"),
    Path("data/controls/qwen_instruct_direct_controls.json"),
    Path(
        "vendor/metaphor-understanding-challenge/correct_answers/"
        "for_judgement.csv"
    ),
    Path(
        "vendor/metaphor-understanding-challenge/correct_answers/"
        "for_generation.csv"
    ),
    Path("vendor/metaphor-understanding-challenge/source_manifest.json"),
    Path("vendor/ias-naturalstories/texts_400words.csv"),
    Path("vendor/ias-naturalstories/source_manifest.json"),
)


class BundleError(RuntimeError):
    pass


def _allowed_code_files(project_root: Path) -> Iterable[Path]:
    for relative_dir, suffixes in CODE_ALLOWLIST.items():
        directory = project_root / relative_dir
        if not directory.is_dir():
            raise BundleError(f"missing allowlisted code directory: {relative_dir}")
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                yield path.relative_to(project_root)


def collect_files(project_root: Path) -> list[Path]:
    """Resolve the positive allowlist and reject an incomplete input bundle."""
    project_root = project_root.resolve()
    required = (*STATIC_FILES, *INPUT_FILES)
    missing = [relative.as_posix() for relative in required if not (project_root / relative).is_file()]
    if missing:
        raise BundleError("missing required bundle input(s): " + ", ".join(missing))

    selected = set(required)
    selected.update(_allowed_code_files(project_root))
    return sorted(selected, key=lambda path: path.as_posix())


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o755 if info.name.endswith(".sh") else 0o644
    return info


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(project_root: Path, output_path: Path) -> tuple[list[Path], str]:
    project_root = project_root.resolve()
    output_path = output_path.resolve()
    files = collect_files(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            for relative in files:
                archive.add(
                    project_root / relative,
                    arcname=f"{PROJECT_NAME}/{relative.as_posix()}",
                    recursive=False,
                    filter=_archive_filter,
                )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return files, sha256_file(output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        files, digest = build_bundle(args.project_root, args.output)
    except (BundleError, OSError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"bundle={args.output.resolve()}")
    print(f"files={len(files)}")
    print(f"bundle_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
