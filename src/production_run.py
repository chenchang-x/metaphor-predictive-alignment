"""Crash-safe, duplicate-safe Qwen continuation production runs."""

from __future__ import annotations

import base64
import json
import os
import platform
import random
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .artifacts import render_json, sha256_file, write_text_atomic
from .continuation_input import ContinuationError
from .continuation_records import validate_record
from .contracts import (
    CONDITIONS,
    DO_SAMPLE,
    EOS_STOPPING,
    H_QWEN_TOKENS,
    Q_ORTHOGRAPHIC_WORDS,
    SAMPLES_PER_CONDITION,
    SEEDS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from .model_runtime import DEFAULT_MODEL_ID, DEFAULT_REVISION


RUN_SCHEMA = "munch-qwen3.5-9b-production-run/v1"
CHECKPOINT_SCHEMA = "munch-qwen3.5-9b-generation-checkpoint/v1"
SUMMARY_SCHEMA = "munch-qwen3.5-9b-production-summary/v1"
PARTIAL_SCHEMA = "munch-qwen3.5-9b-production-partial/v1"
CONFIG_ID = "munch-qwen3.5-9b-q3-h5-k32-seeds11-23-37-v1"
RUN_ID_PREFIX = "munch-qwen3p5-9b-"
MANIFEST_NAME = "run_manifest.json"
CHECKPOINT_NAME = "generation_checkpoint.sqlite3"
OUTPUT_NAME = "continuations.jsonl"
SUMMARY_NAME = "completion_summary.json"
LOCK_NAME = ".generation.lock"


@dataclass(frozen=True)
class ExpectedGroup:
    ordinal: int
    seed: int
    condition: str
    row: Mapping[str, str]

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.row["triple_id"], self.condition, self.seed)


@dataclass(frozen=True)
class StoredGroup:
    ordinal: int
    triple_id: str
    condition: str
    seed: int
    record_json: str
    rng_before_json: str
    rng_after_json: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.triple_id, self.condition, self.seed)


def expected_groups(rows: Sequence[Mapping[str, str]]) -> list[ExpectedGroup]:
    groups: list[ExpectedGroup] = []
    ordinal = 0
    for seed in SEEDS:
        for row in rows:
            for condition in CONDITIONS:
                groups.append(ExpectedGroup(ordinal, seed, condition, row))
                ordinal += 1
    return groups


def _validate_runtime_identity(identity: Mapping[str, object], device: str) -> None:
    try:
        model = identity["model"]
        load = identity["load"]
        gpu = identity["gpu"]
        software = identity["software"]
    except KeyError as exc:
        raise ContinuationError(
            f"runtime identity is missing section {exc.args[0]!r}"
        ) from exc
    if not all(isinstance(value, Mapping) for value in (model, load, gpu, software)):
        raise ContinuationError("runtime identity sections must be objects")
    assert isinstance(model, Mapping)
    assert isinstance(load, Mapping)
    assert isinstance(gpu, Mapping)
    assert isinstance(software, Mapping)
    checks = {
        "model identifier": model.get("identifier") == DEFAULT_MODEL_ID,
        "model revision": model.get("requested_revision") == DEFAULT_REVISION,
        "resolved commit": model.get("resolved_commit") == DEFAULT_REVISION,
        "BF16": load.get("dtype") == "bfloat16",
        "eager attention": load.get("attention_implementation") == "eager",
        "single GPU": load.get("single_gpu") is True,
        "local files only": load.get("local_files_only") is True,
        "no quantization": load.get("quantization") is None,
        "selected CUDA device": str(gpu.get("selected_device", "")).startswith("cuda:"),
        "CUDA runtime": bool(software.get("cuda_runtime")),
        "PyTorch version": bool(software.get("torch")),
        "Transformers version": bool(software.get("transformers")),
    }
    if not str(device).startswith("cuda"):
        checks["requested CUDA device"] = False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ContinuationError(
            "runtime identity violates the Qwen production contract: "
            + ", ".join(failed)
        )


def build_run_identity(
    *,
    project_root: Path,
    input_path: Path,
    rows: Sequence[Mapping[str, str]],
    dataset_id: str,
    device: str,
    runtime_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not dataset_id.strip():
        raise ContinuationError("production dataset_id must not be empty")
    if not str(device).startswith("cuda"):
        raise ContinuationError("production runs require an explicit CUDA device")
    if runtime_identity is not None:
        _validate_runtime_identity(runtime_identity, device)
    try:
        relative_input = input_path.resolve().relative_to(
            project_root.resolve()
        ).as_posix()
    except ValueError:
        relative_input = input_path.name
    groups = len(rows) * len(CONDITIONS) * len(SEEDS)
    return {
        "config_id": CONFIG_ID,
        "project_namespace": "qwen3.5-9b",
        "dataset": {
            "dataset_id": dataset_id,
            "input_file": relative_input,
            "input_sha256": sha256_file(input_path),
            "triple_rows": len(rows),
            "sentence_ids": len({row["sentence_id"] for row in rows}),
            "source_sids": len({row["source_sid"] for row in rows}),
        },
        "model": {
            "identifier": DEFAULT_MODEL_ID,
            "revision": DEFAULT_REVISION,
            "instruction_tuned": True,
        },
        "sampling": {
            "conditions": list(CONDITIONS),
            "q_orthographic_words": Q_ORTHOGRAPHIC_WORDS,
            "h_qwen_tokens": H_QWEN_TOKENS,
            "samples_per_condition_per_seed": SAMPLES_PER_CONDITION,
            "seeds": list(SEEDS),
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "eos_stopping": EOS_STOPPING,
            "tokenizer_add_special_tokens": False,
            "chat_template_used_for_rq1": False,
            "seed_scope": "set once per seed; stable item then M/A/I iteration",
        },
        "expected": {
            "groups": groups,
            "continuations": groups * SAMPLES_PER_CONDITION,
            "groups_per_seed": len(rows) * len(CONDITIONS),
            "groups_per_condition": len(rows) * len(SEEDS),
        },
        "runtime": (
            dict(runtime_identity)
            if runtime_identity is not None
            else {
                "unresolved_test_identity": True,
                "requested_device": device,
                "python": platform.python_version(),
            }
        ),
    }


def _resume_contract(identity: Mapping[str, object]) -> dict[str, object]:
    """Return fields that must match when continuing one RNG chain."""
    fields = (
        "config_id",
        "project_namespace",
        "dataset",
        "model",
        "sampling",
        "expected",
    )
    contract = {field: identity.get(field) for field in fields}
    runtime = identity.get("runtime")
    if not isinstance(runtime, Mapping):
        contract["runtime"] = runtime
        return contract

    def section(name: str) -> Mapping[str, object]:
        value = runtime.get(name)
        return value if isinstance(value, Mapping) else {}

    runtime_model = section("model")
    load = section("load")
    gpu = section("gpu")
    software = section("software")
    contract["runtime"] = {
        "model": {
            key: runtime_model.get(key)
            for key in ("identifier", "requested_revision", "resolved_commit")
        },
        "load": {
            key: load.get(key)
            for key in (
                "dtype",
                "attention_implementation",
                "single_gpu",
                "quantization",
            )
        },
        "gpu": {"name": gpu.get("name")},
        "software": {
            key: software.get(key)
            for key in ("torch", "transformers", "cuda_runtime")
        },
    }
    return contract


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Hold a process lock that the operating system releases after a crash."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / LOCK_NAME
    handle = lock_path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ContinuationError(f"production run is already active: {run_dir}") from exc
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def initialise_or_resume_manifest(
    run_dir: Path, identity: Mapping[str, object], *, resume: bool
) -> dict[str, object]:
    manifest_path = run_dir / MANIFEST_NAME
    if resume:
        if not manifest_path.is_file():
            raise ContinuationError("--resume requires an existing run_manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContinuationError(f"could not read production manifest: {exc}") from exc
        if manifest.get("schema") != RUN_SCHEMA:
            raise ContinuationError("production manifest schema mismatch")
        saved_identity = manifest.get("identity")
        if not isinstance(saved_identity, dict) or _resume_contract(
            saved_identity
        ) != _resume_contract(identity):
            raise ContinuationError(
                "resume identity mismatch: scientific or runtime contract changed"
            )
        return manifest

    existing = [path.name for path in run_dir.iterdir() if path.name != LOCK_NAME]
    if existing:
        raise ContinuationError(
            "new production run directory is not empty; use --resume only for its original run"
        )
    now = datetime.now(timezone.utc)
    run_id = f"{RUN_ID_PREFIX}{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    manifest: dict[str, object] = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_utc": now.isoformat(),
        "identity": dict(identity),
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "file": CHECKPOINT_NAME,
            "durability": "SQLite FULL transaction after every item x condition x seed group",
            "unique_key": ["run_id", "triple_id", "condition", "seed"],
            "rng_state": "PyTorch CPU and selected-CUDA states before and after every group",
        },
    }
    write_text_atomic(manifest_path, render_json(manifest))
    return manifest


class CheckpointStore:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.connection = sqlite3.connect(self.path, timeout=10.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                checkpoint_schema TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                triple_id TEXT NOT NULL,
                i0 INTEGER NOT NULL,
                condition TEXT NOT NULL,
                seed INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                rng_before_json TEXT NOT NULL,
                rng_after_json TEXT NOT NULL,
                PRIMARY KEY (run_id, triple_id, condition, seed),
                UNIQUE (run_id, ordinal)
            )
            """
        )
        existing = self.connection.execute(
            "SELECT run_id, checkpoint_schema FROM run_metadata"
        ).fetchall()
        if not existing:
            self.connection.execute(
                "INSERT INTO run_metadata VALUES (?, ?)",
                (run_id, CHECKPOINT_SCHEMA),
            )
            self.connection.commit()
        elif existing != [(run_id, CHECKPOINT_SCHEMA)]:
            self.connection.close()
            raise ContinuationError("checkpoint belongs to a different Qwen run or schema")

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def integrity_check(self) -> None:
        result = self.connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ContinuationError(f"SQLite checkpoint integrity failure: {result}")

    def load(self) -> dict[int, StoredGroup]:
        rows = self.connection.execute(
            """
            SELECT ordinal, triple_id, condition, seed, record_json,
                   rng_before_json, rng_after_json
            FROM generations WHERE run_id = ? ORDER BY ordinal
            """,
            (self.run_id,),
        ).fetchall()
        return {int(row[0]): StoredGroup(*row) for row in rows}

    def insert(
        self,
        group: ExpectedGroup,
        record: Mapping[str, object],
        rng_before_json: str,
        rng_after_json: str,
    ) -> None:
        if record.get("run_id") != self.run_id:
            raise ContinuationError("production record run_id does not match checkpoint")
        record_json = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                INSERT INTO generations (
                    run_id, ordinal, triple_id, i0, condition, seed,
                    record_json, rng_before_json, rng_after_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    group.ordinal,
                    group.row["triple_id"],
                    int(group.row["i0"]),
                    group.condition,
                    group.seed,
                    record_json,
                    rng_before_json,
                    rng_after_json,
                ),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            self.connection.rollback()
            raise ContinuationError(
                f"duplicate generation refused for key={group.key}"
            ) from exc
        except Exception:
            self.connection.rollback()
            raise

    def counts(self) -> dict[str, object]:
        total = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM generations WHERE run_id = ?", (self.run_id,)
            ).fetchone()[0]
        )
        by_seed = {
            str(seed): int(count)
            for seed, count in self.connection.execute(
                "SELECT seed, COUNT(*) FROM generations WHERE run_id = ? GROUP BY seed",
                (self.run_id,),
            )
        }
        by_condition = {
            str(condition): int(count)
            for condition, count in self.connection.execute(
                "SELECT condition, COUNT(*) FROM generations WHERE run_id = ? GROUP BY condition",
                (self.run_id,),
            )
        }
        item_count = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT triple_id) FROM generations WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()[0]
        )
        by_seed_condition = {
            f"{seed}:{condition}": int(count)
            for seed, condition, count in self.connection.execute(
                """
                SELECT seed, condition, COUNT(*) FROM generations
                WHERE run_id = ? GROUP BY seed, condition
                """,
                (self.run_id,),
            )
        }
        return {
            "groups": total,
            "items": item_count,
            "groups_by_seed": by_seed,
            "groups_by_condition": by_condition,
            "groups_by_seed_condition": by_seed_condition,
        }


def _tensor_state_to_text(state: Any) -> str:
    values = state.detach().cpu().tolist()
    return base64.b64encode(bytes(values)).decode("ascii")


def capture_rng_state(torch: Any, device: Any) -> str:
    state = {
        "cpu": _tensor_state_to_text(torch.get_rng_state()),
        "cuda_device": str(device),
        "cuda": _tensor_state_to_text(torch.cuda.get_rng_state(device)),
    }
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def _text_to_tensor_state(value: str, torch: Any) -> Any:
    return torch.tensor(list(base64.b64decode(value)), dtype=torch.uint8)


def restore_rng_state(state_json: str, torch: Any, device: Any) -> None:
    try:
        state = json.loads(state_json)
        if state["cuda_device"] != str(device):
            raise ContinuationError("checkpoint CUDA device does not match runtime")
        torch.set_rng_state(_text_to_tensor_state(state["cpu"], torch))
        torch.cuda.set_rng_state(
            _text_to_tensor_state(state["cuda"], torch), device=device
        )
    except ContinuationError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContinuationError("checkpoint contains invalid RNG state") from exc


def validate_checkpoint(
    stored: Mapping[int, StoredGroup],
    groups: Sequence[ExpectedGroup],
    *,
    run_id: str | None = None,
) -> None:
    expected_by_ordinal = {group.ordinal: group for group in groups}
    for ordinal, saved in stored.items():
        expected = expected_by_ordinal.get(ordinal)
        if expected is None or saved.key != expected.key:
            raise ContinuationError(f"checkpoint has unexpected group at ordinal={ordinal}")
        try:
            record = json.loads(saved.record_json)
        except json.JSONDecodeError as exc:
            raise ContinuationError(f"checkpoint record {ordinal} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ContinuationError(f"checkpoint record {ordinal} is not an object")
        if run_id is not None and record.get("run_id") != run_id:
            raise ContinuationError(f"checkpoint record {ordinal} run_id mismatch")
        validate_record(
            record,
            expected.row,
            label=f"checkpoint record ordinal={ordinal}",
        )
        if (
            record["condition"] != expected.condition
            or record["seed"] != expected.seed
            or record["triple_id"] != expected.row["triple_id"]
        ):
            raise ContinuationError(f"checkpoint record {ordinal} key mismatch")


def _seed_rng(runtime: Mapping[str, Any], seed: int) -> None:
    torch = runtime["torch"]
    numpy_module = runtime.get("numpy")
    random.seed(seed)
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_checkpointed_generation(
    *,
    store: CheckpointStore,
    rows: Sequence[Mapping[str, str]],
    runtime: Mapping[str, Any],
    record_builder: Callable[[Mapping[str, str], str, int], Mapping[str, object]],
    progress: Callable[[int, int, int], None] | None = None,
    stop_after_new_groups: int | None = None,
) -> dict[str, int | bool]:
    if stop_after_new_groups is not None and stop_after_new_groups < 1:
        raise ContinuationError("stop_after_new_groups must be positive")
    groups = expected_groups(rows)
    store.integrity_check()
    stored = store.load()
    validate_checkpoint(stored, groups, run_id=store.run_id)
    torch = runtime["torch"]
    device = runtime["device"]
    if not str(device).startswith("cuda:"):
        raise ContinuationError("checkpointed Qwen generation requires one CUDA device")
    generated = 0
    skipped = 0
    current_seed: int | None = None
    for group in groups:
        if group.seed != current_seed:
            current_seed = group.seed
            _seed_rng(runtime, group.seed)
        rng_before = capture_rng_state(torch, device)
        saved = stored.get(group.ordinal)
        if saved is not None:
            if saved.rng_before_json != rng_before:
                raise ContinuationError(
                    f"RNG chain mismatch before completed group ordinal={group.ordinal}"
                )
            restore_rng_state(saved.rng_after_json, torch, device)
            skipped += 1
        else:
            record = dict(record_builder(group.row, group.condition, group.seed))
            record["run_id"] = store.run_id
            validate_record(record, group.row, label=f"new group ordinal={group.ordinal}")
            if record["condition"] != group.condition or record["seed"] != group.seed:
                raise ContinuationError(
                    f"record builder returned wrong key for ordinal={group.ordinal}"
                )
            rng_after = capture_rng_state(torch, device)
            store.insert(group, record, rng_before, rng_after)
            generated += 1
            if stop_after_new_groups is not None and generated >= stop_after_new_groups:
                return {
                    "complete": False,
                    "generated_this_invocation": generated,
                    "skipped_completed": skipped,
                    "observed_groups": int(store.counts()["groups"]),
                }
        if progress is not None:
            progress(group.ordinal + 1, len(groups), generated)
    return {
        "complete": True,
        "generated_this_invocation": generated,
        "skipped_completed": skipped,
        "observed_groups": int(store.counts()["groups"]),
    }


def finalize_run(
    *,
    run_dir: Path,
    manifest: Mapping[str, object],
    store: CheckpointStore,
    rows: Sequence[Mapping[str, str]],
    invocation: Mapping[str, int | bool],
) -> dict[str, object]:
    groups = expected_groups(rows)
    stored = store.load()
    validate_checkpoint(stored, groups, run_id=store.run_id)
    missing = [group.key for group in groups if group.ordinal not in stored]
    expected_keys = {group.key for group in groups}
    unexpected = [saved.key for saved in stored.values() if saved.key not in expected_keys]
    observed_keys = [saved.key for saved in stored.values()]
    duplicate_count = len(observed_keys) - len(set(observed_keys))
    if missing or unexpected or duplicate_count:
        raise ContinuationError(
            f"cannot finalize: missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={duplicate_count}"
        )

    output_path = run_dir / OUTPUT_NAME
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for group in groups:
            handle.write(stored[group.ordinal].record_json)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)

    observed = store.counts()
    expected_group_count = len(groups)
    expected_per_seed = len(rows) * len(CONDITIONS)
    expected_per_condition = len(rows) * len(SEEDS)
    count_checks = {
        "expected_items": len(rows),
        "observed_items": observed["items"],
        "expected_groups": expected_group_count,
        "observed_groups": observed["groups"],
        "missing_groups": 0,
        "unexpected_groups": 0,
        "duplicate_groups": 0,
        "expected_continuations": expected_group_count * SAMPLES_PER_CONDITION,
        "observed_continuations": int(observed["groups"]) * SAMPLES_PER_CONDITION,
        "expected_groups_per_seed": expected_per_seed,
        "observed_groups_by_seed": observed["groups_by_seed"],
        "expected_groups_per_condition": expected_per_condition,
        "observed_groups_by_condition": observed["groups_by_condition"],
        "expected_groups_per_seed_condition": len(rows),
        "observed_groups_by_seed_condition": observed["groups_by_seed_condition"],
    }
    counts_pass = (
        observed["items"] == len(rows)
        and observed["groups"] == expected_group_count
        and all(
            observed["groups_by_seed"].get(str(seed)) == expected_per_seed
            for seed in SEEDS
        )
        and all(
            observed["groups_by_condition"].get(condition) == expected_per_condition
            for condition in CONDITIONS
        )
        and all(
            observed["groups_by_seed_condition"].get(f"{seed}:{condition}")
            == len(rows)
            for seed in SEEDS
            for condition in CONDITIONS
        )
    )
    if not counts_pass:
        raise ContinuationError("final expected-count validation failed")
    summary: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "run_id": manifest["run_id"],
        "status": "complete",
        "identity": manifest["identity"],
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "file": CHECKPOINT_NAME,
            "durable_groups": observed["groups"],
            "unique_constraint": True,
            "sqlite_integrity": "ok",
        },
        "invocation": dict(invocation),
        "validation": {**count_checks, "status": "pass"},
        "output": {
            "file": OUTPUT_NAME,
            "format": "UTF-8 JSON Lines; stable seed/item/condition order",
            "sha256": sha256_file(output_path),
        },
    }
    write_text_atomic(run_dir / SUMMARY_NAME, render_json(summary))
    return summary


__all__ = [
    "CHECKPOINT_NAME",
    "CHECKPOINT_SCHEMA",
    "CONFIG_ID",
    "CheckpointStore",
    "MANIFEST_NAME",
    "OUTPUT_NAME",
    "PARTIAL_SCHEMA",
    "RUN_ID_PREFIX",
    "RUN_SCHEMA",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA",
    "build_run_identity",
    "capture_rng_state",
    "expected_groups",
    "finalize_run",
    "initialise_or_resume_manifest",
    "restore_rng_state",
    "run_checkpointed_generation",
    "run_lock",
    "validate_checkpoint",
]
