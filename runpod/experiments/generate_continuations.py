#!/usr/bin/env python3
"""Generate frozen Qwen3.5-9B continuations from raw M/A/I prefixes.

Every record is one item x condition x seed group containing 32 independent
five-token samples.  Production mode commits each group to SQLite and can
resume without repeating a completed sample or changing the RNG chain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment import model_runtime as ENVIRONMENT
from qwen_alignment.artifacts import render_json, sha256_file, write_text_atomic
from qwen_alignment.continuation_input import (
    ContinuationError,
    canonical_int,
    orthographic_words,
    read_input_rows,
    validate_input_rows,
)
from qwen_alignment.continuation_records import (
    read_jsonl,
    render_jsonl as _render_jsonl,
    validate_record,
    validate_records,
)
from qwen_alignment.contracts import (
    CONDITIONS,
    CONTINUATION_FORMAT_NAME as FORMAT_NAME,
    CONTINUATION_FORMAT_VERSION as FORMAT_VERSION,
    CONTINUATION_INPUT_COLUMNS as INPUT_COLUMNS,
    CONTINUATION_OUTPUT_JSONL as OUTPUT_JSONL,
    CONTINUATION_OUTPUT_SUMMARY as OUTPUT_SUMMARY,
    CONTINUATION_SCHEMA as SCHEMA,
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
from qwen_alignment.production_run import (
    CHECKPOINT_NAME,
    PARTIAL_SCHEMA,
    CheckpointStore,
    build_run_identity,
    finalize_run,
    initialise_or_resume_manifest,
    run_checkpointed_generation,
    run_lock,
)


BENCHMARK_SAFETY_FACTOR = 1.20


def resolve_runtime(cache_dir: Path, requested_device: str) -> dict[str, Any]:
    """Resolve a production runtime without any network fallback."""
    try:
        return ENVIRONMENT.resolve_runtime(
            cache_dir, requested_device, allow_download=False
        )
    except RuntimeError as exc:
        raise ContinuationError(str(exc)) from exc


def fixed_generation_config(
    tokenizer: Any, *, generation_config_class: Any | None = None
) -> Any:
    if generation_config_class is None:
        from transformers import GenerationConfig

        generation_config_class = GenerationConfig
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ContinuationError("Qwen tokenizer has no pad_token_id")
    # EOS remains a legal sampled token but is not a stopping event.  Both
    # limits are five so every returned alternative has exactly h=5 tokens.
    return generation_config_class(
        do_sample=DO_SAMPLE,
        min_new_tokens=H_QWEN_TOKENS,
        max_new_tokens=H_QWEN_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        eos_token_id=None,
        forced_eos_token_id=None,
        pad_token_id=pad_token_id,
        use_cache=True,
    )


def _generation_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _validate_generation_config(config: Any) -> None:
    expected = {
        "do_sample": True,
        "min_new_tokens": H_QWEN_TOKENS,
        "max_new_tokens": H_QWEN_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "eos_token_id": None,
        "forced_eos_token_id": None,
    }
    failed = [
        f"{name}={_generation_value(config, name)!r}"
        for name, value in expected.items()
        if _generation_value(config, name) != value
    ]
    if failed:
        raise ContinuationError(
            "generation config violates the frozen protocol: " + ", ".join(failed)
        )


def generate_record(
    row: Mapping[str, str],
    condition: str,
    seed: int,
    runtime: Mapping[str, Any],
    *,
    generation_config: Any | None = None,
    samples_per_condition: int = SAMPLES_PER_CONDITION,
) -> dict[str, object]:
    if condition not in CONDITIONS:
        raise ContinuationError(f"invalid condition {condition!r}")
    if seed not in SEEDS:
        raise ContinuationError(f"invalid frozen seed {seed}")
    if samples_per_condition != SAMPLES_PER_CONDITION:
        raise ContinuationError(
            f"samples_per_condition must equal frozen value {SAMPLES_PER_CONDITION}"
        )
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    if not str(device).startswith("cuda:"):
        raise ContinuationError("Qwen continuation generation requires one CUDA device")
    if generation_config is None:
        generation_config = fixed_generation_config(tokenizer)
    _validate_generation_config(generation_config)

    max_positions = int(model.config.max_position_embeddings)
    lower = condition.lower()
    prefix_text = row[f"{lower}_prefix_q3"]
    encoded = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prefix_token_ids = encoded["input_ids"][0].tolist()
    prefix_token_count = len(prefix_token_ids)
    if not prefix_token_ids:
        raise ContinuationError(f"i0={row['i0']} {condition} encoded to zero tokens")
    if prefix_token_count + H_QWEN_TOKENS > max_positions:
        raise ContinuationError(
            f"i0={row['i0']} {condition} prefix has {prefix_token_count} tokens; "
            f"h={H_QWEN_TOKENS} exceeds context limit {max_positions}"
        )
    model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        sequences = model.generate(
            **model_inputs,
            generation_config=generation_config,
            num_return_sequences=samples_per_condition,
        )
    sequence_rows = sequences.detach().cpu().tolist()
    if len(sequence_rows) != samples_per_condition:
        raise ContinuationError(
            f"i0={row['i0']} {condition} returned {len(sequence_rows)} sequences, "
            f"expected {samples_per_condition}"
        )

    continuations: list[dict[str, object]] = []
    for sample_index, sequence in enumerate(sequence_rows):
        if sequence[:prefix_token_count] != prefix_token_ids:
            raise ContinuationError(
                f"i0={row['i0']} {condition} sample={sample_index} "
                "does not preserve the encoded prefix"
            )
        token_ids = sequence[prefix_token_count:]
        if len(token_ids) != H_QWEN_TOKENS:
            raise ContinuationError(
                f"i0={row['i0']} {condition} seed={seed} sample={sample_index} "
                f"has {len(token_ids)} continuation tokens, expected {H_QWEN_TOKENS}"
            )
        continuations.append(
            {
                "sample_index": sample_index,
                "token_ids": token_ids,
                "tokens": list(tokenizer.convert_ids_to_tokens(token_ids)),
                "text": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
    record: dict[str, object] = {
        "schema": SCHEMA,
        "triple_id": row["triple_id"],
        "i0": int(row["i0"]),
        "sentence_id": int(row["sentence_id"]),
        "source_sid": row["source_sid"],
        "condition": condition,
        "target": row[f"{lower}_word"],
        "q_orthographic_words": Q_ORTHOGRAPHIC_WORDS,
        "q3_right_context": row["q3_right_context"],
        "prefix_text": prefix_text,
        "prefix_token_ids": prefix_token_ids,
        "prefix_token_count": prefix_token_count,
        "seed": seed,
        "h_qwen_tokens": H_QWEN_TOKENS,
        "continuations": continuations,
    }
    validate_record(record, row, label=f"i0={row['i0']} {condition} seed={seed}")
    return record


def _seed_runtime(runtime: Mapping[str, Any], seed: int) -> None:
    torch = runtime["torch"]
    numpy_module = runtime.get("numpy")
    random.seed(seed)
    if numpy_module is not None:
        numpy_module.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_records(
    rows: Sequence[Mapping[str, str]],
    runtime: Mapping[str, Any],
    *,
    seeds: Sequence[int] = SEEDS,
    samples_per_condition: int = SAMPLES_PER_CONDITION,
    generation_config: Any | None = None,
) -> list[dict[str, object]]:
    if tuple(seeds) != SEEDS:
        raise ContinuationError(f"seeds must equal frozen values {SEEDS}")
    if samples_per_condition != SAMPLES_PER_CONDITION:
        raise ContinuationError(
            f"samples_per_condition must equal frozen value {SAMPLES_PER_CONDITION}"
        )
    if generation_config is None:
        generation_config = fixed_generation_config(runtime["tokenizer"])
    _validate_generation_config(generation_config)

    records: list[dict[str, object]] = []
    for seed in seeds:
        _seed_runtime(runtime, seed)
        for row in rows:
            for condition in CONDITIONS:
                records.append(
                    generate_record(
                        row,
                        condition,
                        seed,
                        runtime,
                        generation_config=generation_config,
                        samples_per_condition=samples_per_condition,
                    )
                )
    validate_records(
        records,
        rows,
        seeds=seeds,
        samples_per_condition=samples_per_condition,
    )
    return records


def reset_generation_peak_memory(runtime: Mapping[str, Any]) -> None:
    """Start a CUDA peak measurement with only persistent model memory retained."""
    torch = runtime["torch"]
    device = runtime["device"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def generation_memory_facts(runtime: Mapping[str, Any]) -> dict[str, int]:
    """Capture both peak and post-generation CUDA allocator measurements."""
    torch = runtime["torch"]
    device = runtime["device"]
    torch.cuda.synchronize(device)
    return {
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes_after_generation": int(torch.cuda.memory_reserved(device)),
        "allocated_bytes_after_generation": int(torch.cuda.memory_allocated(device)),
    }


def write_outputs(
    records: Sequence[Mapping[str, object]],
    input_rows: Sequence[Mapping[str, str]],
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    runtime: Mapping[str, Any],
    *,
    generation_gpu_memory: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if output_path.resolve() == summary_path.resolve():
        raise ContinuationError("JSONL output and summary paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_records(records, input_rows)
    output_tmp = output_path.with_name(output_path.name + ".tmp")
    output_tmp.write_text(_render_jsonl(records), encoding="utf-8", newline="\n")
    loaded_records = read_jsonl(output_tmp)
    if loaded_records != list(records):
        raise ContinuationError("JSONL save/read round trip changed the records")
    os.replace(output_tmp, output_path)

    summary: dict[str, object] = {
        "purpose": "Qwen3.5-9B raw-prefix continuation smoke test",
        "schema": SCHEMA,
        "input": {
            "file": input_path.name,
            "rows": len(input_rows),
        },
        "runtime_identity": runtime.get("runtime_identity", {"fake_runtime": True}),
        "generation_gpu_memory": (
            dict(generation_gpu_memory)
            if generation_gpu_memory is not None
            else None
        ),
        "protocol": {
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
            "raw_tokenizer_add_special_tokens": False,
            "chat_template_used": False,
            "runtime_chat_template_preserved_for_rq2": bool(
                getattr(runtime["tokenizer"], "chat_template", None)
            ),
            "seed_scope": "set once per seed before stable item then M/A/I iteration",
        },
        "output": {
            "file": output_path.name,
            "format": "UTF-8 JSON Lines; LF; no BOM; one group per line",
            "groups": validation["groups"],
            "continuations": validation["continuations"],
        },
        "validation": {
            **validation,
            "input_m_a_i_and_q3_checked": True,
            "all_continuations_exactly_h_tokens": (
                validation["continuation_token_lengths"] == [H_QWEN_TOKENS]
            ),
            "save_read_round_trip_equal": True,
        },
    }
    rendered = render_json(summary)
    write_text_atomic(summary_path, rendered)
    if json.loads(summary_path.read_text("utf-8")) != summary:
        raise ContinuationError("summary save/read round trip changed the data")
    return summary


def generate_outputs(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    cache_dir: Path,
    requested_device: str,
) -> dict[str, object]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()
    if len({input_path, output_path, summary_path}) != 3:
        raise ContinuationError("input, JSONL output, and summary paths must differ")
    rows = read_input_rows(input_path)
    runtime = resolve_runtime(cache_dir.resolve(), requested_device)
    reset_generation_peak_memory(runtime)
    records = build_records(rows, runtime)
    memory = generation_memory_facts(runtime)
    return write_outputs(
        records,
        rows,
        input_path,
        output_path,
        summary_path,
        runtime,
        generation_gpu_memory=memory,
    )


def _prefix_token_count(row: Mapping[str, str], tokenizer: Any) -> int:
    return max(
        len(
            tokenizer(
                row[f"{condition.lower()}_prefix_q3"], add_special_tokens=False
            )["input_ids"]
        )
        for condition in CONDITIONS
    )


def select_longest_prefix(
    rows: Sequence[Mapping[str, str]], tokenizer: Any
) -> tuple[Mapping[str, str], str, int]:
    """Select the exact longest raw M/A/I prefix, with stable tie-breaking."""
    if not rows:
        raise ContinuationError("cannot select a memory probe from zero rows")
    cases: list[tuple[int, int, int, Mapping[str, str], str]] = []
    for row in rows:
        i0 = canonical_int(row["i0"], "i0", 0)
        for condition_index, condition in enumerate(CONDITIONS):
            token_count = len(
                tokenizer(
                    row[f"{condition.lower()}_prefix_q3"],
                    add_special_tokens=False,
                )["input_ids"]
            )
            cases.append((-token_count, i0, condition_index, row, condition))
    negative_count, _i0, _condition_index, row, condition = min(
        cases, key=lambda case: case[:3]
    )
    return row, condition, -negative_count


def select_timing_rows(
    rows: Sequence[Mapping[str, str]], tokenizer: Any, item_count: int
) -> list[tuple[Mapping[str, str], int]]:
    if not 1 <= item_count <= len(rows):
        raise ContinuationError(
            f"benchmark item count must be between 1 and {len(rows)}, found {item_count}"
        )
    profiled = sorted(
        ((row, _prefix_token_count(row, tokenizer)) for row in rows),
        key=lambda pair: (pair[1], canonical_int(pair[0]["i0"], "i0", 0)),
    )
    indices = [
        math.floor((rank + 0.5) * len(profiled) / item_count)
        for rank in range(item_count)
    ]
    selected = [profiled[index] for index in indices]
    if len({row["triple_id"] for row, _ in selected}) != item_count:
        raise ContinuationError("benchmark quantile selection repeated a triple")
    return selected


def benchmark_generation(
    rows: Sequence[Mapping[str, str]],
    runtime: Mapping[str, Any],
    *,
    item_count: int = 12,
    block_count: int = 3,
    full_item_count: int = 880,
    model_load_seconds: float = 0.0,
    generation_config: Any | None = None,
) -> dict[str, object]:
    if block_count < 1 or item_count % block_count:
        raise ContinuationError("benchmark item count must divide evenly into blocks")
    selected = select_timing_rows(rows, runtime["tokenizer"], item_count)
    if generation_config is None:
        generation_config = fixed_generation_config(runtime["tokenizer"])
    _validate_generation_config(generation_config)

    warmup_row = selected[item_count // 2][0]
    started = time.perf_counter()
    build_records([warmup_row], runtime, generation_config=generation_config)
    warmup_seconds = time.perf_counter() - started

    block_size = item_count // block_count
    blocks: list[dict[str, object]] = []
    total_seconds = 0.0
    for block_index in range(block_count):
        block_pairs = selected[
            block_index * block_size : (block_index + 1) * block_size
        ]
        block_rows = [row for row, _ in block_pairs]
        started = time.perf_counter()
        generated = build_records(
            block_rows, runtime, generation_config=generation_config
        )
        elapsed = time.perf_counter() - started
        del generated
        total_seconds += elapsed
        blocks.append(
            {
                "block": block_index + 1,
                "items": len(block_rows),
                "max_prefix_token_range": [
                    min(length for _, length in block_pairs),
                    max(length for _, length in block_pairs),
                ],
                "seconds": elapsed,
                "seconds_per_item": elapsed / len(block_rows),
            }
        )
    seconds_per_item = total_seconds / item_count
    slowest = max(float(block["seconds_per_item"]) for block in blocks)

    longest_row, longest_condition, longest_count = select_longest_prefix(
        rows, runtime["tokenizer"]
    )
    _seed_runtime(runtime, SEEDS[0])
    reset_generation_peak_memory(runtime)
    memory_probe_record = generate_record(
        longest_row,
        longest_condition,
        SEEDS[0],
        runtime,
        generation_config=generation_config,
        samples_per_condition=SAMPLES_PER_CONDITION,
    )
    memory_probe = {
        "measurement": (
            "exact longest raw M/A/I prefix in the benchmark input; one real "
            "RQ1 generate call after reset_peak_memory_stats"
        ),
        "triple_id": longest_row["triple_id"],
        "i0": canonical_int(longest_row["i0"], "i0", 0),
        "condition": longest_condition,
        "prefix_token_count": longest_count,
        "num_return_sequences": SAMPLES_PER_CONDITION,
        "h_qwen_tokens": H_QWEN_TOKENS,
        **generation_memory_facts(runtime),
    }
    del memory_probe_record
    return {
        "schema": "munch-qwen3.5-9b-generation-speed-benchmark/v1",
        "purpose": "pre-results wall-time planning; continuation content discarded",
        "runtime_identity": runtime["runtime_identity"],
        "production_workload": {
            "items": full_item_count,
            "conditions": len(CONDITIONS),
            "seeds": len(SEEDS),
            "samples_per_condition_per_seed": SAMPLES_PER_CONDITION,
            "generate_calls": full_item_count * len(CONDITIONS) * len(SEEDS),
            "continuations": full_item_count
            * len(CONDITIONS)
            * len(SEEDS)
            * SAMPLES_PER_CONDITION,
        },
        "selection": {
            "method": "midpoint quantiles of max M/A/I Qwen prefix token count",
            "items": item_count,
            "blocks": block_count,
            "triple_ids": [row["triple_id"] for row, _ in selected],
            "max_prefix_token_counts": [length for _, length in selected],
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "warmup_seconds_excluded": warmup_seconds,
            "timed_seconds": total_seconds,
            "seconds_per_item": seconds_per_item,
            "items_per_minute": 60.0 / seconds_per_item,
            "blocks": blocks,
            "point_estimate_seconds": model_load_seconds
            + seconds_per_item * full_item_count,
            "planning_upper_seconds": model_load_seconds
            + slowest * full_item_count * BENCHMARK_SAFETY_FACTOR,
            "planning_upper_rule": "slowest block x full item count x 1.20, plus model load",
        },
        "gpu_memory_probe": memory_probe,
        "content_inspection": False,
    }


def run_production(
    *,
    input_path: Path,
    run_dir: Path,
    cache_dir: Path,
    device: str,
    dataset_id: str,
    resume: bool,
    stop_after_new_groups: int | None = None,
) -> dict[str, object]:
    input_path = input_path.resolve()
    run_dir = run_dir.resolve()
    rows = read_input_rows(input_path)
    with run_lock(run_dir):
        runtime = resolve_runtime(cache_dir.resolve(), device)
        identity = build_run_identity(
            project_root=PROJECT_ROOT,
            input_path=input_path,
            rows=rows,
            dataset_id=dataset_id,
            device=str(runtime["device"]),
            runtime_identity=runtime["runtime_identity"],
        )
        manifest = initialise_or_resume_manifest(run_dir, identity, resume=resume)
        with CheckpointStore(
            run_dir / CHECKPOINT_NAME, str(manifest["run_id"])
        ) as store:
            generation_config = fixed_generation_config(runtime["tokenizer"])

            def record_builder(
                row: Mapping[str, str], condition: str, seed: int
            ) -> Mapping[str, object]:
                return generate_record(
                    row,
                    condition,
                    seed,
                    runtime,
                    generation_config=generation_config,
                )

            def progress(completed: int, expected: int, generated: int) -> None:
                if completed == expected or completed % 25 == 0:
                    print(
                        f"production_progress={completed}/{expected} "
                        f"generated_this_invocation={generated}",
                        flush=True,
                    )

            invocation = run_checkpointed_generation(
                store=store,
                rows=rows,
                runtime=runtime,
                record_builder=record_builder,
                progress=progress,
                stop_after_new_groups=stop_after_new_groups,
            )
            if not invocation["complete"]:
                return {
                    "schema": PARTIAL_SCHEMA,
                    "run_id": manifest["run_id"],
                    "status": "partial",
                    "run_dir": str(run_dir),
                    "invocation": invocation,
                }
            return finalize_run(
                run_dir=run_dir,
                manifest=manifest,
                store=store,
                rows=rows,
                invocation=invocation,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "smoke" / "munch_smoke_subset.csv",
        help="Exact 23-column MUNCH analysis/smoke CSV in this Qwen project",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "smoke" / OUTPUT_JSONL,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data" / "smoke" / OUTPUT_SUMMARY,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "huggingface",
        help="Local cache populated by check_environment.py --allow-download",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="One explicit visible CUDA device; CPU and automatic placement are prohibited",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-items", type=int, default=12)
    parser.add_argument("--benchmark-blocks", type=int, default=3)
    parser.add_argument("--full-item-count", type=int, default=880)
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "generation_speed_benchmark.json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Dedicated Qwen run directory containing identity, checkpoint, and output",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dataset-id",
        default="munch-5b78a540-q3-qwen3p5-9b-project-copy-v1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and args.run_dir is None:
        print("ERROR: --resume requires --run-dir", file=sys.stderr)
        return 1
    if args.run_dir is not None and args.benchmark:
        print("ERROR: --run-dir and --benchmark are mutually exclusive", file=sys.stderr)
        return 1
    try:
        if args.run_dir is not None:
            summary = run_production(
                input_path=args.input,
                run_dir=args.run_dir,
                cache_dir=args.cache_dir,
                device=args.device,
                dataset_id=args.dataset_id,
                resume=args.resume,
            )
            print("Qwen3.5-9B production continuation run finished")
            print(f"run_id={summary['run_id']}")
            print(f"status={summary['status']}")
            if summary["status"] == "complete":
                for name in (
                    "observed_items",
                    "expected_items",
                    "observed_groups",
                    "expected_groups",
                    "observed_continuations",
                    "expected_continuations",
                    "duplicate_groups",
                    "missing_groups",
                ):
                    print(f"{name}={summary['validation'][name]}")
            return 0

        if args.benchmark:
            rows = read_input_rows(args.input.resolve())
            started = time.perf_counter()
            runtime = resolve_runtime(args.cache_dir.resolve(), args.device)
            model_load_seconds = time.perf_counter() - started
            summary = benchmark_generation(
                rows,
                runtime,
                item_count=args.benchmark_items,
                block_count=args.benchmark_blocks,
                full_item_count=args.full_item_count,
                model_load_seconds=model_load_seconds,
            )
            summary["input"] = {
                "file": args.input.name,
                "rows": len(rows),
                "sha256": sha256_file(args.input.resolve()),
            }
            write_text_atomic(args.benchmark_summary.resolve(), render_json(summary))
            timing = summary["timing"]
            print("Qwen3.5-9B continuation speed benchmark completed")
            print(f"timed_items={summary['selection']['items']}")
            print(f"seconds_per_item={timing['seconds_per_item']:.3f}")
            print(f"point_estimate_seconds={timing['point_estimate_seconds']:.1f}")
            print(f"planning_upper_seconds={timing['planning_upper_seconds']:.1f}")
            memory = summary["gpu_memory_probe"]
            print(f"peak_reserved_bytes={memory['peak_reserved_bytes']}")
            print(f"peak_allocated_bytes={memory['peak_allocated_bytes']}")
            print("continuation_content_inspected=false")
            return 0

        summary = generate_outputs(
            args.input,
            args.output,
            args.summary,
            args.cache_dir,
            args.device,
        )
    except (ContinuationError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Qwen3.5-9B continuation smoke generation completed")
    print(f"input_rows={summary['input']['rows']}")
    print(f"conditions={','.join(CONDITIONS)}")
    print(f"seeds={','.join(str(seed) for seed in SEEDS)}")
    print(f"groups={summary['output']['groups']}")
    print(f"continuations={summary['output']['continuations']}")
    print(f"h_qwen_tokens={H_QWEN_TOKENS}")
    print("save_read_round_trip_equal=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
