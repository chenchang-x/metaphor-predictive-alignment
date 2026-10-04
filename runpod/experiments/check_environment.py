#!/usr/bin/env python3
"""Audit the pinned Qwen3.5-9B BF16 single-GPU environment.

Network access is disabled by default.  ``--allow-download`` is the only
supported way to populate the pinned Hugging Face snapshot before a run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qwen_alignment.artifacts import render_json, write_text_atomic
from qwen_alignment.model_runtime import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_LAYERS,
    EXPECTED_VOCAB_SIZE,
    resolve_runtime,
)


SMOKE_TEXT = "The metaphor was understood."
DEFAULT_OUTPUT = PROJECT_ROOT / "environment" / "qwen3_5_9b_environment.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=(DEFAULT_MODEL_ID,), default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", choices=(DEFAULT_REVISION,), default=DEFAULT_REVISION)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="One explicit visible CUDA device, normally cuda:0",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly permit downloading only the pinned snapshot artifacts",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run_smoke(runtime: dict[str, Any]) -> dict[str, object]:
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]

    torch.cuda.reset_peak_memory_stats(device)
    encoded = tokenizer(
        SMOKE_TEXT,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    input_length = int(input_ids.shape[1])
    model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    final_hidden_shape = list(output.hidden_states[-1].shape)
    logits_shape = list(output.logits.shape)
    checks = {
        "all_logits_finite": bool(torch.isfinite(output.logits).all().item()),
        "eval_mode": not model.training,
        "raw_tokenizer_add_special_tokens_false": True,
        "chat_template_preserved_for_rq2": bool(
            getattr(tokenizer, "chat_template", None)
        ),
        "final_hidden_shape_correct": final_hidden_shape
        == [1, input_length, EXPECTED_HIDDEN_SIZE],
        "hidden_state_count_correct": len(output.hidden_states) == EXPECTED_LAYERS + 1,
        "logits_shape_correct": logits_shape
        == [1, input_length, EXPECTED_VOCAB_SIZE],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Qwen environment smoke checks failed: {failed}")
    return {
        "text": SMOKE_TEXT,
        "input_token_count": input_length,
        "final_hidden_shape": final_hidden_shape,
        "hidden_state_count": len(output.hidden_states),
        "logits_shape": logits_shape,
        "checks": checks,
        "gpu_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def build_report(runtime: dict[str, Any], smoke: dict[str, object], *, allow_download: bool) -> dict[str, object]:
    return {
        "schema": "munch-qwen3.5-9b-environment-audit/v1",
        "status": "pass",
        "snapshot_resolution": {
            "network_download_allowed": allow_download,
            "formal_generation_local_files_only": True,
        },
        "runtime_identity": runtime["runtime_identity"],
        "smoke_test": smoke,
        "host": {
            "machine": platform.machine(),
            "os": platform.system(),
            "os_release": platform.release(),
            "python": platform.python_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runtime = resolve_runtime(
            args.cache_dir.resolve(),
            args.device,
            allow_download=args.allow_download,
        )
        smoke = run_smoke(runtime)
        report = build_report(runtime, smoke, allow_download=args.allow_download)
        rendered = render_json(report)
        write_text_atomic(args.output.resolve(), rendered)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
