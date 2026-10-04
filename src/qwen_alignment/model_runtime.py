"""Pinned Qwen3.5-9B text-only verification and single-GPU loading."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_REVISION

EXPECTED_REPOSITORY_MODEL_TYPE = "qwen3_5"
EXPECTED_REPOSITORY_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5_text"
EXPECTED_ARCHITECTURE = "Qwen3_5ForCausalLM"
EXPECTED_LAYERS = 32
EXPECTED_HIDDEN_SIZE = 4_096
EXPECTED_VOCAB_SIZE = 248_320
EXPECTED_MAX_POSITIONS = 262_144
EXPECTED_ATTENTION_HEADS = 16
EXPECTED_KEY_VALUE_HEADS = 4
EXPECTED_PARAMETER_COUNT = 8_953_803_264
EXPECTED_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "eager"

# These are the complete model/tokenizer inputs used by Transformers at the
# pinned commit.  Documentation files are intentionally excluded from the
# scientific artifact manifest.
MODEL_ARTIFACT_FILES = (
    "config.json",
    "chat_template.jinja",
    "merges.txt",
    "model.safetensors-00001-of-00004.safetensors",
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_SHARDS = tuple(
    name for name in MODEL_ARTIFACT_FILES if name.endswith(".safetensors")
)


def _config_value(config: Any, name: str) -> Any:
    """Read a Qwen config field from either an object or a test mapping."""
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _text_config(config: Any) -> Any:
    """Unwrap the text config from the repository's composite Qwen config."""
    if _config_value(config, "model_type") == EXPECTED_REPOSITORY_MODEL_TYPE:
        return _config_value(config, "text_config")
    return config


def architecture_contract(config: Any) -> dict[str, object]:
    """Return frozen text fields from a composite or loaded causal config."""
    config = _text_config(config)
    return {
        "model_type": _config_value(config, "model_type"),
        "num_hidden_layers": _config_value(config, "num_hidden_layers"),
        "hidden_size": _config_value(config, "hidden_size"),
        "vocab_size": _config_value(config, "vocab_size"),
        "max_position_embeddings": _config_value(
            config, "max_position_embeddings"
        ),
        "num_attention_heads": _config_value(config, "num_attention_heads"),
        "num_key_value_heads": _config_value(config, "num_key_value_heads"),
    }


def validate_architecture_config(config: Any) -> dict[str, object]:
    """Reject any architecture other than the frozen unquantised text model."""
    text_config = _text_config(config)
    observed = architecture_contract(config)
    expected = {
        "model_type": EXPECTED_MODEL_TYPE,
        "num_hidden_layers": EXPECTED_LAYERS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "max_position_embeddings": EXPECTED_MAX_POSITIONS,
        "num_attention_heads": EXPECTED_ATTENTION_HEADS,
        "num_key_value_heads": EXPECTED_KEY_VALUE_HEADS,
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, observed {observed[name]!r}"
        for name in expected
        if observed[name] != expected[name]
    ]
    configured_dtype = _config_value(text_config, "dtype")
    if configured_dtype is None:
        configured_dtype = _config_value(text_config, "torch_dtype")
    if configured_dtype is not None:
        configured_dtype = str(configured_dtype).removeprefix("torch.")
    if configured_dtype != EXPECTED_DTYPE:
        mismatches.append(
            f"dtype: expected {EXPECTED_DTYPE!r}, observed {configured_dtype!r}"
        )
    if (
        _config_value(config, "quantization_config") is not None
        or _config_value(text_config, "quantization_config") is not None
    ):
        mismatches.append("quantization_config must be absent")
    if mismatches:
        raise RuntimeError(
            "pinned Qwen3.5-9B text architecture verification failed: "
            + "; ".join(mismatches)
        )
    return observed


def verify_artifacts(snapshot_path: Path) -> dict[str, dict[str, int]]:
    """Check that the pinned snapshot contains every required runtime file."""
    snapshot_path = Path(snapshot_path)
    missing = [name for name in MODEL_ARTIFACT_FILES if not (snapshot_path / name).is_file()]
    if missing:
        raise RuntimeError(
            "pinned Qwen3.5-9B snapshot is incomplete: "
            + ", ".join(missing)
        )

    try:
        raw_config = json.loads((snapshot_path / "config.json").read_text("utf-8"))
        index = json.loads(
            (snapshot_path / "model.safetensors.index.json").read_text("utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read pinned Qwen metadata: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise RuntimeError("pinned Qwen config.json is not an object")
    validate_architecture_config(raw_config)
    repository_model_type = raw_config.get("model_type")
    if repository_model_type != EXPECTED_REPOSITORY_MODEL_TYPE:
        raise RuntimeError(
            "pinned Qwen repository model type mismatch: "
            f"expected {EXPECTED_REPOSITORY_MODEL_TYPE!r}, "
            f"observed {repository_model_type!r}"
        )
    architectures = raw_config.get("architectures")
    if architectures != [EXPECTED_REPOSITORY_ARCHITECTURE]:
        raise RuntimeError(
            "pinned Qwen repository architecture class mismatch: "
            f"expected {[EXPECTED_REPOSITORY_ARCHITECTURE]!r}, "
            f"observed {architectures!r}"
        )
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise RuntimeError("pinned Qwen safetensors index has no weight_map")
    indexed_shards = set(index["weight_map"].values())
    if indexed_shards != set(MODEL_SHARDS):
        raise RuntimeError(
            "pinned Qwen shard index mismatch: "
            f"expected {sorted(MODEL_SHARDS)!r}, observed {sorted(indexed_shards)!r}"
        )

    return {
        name: {
            "bytes": (snapshot_path / name).stat().st_size,
        }
        for name in MODEL_ARTIFACT_FILES
    }


def resolve_snapshot(
    cache_dir: Path,
    *,
    allow_download: bool = False,
    snapshot_download: Any | None = None,
) -> Path:
    """Resolve only the pinned commit; networking is an explicit opt-in."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot_download

        snapshot_download = hub_snapshot_download
    snapshot_path = Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            revision=DEFAULT_REVISION,
            cache_dir=Path(cache_dir),
            local_files_only=not allow_download,
            allow_patterns=MODEL_ARTIFACT_FILES,
        )
    )
    if snapshot_path.name != DEFAULT_REVISION:
        raise RuntimeError(
            "checkpoint revision mismatch: "
            f"expected {DEFAULT_REVISION}, found {snapshot_path.name}"
        )
    return snapshot_path


def select_device(requested: str, torch: Any) -> Any:
    """Select one visible CUDA device and reject CPU/automatic placement."""
    try:
        device = torch.device(requested)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"invalid CUDA device {requested!r}") from exc
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("Qwen production inference requires one explicit CUDA GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    index = 0 if getattr(device, "index", None) is None else int(device.index)
    if index != 0:
        raise RuntimeError(
            "the frozen runtime uses cuda:0 after CUDA_VISIBLE_DEVICES=0"
        )
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; visible device count is {count}"
        )
    selected = torch.device(f"cuda:{index}")
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(supports_bf16) and not bool(supports_bf16()):
        raise RuntimeError(f"{selected} does not report BF16 support")
    return selected


def configure_determinism(torch: Any) -> dict[str, object]:
    """Set and report deterministic inference controls before model loading."""
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace != ":4096:8":
        raise RuntimeError(
            "the frozen runtime requires CUBLAS_WORKSPACE_CONFIG=:4096:8; "
            f"observed {workspace!r}"
        )
    torch.use_deterministic_algorithms(True)
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        if hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = False
    cuda_backend = getattr(backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = False
    return determinism_identity(torch)


def determinism_identity(torch: Any) -> dict[str, object]:
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    cuda_backend = getattr(backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    deterministic_enabled = getattr(
        torch, "are_deterministic_algorithms_enabled", lambda: False
    )
    return {
        "algorithms": "torch deterministic algorithms",
        "deterministic_algorithms_enabled": bool(deterministic_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cuda_matmul_allow_tf32": getattr(matmul, "allow_tf32", None),
        "seed_policy": (
            "seed Python, NumPy, CPU PyTorch, and all visible CUDA generators "
            "once per seed; stable item then M/A/I group order"
        ),
        "reproducibility_scope": (
            "exact resume within the recorded model, artifacts, GPU, driver, "
            "CUDA, PyTorch, and Transformers runtime"
        ),
    }


def _nvidia_smi_facts() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"uuid": "unavailable", "driver": "unavailable"}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return {"uuid": "unavailable", "driver": "unavailable"}
    # CUDA_VISIBLE_DEVICES=0 makes the first physical row the selected device.
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 3:
        return {"uuid": "unavailable", "driver": "unavailable"}
    return {"uuid": parts[1], "driver": parts[2]}


def _parameter_facts(model: Any) -> tuple[int, list[str], list[str]]:
    parameter_count = 0
    dtypes: set[str] = set()
    devices: set[str] = set()
    for parameter in model.parameters():
        parameter_count += int(parameter.numel())
        dtypes.add(str(parameter.dtype).removeprefix("torch."))
        devices.add(str(parameter.device))
    return parameter_count, sorted(dtypes), sorted(devices)


def validate_chat_template(tokenizer: Any) -> None:
    """Require the upstream conversational template while leaving it unchanged."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("pinned Qwen3.5 tokenizer has no chat template")


def validate_loaded_runtime(
    model: Any, tokenizer: Any, device: Any, torch: Any
) -> dict[str, object]:
    architecture = validate_architecture_config(model.config)
    parameter_count, dtypes, devices = _parameter_facts(model)
    attention = _config_value(model.config, "_attn_implementation")
    failures: list[str] = []
    loaded_model_type = _config_value(model.config, "model_type")
    if loaded_model_type != EXPECTED_MODEL_TYPE:
        failures.append(
            f"loaded config model_type is {loaded_model_type!r}, "
            f"expected {EXPECTED_MODEL_TYPE!r}"
        )
    if type(model).__name__ != EXPECTED_ARCHITECTURE:
        failures.append(
            f"model class is {type(model).__name__}, expected {EXPECTED_ARCHITECTURE}"
        )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        failures.append(
            f"parameter count is {parameter_count}, expected {EXPECTED_PARAMETER_COUNT}"
        )
    if dtypes != [EXPECTED_DTYPE]:
        failures.append(f"parameter dtypes are {dtypes!r}, expected BF16 only")
    if devices != [str(device)]:
        failures.append(
            f"parameters span {devices!r}, expected only {str(device)!r}"
        )
    if attention != ATTENTION_IMPLEMENTATION:
        failures.append(
            f"attention implementation is {attention!r}, expected 'eager'"
        )
    if bool(getattr(model, "training", True)):
        failures.append("model is not in eval mode")
    if any(
        bool(getattr(model, name, False))
        for name in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit")
    ):
        failures.append("model reports a quantised load")
    if _config_value(model.config, "quantization_config") is not None:
        failures.append("loaded model config contains quantization_config")
    try:
        validate_chat_template(tokenizer)
    except RuntimeError as exc:
        failures.append(str(exc))
    if failures:
        raise RuntimeError("loaded Qwen runtime contract failed: " + "; ".join(failures))
    return {
        **architecture,
        "architecture": type(model).__name__,
        "parameter_count": parameter_count,
        "parameter_dtypes": dtypes,
        "parameter_devices": devices,
        "attention_implementation": attention,
        "quantized": False,
    }


def build_runtime_identity(
    *,
    torch: Any,
    transformers: Any,
    huggingface_hub: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    snapshot_path: Path,
    artifact_manifest: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    facts = validate_loaded_runtime(model, tokenizer, device, torch)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    nvidia = _nvidia_smi_facts()
    return {
        "model": {
            "identifier": DEFAULT_MODEL_ID,
            "requested_revision": DEFAULT_REVISION,
            "resolved_commit": Path(snapshot_path).name,
            **facts,
            "artifact_files": {
                name: dict(details)
                for name, details in sorted(artifact_manifest.items())
            },
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "chat_template_available": True,
            "chat_template_preserved": True,
            "rq1_interface": {
                "raw_text_prefixes": True,
                "add_special_tokens": False,
                "chat_template_used": False,
            },
        },
        "load": {
            "dtype": EXPECTED_DTYPE,
            "low_cpu_mem_usage": True,
            "device_map": {"": str(device)},
            "single_gpu": True,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "local_files_only": True,
            "trust_remote_code": False,
            "quantization": None,
        },
        "gpu": {
            "selected_device": str(device),
            "name": str(properties.name),
            "uuid": nvidia["uuid"],
            "compute_capability": [int(capability[0]), int(capability[1])],
            "total_memory_bytes": int(properties.total_memory),
            "driver": nvidia["driver"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": str(__import__("numpy").__version__),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": str(huggingface_hub.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
        },
        "determinism": determinism_identity(torch),
    }


def resolve_runtime(
    cache_dir: Path,
    requested_device: str,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Load the frozen BF16 model onto exactly one GPU.

    ``allow_download`` exists for the explicit environment-preparation stage.
    Production callers must retain its default ``False``.
    """
    # This must be set before the first CUDA operation in a fresh process.
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace != ":4096:8":
        raise RuntimeError(
            "the frozen runtime requires CUBLAS_WORKSPACE_CONFIG=:4096:8; "
            f"observed {workspace!r}"
        )
    visible_devices = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if visible_devices != "0":
        raise RuntimeError(
            "the frozen runtime requires CUDA_VISIBLE_DEVICES=0; "
            f"observed {visible_devices!r}"
        )

    import huggingface_hub
    import numpy
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    device = select_device(requested_device, torch)
    determinism = configure_determinism(torch)
    snapshot_path = resolve_snapshot(
        Path(cache_dir),
        allow_download=allow_download,
        snapshot_download=huggingface_hub.snapshot_download,
    )
    artifact_manifest = verify_artifacts(snapshot_path)

    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, **common)
    validate_chat_template(tokenizer)
    model = Qwen3_5ForCausalLM.from_pretrained(
        snapshot_path,
        **common,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        device_map={"": str(device)},
    )
    model.eval()
    model.requires_grad_(False)

    runtime_identity = build_runtime_identity(
        torch=torch,
        transformers=transformers,
        huggingface_hub=huggingface_hub,
        tokenizer=tokenizer,
        model=model,
        device=device,
        snapshot_path=snapshot_path,
        artifact_manifest=artifact_manifest,
    )
    # Preserve the state captured immediately after deterministic setup.  The
    # second value proves that validation did not relax a control.
    runtime_identity["determinism"] = {
        **determinism,
        **determinism_identity(torch),
    }
    return {
        "torch": torch,
        "transformers": transformers,
        "huggingface_hub": huggingface_hub,
        "numpy": numpy,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "snapshot_path": snapshot_path,
        "artifact_manifest": artifact_manifest,
        "runtime_identity": runtime_identity,
    }


__all__ = [
    "ATTENTION_IMPLEMENTATION",
    "DEFAULT_MODEL_ID",
    "DEFAULT_REVISION",
    "EXPECTED_DTYPE",
    "EXPECTED_HIDDEN_SIZE",
    "EXPECTED_LAYERS",
    "EXPECTED_MAX_POSITIONS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_REPOSITORY_ARCHITECTURE",
    "EXPECTED_REPOSITORY_MODEL_TYPE",
    "EXPECTED_VOCAB_SIZE",
    "MODEL_ARTIFACT_FILES",
    "MODEL_ID",
    "MODEL_REVISION",
    "architecture_contract",
    "build_runtime_identity",
    "configure_determinism",
    "determinism_identity",
    "resolve_runtime",
    "resolve_snapshot",
    "select_device",
    "validate_architecture_config",
    "validate_chat_template",
    "validate_loaded_runtime",
    "verify_artifacts",
]
