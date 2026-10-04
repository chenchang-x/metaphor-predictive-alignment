#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/qwen3p5-9b-metaphor-predictive-alignment"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
CACHE_DIR="$PROJECT_ROOT/.cache/huggingface"
FORMAL_INPUT="$PROJECT_ROOT/data/processed/analysis_items.csv"
REFERENCE_STATS="$PROJECT_ROOT/data/reference/ias_naturalstories_qwen3_5_9b_h5_final_stats.json"
REFERENCE_MANIFEST="$PROJECT_ROOT/data/reference/ias_naturalstories_qwen3_5_9b_h5_final_manifest.json"
PLANNING_GATE="$PROJECT_ROOT/environment/latest_end_to_end_planning.json"
PLANNING_SCHEMA="munch-qwen3.5-9b-end-to-end-runtime-plan/v3"
MODE="${1:-}"

case "$MODE" in
  new)
    RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
    FORMAL_RUN_DIR="$PROJECT_ROOT/runs/formal/$RUN_TAG"
    GENERATION_RESUME=()
    ;;
  resume)
    if [[ $# -ne 2 ]]; then
      echo "usage: bash runpod/formal.sh resume $PROJECT_ROOT/runs/formal/<run-tag>" >&2
      exit 2
    fi
    FORMAL_RUN_DIR="${2%/}"
    case "$FORMAL_RUN_DIR" in
      "$PROJECT_ROOT"/runs/formal/*) ;;
      *)
        echo "ERROR: resume directory must be below $PROJECT_ROOT/runs/formal" >&2
        exit 2
        ;;
    esac
    GENERATION_RESUME=(--resume)
    ;;
  *)
    echo "usage: bash runpod/formal.sh new" >&2
    echo "   or: bash runpod/formal.sh resume $PROJECT_ROOT/runs/formal/<run-tag>" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: run bash runpod/bootstrap.sh first" >&2
  exit 1
fi
for required in "$FORMAL_INPUT" "$REFERENCE_STATS" "$REFERENCE_MANIFEST" "$PLANNING_GATE"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing prerequisite $required; complete smoke first" >&2
    exit 1
  fi
done
if ! "$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); gate=value.get("gpu_memory_gate", {}); total=gate.get("total_memory_bytes"); identity=gate.get("runtime_identity", {}); ok=(value.get("schema")==sys.argv[2] and gate.get("status")=="pass" and type(total) is int and total>0 and identity.get("gpu_total_memory_bytes")==total); raise SystemExit(0 if ok else 1)' "$PLANNING_GATE" "$PLANNING_SCHEMA"; then
  echo "ERROR: formal requires a current v3 passing dynamic GPU memory gate with total VRAM" >&2
  echo "Run bash runpod/smoke.sh on this GPU first." >&2
  exit 1
fi
if ! SMOKE_GPU_NAME="$("$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); name=value.get("gpu_memory_gate", {}).get("runtime_identity", {}).get("gpu_name"); sys.exit(1) if not isinstance(name, str) or not name else print(name)' "$PLANNING_GATE")"; then
  echo "ERROR: planning gate has no smoke GPU name; rerun smoke on this GPU" >&2
  exit 1
fi
if ! CURRENT_GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
  echo "ERROR: cannot read current GPU name; rerun smoke after GPU access is available" >&2
  exit 1
fi
CURRENT_GPU_NAME="${CURRENT_GPU_NAMES%%$'\n'*}"
CURRENT_GPU_NAME="${CURRENT_GPU_NAME%$'\r'}"
if [[ -z "$CURRENT_GPU_NAME" || "$CURRENT_GPU_NAME" != "$SMOKE_GPU_NAME" ]]; then
  echo "ERROR: current GPU '$CURRENT_GPU_NAME' differs from smoke GPU '$SMOKE_GPU_NAME'" >&2
  echo "Run bash runpod/smoke.sh on this GPU before formal." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$FORMAL_RUN_DIR" "$PROJECT_ROOT/.tmp"
exec > >(tee -a "$FORMAL_RUN_DIR/run.log") 2>&1

FORMAL_STAGE_TIMINGS="$FORMAL_RUN_DIR/formal_stage_timings.tsv"
if [[ ! -f "$FORMAL_STAGE_TIMINGS" ]]; then
  printf 'stage\telapsed_seconds\tstatus\n' > "$FORMAL_STAGE_TIMINGS"
fi
run_timed() {
  local stage="$1"
  shift
  local started ended elapsed
  started="$(date +%s%N)"
  "$@"
  ended="$(date +%s%N)"
  elapsed="$("$PYTHON" -c "print(($ended - $started) / 1000000000)")"
  printf '%s\t%s\tpass\n' "$stage" "$elapsed" >> "$FORMAL_STAGE_TIMINGS"
  echo "stage_elapsed_seconds[$stage]=$elapsed"
}

export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export TEMP="$PROJECT_ROOT/.tmp"
export TMP="$PROJECT_ROOT/.tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

run_timed reference_validation "$PYTHON" scripts/validate_reference_stats.py \
  --stats "$REFERENCE_STATS" \
  --manifest "$REFERENCE_MANIFEST"

run_timed continuation_generation "$PYTHON" scripts/generate_continuations.py \
  --input "$FORMAL_INPUT" \
  --run-dir "$FORMAL_RUN_DIR/continuations" \
  "${GENERATION_RESUME[@]}" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0

mkdir -p "$FORMAL_RUN_DIR/distances"
run_timed distance "$PYTHON" scripts/compute_distances.py \
  --input "$FORMAL_RUN_DIR/continuations/continuations.jsonl" \
  --items "$FORMAL_INPUT" \
  --stats "$REFERENCE_STATS" \
  --output-dir "$FORMAL_RUN_DIR/distances" \
  --output-prefix qwen3p5_9b_formal \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size "${DISTANCE_BATCH_SIZE:-64}"

run_timed primary_analysis "$PYTHON" scripts/run_primary_analysis.py \
  --sentence-distances "$FORMAL_RUN_DIR/distances/qwen3p5_9b_formal_sentence_distances.csv" \
  --distance-summary "$FORMAL_RUN_DIR/distances/qwen3p5_9b_formal_distance_summary.json" \
  --run-manifest "$FORMAL_RUN_DIR/continuations/run_manifest.json" \
  --output-dir "$FORMAL_RUN_DIR/primary"

run_timed surprisal "$PYTHON" scripts/run_surprisal.py \
  --input "$FORMAL_INPUT" \
  --output-dir "$FORMAL_RUN_DIR/surprisal" \
  --scope formal \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size "${SURPRISAL_BATCH_SIZE:-8}"

run_timed rq2_direct "$PYTHON" scripts/run_direct_judgement.py \
  --scope formal \
  --analysis-items "$FORMAL_INPUT" \
  --controls "$PROJECT_ROOT/data/controls/qwen_instruct_direct_controls.json" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0 \
  --output-dir "$FORMAL_RUN_DIR/direct"

run_timed rq2_analysis "$PYTHON" scripts/run_rq2_analysis.py \
  --direct-target "$FORMAL_RUN_DIR/direct/direct_judgement_target.csv" \
  --direct-manifest "$FORMAL_RUN_DIR/direct/direct_judgement_manifest.json" \
  --sentence-distances "$FORMAL_RUN_DIR/distances/qwen3p5_9b_formal_sentence_distances.csv" \
  --output-dir "$FORMAL_RUN_DIR/rq2"

echo "FORMAL_STATUS=pass"
echo "FORMAL_RUN_DIR=$FORMAL_RUN_DIR"
