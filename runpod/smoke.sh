#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/qwen3p5-9b-metaphor-predictive-alignment"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
CACHE_DIR="$PROJECT_ROOT/.cache/huggingface"
SMOKE_INPUT="$PROJECT_ROOT/data/smoke/munch_smoke_subset.csv"
FORMAL_INPUT="$PROJECT_ROOT/data/processed/analysis_items.csv"
REFERENCE_INPUT="$PROJECT_ROOT/vendor/ias-naturalstories/texts_400words.csv"
REFERENCE_SOURCE_MANIFEST="$PROJECT_ROOT/vendor/ias-naturalstories/source_manifest.json"
REFERENCE_STATS="$PROJECT_ROOT/data/reference/ias_naturalstories_qwen3_5_9b_h5_final_stats.json"
REFERENCE_MANIFEST="$PROJECT_ROOT/data/reference/ias_naturalstories_qwen3_5_9b_h5_final_manifest.json"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_RUN_DIR="$PROJECT_ROOT/results/smoke/$RUN_TAG"
BENCHMARK_SUMMARY="$SMOKE_RUN_DIR/generation_speed_benchmark.json"
AUXILIARY_BENCHMARK="$SMOKE_RUN_DIR/auxiliary_stage_benchmark.json"
STAGE_TIMINGS="$SMOKE_RUN_DIR/stage_timings.tsv"
PLANNING_SUMMARY="$SMOKE_RUN_DIR/end_to_end_planning.json"
LATEST_PLANNING_GATE="$PROJECT_ROOT/environment/latest_end_to_end_planning.json"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: run bash runpod/bootstrap.sh first" >&2
  exit 1
fi
for required in "$SMOKE_INPUT" "$FORMAL_INPUT" "$REFERENCE_INPUT" "$REFERENCE_SOURCE_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing input $required" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
mkdir -p \
  "$SMOKE_RUN_DIR" \
  "$PROJECT_ROOT/data/reference" \
  "$PROJECT_ROOT/environment" \
  "$PROJECT_ROOT/.tmp"
rm -f "$LATEST_PLANNING_GATE"
exec > >(tee -a "$SMOKE_RUN_DIR/run.log") 2>&1

printf 'stage\telapsed_seconds\tstatus\n' > "$STAGE_TIMINGS"
run_timed() {
  local stage="$1"
  shift
  local started ended elapsed
  started="$(date +%s%N)"
  "$@"
  ended="$(date +%s%N)"
  elapsed="$("$PYTHON" -c "print(($ended - $started) / 1000000000)")"
  printf '%s\t%s\tpass\n' "$stage" "$elapsed" >> "$STAGE_TIMINGS"
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

if [[ ! -s "$REFERENCE_STATS" || ! -s "$REFERENCE_MANIFEST" ]]; then
  run_timed reference_build "$PYTHON" runpod/experiments/build_reference_stats.py \
    --input "$REFERENCE_INPUT" \
    --source-manifest "$REFERENCE_SOURCE_MANIFEST" \
    --output "$REFERENCE_STATS" \
    --manifest "$REFERENCE_MANIFEST" \
    --cache-dir "$CACHE_DIR" \
    --device cuda \
    --batch-size "${REFERENCE_BATCH_SIZE:-128}"
else
  printf 'reference_build\t0\treused\n' >> "$STAGE_TIMINGS"
fi

run_timed reference_validation "$PYTHON" runpod/experiments/validate_reference_stats.py \
  --stats "$REFERENCE_STATS" \
  --manifest "$REFERENCE_MANIFEST"

run_timed rq1_generation_smoke "$PYTHON" runpod/experiments/generate_continuations.py \
  --input "$SMOKE_INPUT" \
  --output "$SMOKE_RUN_DIR/continuations.jsonl" \
  --summary "$SMOKE_RUN_DIR/continuations_summary.json" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0

mkdir -p "$SMOKE_RUN_DIR/distances"
run_timed distance_smoke "$PYTHON" runpod/experiments/compute_distances.py \
  --input "$SMOKE_RUN_DIR/continuations.jsonl" \
  --items "$SMOKE_INPUT" \
  --stats "$REFERENCE_STATS" \
  --output-dir "$SMOKE_RUN_DIR/distances" \
  --output-prefix qwen3p5_9b_smoke \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size "${DISTANCE_BATCH_SIZE:-64}"

run_timed surprisal_smoke "$PYTHON" runpod/experiments/run_surprisal.py \
  --input "$SMOKE_INPUT" \
  --output-dir "$SMOKE_RUN_DIR/surprisal" \
  --scope smoke \
  --cache-dir "$CACHE_DIR" \
  --device cuda \
  --batch-size "${SURPRISAL_BATCH_SIZE:-8}"

run_timed rq2_controls "$PYTHON" runpod/experiments/run_direct_judgement.py \
  --scope controls \
  --controls "$PROJECT_ROOT/data/controls/qwen_instruct_direct_controls.json" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0 \
  --output-dir "$SMOKE_RUN_DIR/direct-controls"

run_timed continuation_benchmark "$PYTHON" runpod/experiments/generate_continuations.py \
  --input "$FORMAL_INPUT" \
  --benchmark \
  --benchmark-items "${BENCHMARK_ITEMS:-12}" \
  --benchmark-blocks "${BENCHMARK_BLOCKS:-3}" \
  --full-item-count 880 \
  --benchmark-summary "$BENCHMARK_SUMMARY" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0

run_timed auxiliary_benchmarks "$PYTHON" runpod/experiments/benchmark_auxiliary_stages.py \
  --input "$FORMAL_INPUT" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0 \
  --items 12 \
  --surprisal-batch-size "${SURPRISAL_BATCH_SIZE:-8}" \
  --output "$AUXILIARY_BENCHMARK"

"$PYTHON" runpod/experiments/summarize_smoke_timing.py \
  --timings "$STAGE_TIMINGS" \
  --generation-benchmark "$BENCHMARK_SUMMARY" \
  --auxiliary-benchmark "$AUXILIARY_BENCHMARK" \
  --output "$PLANNING_SUMMARY" \
  --smoke-items 3 \
  --formal-items 880 \
  --rq2-controls 12 \
  --safety-factor 1.20 \
  --cpu-postprocess-allowance-seconds 300
cp "$PLANNING_SUMMARY" "$LATEST_PLANNING_GATE"

echo "SMOKE_STATUS=pass"
echo "SMOKE_RUN_DIR=$SMOKE_RUN_DIR"
echo "BENCHMARK_SUMMARY=$BENCHMARK_SUMMARY"
echo "AUXILIARY_BENCHMARK=$AUXILIARY_BENCHMARK"
echo "END_TO_END_PLANNING_SUMMARY=$PLANNING_SUMMARY"
echo "LATEST_PLANNING_GATE=$LATEST_PLANNING_GATE"
