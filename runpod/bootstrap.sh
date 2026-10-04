#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/qwen3p5-9b-metaphor-predictive-alignment"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "ERROR: project is not extracted at $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p \
  "$PROJECT_ROOT/.cache/huggingface" \
  "$PROJECT_ROOT/.cache/torch" \
  "$PROJECT_ROOT/.cache/pycache" \
  "$PROJECT_ROOT/.tmp" \
  "$PROJECT_ROOT/environment" \
  "$PROJECT_ROOT/results" \
  "$PROJECT_ROOT/transfer"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/pycache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export TEMP="$PROJECT_ROOT/.tmp"
export TMP="$PROJECT_ROOT/.tmp"
export PIP_NO_CACHE_DIR=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv --system-site-packages "$PROJECT_ROOT/.venv"
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check -r requirements-runpod.txt
"$PYTHON" -m pip install --disable-pip-version-check --no-deps -e .
"$PYTHON" -m unittest discover -s tests -v

"$PYTHON" runpod/experiments/check_environment.py \
  --allow-download \
  --device cuda:0 \
  --cache-dir "$PROJECT_ROOT/.cache/huggingface" \
  --output "$PROJECT_ROOT/environment/qwen3_5_9b_environment.json"

"$PYTHON" runpod/experiments/tokenization_audit.py \
  --input "$PROJECT_ROOT/data/processed/analysis_items.csv" \
  --cache-dir "$PROJECT_ROOT/.cache/huggingface" \
  --output "$PROJECT_ROOT/environment/tokenization_audit_summary.json"

echo "BOOTSTRAP_STATUS=pass"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MODEL_CACHE=$PROJECT_ROOT/.cache/huggingface"
