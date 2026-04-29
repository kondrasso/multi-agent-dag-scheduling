#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_full_mlvp.sh [options]

Full-scale launcher for the MLVP experiment.
Defaults:
  - workspaces: 1,2,3,4
  - DAG sizes: 3000,6000,12000,24000
  - topology classes per workspace: 48
  - train/eval split per class: 3 / 10
  - train mode: full train split
  - eval policies: mlvp,donf,fifo,minmin,maxmin
  - tuning objective: best-baseline-ratio against donf,fifo,minmin,maxmin

Any option accepted by run_long_training.sh may be passed through here.
Later arguments override these defaults.

Examples:
  ./cpp/mlvp/scripts/run_full_mlvp.sh
  ./cpp/mlvp/scripts/run_full_mlvp.sh --jobs 24 --population 48 --generations 60
  ./cpp/mlvp/scripts/run_full_mlvp.sh --root /data/mlvp_runs --run-name mlvp_full
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LONG_RUN_SCRIPT="${SCRIPT_DIR}/run_long_training.sh"

if [[ ! -x "${LONG_RUN_SCRIPT}" ]]; then
  chmod +x "${LONG_RUN_SCRIPT}"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

DEFAULT_RUN_NAME="full_mlvp_$(date +"%Y%m%d_%H%M%S")"

exec "${LONG_RUN_SCRIPT}" \
  --run-name "${DEFAULT_RUN_NAME}" \
  --workspace 1,2,3,4 \
  --train-per-class 3 \
  --eval-per-class 10 \
  --full-train \
  --train-sample-mode stratified \
  --population 32 \
  --generations 40 \
  --mutation-prob 0.8 \
  --mutation-sigma 0.25 \
  --candidate-cap 8 \
  --gamma 0.2 \
  --epsilon 0.05 \
  --max-iterations 8 \
  --objective best-baseline-ratio \
  --baselines donf,fifo,minmin,maxmin \
  --policies mlvp,donf,fifo,minmin,maxmin \
  --assign-types alpha \
  --seed 12345 \
  "$@"
