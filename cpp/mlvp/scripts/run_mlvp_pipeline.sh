#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_mlvp_pipeline.sh --corpus-root DIR --report-root DIR [options]

Options:
  --workspace csv              Workspace ids to process. Default: 1,2,3,4
  --train-per-class N          Frozen train instances per class. Default: 3
  --eval-per-class N           Frozen eval instances per class. Default: 10
  --train-sample-size N        Number of train DAGs used during tuning. Default: all
  --train-sample-mode STR      random|stratified. Default: stratified
  --assign-types STR           random|alpha. Default: alpha
  --objective STR              makespan|mean-ratio|best-baseline-ratio.
                               Default: best-baseline-ratio
  --baselines csv              Baselines for tuning objective. Default: donf
  --policies csv               Policies for eval matrix.
                               Default: mlvp,donf,fifo,minmin,maxmin
  --population N               GA population. Default: 32
  --generations N              GA generations. Default: 40
  --mutation-prob X            GA mutation probability. Default: 0.8
  --mutation-sigma X           GA mutation sigma. Default: 0.25
  --jobs N                     Parallel jobs for tuning. Default: 1
  --candidate-cap N            MLVP candidate cap. Default: 8
  --gamma X                    MLVP gamma. Default: 0.2
  --epsilon X                  MLVP epsilon. Default: 0.05
  --max-iterations N           MLVP max iterations. Default: 8
  --seed N                     Base seed. Default: 0
  --daggen-binary PATH         DAGGEN executable path.
  --refreeze                   Delete and regenerate existing workspace corpora.
  --skip-build                 Reuse existing binaries instead of running make.
  --help                       Show this message.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
MLVP_ROOT="${REPO_ROOT}/cpp/mlvp"
BIN_DIR="${MLVP_ROOT}/build/bin"
FREEZE_BIN="${BIN_DIR}/mlvp_freeze_corpus"
TUNE_BIN="${BIN_DIR}/mlvp_tune_weights"
EVAL_BIN="${BIN_DIR}/mlvp_eval_matrix"

CORPUS_ROOT=""
REPORT_ROOT=""
WORKSPACES="1,2,3,4"
TRAIN_PER_CLASS=3
EVAL_PER_CLASS=10
TRAIN_SAMPLE_SIZE=0
TRAIN_SAMPLE_MODE="stratified"
ASSIGN_TYPES="alpha"
OBJECTIVE="best-baseline-ratio"
BASELINES="donf"
POLICIES="mlvp,donf,fifo,minmin,maxmin"
POPULATION=32
GENERATIONS=40
MUTATION_PROB=0.8
MUTATION_SIGMA=0.25
JOBS=1
CANDIDATE_CAP=8
GAMMA=0.2
EPSILON=0.05
MAX_ITERATIONS=8
SEED=0
DAGGEN_BINARY="${REPO_ROOT}/daggen/daggen"
REFREEZE=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus-root)
      CORPUS_ROOT="$2"
      shift 2
      ;;
    --report-root)
      REPORT_ROOT="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACES="$2"
      shift 2
      ;;
    --train-per-class)
      TRAIN_PER_CLASS="$2"
      shift 2
      ;;
    --eval-per-class)
      EVAL_PER_CLASS="$2"
      shift 2
      ;;
    --train-sample-size)
      TRAIN_SAMPLE_SIZE="$2"
      shift 2
      ;;
    --train-sample-mode)
      TRAIN_SAMPLE_MODE="$2"
      shift 2
      ;;
    --assign-types)
      ASSIGN_TYPES="$2"
      shift 2
      ;;
    --objective)
      OBJECTIVE="$2"
      shift 2
      ;;
    --baselines)
      BASELINES="$2"
      shift 2
      ;;
    --policies)
      POLICIES="$2"
      shift 2
      ;;
    --population)
      POPULATION="$2"
      shift 2
      ;;
    --generations)
      GENERATIONS="$2"
      shift 2
      ;;
    --mutation-prob)
      MUTATION_PROB="$2"
      shift 2
      ;;
    --mutation-sigma)
      MUTATION_SIGMA="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --candidate-cap)
      CANDIDATE_CAP="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --epsilon)
      EPSILON="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --daggen-binary)
      DAGGEN_BINARY="$2"
      shift 2
      ;;
    --refreeze)
      REFREEZE=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CORPUS_ROOT}" || -z "${REPORT_ROOT}" ]]; then
  echo "error: --corpus-root and --report-root are required" >&2
  usage >&2
  exit 1
fi

if [[ ${SKIP_BUILD} -eq 0 ]]; then
  make -C "${MLVP_ROOT}" all
fi

mkdir -p "${CORPUS_ROOT}" "${REPORT_ROOT}"

COMBINED_CSV="${REPORT_ROOT}/summary_eval.csv"
COMBINED_INSTANCE_CSV="${REPORT_ROOT}/instances_eval.csv"
COMBINED_JSON_INDEX="${REPORT_ROOT}/summary_index.txt"
: > "${COMBINED_CSV}"
: > "${COMBINED_INSTANCE_CSV}"
: > "${COMBINED_JSON_INDEX}"
echo "workspace,dag_size,instances,policy,mean_makespan,mlvp_improvement_pct" >> "${COMBINED_CSV}"
echo "workspace,dag_size,instance,instance_name,policy,makespan,completed_tasks,cycles,max_ready_width,max_visible_width,mlvp_improvement_pct" >> "${COMBINED_INSTANCE_CSV}"

IFS=',' read -r -a WORKSPACE_LIST <<< "${WORKSPACES}"

for RAW_WS in "${WORKSPACE_LIST[@]}"; do
  WS=$(echo "${RAW_WS}" | tr -d '[:space:]')
  if [[ -z "${WS}" ]]; then
    continue
  fi
  WS_SEED=$((SEED + WS))

  WS_CORPUS_DIR="${CORPUS_ROOT}/ws${WS}"
  WS_REPORT_DIR="${REPORT_ROOT}/ws${WS}"
  WS_WEIGHTS="${WS_REPORT_DIR}/weights.txt"
  WS_HISTORY="${WS_REPORT_DIR}/history.csv"
  WS_EVAL_CSV="${WS_REPORT_DIR}/eval.csv"
  WS_INSTANCE_CSV="${WS_REPORT_DIR}/instances.csv"
  WS_EVAL_JSON="${WS_REPORT_DIR}/eval.json"

  mkdir -p "${WS_REPORT_DIR}"

  if [[ ${REFREEZE} -eq 1 && -d "${WS_CORPUS_DIR}" ]]; then
    rm -rf "${WS_CORPUS_DIR}"
  fi

  if [[ ! -d "${WS_CORPUS_DIR}" ]]; then
    "${FREEZE_BIN}" \
      --out-root "${CORPUS_ROOT}" \
      --workspace "${WS}" \
      --assign-types "${ASSIGN_TYPES}" \
      --train-per-class "${TRAIN_PER_CLASS}" \
      --eval-per-class "${EVAL_PER_CLASS}" \
      --seed "${WS_SEED}" \
      --daggen-binary "${DAGGEN_BINARY}" \
      --overwrite
  fi

  "${TUNE_BIN}" \
    --corpus-root "${CORPUS_ROOT}" \
    --workspace "${WS}" \
    --split train \
    --objective "${OBJECTIVE}" \
    --baselines "${BASELINES}" \
    --sample-size "${TRAIN_SAMPLE_SIZE}" \
    --sample-mode "${TRAIN_SAMPLE_MODE}" \
    --population "${POPULATION}" \
    --generations "${GENERATIONS}" \
    --mutation-prob "${MUTATION_PROB}" \
    --mutation-sigma "${MUTATION_SIGMA}" \
    --jobs "${JOBS}" \
    --assign-types "${ASSIGN_TYPES}" \
    --candidate-cap "${CANDIDATE_CAP}" \
    --gamma "${GAMMA}" \
    --epsilon "${EPSILON}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --seed "${WS_SEED}" \
    --out "${WS_WEIGHTS}" \
    --history-csv "${WS_HISTORY}"

  "${EVAL_BIN}" \
    --dot-root "${CORPUS_ROOT}" \
    --workspace "${WS}" \
    --split eval \
    --weights-file "${WS_WEIGHTS}" \
    --policies "${POLICIES}" \
    --candidate-cap "${CANDIDATE_CAP}" \
    --gamma "${GAMMA}" \
    --epsilon "${EPSILON}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --seed "${WS_SEED}" \
    --summary-csv "${WS_EVAL_CSV}" \
    --instances-csv "${WS_INSTANCE_CSV}" \
    --summary-json "${WS_EVAL_JSON}"

  tail -n +2 "${WS_EVAL_CSV}" >> "${COMBINED_CSV}"
  tail -n +2 "${WS_INSTANCE_CSV}" >> "${COMBINED_INSTANCE_CSV}"
  echo "ws${WS},${WS_EVAL_JSON}" >> "${COMBINED_JSON_INDEX}"
done

echo "combined summary -> ${COMBINED_CSV}"
echo "combined instances -> ${COMBINED_INSTANCE_CSV}"
echo "json index -> ${COMBINED_JSON_INDEX}"
