#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_long_training.sh [options]

Long-running launcher for MLVP training/evaluation.
It wraps the MLVP pipeline with heavier defaults, timestamped output directories,
and a persistent log file.

Options:
  --workspace csv              Workspace ids. Default: 1,2,3,4
  --root DIR                   Root directory for runs. Default: ./cpp/mlvp/runs
  --run-name NAME              Run directory name. Default: auto timestamp
  --train-per-class N          Default: 3
  --eval-per-class N           Default: 10
  --train-sample-size N        Default: 96
  --train-sample-mode STR      random|stratified. Default: stratified
  --population N               Default: 24
  --generations N              Default: 20
  --mutation-prob X            Default: 0.8
  --mutation-sigma X           Default: 0.25
  --jobs N                     Default: max(1, nproc-2)
  --candidate-cap N            Default: 16
  --gamma X                    Default: 0.2
  --epsilon X                  Default: 0.05
  --max-iterations N           Default: 8
  --objective STR              Default: best-baseline-ratio
  --baselines csv              Default: donf
  --policies csv               Default: mlvp,donf,fifo,minmin,maxmin
  --assign-types STR           Default: alpha
  --seed N                     Default: 12345
  --daggen-binary PATH         Override DAGGEN binary path
  --full-train                 Disable train sampling (sample-size=0)
  --refreeze                   Regenerate saved corpora
  --skip-build                 Skip make step
  --help                       Show this message

Examples:
  ./cpp/mlvp/scripts/run_long_training.sh
  ./cpp/mlvp/scripts/run_long_training.sh --workspace 2,3,4 --jobs 24
  ./cpp/mlvp/scripts/run_long_training.sh --full-train --population 32 --generations 30
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_mlvp_pipeline.sh"

ROOT_DIR="${REPO_ROOT}/cpp/mlvp/runs"
RUN_NAME=""
WORKSPACES="1,2,3,4"
TRAIN_PER_CLASS=3
EVAL_PER_CLASS=10
TRAIN_SAMPLE_SIZE=96
TRAIN_SAMPLE_MODE="stratified"
POPULATION=24
GENERATIONS=20
MUTATION_PROB=0.8
MUTATION_SIGMA=0.25
CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)
if [[ "${CPU_COUNT}" -gt 2 ]]; then
  JOBS=$((CPU_COUNT - 2))
else
  JOBS=1
fi
CANDIDATE_CAP=16
GAMMA=0.2
EPSILON=0.05
MAX_ITERATIONS=8
OBJECTIVE="best-baseline-ratio"
BASELINES="donf"
POLICIES="mlvp,donf,fifo,minmin,maxmin"
ASSIGN_TYPES="alpha"
SEED=12345
DAGGEN_BINARY=""
REFREEZE=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACES="$2"
      shift 2
      ;;
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
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
    --assign-types)
      ASSIGN_TYPES="$2"
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
    --full-train)
      TRAIN_SAMPLE_SIZE=0
      shift
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

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME=$(date +"%Y%m%d_%H%M%S")
fi

RUN_DIR="${ROOT_DIR}/${RUN_NAME}"
CORPUS_DIR="${RUN_DIR}/corpus"
REPORT_DIR="${RUN_DIR}/reports"
LOG_PATH="${RUN_DIR}/pipeline.log"
mkdir -p "${RUN_DIR}"

PIPELINE_CMD=(
  "${PIPELINE_SCRIPT}"
  "--corpus-root" "${CORPUS_DIR}"
  "--report-root" "${REPORT_DIR}"
  "--workspace" "${WORKSPACES}"
  "--train-per-class" "${TRAIN_PER_CLASS}"
  "--eval-per-class" "${EVAL_PER_CLASS}"
  "--train-sample-size" "${TRAIN_SAMPLE_SIZE}"
  "--train-sample-mode" "${TRAIN_SAMPLE_MODE}"
  "--population" "${POPULATION}"
  "--generations" "${GENERATIONS}"
  "--mutation-prob" "${MUTATION_PROB}"
  "--mutation-sigma" "${MUTATION_SIGMA}"
  "--jobs" "${JOBS}"
  "--candidate-cap" "${CANDIDATE_CAP}"
  "--gamma" "${GAMMA}"
  "--epsilon" "${EPSILON}"
  "--max-iterations" "${MAX_ITERATIONS}"
  "--objective" "${OBJECTIVE}"
  "--baselines" "${BASELINES}"
  "--policies" "${POLICIES}"
  "--assign-types" "${ASSIGN_TYPES}"
  "--seed" "${SEED}"
)

if [[ -n "${DAGGEN_BINARY}" ]]; then
  PIPELINE_CMD+=("--daggen-binary" "${DAGGEN_BINARY}")
fi
if [[ ${REFREEZE} -eq 1 ]]; then
  PIPELINE_CMD+=("--refreeze")
fi
if [[ ${SKIP_BUILD} -eq 1 ]]; then
  PIPELINE_CMD+=("--skip-build")
fi

{
  echo "run_dir=${RUN_DIR}"
  echo "corpus_dir=${CORPUS_DIR}"
  echo "report_dir=${REPORT_DIR}"
  echo "log_path=${LOG_PATH}"
  echo "hostname=$(hostname)"
  echo "cpu_count=${CPU_COUNT}"
  echo "jobs=${JOBS}"
  echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf 'command='
  printf '%q ' "${PIPELINE_CMD[@]}"
  echo
  echo
} | tee "${LOG_PATH}"

"${PIPELINE_CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"

{
  echo
  echo "end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "summary_csv=${REPORT_DIR}/summary_eval.csv"
  echo "instances_csv=${REPORT_DIR}/instances_eval.csv"
  echo "summary_index=${REPORT_DIR}/summary_index.txt"
} | tee -a "${LOG_PATH}"
