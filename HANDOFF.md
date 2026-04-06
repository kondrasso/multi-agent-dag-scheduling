# Handoff

This file is a compact state summary for continuing work on another machine.

## Current state

- `MLVP` is implemented in `C++` under `cpp/mlvp`
- offline Python side includes:
  - heuristic baselines
  - MARL
  - MCTS
  - NN hyper-heuristic
  - MILP reference model

## MLVP

Main path:

- [cpp/mlvp/scripts/run_full_thesis_wsl.sh](/Users/kondrasso/Projects/phd_code/cpp/mlvp/scripts/run_full_thesis_wsl.sh#L1)
- [cpp/mlvp/FULL_THESIS_WSL.md](/Users/kondrasso/Projects/phd_code/cpp/mlvp/FULL_THESIS_WSL.md#L1)

Notes:

- chapter-4 full run wrapper exists
- thesis-scale split defaults are wired in
- final eval includes:
  - `mlvp`
  - `donf`
  - `fifo`
  - `minmin`
  - `maxmin`

## MILP

Shared formulation:

- [src/dag_scheduling/milp/model.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/milp/model.py#L13)
- [src/dag_scheduling/milp/solve.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/milp/solve.py#L15)

Backends:

- `HiGHS` via `pyomo + highspy`
- `Gurobi` via `pyomo + gurobipy`

Single-instance CLI:

- [src/dag_scheduling/milp/cli.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/milp/cli.py#L29)

Reference evaluation runner:

- [src/dag_scheduling/evaluation/milp_reference.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/evaluation/milp_reference.py#L95)

Tests:

- [tests/test_milp.py](/Users/kondrasso/Projects/phd_code/tests/test_milp.py#L35)

## Benchmark integration

Existing offline benchmark now supports optional MILP-reference runs and CSV export:

- [src/dag_scheduling/evaluation/benchmark.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/evaluation/benchmark.py#L76)
- [src/dag_scheduling/evaluation/benchmark.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/evaluation/benchmark.py#L327)

Important flags:

- `--milp_solver highs|gurobi`
- `--milp_limit N`
- `--milp_time_limit`
- `--milp_gap`
- `--summary_csv`
- `--milp_summary_csv`

## WSL / machine prep

Preflight:

- [scripts/wsl_preflight.sh](/Users/kondrasso/Projects/phd_code/scripts/wsl_preflight.sh#L1)

Runbook:

- [WSL_RUNS.md](/Users/kondrasso/Projects/phd_code/WSL_RUNS.md#L1)

Practical split:

- GPU-usable:
  - `MARL`
- CPU-bound:
  - `MILP`
  - `NN`
  - `MCTS`
  - heuristics
  - `MLVP`

## Start here on the other machine

```bash
cd /path/to/phd_code
uv sync --extra gurobi
chmod +x scripts/wsl_preflight.sh
./scripts/wsl_preflight.sh
```

Then decide:

- CPU-heavy benchmark/MILP runs
- GPU MARL runs
- full C++ MLVP thesis run

## Useful commands

MILP benchmark with CSV export:

```bash
uv run python -m dag_scheduling.evaluation.benchmark \
  --ws 1 2 3 \
  --n 30 60 90 \
  --milp_solver highs \
  --milp_limit 5 \
  --milp_time_limit 900 \
  --milp_gap 0.03 \
  --summary_csv results/offline_summary.csv \
  --milp_summary_csv results/offline_milp_summary.csv
```

GPU MARL:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m dag_scheduling.algorithms.marl.train \
  --ws 1 \
  --n 30 \
  --iters 350000 \
  --gpus 1 \
  --out results/marl_ws1_n30
```

Full MLVP:

```bash
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

## What to paste into the next chat

- output of `./scripts/wsl_preflight.sh`
- machine specs
- whether Gurobi is full/academic or restricted
- what you want to run first
