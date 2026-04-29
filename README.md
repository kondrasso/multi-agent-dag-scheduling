# Multi-Agent DAG Scheduling

Research code for heterogeneous DAG scheduling experiments. The repository
contains Python implementations for offline scheduling experiments and a C++
simulator for online MLVP experiments.

The code models typed tasks and typed executors (`CPU`, `GPU`, `IO`),
communication delays, DAGGEN-generated workloads, heuristic baselines,
learned schedulers, and MILP reference solves.

## Contents

- `src/dag_scheduling/protocol.py` - shared experiment protocol: topology
  grids, workspace IDs, DAG sizes, seed offsets, and corpus builders.
- `src/dag_scheduling/baselines/` - DONF, CPOP, HCPT, HPS, and PETS.
- `src/dag_scheduling/algorithms/` - MARL, MCTS, NN hyper-heuristic, and GA.
- `src/dag_scheduling/milp/` - Pyomo MILP model and HiGHS/Gurobi solve wrappers.
- `src/dag_scheduling/evaluation/` - benchmark, training, data, and MILP-reference
  entrypoints.
- `cpp/mlvp/` - C++ MLVP simulator, tuning tools, and full-scale pipeline.
- `tests/` - Python smoke and regression tests.

Generated outputs such as `results/`, `cpp/mlvp/runs/`, build directories,
virtual environments, caches, and local manuscript/reference drops are ignored
by Git.

## Setup

Use Python 3.12 and `uv`.

```bash
git submodule update --init --recursive
uv sync
```

For optional Gurobi support:

```bash
uv sync --extra gurobi
```

Build DAGGEN:

```bash
make daggen
```

## Test

Run the normal smoke suite:

```bash
make test
```

Run the additional MLVP benchmark smoke:

```bash
make smoke
```

Run lightweight NN, MCTS, and MARL training smokes:

```bash
make smoke-training
```

Check a Linux/WSL machine for solver, CUDA, and build readiness:

```bash
make preflight
```

## Offline Python Experiments

Evaluate the small-scale offline protocol:

```bash
uv run python -m dag_scheduling.evaluation.benchmark \
  --ws 1 2 3 \
  --n 30 60 90 \
  --summary_csv results/offline_summary.csv
```

Add a MILP reference prefix with HiGHS:

```bash
uv run python -m dag_scheduling.evaluation.benchmark \
  --ws 1 \
  --n 30 \
  --milp_solver highs \
  --milp_limit 1 \
  --milp_time_limit 20 \
  --milp_gap 0.03 \
  --summary_csv results/offline_summary.csv \
  --milp_summary_csv results/offline_milp_summary.csv
```

Train NN and MCTS hyper-heuristics:

```bash
uv run python -m dag_scheduling.evaluation.train_all \
  --alg both \
  --ws 1 2 3 \
  --n 30 60 90 \
  --out_dir results
```

Train MARL with RLlib:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m dag_scheduling.algorithms.marl.train \
  --ws 1 \
  --n 30 \
  --iters 350000 \
  --gpus 1 \
  --out results/marl_ws1_n30
```

## MLVP C++ Experiments

Build MLVP tools:

```bash
make mlvp
```

Run a quick generated benchmark:

```bash
./cpp/mlvp/build/bin/mlvp_benchmark \
  --generate 2 \
  --n 20 \
  --workspace 1 \
  --assign-types alpha \
  --policies mlvp,donf,fifo,minmin,maxmin
```

Run the full-scale MLVP pipeline:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

The pipeline writes:

- `reports/summary_eval.csv` - aggregate policy means.
- `reports/instances_eval.csv` - per-instance rows for plots and diagnostics.
- `reports/ws*/weights.txt` - tuned MLVP weights per workspace.
- `reports/ws*/history.csv` - tuning history per workspace.

See [cpp/mlvp/MLVP_FULL_RUN.md](cpp/mlvp/MLVP_FULL_RUN.md) for the longer
MLVP runbook.

## Cleaning

Remove Python caches and C++ build outputs:

```bash
make clean
```

Remove generated result directories and local run artifacts as well:

```bash
make distclean
```

`make distclean` intentionally does not remove `.venv/` or ignored local
manuscript/reference folders. Delete those manually only when you are sure you
no longer need them.

## Notes for Public Release

- This repository contains the implementation code for algorithms used in thesis
  experiments on heterogeneous DAG scheduling for "Heterogeneous Computational
  DAG Scheduling: Adaptive Learning and Search under Compatibility and
  Communication Constraints".
- Manuscript files are not part of this repository.

## License

The original code in this repository is released under the Zero-Clause BSD
License. Third-party dependencies and submodules retain their own licenses.
