# Heterogeneous DAG Scheduling Experiments

This repository contains reproducible implementations for the dissertation
experiments on heterogeneous DAG scheduling with type compatibility and
communication delays.

## Layout

- `src/dag_scheduling/` - Python offline schedulers, environments, evaluation, and MILP reference model.
- `src/dag_scheduling/protocol.py` - shared thesis topology grids, seed offsets, and corpus builders.
- `cpp/mlvp/` - C++ online MLVP simulator and chapter-4 thesis-scale pipeline.
- `daggen/` - DAGGEN submodule used for synthetic workflow generation.
- `scripts/` - machine/preflight helpers.
- `tests/` - Python smoke and regression tests.

The local thesis source folder is intentionally ignored by Git and is used only
as a reference for matching the protocol.

## Core Protocols

Offline chapters evaluate DAG sizes `30, 60, 90` on workspaces `1, 2, 3`.
The MLVP chapter evaluates the online sliding-window protocol on:

- workspaces `1, 2, 3, 4`
- executor counts `3, 6, 12, 24`
- DAG sizes `3000, 6000, 12000, 24000`
- `48` topology classes
- `3` training and `10` evaluation instances per class
- policies `mlvp, donf, fifo, minmin, maxmin`

## Setup

```bash
uv sync --extra gurobi
git submodule update --init --recursive
git -C daggen apply ../patches/daggen-linkage-fix.patch
make -C daggen
```

The DAGGEN patch fixes modern C linkage and adds `--seed` so frozen corpora are
reproducible.

## Smoke Tests

```bash
uv run python -m unittest discover -s tests
make -C cpp/mlvp all test
./scripts/wsl_preflight.sh
```

## MLVP Thesis Run

```bash
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

The run writes aggregate results to `reports/summary_eval.csv` and per-instance
results for box plots to `reports/instances_eval.csv`.
