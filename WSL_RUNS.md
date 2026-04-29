# WSL Runs

This note separates GPU-usable runs from CPU-bound runs for this repo.

## What uses the NVIDIA GPU

Current state of this codebase:

- `MARL`: GPU-capable
  - Uses Torch + RLlib in [marl/train.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/algorithms/marl/train.py:70)
- `MILP`: CPU-bound for practical purposes here
  - `HiGHS` backend is CPU
  - `Gurobi` may use GPU only for parts of LP/root work, not as a general MIP accelerator
- `NN` hyper-heuristic: CPU
  - NumPy + GA in [nn/train.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/algorithms/nn/train.py:79)
- `MCTS`: CPU
  - NumPy + search/rollouts in [mcts/train.py](/Users/kondrasso/Projects/phd_code/src/dag_scheduling/algorithms/mcts/train.py:71)
- `Heuristic baselines`: CPU
- `MLVP` C++: CPU

So for a WSL box with a `3060`:

- use the GPU for `MARL`
- use CPU cores and RAM for `MILP`, `NN`, `MCTS`, heuristics, and `MLVP`

## Preflight

Script: [wsl_preflight.sh](/Users/kondrasso/Projects/phd_code/scripts/wsl_preflight.sh)

Run:

```bash
cd /path/to/phd_code
chmod +x scripts/wsl_preflight.sh
./scripts/wsl_preflight.sh
```

It reports:

- WSL/basic host info
- CPU count and memory
- C/C++ build tool availability
- local `daggen` source/binary availability
- visible NVIDIA GPU from WSL
- Torch CUDA visibility
- `HiGHS` and `Gurobi` availability
- tiny MILP smoke solves on both backends

## Environment setup

Recommended:

- keep the repo on the Linux filesystem, not under `/mnt/c/...`
- use `uv` inside WSL

Setup:

```bash
cd /path/to/phd_code
uv sync --extra gurobi
```

If the machine is only for open-source MILP runs, plain `uv sync` is enough.

## GPU run: MARL

Single run:

```bash
cd /path/to/phd_code
CUDA_VISIBLE_DEVICES=0 uv run python -m dag_scheduling.algorithms.marl.train \
  --ws 1 \
  --n 30 \
  --iters 350000 \
  --gpus 1 \
  --out results/marl_ws1_n30
```

Repeat across the thesis grid:

```bash
cd /path/to/phd_code
for ws in 1 2 3; do
  for n in 30 60 90; do
    CUDA_VISIBLE_DEVICES=0 uv run python -m dag_scheduling.algorithms.marl.train \
      --ws "${ws}" \
      --n "${n}" \
      --iters 350000 \
      --gpus 1 \
      --out "results/marl_ws${ws}_n${n}"
  done
done
```

## CPU run: MILP benchmark with CSV export

Open-source solver:

```bash
cd /path/to/phd_code
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

Gurobi:

```bash
cd /path/to/phd_code
uv run python -m dag_scheduling.evaluation.benchmark \
  --ws 1 2 3 \
  --n 30 60 90 \
  --milp_solver gurobi \
  --milp_limit 5 \
  --milp_time_limit 900 \
  --milp_gap 0.03 \
  --summary_csv results/offline_summary.csv \
  --milp_summary_csv results/offline_milp_summary.csv
```

Important:

- `--milp_limit` is the tractable prefix per cell, not the whole 480-instance cell
- on a restricted Gurobi license, larger cells may still fail

## CPU run: NN and MCTS

NN only:

```bash
cd /path/to/phd_code
uv run python -m dag_scheduling.evaluation.train_all \
  --alg nn \
  --ws 1 2 3 \
  --n 30 60 90 \
  --out_dir results/nn
```

MCTS only:

```bash
cd /path/to/phd_code
uv run python -m dag_scheduling.evaluation.train_all \
  --alg mcts \
  --ws 1 2 3 \
  --n 30 60 90 \
  --out_dir results/mcts
```

## CPU run: MLVP

Chapter-4 run:

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

## Detached runs

Example for a CPU-heavy benchmark:

```bash
cd /path/to/phd_code
nohup uv run python -m dag_scheduling.evaluation.benchmark \
  --ws 1 2 3 \
  --n 30 60 90 \
  --milp_solver highs \
  --milp_limit 5 \
  --milp_time_limit 900 \
  --milp_gap 0.03 \
  --summary_csv results/offline_summary.csv \
  --milp_summary_csv results/offline_milp_summary.csv \
  > /tmp/offline_benchmark.nohup 2>&1 &
```

Example for GPU MARL:

```bash
cd /path/to/phd_code
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 uv run python -m dag_scheduling.algorithms.marl.train \
  --ws 1 --n 30 --iters 350000 --gpus 1 --out results/marl_ws1_n30' \
  > /tmp/marl_ws1_n30.nohup 2>&1 &
```
