# MLVP Full Run

This runbook covers the C++ MLVP full-scale pipeline.

## Pipeline

Launcher:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh
```

The wrapper calls:

- `cpp/mlvp/scripts/run_long_training.sh`
- `cpp/mlvp/scripts/run_mlvp_pipeline.sh`

Default protocol:

- workspaces `1,2,3,4`
- executor counts `3,6,12,24`
- DAG sizes `3000,6000,12000,24000`
- `48` topology classes per workspace
- `3` train instances per topology class
- `10` eval instances per topology class
- type assignment strategy `alpha`
- policies `mlvp,donf,fifo,minmin,maxmin`

The protocol constants are implemented in `cpp/mlvp/src/core.cpp` and shared
through `cpp/mlvp/include/mlvp/core.hpp`.

## Tuning Defaults

- `--full-train`
- `--population 32`
- `--generations 40`
- `--mutation-prob 0.8`
- `--mutation-sigma 0.25`
- `--candidate-cap 8`
- `--gamma 0.2`
- `--epsilon 0.05`
- `--max-iterations 8`
- `--objective best-baseline-ratio`
- `--baselines donf,fifo,minmin,maxmin`
- `--assign-types alpha`
- `--seed 12345`

`best-baseline-ratio` tunes MLVP against the strongest enabled baseline on each
training instance. The seed is passed through platform generation, type
assignment, MLVP sampling, and DAGGEN generation.

## Prerequisites

```bash
uv sync
git submodule update --init --recursive
make daggen
make mlvp
```

## Launch

Standard run:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh
```

Larger tuning run:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

Custom run directory:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --root /data/mlvp_runs \
  --run-name mlvp_full_2026_04 \
  --jobs 24
```

Regenerate frozen corpora:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --root /data/mlvp_runs \
  --run-name mlvp_full_refreeze \
  --refreeze
```

Reuse existing binaries:

```bash
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --skip-build \
  --jobs 24
```

## Detached Run

```bash
nohup ./cpp/mlvp/scripts/run_full_mlvp.sh \
  --root /data/mlvp_runs \
  --run-name mlvp_full_nohup \
  --jobs 24 \
  > /tmp/mlvp_full.nohup 2>&1 &
```

Using `tmux` is usually cleaner for very long runs:

```bash
tmux new -s mlvp_full
./cpp/mlvp/scripts/run_full_mlvp.sh \
  --root /data/mlvp_runs \
  --run-name mlvp_full_tmux \
  --jobs 24
```

Detach with `Ctrl-b d`.

## Outputs

Each run creates a directory under `cpp/mlvp/runs/` unless `--root` or
`--run-name` overrides the location.

Typical files:

- `corpus/`
- `pipeline.log`
- `reports/summary_eval.csv`
- `reports/instances_eval.csv`
- `reports/summary_index.txt`
- `reports/ws*/weights.txt`
- `reports/ws*/history.csv`
- `reports/ws*/eval.csv`
- `reports/ws*/instances.csv`
- `reports/ws*/eval.json`

Follow progress:

```bash
tail -f cpp/mlvp/runs/<run_name>/pipeline.log
```

Inspect aggregate results:

```bash
column -s, -t < cpp/mlvp/runs/<run_name>/reports/summary_eval.csv
```

Inspect per-instance results:

```bash
head cpp/mlvp/runs/<run_name>/reports/instances_eval.csv
```

## Notes

- `WS4` is the expensive case and normally dominates wall-clock time.
- The final matrix evaluates all online baselines even though only MLVP is
  tuned.
- Generated corpora and reports are ignored by Git.
