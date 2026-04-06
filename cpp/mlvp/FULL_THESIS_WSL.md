# Full Thesis Run in WSL

This runbook is for the chapter-4 MLVP experiment in WSL with the full thesis-scale split and all online baseline algorithms in the final evaluation.

## What the wrapper runs

Script: [run_full_thesis_wsl.sh](/Users/kondrasso/Projects/phd_code/cpp/mlvp/scripts/run_full_thesis_wsl.sh)

By default it launches:

- Workspaces `1,2,3,4`
- Workspace sizes:
  - `WS1`: `p=3`, `n=3000`
  - `WS2`: `p=6`, `n=6000`
  - `WS3`: `p=12`, `n=12000`
  - `WS4`: `p=24`, `n=24000`
- `48` DAG topology classes per workspace
- `3` train instances per topology class
- `10` eval instances per topology class
- Full-train tuning on the saved train split
- Final eval policies:
  - `mlvp`
  - `donf`
  - `fifo`
  - `minmin`
  - `maxmin`

The split counts above come from [chapter_4.tex](/Users/kondrasso/Projects/phd_code/phd_thesis_source/chapter_4.tex:434) and [chapter_4.tex](/Users/kondrasso/Projects/phd_code/phd_thesis_source/chapter_4.tex:493).

The wrapper calls [run_long_training_wsl.sh](/Users/kondrasso/Projects/phd_code/cpp/mlvp/scripts/run_long_training_wsl.sh), which calls [run_chapter4_pipeline.sh](/Users/kondrasso/Projects/phd_code/cpp/mlvp/scripts/run_chapter4_pipeline.sh).

## Default tuning settings

These are the current wrapper defaults:

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

`best-baseline-ratio` means MLVP is tuned against the strongest enabled baseline on each training instance, not only against DONF.

## WSL prerequisites

- Use a Linux filesystem path for the repo if possible. Prefer `/home/<user>/...` over `/mnt/c/...`.
- Install basic build tools:

```bash
sudo apt update
sudo apt install -y build-essential make g++
```

- Make the launchers executable:

```bash
cd /path/to/phd_code
chmod +x cpp/mlvp/scripts/run_chapter4_pipeline.sh
chmod +x cpp/mlvp/scripts/run_long_training_wsl.sh
chmod +x cpp/mlvp/scripts/run_full_thesis_wsl.sh
```

## Launch commands

### Standard attached run

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh
```

### Heavier run for a beefier machine

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --jobs 24 \
  --population 48 \
  --generations 60
```

### Custom output location and run name

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --root /data/mlvp_runs \
  --run-name thesis_full_2026_04 \
  --jobs 24
```

### Force corpus regeneration

Use this only if you want to discard previously frozen train/eval corpora under the selected run directory.

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --root /data/mlvp_runs \
  --run-name thesis_full_refreeze \
  --refreeze
```

### Reuse existing binaries

```bash
cd /path/to/phd_code
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --skip-build \
  --jobs 24
```

## Detached launch

### `nohup`

```bash
cd /path/to/phd_code
nohup ./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --root /data/mlvp_runs \
  --run-name thesis_full_nohup \
  --jobs 24 \
  > /tmp/mlvp_full_thesis.nohup 2>&1 &
```

### `tmux`

```bash
cd /path/to/phd_code
tmux new -s mlvp_full
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --root /data/mlvp_runs \
  --run-name thesis_full_tmux \
  --jobs 24
```

Detach from `tmux` with `Ctrl-b d`.

## Monitoring

Each run creates a timestamped directory under `cpp/mlvp/runs/` unless you override `--root` or `--run-name`.

Typical files:

- `corpus/`
- `reports/`
- `pipeline.log`
- `reports/summary_eval.csv`
- `reports/summary_index.txt`
- `reports/ws*/weights.txt`
- `reports/ws*/history.csv`
- `reports/ws*/eval.csv`
- `reports/ws*/eval.json`

Follow the live pipeline log:

```bash
tail -f cpp/mlvp/runs/<run_name>/pipeline.log
```

Inspect the combined summary:

```bash
cat cpp/mlvp/runs/<run_name>/reports/summary_eval.csv
```

Pretty-print the combined summary if `column` is available:

```bash
column -s, -t < cpp/mlvp/runs/<run_name>/reports/summary_eval.csv
```

Inspect one workspace history file:

```bash
cat cpp/mlvp/runs/<run_name>/reports/ws1/history.csv
```

## Notes

- `WS4` is the expensive case. Expect that workspace to dominate wall-clock time.
- The wrapper evaluates all online baselines in the final matrix even though only MLVP is tuned.
- If you want to change the training objective, pass an override at the end. Example:

```bash
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --objective makespan
```

- If you want to tune only against DONF instead of the best enabled baseline:

```bash
./cpp/mlvp/scripts/run_full_thesis_wsl.sh \
  --objective best-baseline-ratio \
  --baselines donf
```
