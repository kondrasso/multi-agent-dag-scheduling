"""
Pre-generate and cache training / test datasets as pickle files.

Running this once saves time on repeated training runs by eliminating
daggen subprocess overhead and DAG construction from every training call.

Small-scale corpus layout (saved under --data_dir):
  train_nn_ws{ws}_n{n}.pkl    72 DAGs  (24 classes × 3)   ws=1-3, n=30/60/90
  train_mcts_ws{ws}_n{n}.pkl  144 DAGs (48 classes × 3)   ws=1-3, n=30/60/90
  test_ws{ws}_n{n}.pkl        480 DAGs (48 classes × 10)  ws=1-3, n=30/60/90

Large-scale corpus layout (chapter 3 NN, --large flag):
  train_nn_large_ws{ws}.pkl   3000 DAGs   ws=4-9
  test_large_ws{ws}.pkl       10000 DAGs  ws=4-9

Usage:
  # small-scale only
  uv run python -m dag_scheduling.evaluation.generate_data --data_dir data/

  # large-scale only (slow, large files)
  uv run python -m dag_scheduling.evaluation.generate_data --large --data_dir data/

  # subset
  uv run python -m dag_scheduling.evaluation.generate_data \\
      --ws 1 --n 30 60 --data_dir data/
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

from dag_scheduling.protocol import (
    LARGE_TEST_SEED,
    LARGE_TRAIN_SEED,
    LARGE_WORKSPACES,
    FULL_TOPOLOGIES,
    NN_TOPOLOGIES,
    OFFLINE_DAG_SIZES,
    SMALL_WORKSPACES,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    LARGE_SCALE_N,
    make_large_random_corpus,
    make_mcts_training_corpus,
    make_nn_training_corpus,
    make_test_corpus,
)

_WS_VALUES = list(SMALL_WORKSPACES)
_N_VALUES = list(OFFLINE_DAG_SIZES)
_LARGE_WS = list(LARGE_WORKSPACES)
_NN_TRAIN_COUNT = len(NN_TOPOLOGIES) * TRAIN_PER_CLASS
_MCTS_TRAIN_COUNT = len(FULL_TOPOLOGIES) * TRAIN_PER_CLASS
_TEST_COUNT = len(FULL_TOPOLOGIES) * TEST_PER_CLASS


def _large_corpus(ws: int, count: int, seed_offset: int):
    """
    Random sampling from the thesis parameter ranges — 3000/10000 instances
    are not divisible by 48 topology classes, so large-scale uses random draw.
    """
    return make_large_random_corpus(ws=ws, count=count, seed_offset=seed_offset)


def generate_all(
    ws_values: list[int] = _WS_VALUES,
    n_values: list[int] = _N_VALUES,
    data_dir: str = "data",
    large: bool = False,
    overwrite: bool = False,
):
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ws in ws_values:
        for n in n_values:
            _save(out / f"train_nn_ws{ws}_n{n}.pkl",
                  lambda ws=ws, n=n: make_nn_training_corpus(n=n, ws=ws),
                  f"train NN   ws={ws} n={n} ({_NN_TRAIN_COUNT} DAGs)", overwrite)

            _save(out / f"train_mcts_ws{ws}_n{n}.pkl",
                  lambda ws=ws, n=n: make_mcts_training_corpus(n=n, ws=ws),
                  f"train MCTS ws={ws} n={n} ({_MCTS_TRAIN_COUNT} DAGs)", overwrite)

            _save(out / f"test_ws{ws}_n{n}.pkl",
                  lambda ws=ws, n=n: make_test_corpus(n=n, ws=ws),
                  f"test       ws={ws} n={n} ({_TEST_COUNT} DAGs)", overwrite)

    if large:
        for ws in _LARGE_WS:
            n = LARGE_SCALE_N[ws]
            _save(out / f"train_nn_large_ws{ws}.pkl",
                  lambda ws=ws: _large_corpus(ws, 3000, _LARGE_TRAIN_SEED + ws * 10000),
                  f"train NN large ws={ws} n={n} (3000 DAGs)", overwrite)

            _save(out / f"test_large_ws{ws}.pkl",
                  lambda ws=ws: _large_corpus(ws, 10000, _LARGE_TEST_SEED + ws * 10000),
                  f"test  NN large ws={ws} n={n} (10000 DAGs)", overwrite)


def _save(path: Path, builder, label: str, overwrite: bool):
    if path.exists() and not overwrite:
        print(f"  skip (exists): {path.name}")
        return
    print(f"  generating {label} … ", end="", flush=True)
    data = builder()
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved → {path}")


def load(path: str | Path):
    """Load a previously generated corpus pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ws",        type=int, nargs="+", default=_WS_VALUES)
    p.add_argument("--n",         type=int, nargs="+", default=_N_VALUES)
    p.add_argument("--data_dir",  type=str, default="data")
    p.add_argument("--large",     action="store_true",
                   help="also generate large-scale NN corpora (WS4-9, slow)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    generate_all(
        ws_values=args.ws,
        n_values=args.n,
        data_dir=args.data_dir,
        large=args.large,
        overwrite=args.overwrite,
    )
