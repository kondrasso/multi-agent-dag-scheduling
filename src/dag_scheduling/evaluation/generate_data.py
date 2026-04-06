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

from dag_scheduling.algorithms.nn.train   import make_corpus as nn_corpus
from dag_scheduling.algorithms.mcts.train import make_corpus as mcts_corpus
from dag_scheduling.evaluation.benchmark  import make_test_corpus
from dag_scheduling.data.generator import generate
from dag_scheduling.data.augmentor import augment_random
from dag_scheduling.core.platform import make_workspace, LARGE_SCALE_N
import random

_WS_VALUES = [1, 2, 3]
_N_VALUES  = [30, 60, 90]
_LARGE_WS  = [4, 5, 6, 7, 8, 9]

# large-scale seeds well away from small-scale seeds
_LARGE_TRAIN_SEED = 200_000
_LARGE_TEST_SEED  = 500_000

# same parameter ranges as the small-scale topology grid (chapter 3, Table par)
_FAT        = [0.2, 0.5]
_DENSITY    = [0.1, 0.4, 0.8]
_REGULARITY = [0.2, 0.8]
_JUMP       = [2, 4]
_CCR        = [0.2, 0.8]


def _large_corpus(ws: int, count: int, seed_offset: int):
    """
    Random sampling from the thesis parameter ranges — 3000/10000 instances
    are not divisible by 48 topology classes, so large-scale uses random draw.
    """
    n = LARGE_SCALE_N[ws]
    platform = make_workspace(ws)
    corpus = []
    rng = random.Random(seed_offset)
    for i in range(count):
        dag = generate(
            n=n,
            fat=rng.choice(_FAT),
            regular=rng.choice(_REGULARITY),
            density=rng.choice(_DENSITY),
            jump=rng.choice(_JUMP),
            ccr=int(rng.choice(_CCR) * 10),
        )
        augment_random(dag, seed=seed_offset + i)
        corpus.append((dag, platform))
    return corpus


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
                  lambda ws=ws, n=n: nn_corpus(n, ws, 3),
                  f"train NN   ws={ws} n={n} (72 DAGs)", overwrite)

            _save(out / f"train_mcts_ws{ws}_n{n}.pkl",
                  lambda ws=ws, n=n: mcts_corpus(n, ws, 3),
                  f"train MCTS ws={ws} n={n} (144 DAGs)", overwrite)

            _save(out / f"test_ws{ws}_n{n}.pkl",
                  lambda ws=ws, n=n: make_test_corpus(n, ws),
                  f"test       ws={ws} n={n} (480 DAGs)", overwrite)

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
