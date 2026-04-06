"""
Train NN and MCTS hyper-heuristics for all (ws, n) combinations.

Thesis protocol:
  NN:   n ∈ {30, 60, 90},  ws ∈ {1, 2, 3},  1000 GA generations, pop=150
  MCTS: n ∈ {30, 60, 90},  ws ∈ {1, 2, 3},   150 GA generations, pop=150

Weights + training history saved to --out_dir:
  nn_ws{ws}_n{n}.npy
  nn_ws{ws}_n{n}_history.npy
  mcts_ws{ws}_n{n}.npy
  mcts_ws{ws}_n{n}_history.npy

If --data_dir is set, pre-generated corpora are loaded from there
(see generate_data.py). Otherwise, corpora are generated on the fly.

Usage:
  # train everything
  uv run python -m dag_scheduling.evaluation.train_all --out_dir results/

  # with pre-generated data (faster, reproducible)
  uv run python -m dag_scheduling.evaluation.train_all \\
      --data_dir data/ --out_dir results/

  # single cell
  uv run python -m dag_scheduling.evaluation.train_all \\
      --alg nn --ws 1 --n 30 --out_dir results/
"""

from __future__ import annotations
import argparse
from pathlib import Path

from dag_scheduling.algorithms.nn.train   import train as train_nn
from dag_scheduling.algorithms.mcts.train import train as train_mcts

_WS_VALUES = [1, 2, 3]
_N_VALUES  = [30, 60, 90]


def _load_corpus(data_dir: Path | None, filename: str):
    """Return pre-loaded corpus or None (train script generates it on the fly)."""
    if data_dir is None:
        return None
    path = data_dir / filename
    if not path.exists():
        print(f"  [warn] corpus not found at {path}, will generate on the fly")
        return None
    from dag_scheduling.evaluation.generate_data import load
    print(f"  loading corpus from {path}")
    return load(path)


def train_all(
    alg: str = "both",
    ws_values: list[int] = _WS_VALUES,
    n_values: list[int] = _N_VALUES,
    out_dir: str = "results",
    data_dir: str | None = None,
    seed: int = 0,
    verbose: bool = True,
):
    out = Path(out_dir)
    data = Path(data_dir) if data_dir else None

    for ws in ws_values:
        for n in n_values:
            if alg in ("nn", "both"):
                corpus = _load_corpus(data, f"train_nn_ws{ws}_n{n}.pkl")
                weights_path = str(out / f"nn_ws{ws}_n{n}.npy")
                print(f"\n{'='*60}")
                print(f"Training NN   ws={ws}  n={n}  → {weights_path}")
                print(f"{'='*60}")
                train_nn(ws=ws, n=n, pop_size=150, n_gens=1000,
                         seed=seed, out=weights_path,
                         corpus=corpus, verbose=verbose)

            if alg in ("mcts", "both"):
                corpus = _load_corpus(data, f"train_mcts_ws{ws}_n{n}.pkl")
                weights_path = str(out / f"mcts_ws{ws}_n{n}.npy")
                print(f"\n{'='*60}")
                print(f"Training MCTS ws={ws}  n={n}  → {weights_path}")
                print(f"{'='*60}")
                train_mcts(ws=ws, n=n, pop_size=150, n_gens=150,
                           seed=seed, out=weights_path,
                           corpus=corpus, verbose=verbose)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alg",      choices=["nn", "mcts", "both"], default="both")
    p.add_argument("--ws",       type=int, nargs="+", default=_WS_VALUES)
    p.add_argument("--n",        type=int, nargs="+", default=_N_VALUES)
    p.add_argument("--out_dir",  type=str, default="results")
    p.add_argument("--data_dir", type=str, default=None,
                   help="directory with pre-generated .pkl corpora (generate_data.py)")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--quiet",    action="store_true")
    args = p.parse_args()

    train_all(
        alg=args.alg,
        ws_values=args.ws,
        n_values=args.n,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )
