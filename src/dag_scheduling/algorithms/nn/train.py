"""
NN hyper-heuristic training — chapter 3.

GA configuration (chapter 3, §Training):
  population:  150
  generations: 1000
  mutation:    p=0.8, Gaussian sigma=0.1
  crossover:   SBX / TPC selected at random
  init:        U[-1, 1]
  elitism:     yes
  fitness:     -mean(makespan) over training corpus

Training corpus: 3 instances per topology class × 24 classes = 72 DAGs per n.

Usage:
  uv run python -m dag_scheduling.algorithms.nn.train \
      --ws 1 --n 30 --gens 1000 --out results/nn_ws1_n30.npy
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path

from dag_scheduling.core.platform import Platform
from dag_scheduling.core.simulator import ScheduleState
from dag_scheduling.core.metrics import compute_metrics, normalise
from dag_scheduling.algorithms.ga import run_ga
from dag_scheduling.algorithms.nn.model import score_tasks, CHROMOSOME_LEN
from dag_scheduling.protocol import TRAIN_PER_CLASS, make_nn_training_corpus

N_TRAIN_PER_CLASS = TRAIN_PER_CLASS


def make_corpus(n: int, ws: int, n_per_class: int, seed_offset: int = 0):
    return make_nn_training_corpus(
        n=n,
        ws=ws,
        n_per_class=n_per_class,
        seed_offset=seed_offset,
    )


def nn_schedule(dag, platform: Platform, weights: np.ndarray) -> float:
    """Run one NN-driven schedule and return makespan."""
    M = normalise(compute_metrics(dag, platform)).astype(np.float32)
    state = ScheduleState(dag, platform)
    while not state.is_done():
        ready = list(state.ready)
        if not ready:
            break
        scores = score_tasks(ready, M, weights)
        best = ready[int(np.argmax(scores))]
        state.schedule_task(best)
    return state.makespan


def make_fitness_fn(corpus):
    """Return a fitness function (higher = better) for the GA."""
    def fitness(weights: np.ndarray) -> float:
        makespans = [nn_schedule(dag, platform, weights) for dag, platform in corpus]
        return -float(np.mean(makespans))
    return fitness


def train(
    ws: int = 1,
    n: int = 30,
    pop_size: int = 150,
    n_gens: int = 1000,
    seed: int = 0,
    out: str = "results/nn.npy",
    corpus=None,   # pre-loaded list of (dag, platform); generated if None
    verbose: bool = True,
) -> np.ndarray:
    if corpus is None:
        corpus = make_corpus(n, ws, N_TRAIN_PER_CLASS)
    print(f"corpus: {len(corpus)} DAGs  |  chromosome_len={CHROMOSOME_LEN}")

    fitness_fn = make_fitness_fn(corpus)

    best_weights, history = run_ga(
        fitness_fn=fitness_fn,
        chromosome_len=CHROMOSOME_LEN,
        pop_size=pop_size,
        n_generations=n_gens,
        mutation_prob=0.8,
        seed=seed,
        verbose=verbose,
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, best_weights)
    log_path = out.replace(".npy", "_history.npy")
    np.save(log_path, np.array(history))
    print(f"saved weights → {out}  (best fitness={history[-1]:.4f})")
    print(f"saved history → {log_path}")
    return best_weights


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ws",   type=int, default=1)
    p.add_argument("--n",    type=int, default=30)
    p.add_argument("--gens", type=int, default=1000)
    p.add_argument("--pop",  type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out",  type=str, default="results/nn.npy")
    args = p.parse_args()
    train(ws=args.ws, n=args.n, pop_size=args.pop,
          n_gens=args.gens, seed=args.seed, out=args.out)
