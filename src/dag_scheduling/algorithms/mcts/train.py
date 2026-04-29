"""
MCTS training — chapter 2.

Same neural network and GA as NN (chapter 3), but:
  - GA: 150 generations (not 1000)
  - Training corpus: fat={0.2, 0.5} → 48 topology classes × 3 = 144 DAGs per n
  - Fitness evaluated by running full MCTS schedule (not plain NN schedule)

MCTS hyperparameters (k, I, h) used during GA fitness evaluation should
match inference time. Defaults are conservative to keep training fast;
increase for better quality at cost of training time.

Usage:
  uv run python -m dag_scheduling.algorithms.mcts.train \
      --ws 1 --n 30 --gens 150 --out results/mcts_ws1_n30.npy
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
from dag_scheduling.algorithms.mcts.search import mcts_select, _ready_candidates
from dag_scheduling.protocol import TRAIN_PER_CLASS, make_mcts_training_corpus

N_TRAIN_PER_CLASS = TRAIN_PER_CLASS   # 48 x 3 = 144


def make_corpus(n: int, ws: int, n_per_class: int, seed_offset: int = 0):
    return make_mcts_training_corpus(
        n=n,
        ws=ws,
        n_per_class=n_per_class,
        seed_offset=seed_offset,
    )


def mcts_schedule(dag, platform: Platform, weights: np.ndarray,
                  k: int = 5, n_iter: int = 20, h: int = 5) -> float:
    """Run one MCTS-driven schedule and return makespan."""
    M = normalise(compute_metrics(dag, platform)).astype(np.float32)
    state = ScheduleState(dag, platform)
    while not state.is_done():
        ready = list(state.ready)
        if not ready:
            break
        # top-k candidates by neural score
        candidates = _ready_candidates(state, M, weights, k)
        if not candidates:
            break
        # MCTS picks the best task
        task = mcts_select(state, candidates, M, weights, n_iter, h, k)
        state.schedule_task(task)
    return state.makespan


def make_fitness_fn(corpus, k: int, n_iter: int, h: int):
    def fitness(weights: np.ndarray) -> float:
        makespans = [
            mcts_schedule(dag, platform, weights, k=k, n_iter=n_iter, h=h)
            for dag, platform in corpus
        ]
        return -float(np.mean(makespans))
    return fitness


def train(
    ws: int = 1,
    n: int = 30,
    pop_size: int = 150,
    n_gens: int = 150,
    k: int = 5,
    n_iter: int = 20,
    h: int = 5,
    seed: int = 0,
    out: str = "results/mcts.npy",
    corpus=None,   # pre-loaded list of (dag, platform); generated if None
    verbose: bool = True,
) -> np.ndarray:
    if corpus is None:
        corpus = make_corpus(n, ws, N_TRAIN_PER_CLASS)
    print(f"corpus: {len(corpus)} DAGs  |  chromosome_len={CHROMOSOME_LEN}")
    print(f"MCTS params: k={k}, I={n_iter}, h={h}")

    fitness_fn = make_fitness_fn(corpus, k=k, n_iter=n_iter, h=h)

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
    p.add_argument("--ws",     type=int, default=1)
    p.add_argument("--n",      type=int, default=30)
    p.add_argument("--gens",   type=int, default=150)
    p.add_argument("--pop",    type=int, default=150)
    p.add_argument("--k",      type=int, default=5,  help="candidate cap")
    p.add_argument("--n_iter", type=int, default=20, help="MCTS iterations per step")
    p.add_argument("--h",      type=int, default=5,  help="rollout horizon")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--out",    type=str, default="results/mcts.npy")
    args = p.parse_args()
    train(ws=args.ws, n=args.n, pop_size=args.pop, n_gens=args.gens,
          k=args.k, n_iter=args.n_iter, h=args.h, seed=args.seed, out=args.out)
