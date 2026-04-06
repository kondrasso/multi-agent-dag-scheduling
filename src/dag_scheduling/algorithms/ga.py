"""
Generic real-valued Genetic Algorithm used by NN, MCTS, and MLVP.

Variation operators (chapter 2/3 of the thesis):
  - Two-Point Crossover (TPC)
  - Simulated Binary Crossover (SBX)
  Crossover operator is selected at random per mating event.
  Mutation: additive Gaussian perturbation, applied per-offspring with
  probability p_m (default 0.8).

Selection: tournament (size 3) for parent selection.
Elitism: best individual is always kept.

fitness_fn(chromosome: np.ndarray) -> float
  Higher = better. Caller wraps makespan as -makespan.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Callable

FitnessFn = Callable[[np.ndarray], float]


def _tpc(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-Point Crossover."""
    n = len(p1)
    pts = sorted(rng.integers(0, n, size=2))
    child = p1.copy()
    child[pts[0]:pts[1]] = p2[pts[0]:pts[1]]
    return child


def _sbx(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator,
         eta: float = 2.0) -> np.ndarray:
    """Simulated Binary Crossover (SBX)."""
    u = rng.random(len(p1))
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1)),
    )
    return 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)


def _crossover(p1: np.ndarray, p2: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Randomly select TPC or SBX."""
    if rng.random() < 0.5:
        return _tpc(p1, p2, rng)
    return _sbx(p1, p2, rng)


def _mutate(child: np.ndarray, rng: np.random.Generator,
            sigma: float = 0.1) -> np.ndarray:
    """Gaussian additive mutation."""
    return child + rng.normal(0.0, sigma, size=child.shape)


def _tournament(fitnesses: np.ndarray, rng: np.random.Generator,
                k: int = 3) -> int:
    """Tournament selection, returns index of winner."""
    contestants = rng.integers(0, len(fitnesses), size=k)
    return int(contestants[np.argmax(fitnesses[contestants])])


def run_ga(
    fitness_fn: FitnessFn,
    chromosome_len: int,
    pop_size: int = 150,
    n_generations: int = 1000,
    mutation_prob: float = 0.8,
    mutation_sigma: float = 0.1,
    seed: int | None = None,
    verbose: bool = False,
    log_interval: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """
    Run the genetic algorithm.

    Returns (best_chromosome, history) where history[g] = best fitness at gen g.
    """
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()

    # initialise population in U[-1, 1] (chapter 2/3 spec)
    population = rng.uniform(-1.0, 1.0, size=(pop_size, chromosome_len))
    fitnesses = np.array([fitness_fn(ind) for ind in population])

    best_idx = int(np.argmax(fitnesses))
    best_chr = population[best_idx].copy()
    best_fit = float(fitnesses[best_idx])
    history: list[float] = [best_fit]

    if verbose:
        print(f"gen   0/{n_generations}  best={best_fit:.4f}"
              f"  mean={fitnesses.mean():.4f}  elapsed=0s")

    for gen in range(1, n_generations + 1):
        new_pop = np.empty_like(population)
        new_pop[0] = best_chr  # elitism: keep best

        for i in range(1, pop_size):
            p1 = population[_tournament(fitnesses, rng)]
            p2 = population[_tournament(fitnesses, rng)]
            child = _crossover(p1, p2, rng)
            if rng.random() < mutation_prob:
                child = _mutate(child, rng, sigma=mutation_sigma)
            new_pop[i] = child

        population = new_pop
        fitnesses = np.array([fitness_fn(ind) for ind in population])

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = float(fitnesses[gen_best_idx])
            best_chr = population[gen_best_idx].copy()

        history.append(best_fit)
        if verbose and gen % log_interval == 0:
            elapsed = time.monotonic() - t0
            print(f"gen {gen:>4}/{n_generations}"
                  f"  best={best_fit:.4f}"
                  f"  mean={fitnesses.mean():.4f}"
                  f"  elapsed={elapsed:.0f}s")

    return best_chr, history
