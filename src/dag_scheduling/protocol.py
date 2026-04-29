"""Shared thesis experiment protocol helpers.

This module owns the topology grids, seed offsets, and corpus construction used
by the Python training and evaluation scripts. Keeping the protocol here avoids
quiet drift between NN, MCTS, MARL, benchmark, and MILP-reference entrypoints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import LARGE_SCALE_N, Platform, make_workspace
from dag_scheduling.data.augmentor import augment_alpha_based, augment_random
from dag_scheduling.data.generator import generate


SMALL_WORKSPACES = (1, 2, 3)
OFFLINE_DAG_SIZES = (30, 60, 90)
LARGE_WORKSPACES = (4, 5, 6, 7, 8, 9)

FAT_VALUES = (0.2, 0.5)
NN_FAT_VALUES = (0.5,)
DENSITY_VALUES = (0.1, 0.4, 0.8)
REGULARITY_VALUES = (0.2, 0.8)
JUMP_VALUES = (2, 4)
CCR_VALUES = (0.2, 0.8)

TRAIN_PER_CLASS = 3
TEST_PER_CLASS = 10

TEST_SEED_OFFSET = 100_000
LARGE_TRAIN_SEED = 200_000
LARGE_TEST_SEED = 500_000

NodeTypeStrategy = Literal["random", "alpha"]
Corpus = list[tuple[SchedulingDAG, Platform]]


@dataclass(frozen=True)
class Topology:
    fat: float
    density: float
    regularity: float
    jump: int
    ccr: float

    @property
    def daggen_ccr(self) -> int:
        return int(self.ccr * 10)


def topology_grid(fat_values: Sequence[float] = FAT_VALUES) -> list[Topology]:
    """Return the deterministic Cartesian product used by DAGGEN corpora."""
    return [
        Topology(
            fat=fat,
            density=density,
            regularity=regularity,
            jump=jump,
            ccr=ccr,
        )
        for fat in fat_values
        for density in DENSITY_VALUES
        for regularity in REGULARITY_VALUES
        for jump in JUMP_VALUES
        for ccr in CCR_VALUES
    ]


NN_TOPOLOGIES = tuple(topology_grid(NN_FAT_VALUES))
FULL_TOPOLOGIES = tuple(topology_grid(FAT_VALUES))


def test_seed_offset(n: int, ws: int) -> int:
    """Seed offset for the standard test corpus for one ``(n, ws)`` cell."""
    return TEST_SEED_OFFSET + n * 1000 + ws * 100


def assign_node_types(
    dag: SchedulingDAG,
    strategy: NodeTypeStrategy = "random",
    seed: int | None = None,
) -> SchedulingDAG:
    if strategy == "random":
        return augment_random(dag, seed=seed)
    if strategy == "alpha":
        return augment_alpha_based(dag)
    raise ValueError(f"unknown node type strategy: {strategy!r}")


def generate_topology_corpus(
    *,
    n: int,
    ws: int,
    n_per_class: int,
    topologies: Sequence[Topology],
    seed_offset: int = 0,
    type_strategy: NodeTypeStrategy = "random",
) -> Corpus:
    """Generate one corpus by walking a topology grid in deterministic order."""
    platform = make_workspace(ws)
    corpus: Corpus = []
    seed = seed_offset
    for topology in topologies:
        for _ in range(n_per_class):
            dag = generate(
                n=n,
                fat=topology.fat,
                regular=topology.regularity,
                density=topology.density,
                jump=topology.jump,
                ccr=topology.daggen_ccr,
                seed=seed,
            )
            assign_node_types(dag, strategy=type_strategy, seed=seed)
            corpus.append((dag, platform))
            seed += 1
    return corpus


def make_nn_training_corpus(
    n: int,
    ws: int,
    n_per_class: int = TRAIN_PER_CLASS,
    seed_offset: int = 0,
    type_strategy: NodeTypeStrategy = "random",
) -> Corpus:
    return generate_topology_corpus(
        n=n,
        ws=ws,
        n_per_class=n_per_class,
        topologies=NN_TOPOLOGIES,
        seed_offset=seed_offset,
        type_strategy=type_strategy,
    )


def make_mcts_training_corpus(
    n: int,
    ws: int,
    n_per_class: int = TRAIN_PER_CLASS,
    seed_offset: int = 0,
    type_strategy: NodeTypeStrategy = "random",
) -> Corpus:
    return generate_topology_corpus(
        n=n,
        ws=ws,
        n_per_class=n_per_class,
        topologies=FULL_TOPOLOGIES,
        seed_offset=seed_offset,
        type_strategy=type_strategy,
    )


def make_test_corpus(
    n: int,
    ws: int,
    n_per_class: int = TEST_PER_CLASS,
    seed_offset: int | None = None,
    type_strategy: NodeTypeStrategy = "random",
) -> Corpus:
    return generate_topology_corpus(
        n=n,
        ws=ws,
        n_per_class=n_per_class,
        topologies=FULL_TOPOLOGIES,
        seed_offset=test_seed_offset(n, ws) if seed_offset is None else seed_offset,
        type_strategy=type_strategy,
    )


def make_large_random_corpus(
    ws: int,
    count: int,
    seed_offset: int,
    type_strategy: NodeTypeStrategy = "random",
) -> Corpus:
    """Generate a large-scale corpus by random draws from the full topology grid."""
    n = LARGE_SCALE_N[ws]
    platform = make_workspace(ws)
    topologies = tuple(FULL_TOPOLOGIES)
    rng = random.Random(seed_offset)
    corpus: Corpus = []
    for i in range(count):
        topology = rng.choice(topologies)
        seed = seed_offset + i
        dag = generate(
            n=n,
            fat=topology.fat,
            regular=topology.regularity,
            density=topology.density,
            jump=topology.jump,
            ccr=topology.daggen_ccr,
            seed=seed,
        )
        assign_node_types(dag, strategy=type_strategy, seed=seed)
        corpus.append((dag, platform))
    return corpus
