"""
MCTS-guided task selection — chapter 2, Algorithm 1.

MCTSSelect(state, candidates, scorer, I, h):
  Runs I iterations of Selection → Expansion → Simulation → Backprop.
  Returns the task from candidates with the most visits (best child).

Tree nodes hold shallow copies of ScheduleState so the DAG graph
object itself is never copied (it's read-only).

UCB1 exploration constant C = sqrt(2) (standard MCTS default).
"""

from __future__ import annotations
import math
import numpy as np

from dag_scheduling.core.simulator import ScheduleState
from dag_scheduling.algorithms.nn.model import score_tasks

C_EXPLORE = math.sqrt(2)


class _Node:
    __slots__ = ("state", "parent", "task_idx",
                 "children", "untried", "Q", "N")

    def __init__(self, state: ScheduleState,
                 parent: "_Node | None",
                 task_idx: int | None,
                 candidates: list[int]):
        self.state = state
        self.parent = parent
        self.task_idx = task_idx        # task that led to this node
        self.children: dict[int, _Node] = {}
        self.untried: list[int] = list(candidates)  # tasks not yet expanded
        self.Q: float = 0.0
        self.N: int = 0

    def ucb1(self, parent_n: int) -> float:
        if self.N == 0:
            return math.inf
        return self.Q / self.N + C_EXPLORE * math.sqrt(math.log(parent_n) / self.N)

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def best_child(self) -> "_Node":
        return max(self.children.values(), key=lambda c: c.ucb1(self.N))

    def most_visited_child(self) -> "_Node":
        return max(self.children.values(), key=lambda c: c.N)


def _ready_candidates(state: ScheduleState, metrics: np.ndarray,
                      weights: np.ndarray, k: int) -> list[int]:
    """Top-k ready tasks by neural score."""
    ready = list(state.ready)
    if not ready:
        return []
    if len(ready) <= k:
        return ready
    scores = score_tasks(ready, metrics, weights)
    order = np.argsort(scores)[::-1]
    return [ready[i] for i in order[:k]]


def _simulate(node: "_Node", metrics: np.ndarray,
              weights: np.ndarray, h: int, k: int) -> float:
    """
    Neural-guided rollout up to depth h from node's state.
    Returns -makespan of the partial/completed schedule.
    """
    state = node.state.shallow_copy()
    depth = 0
    while not state.is_done() and depth < h:
        candidates = _ready_candidates(state, metrics, weights, k)
        if not candidates:
            break
        scores = score_tasks(candidates, metrics, weights)
        task = candidates[int(np.argmax(scores))]
        state.schedule_task(task)
        depth += 1

    # complete remainder greedily (NN priority) if not done
    if not state.is_done():
        def prio(t, _state):
            s = score_tasks([t], metrics, weights)
            return float(s[0])
        state.greedy_rollout(priority_fn=prio)

    return -state.makespan


def mcts_select(
    state: ScheduleState,
    candidates: list[int],
    metrics: np.ndarray,
    weights: np.ndarray,
    n_iter: int,
    h: int,
    k: int,
) -> int:
    """
    Run MCTS and return the selected task index.

    state:      current (committed) ScheduleState — not modified
    candidates: top-k tasks from the ready set
    metrics:    normalised (max_idx+1, 13) metric array
    weights:    NN chromosome (95,)
    n_iter:     search budget I
    h:          rollout horizon
    k:          candidate cap for deeper nodes
    """
    if len(candidates) == 1:
        return candidates[0]

    root = _Node(state.shallow_copy(), parent=None,
                 task_idx=None, candidates=list(candidates))

    for _ in range(n_iter):
        # --- Selection ---
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # --- Expansion ---
        if node.untried and not node.state.is_done():
            task = node.untried.pop()
            child_state = node.state.shallow_copy()
            child_state.schedule_task(task)
            child_cands = _ready_candidates(child_state, metrics, weights, k)
            child = _Node(child_state, parent=node,
                          task_idx=task, candidates=child_cands)
            node.children[task] = child
            node = child

        # --- Simulation ---
        value = _simulate(node, metrics, weights, h, k)

        # --- Backpropagation ---
        while node is not None:
            node.N += 1
            node.Q += value
            node = node.parent

    # pick child of root with most visits
    if not root.children:
        return candidates[0]
    best = root.most_visited_child()
    return best.task_idx
