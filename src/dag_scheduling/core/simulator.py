"""
ScheduleState: mutable state for one scheduling episode.

Tracks which tasks are scheduled, the ready set, executor availability,
and actual finish times. All algorithms drive the scheduler by calling
commit() after each placement decision.
"""

from __future__ import annotations
import math
from copy import copy

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform
from dag_scheduling.core.eft import eft_place


class ScheduleState:
    def __init__(self, dag: SchedulingDAG, platform: Platform) -> None:
        self.dag = dag
        self.platform = platform

        self.aft: dict[int, float] = {}           # task_idx -> finish time
        self.assigned: dict[int, int] = {}         # task_idx -> executor_id
        self.executor_available: dict[int, float] = {
            e.id: 0.0 for e in platform.executors
        }
        self.scheduled: set[int] = set()
        self.ready: set[int] = set(dag.entry_nodes())

    # ------------------------------------------------------------------

    def commit(self, task_idx: int, executor_id: int,
               start_time: float, finish_time: float) -> None:
        """Record a placement decision and update ready set."""
        self.aft[task_idx] = finish_time
        self.assigned[task_idx] = executor_id
        self.executor_available[executor_id] = finish_time
        self.scheduled.add(task_idx)
        self.ready.discard(task_idx)

        # unlock successors whose all predecessors are now scheduled
        for succ in self.dag.successors(task_idx):
            if succ not in self.scheduled and all(
                p in self.scheduled for p in self.dag.predecessors(succ)
            ):
                self.ready.add(succ)

    def eft_place(self, task_idx: int) -> tuple[int, float, float]:
        """Convenience: run EFT for a task given current state."""
        return eft_place(
            task_idx, self.dag, self.platform,
            self.executor_available, self.aft, self.assigned,
        )

    def schedule_task(self, task_idx: int) -> tuple[int, float, float]:
        """EFT-place and immediately commit."""
        exc_id, start, finish = self.eft_place(task_idx)
        self.commit(task_idx, exc_id, start, finish)
        return exc_id, start, finish



    def is_done(self) -> bool:
        return len(self.scheduled) == len(self.dag)

    @property
    def makespan(self) -> float:
        if not self.aft:
            return 0.0
        return max(self.aft.values())

    def shallow_copy(self) -> "ScheduleState":
        """Copy only mutable state (not the read-only dag/platform refs)."""
        s = ScheduleState.__new__(ScheduleState)
        s.dag = self.dag
        s.platform = self.platform
        s.aft = dict(self.aft)
        s.assigned = dict(self.assigned)
        s.executor_available = dict(self.executor_available)
        s.scheduled = set(self.scheduled)
        s.ready = set(self.ready)
        return s

    # ------------------------------------------------------------------
    # Greedy rollout helper used by MCTS simulation phase
    # ------------------------------------------------------------------

    def greedy_rollout(self, priority_fn=None) -> float:
        """
        Complete the schedule greedily from the current state.
        priority_fn(task_idx, state) -> float  (higher = schedule first)
        Defaults to FIFO (arbitrary) if None.
        Returns makespan.
        """
        state = self.shallow_copy()
        while not state.is_done():
            if not state.ready:
                break  # disconnected graph — shouldn't happen with daggen output
            if priority_fn is not None:
                task = max(state.ready, key=lambda t: priority_fn(t, state))
            else:
                task = next(iter(state.ready))
            state.schedule_task(task)
        return state.makespan
