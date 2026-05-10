"""
Offline DAG scheduling baselines.

All five heuristics follow the same schedule-construction loop:
  1. prioritise ready tasks using a method-specific rule
  2. select the top task
  3. assign to a compatible executor using the method-specific placement rule
  4. commit and update ready set

HPS and PETS use insertion-aware EFT placement: if an already-built executor
timeline has an idle gap large enough for the selected task after its data-ready
time, the task can be inserted into that gap.

Returns makespan (float).
"""

from __future__ import annotations
from dataclasses import dataclass
import math

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform, BANDWIDTH
from dag_scheduling.core.simulator import ScheduleState


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------

PriorityValue = float | tuple[float, ...]


@dataclass(frozen=True)
class _Interval:
    start: float
    finish: float

def _avg_proc(dag: SchedulingDAG, idx: int, platform: Platform) -> float:
    """Average processing time over compatible executors (w_bar)."""
    execs = platform.compatible(dag.node_type(idx))
    if not execs:
        return 0.0
    return sum(e.processing_time(dag.compute_cost(idx)) for e in execs) / len(execs)


def _comm_time(volume: float) -> float:
    return volume / BANDWIDTH


def _upward_rank(dag: SchedulingDAG, platform: Platform) -> dict[int, float]:
    """rank_u(vi) = w_bar(vi) + max_{vj in succ}(b_bar(i,j) + rank_u(vj))"""
    topo = dag.topological_order()
    rank: dict[int, float] = {}
    for idx in reversed(topo):
        w = _avg_proc(dag, idx, platform)
        succs = dag.successors(idx)
        if not succs:
            rank[idx] = w
        else:
            rank[idx] = w + max(
                _comm_time(dag.comm_cost(idx, s)) + rank.get(s, 0.0)
                for s in succs
            )
    return rank


def _downward_rank(dag: SchedulingDAG, platform: Platform) -> dict[int, float]:
    """rank_d(vi) = max_{vj in pred}(rank_d(vj) + w_bar(vj) + b_bar(j,i))"""
    topo = dag.topological_order()
    rank: dict[int, float] = {}
    for idx in topo:
        preds = dag.predecessors(idx)
        if not preds:
            rank[idx] = 0.0
        else:
            rank[idx] = max(
                rank.get(p, 0.0) + _avg_proc(dag, p, platform)
                + _comm_time(dag.comm_cost(p, idx))
                for p in preds
            )
    return rank


def _data_ready_time(
    dag: SchedulingDAG,
    task_idx: int,
    executor_id: int,
    state: ScheduleState,
) -> float:
    """Latest predecessor finish plus transfer delay for one executor."""
    data_ready = 0.0
    for pred_idx in dag.predecessors(task_idx):
        pred_aft = state.aft.get(pred_idx)
        if pred_aft is None:
            return math.inf
        delta = (
            0.0 if state.assigned.get(pred_idx) == executor_id
            else _comm_time(dag.comm_cost(pred_idx, task_idx))
        )
        data_ready = max(data_ready, pred_aft + delta)
    return data_ready


def _find_insertion_slot(
    dag: SchedulingDAG,
    platform: Platform,
    task_idx: int,
    executor_id: int,
    state: ScheduleState,
    intervals: dict[int, list[_Interval]],
) -> tuple[float, float]:
    """Earliest feasible non-overlapping slot on an executor timeline."""
    exc = platform.by_id(executor_id)
    duration = exc.processing_time(dag.compute_cost(task_idx))
    start = _data_ready_time(dag, task_idx, executor_id, state)
    if not math.isfinite(start):
        return math.inf, math.inf

    for interval in intervals.get(executor_id, []):
        finish = start + duration
        if finish <= interval.start + 1e-12:
            return start, finish
        start = max(start, interval.finish)
    return start, start + duration


def _insertion_place(
    dag: SchedulingDAG,
    platform: Platform,
    task_idx: int,
    state: ScheduleState,
    intervals: dict[int, list[_Interval]],
) -> tuple[int, float, float]:
    """Insertion-based EFT placement over compatible executors."""
    candidates = platform.compatible(dag.node_type(task_idx))
    if not candidates:
        raise ValueError(f"No executor of type {dag.node_type(task_idx)!r} in platform")

    best: tuple[float, float, int] | None = None
    for exc in candidates:
        start, finish = _find_insertion_slot(dag, platform, task_idx, exc.id, state, intervals)
        candidate = (finish, start, exc.id)
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    finish, start, executor_id = best
    return executor_id, start, finish


def _commit_with_interval(
    state: ScheduleState,
    intervals: dict[int, list[_Interval]],
    task_idx: int,
    executor_id: int,
    start: float,
    finish: float,
) -> None:
    """Commit an insertion placement while preserving executor tail time."""
    previous_tail = state.executor_available.get(executor_id, 0.0)
    state.commit(task_idx, executor_id, start, finish)
    state.executor_available[executor_id] = max(previous_tail, finish)
    intervals.setdefault(executor_id, []).append(_Interval(start, finish))
    intervals[executor_id].sort(key=lambda item: (item.start, item.finish))


def _schedule_greedy(
    dag: SchedulingDAG,
    platform: Platform,
    priority: dict[int, PriorityValue],
    *,
    insertion: bool = False,
) -> float:
    """Generic greedy scheduler: highest priority ready task, then placement."""
    state = ScheduleState(dag, platform)
    intervals: dict[int, list[_Interval]] = {e.id: [] for e in platform.executors}
    while not state.is_done():
        if not state.ready:
            break
        task = max(state.ready, key=lambda t: priority[t])
        if insertion:
            exc_id, start, finish = _insertion_place(dag, platform, task, state, intervals)
            _commit_with_interval(state, intervals, task, exc_id, start, finish)
        else:
            state.schedule_task(task)
    return state.makespan


# ------------------------------------------------------------------
# DONF  (Degree of Node First)
# WOD priority, EFT placement.
# ------------------------------------------------------------------

def donf(dag: SchedulingDAG, platform: Platform) -> float:
    """
    DONF weighted-out-degree rule.
    WOD(vi) = sum_{vj in succ(vi)} 1/in_degree(vj)
    """
    priority = {}
    for idx in dag.indices():
        wod = sum(
            1.0 / dag.in_degree(s)
            for s in dag.successors(idx)
            if dag.in_degree(s) > 0
        )
        priority[idx] = wod
    return _schedule_greedy(dag, platform, priority)


# ------------------------------------------------------------------
# CPOP  (Critical Path on a Processor)
# priority = rank_u + rank_d; CP tasks → CP-processor, rest → EFT.
# ------------------------------------------------------------------

def cpop(dag: SchedulingDAG, platform: Platform) -> float:
    """
    CPOP — eqs. (cpop_rank_u), (cpop_rank_d), (cpop_priority).
    CP processor = executor minimising cumulative processing time of CP tasks.
    """
    rank_u = _upward_rank(dag, platform)
    rank_d = _downward_rank(dag, platform)
    priority = {idx: rank_u[idx] + rank_d[idx] for idx in dag.indices()}

    cp_threshold = max(priority.values())
    cp_tasks = {idx for idx, p in priority.items() if math.isclose(p, cp_threshold, rel_tol=1e-9)}

    # CP processor: executor minimising sum of processing times for CP tasks
    # (restricted to a single executor type that can run all CP tasks)
    # In practice CP tasks may span types; fall back to EFT per CP task.
    # We find the executor minimising the sum over CP tasks of w_{j,i}
    # for tasks of compatible type.
    best_cp_exec: dict[str, int] = {}  # node_type -> best executor id
    for ntype in ("CPU", "GPU", "IO"):
        execs = platform.compatible(ntype)
        if not execs:
            continue
        cp_of_type = [t for t in cp_tasks if dag.node_type(t) == ntype]
        if not cp_of_type:
            continue
        best_e = min(
            execs,
            key=lambda e: sum(e.processing_time(dag.compute_cost(t)) for t in cp_of_type),
        )
        best_cp_exec[ntype] = best_e.id

    state = ScheduleState(dag, platform)
    while not state.is_done():
        if not state.ready:
            break
        task = max(state.ready, key=lambda t: priority[t])
        ntype = dag.node_type(task)
        if task in cp_tasks and ntype in best_cp_exec:
            # force assignment to CP executor
            cp_exec_id = best_cp_exec[ntype]
            exc = platform.by_id(cp_exec_id)
            # compute EST/EFT manually for this executor
            data_ready = 0.0
            for pred in dag.predecessors(task):
                pred_aft = state.aft.get(pred, 0.0)
                delta = (
                    0.0 if state.assigned.get(pred) == exc.id
                    else _comm_time(dag.comm_cost(pred, task))
                )
                data_ready = max(data_ready, pred_aft + delta)
            est = max(state.executor_available.get(exc.id, 0.0), data_ready)
            finish = est + exc.processing_time(dag.compute_cost(task))
            state.commit(task, exc.id, est, finish)
        else:
            state.schedule_task(task)

    return state.makespan


# ------------------------------------------------------------------
# HCPT  (Heterogeneous Critical Parent Trees)
# Critical parent is the predecessor that determines the average earliest start;
# ready tasks are prioritised by low slack, then by larger critical-parent-tree
# subtree size and upward rank.
# ------------------------------------------------------------------

def hcpt(dag: SchedulingDAG, platform: Platform) -> float:
    """
    HCPT-style critical-parent-tree priority with EFT placement.

    The thesis describes HCPT at the level of critical parent trees plus
    start-time criticality. This implementation makes those quantities explicit
    using average processing/communication times: AEST/ALST slack gives the
    primary criticality signal, and critical-parent subtree size breaks ties.
    """
    topo = dag.topological_order()
    avg_proc = {idx: _avg_proc(dag, idx, platform) for idx in dag.indices()}

    aest: dict[int, float] = {}
    critical_parent: dict[int, int | None] = {}
    for idx in topo:
        preds = dag.predecessors(idx)
        if not preds:
            aest[idx] = 0.0
            critical_parent[idx] = None
            continue
        parent = max(
            preds,
            key=lambda p: aest[p] + avg_proc[p] + _comm_time(dag.comm_cost(p, idx)),
        )
        critical_parent[idx] = parent
        aest[idx] = aest[parent] + avg_proc[parent] + _comm_time(dag.comm_cost(parent, idx))

    exits = dag.exit_nodes()
    project_length = max((aest[idx] + avg_proc[idx] for idx in exits), default=0.0)

    alst: dict[int, float] = {}
    for idx in reversed(topo):
        succs = dag.successors(idx)
        if not succs:
            alst[idx] = project_length - avg_proc[idx]
        else:
            alst[idx] = min(
                alst[s] - _comm_time(dag.comm_cost(idx, s)) - avg_proc[idx]
                for s in succs
            )

    children: dict[int, list[int]] = {idx: [] for idx in dag.indices()}
    for child, parent in critical_parent.items():
        if parent is not None:
            children[parent].append(child)

    subtree_size = {idx: 1 for idx in dag.indices()}
    for idx in reversed(topo):
        subtree_size[idx] = 1 + sum(subtree_size[child] for child in children[idx])

    rank_u = _upward_rank(dag, platform)
    priority: dict[int, PriorityValue] = {}
    for idx in dag.indices():
        slack = max(0.0, alst[idx] - aest[idx])
        priority[idx] = (-slack, float(subtree_size[idx]), rank_u[idx])
    return _schedule_greedy(dag, platform, priority)


# ------------------------------------------------------------------
# HPS  (High-Performance Task Scheduling)
# Priority = Up Link Cost (TW_out) + Down Link Cost (TW_in).
# Insertion-aware EFT placement.
# ------------------------------------------------------------------

def hps(dag: SchedulingDAG, platform: Platform) -> float:
    """
    HPS — link-based priority with insertion-aware EFT placement.
    """
    priority = {}
    for idx in dag.indices():
        ulc = sum(_comm_time(dag.comm_cost(idx, s)) for s in dag.successors(idx))
        dlc = sum(_comm_time(dag.comm_cost(p, idx)) for p in dag.predecessors(idx))
        priority[idx] = ulc + dlc
    return _schedule_greedy(dag, platform, priority, insertion=True)


# ------------------------------------------------------------------
# PETS  (Performance Effective Task Scheduling)
# priority = ACC + DTC + RPT
# ACC = avg processing time; DTC = sum comm to successors;
# RPT = max over pred of (RPT(pred) + ACC(pred) + b_{pred,v})
# Insertion-aware EFT placement.
# ------------------------------------------------------------------

def pets(dag: SchedulingDAG, platform: Platform) -> float:
    """
    PETS — priority = ACC(vi) + DTC(vi) + RPT(vi), insertion-aware placement.
    """
    topo = dag.topological_order()

    acc = {idx: _avg_proc(dag, idx, platform) for idx in dag.indices()}
    dtc = {
        idx: sum(_comm_time(dag.comm_cost(idx, s)) for s in dag.successors(idx))
        for idx in dag.indices()
    }

    # RPT: max over predecessors of (RPT(pred) + ACC(pred) + b_{pred,v})
    rpt: dict[int, float] = {}
    for idx in topo:
        preds = dag.predecessors(idx)
        if not preds:
            rpt[idx] = 0.0
        else:
            rpt[idx] = max(
                rpt.get(p, 0.0) + acc[p] + _comm_time(dag.comm_cost(p, idx))
                for p in preds
            )

    priority = {idx: acc[idx] + dtc[idx] + rpt[idx] for idx in dag.indices()}
    return _schedule_greedy(dag, platform, priority, insertion=True)


# ------------------------------------------------------------------
# convenience: run all baselines on one (dag, platform) pair
# ------------------------------------------------------------------

BASELINES: dict[str, callable] = {
    "DONF": donf,
    "CPOP": cpop,
    "HCPT": hcpt,
    "HPS":  hps,
    "PETS": pets,
}


def run_all(dag: SchedulingDAG, platform: Platform) -> dict[str, float]:
    return {name: fn(dag, platform) for name, fn in BASELINES.items()}
