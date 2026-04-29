"""
Offline DAG scheduling baselines.

All five heuristics follow the same schedule-construction loop:
  1. prioritise ready tasks using a method-specific rule
  2. select the top task
  3. assign to a compatible executor using EFT (or CP-processor for CPOP)
  4. commit and update ready set

Insertion-based EFT (HPS, PETS) is not implemented; standard EFT is used
for all methods, consistent with the non-insertion variant used by DONF/CPOP.

Returns makespan (float).
"""

from __future__ import annotations
import math
import numpy as np

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform, BANDWIDTH
from dag_scheduling.core.simulator import ScheduleState


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------

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


def _schedule_greedy(dag: SchedulingDAG, platform: Platform,
                     priority: dict[int, float]) -> float:
    """Generic greedy scheduler: highest priority ready task → EFT placement."""
    state = ScheduleState(dag, platform)
    while not state.is_done():
        if not state.ready:
            break
        task = max(state.ready, key=lambda t: priority[t])
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
# The protocol gives a high-level description referencing [a0].
# Priority approximated by upward rank (most critical = highest rank_u),
# with EFT placement — faithful to the "criticality" intent described.
# ------------------------------------------------------------------

def hcpt(dag: SchedulingDAG, platform: Platform) -> float:
    """
    HCPT — approximated by upward-rank priority with EFT placement.
    (Exact critical-parent-tree construction follows [a0], not reproduced here.)
    """
    priority = _upward_rank(dag, platform)
    return _schedule_greedy(dag, platform, priority)


# ------------------------------------------------------------------
# HPS  (High-Performance Task Scheduling)
# Priority = Up Link Cost (TW_out); ties broken by Down Link Cost (TW_in).
# EFT placement (insertion variant not implemented).
# ------------------------------------------------------------------

def hps(dag: SchedulingDAG, platform: Platform) -> float:
    """
    HPS — link-based priority: ULC = TW_out (primary), DLC = TW_in (secondary).
    """
    priority = {}
    for idx in dag.indices():
        ulc = sum(_comm_time(dag.comm_cost(idx, s)) for s in dag.successors(idx))
        dlc = sum(_comm_time(dag.comm_cost(p, idx)) for p in dag.predecessors(idx))
        # encode (ulc, dlc) as a single float for max() comparison
        priority[idx] = ulc + dlc * 1e-9
    return _schedule_greedy(dag, platform, priority)


# ------------------------------------------------------------------
# PETS  (Performance Effective Task Scheduling)
# priority = ACC + DTC + RPT
# ACC = avg processing time; DTC = sum comm to successors;
# RPT = max over pred of (RPT(pred) + ACC(pred) + b_{pred,v})
# EFT placement (insertion variant not implemented).
# ------------------------------------------------------------------

def pets(dag: SchedulingDAG, platform: Platform) -> float:
    """
    PETS — priority = ACC(vi) + DTC(vi) + RPT(vi).
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
    return _schedule_greedy(dag, platform, priority)


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
