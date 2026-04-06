"""
EFT (Earliest Finish Time) placement rule — exactly as Definitions 2 & 3
in the thesis problem statement.

EST(v_j, p_i) = max(T_ava(p_i),  max over pred(v_m) of [AFT(v_m) + delta(v_m, v_j, p_i)])

where delta(v_m, v_j, p_i) = 0              if pi(v_m) == p_i
                              b_{m,j}        otherwise

b_{m,j} = edge_data_volume / BANDWIDTH   (comm time in seconds)

EFT(v_j, p_i) = EST(v_j, p_i) + w_{j,i}

The EFT placement rule picks argmin_i EFT(v_j, p_i) over compatible executors.
"""

from __future__ import annotations
import math

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform, BANDWIDTH


def comm_time(data_volume: float) -> float:
    """b_{m,j} = data_volume / bandwidth (seconds)."""
    return data_volume / BANDWIDTH


def eft_place(
    task_idx: int,
    dag: SchedulingDAG,
    platform: Platform,
    executor_available: dict[int, float],   # p_i.id -> T_ava(p_i)
    aft: dict[int, float],                  # task_idx -> AFT
    assigned: dict[int, int],               # task_idx -> executor_id
) -> tuple[int, float, float]:
    """
    Returns (executor_id, start_time, finish_time) for the best placement.
    """
    node_type = dag.node_type(task_idx)
    if node_type is None:
        raise ValueError(f"Task {task_idx} has no node_type assigned")

    candidates = platform.compatible(node_type)
    if not candidates:
        raise ValueError(f"No executor of type {node_type!r} in platform")

    best_exec_id = -1
    best_start = math.inf
    best_finish = math.inf

    for exc in candidates:
        # data-ready time: max over predecessors of [AFT(pred) + delta]
        data_ready = 0.0
        for pred_idx in dag.predecessors(task_idx):
            pred_aft = aft.get(pred_idx)
            if pred_aft is None:
                data_ready = math.inf
                break
            delta = (
                0.0
                if assigned.get(pred_idx) == exc.id
                else comm_time(dag.comm_cost(pred_idx, task_idx))
            )
            data_ready = max(data_ready, pred_aft + delta)

        est = max(executor_available.get(exc.id, 0.0), data_ready)
        finish = est + exc.processing_time(dag.compute_cost(task_idx))

        if finish < best_finish:
            best_finish = finish
            best_start = est
            best_exec_id = exc.id

    return best_exec_id, best_start, best_finish
