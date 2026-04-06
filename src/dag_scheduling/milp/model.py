from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pyomo.environ as pyo

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.eft import comm_time
from dag_scheduling.core.platform import Platform


@dataclass(frozen=True)
class MilpInstance:
    tasks: tuple[int, ...]
    executors: tuple[int, ...]
    precedence_edges: tuple[tuple[int, int], ...]
    unrelated_pairs: tuple[tuple[int, int], ...]
    directed_order_pairs: tuple[tuple[int, int], ...]
    node_ids: dict[int, int]
    processing_time: dict[tuple[int, int], float]
    eligibility: dict[tuple[int, int], int]
    comm_delay: dict[tuple[int, int], float]
    horizon: float


def _reachability(dag: SchedulingDAG) -> dict[int, set[int]]:
    reach: dict[int, set[int]] = {idx: set() for idx in dag.indices()}
    for idx in reversed(dag.topological_order()):
        closure: set[int] = set()
        for succ in dag.successors(idx):
            closure.add(succ)
            closure.update(reach[succ])
        reach[idx] = closure
    return reach


def _safe_horizon(
    dag: SchedulingDAG,
    platform: Platform,
    processing_time: dict[tuple[int, int], float],
) -> float:
    total_processing = 0.0
    for task_idx in dag.indices():
        compatible = [
            processing_time[(task_idx, exc.id)]
            for exc in platform.compatible(dag.node_type(task_idx))
        ]
        if not compatible:
            raise ValueError(
                f"Task {dag.node_id(task_idx)} has no compatible executors"
            )
        total_processing += max(compatible)

    total_comm = 0.0
    for src in dag.indices():
        for dst in dag.successors(src):
            total_comm += comm_time(dag.comm_cost(src, dst))

    return max(total_processing + total_comm, 1.0)


def build_instance(dag: SchedulingDAG, platform: Platform) -> MilpInstance:
    tasks = tuple(dag.indices())
    executors = tuple(exc.id for exc in platform.executors)

    node_ids = {idx: dag.node_id(idx) for idx in tasks}

    processing_time: dict[tuple[int, int], float] = {}
    eligibility: dict[tuple[int, int], int] = {}
    for task_idx in tasks:
        node_type = dag.node_type(task_idx)
        if node_type is None:
            raise ValueError(f"Task {dag.node_id(task_idx)} has no node_type assigned")
        compatible_ids = {exc.id for exc in platform.compatible(node_type)}
        if not compatible_ids:
            raise ValueError(
                f"Task {dag.node_id(task_idx)} has no compatible executors of type {node_type!r}"
            )
        for exc in platform.executors:
            allowed = int(exc.id in compatible_ids)
            eligibility[(task_idx, exc.id)] = allowed
            processing_time[(task_idx, exc.id)] = (
                exc.processing_time(dag.compute_cost(task_idx)) if allowed else 0.0
            )

    precedence_edges = tuple(
        (src, dst)
        for src in tasks
        for dst in dag.successors(src)
    )
    comm_delay = {
        (src, dst): comm_time(dag.comm_cost(src, dst))
        for src, dst in precedence_edges
    }

    reach = _reachability(dag)
    unrelated_pairs = tuple(
        (j, k)
        for j, k in combinations(tasks, 2)
        if k not in reach[j] and j not in reach[k]
    )
    directed_order_pairs = tuple(
        pair
        for j, k in unrelated_pairs
        for pair in ((j, k), (k, j))
    )

    return MilpInstance(
        tasks=tasks,
        executors=executors,
        precedence_edges=precedence_edges,
        unrelated_pairs=unrelated_pairs,
        directed_order_pairs=directed_order_pairs,
        node_ids=node_ids,
        processing_time=processing_time,
        eligibility=eligibility,
        comm_delay=comm_delay,
        horizon=_safe_horizon(dag, platform, processing_time),
    )


def build_pyomo_model(instance: MilpInstance) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(name="heterogeneous_dag_milp")

    model.J = pyo.Set(initialize=instance.tasks, ordered=True)
    model.P = pyo.Set(initialize=instance.executors, ordered=True)
    model.E = pyo.Set(dimen=2, initialize=instance.precedence_edges, ordered=True)
    model.UNRELATED = pyo.Set(
        dimen=2, initialize=instance.unrelated_pairs, ordered=True
    )
    model.ORDER_PAIRS = pyo.Set(
        dimen=2, initialize=instance.directed_order_pairs, ordered=True
    )

    model.B = pyo.Param(
        model.J,
        model.P,
        initialize=instance.eligibility,
        default=0,
        within=pyo.Binary,
    )
    model.D = pyo.Param(
        model.J,
        model.P,
        initialize=instance.processing_time,
        default=0.0,
        within=pyo.NonNegativeReals,
    )
    model.comm = pyo.Param(
        model.E,
        initialize=instance.comm_delay,
        within=pyo.NonNegativeReals,
    )
    model.big_m = pyo.Param(initialize=instance.horizon, within=pyo.PositiveReals)

    model.X = pyo.Var(model.J, model.P, domain=pyo.Binary)
    model.S = pyo.Var(model.J, domain=pyo.NonNegativeReals, bounds=(0.0, instance.horizon))
    model.C_max = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, instance.horizon))
    model.Z = pyo.Var(model.E, domain=pyo.Binary)
    model.K = pyo.Var(model.E, model.P, domain=pyo.Binary)
    model.theta = pyo.Var(model.ORDER_PAIRS, domain=pyo.Binary)

    def objective_rule(m: pyo.ConcreteModel) -> pyo.Expression:
        return m.C_max

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def makespan_rule(m: pyo.ConcreteModel, j: int) -> pyo.Expression:
        return m.C_max >= m.S[j] + sum(m.D[j, i] * m.X[j, i] for i in m.P)

    model.makespan_lb = pyo.Constraint(model.J, rule=makespan_rule)

    def eligibility_rule(m: pyo.ConcreteModel, j: int, i: int) -> pyo.Expression:
        return m.X[j, i] <= m.B[j, i]

    model.eligibility = pyo.Constraint(model.J, model.P, rule=eligibility_rule)

    def assign_once_rule(m: pyo.ConcreteModel, j: int) -> pyo.Expression:
        return sum(m.X[j, i] for i in m.P) == 1

    model.assign_once = pyo.Constraint(model.J, rule=assign_once_rule)

    def same_executor_sum_rule(m: pyo.ConcreteModel, j: int, k: int) -> pyo.Expression:
        return m.Z[j, k] == sum(m.K[j, k, i] for i in m.P)

    model.same_executor_sum = pyo.Constraint(model.E, rule=same_executor_sum_rule)

    def same_executor_ub_left_rule(
        m: pyo.ConcreteModel, j: int, k: int, i: int
    ) -> pyo.Expression:
        return m.K[j, k, i] <= m.X[j, i]

    def same_executor_ub_right_rule(
        m: pyo.ConcreteModel, j: int, k: int, i: int
    ) -> pyo.Expression:
        return m.K[j, k, i] <= m.X[k, i]

    def same_executor_lb_rule(
        m: pyo.ConcreteModel, j: int, k: int, i: int
    ) -> pyo.Expression:
        return m.K[j, k, i] >= m.X[j, i] + m.X[k, i] - 1

    model.same_executor_ub_left = pyo.Constraint(
        model.E, model.P, rule=same_executor_ub_left_rule
    )
    model.same_executor_ub_right = pyo.Constraint(
        model.E, model.P, rule=same_executor_ub_right_rule
    )
    model.same_executor_lb = pyo.Constraint(
        model.E, model.P, rule=same_executor_lb_rule
    )

    def precedence_rule(m: pyo.ConcreteModel, j: int, k: int) -> pyo.Expression:
        return (
            m.S[k]
            >= m.S[j]
            + sum(m.D[j, i] * m.X[j, i] for i in m.P)
            + m.comm[j, k] * (1 - m.Z[j, k])
        )

    model.precedence = pyo.Constraint(model.E, rule=precedence_rule)

    def nooverlap_forward_rule(
        m: pyo.ConcreteModel, j: int, k: int, i: int
    ) -> pyo.Expression:
        return (
            m.S[k]
            >= m.S[j]
            + m.D[j, i]
            - m.big_m * (3 - m.X[j, i] - m.X[k, i] - m.theta[j, k])
        )

    def nooverlap_backward_rule(
        m: pyo.ConcreteModel, j: int, k: int, i: int
    ) -> pyo.Expression:
        return (
            m.S[j]
            >= m.S[k]
            + m.D[k, i]
            - m.big_m * (3 - m.X[j, i] - m.X[k, i] - m.theta[k, j])
        )

    model.nooverlap_forward = pyo.Constraint(
        model.UNRELATED, model.P, rule=nooverlap_forward_rule
    )
    model.nooverlap_backward = pyo.Constraint(
        model.UNRELATED, model.P, rule=nooverlap_backward_rule
    )

    def order_upper_rule(m: pyo.ConcreteModel, j: int, k: int) -> pyo.Expression:
        return m.theta[j, k] + m.theta[k, j] <= 1

    def order_lower_rule(m: pyo.ConcreteModel, j: int, k: int, i: int) -> pyo.Expression:
        return m.theta[j, k] + m.theta[k, j] >= m.X[j, i] + m.X[k, i] - 1

    model.order_upper = pyo.Constraint(model.UNRELATED, rule=order_upper_rule)
    model.order_lower = pyo.Constraint(
        model.UNRELATED, model.P, rule=order_lower_rule
    )

    def start_before_makespan_rule(m: pyo.ConcreteModel, j: int) -> pyo.Expression:
        return m.S[j] <= m.C_max

    model.start_before_makespan = pyo.Constraint(
        model.J, rule=start_before_makespan_rule
    )

    return model
