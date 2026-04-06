from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform
from dag_scheduling.milp.model import build_instance, build_pyomo_model


@dataclass(frozen=True)
class MilpSolverConfig:
    solver: str = "highs"
    time_limit: float | None = 900.0
    mip_gap: float | None = 0.03
    threads: int | None = None
    seed: int = 0
    tee: bool = False


@dataclass(frozen=True)
class MilpSolution:
    solver: str
    status: str
    termination_condition: str
    has_solution: bool
    makespan: float | None
    best_bound: float | None
    mip_gap: float | None
    runtime_sec: float | None
    start_times: dict[int, float]
    assignment: dict[int, int]


def _pyomo_solver_name(solver: str) -> str:
    lowered = solver.lower()
    if lowered == "highs":
        return "appsi_highs"
    if lowered == "gurobi":
        return "gurobi"
    raise ValueError(f"Unsupported MILP solver: {solver!r}")


def is_solver_available(solver: str) -> bool:
    pyomo_name = _pyomo_solver_name(solver)
    try:
        return bool(SolverFactory(pyomo_name).available(False))
    except Exception:
        return False


def _result_item(container: Any, key: str) -> Any:
    if len(container) == 0:
        return None
    item = container[0]
    return item[key] if key in item else None


def _safe_value(obj: Any) -> float | None:
    value = pyo.value(obj, exception=False)
    return None if value is None else float(value)


def _solve_model(
    model: pyo.ConcreteModel,
    config: MilpSolverConfig,
):
    solver_name = _pyomo_solver_name(config.solver)
    solver = SolverFactory(solver_name)
    if not solver.available(False):
        raise RuntimeError(f"Solver {config.solver!r} is not available")

    if solver_name == "appsi_highs":
        if config.time_limit is not None:
            solver.config.time_limit = config.time_limit
        if config.mip_gap is not None:
            solver.config.mip_gap = config.mip_gap
        solver.config.stream_solver = config.tee
        solver.highs_options["random_seed"] = config.seed
        if config.threads is not None:
            solver.highs_options["threads"] = config.threads
        return solver.solve(model, tee=config.tee)

    options: dict[str, Any] = {"Seed": config.seed}
    if config.time_limit is not None:
        options["TimeLimit"] = config.time_limit
    if config.mip_gap is not None:
        options["MIPGap"] = config.mip_gap
    if config.threads is not None:
        options["Threads"] = config.threads
    return solver.solve(model, tee=config.tee, options=options)


def solve_milp(
    dag: SchedulingDAG,
    platform: Platform,
    config: MilpSolverConfig | None = None,
) -> MilpSolution:
    config = config or MilpSolverConfig()
    instance = build_instance(dag, platform)
    model = build_pyomo_model(instance)
    results = _solve_model(model, config)

    status = str(_result_item(results.solver, "Status") or "unknown")
    termination = str(
        _result_item(results.solver, "Termination condition") or "unknown"
    )
    runtime_sec = _result_item(results.solver, "Wall time")
    if runtime_sec is None:
        runtime_sec = _result_item(results.solver, "Time")
    runtime = None if runtime_sec is None else float(runtime_sec)

    lower_bound = _result_item(results.problem, "Lower bound")
    upper_bound = _result_item(results.problem, "Upper bound")
    best_bound = None if lower_bound is None else float(lower_bound)

    makespan = _safe_value(model.C_max)
    has_solution = makespan is not None

    start_times: dict[int, float] = {}
    assignment: dict[int, int] = {}
    if has_solution:
        for task_idx in instance.tasks:
            start = _safe_value(model.S[task_idx])
            if start is not None:
                start_times[task_idx] = start
            for exc_id in instance.executors:
                if _safe_value(model.X[task_idx, exc_id]) and _safe_value(
                    model.X[task_idx, exc_id]
                ) > 0.5:
                    assignment[task_idx] = exc_id
                    break

    incumbent = makespan
    if incumbent is None and upper_bound is not None:
        incumbent = float(upper_bound)

    mip_gap = None
    if incumbent is not None and best_bound is not None and math.isfinite(incumbent):
        mip_gap = max(incumbent - best_bound, 0.0) / max(abs(incumbent), 1e-9)

    return MilpSolution(
        solver=config.solver,
        status=status,
        termination_condition=termination,
        has_solution=has_solution,
        makespan=makespan,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime_sec=runtime,
        start_times=start_times,
        assignment=assignment,
    )
