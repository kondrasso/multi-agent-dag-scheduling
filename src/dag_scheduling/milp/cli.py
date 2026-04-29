from __future__ import annotations

import argparse
import json
from pathlib import Path

from dag_scheduling.baselines.heuristics import run_all
from dag_scheduling.core.platform import make_workspace
from dag_scheduling.data.augmentor import inject_node_types_dot
from dag_scheduling.data.generator import generate_dot
from dag_scheduling.data.parser import parse_dot, parse_dot_str
from dag_scheduling.milp.solve import MilpSolverConfig, solve_milp
from dag_scheduling.protocol import assign_node_types


def _needs_types(dag) -> bool:
    return any(dag.node_type(idx) is None for idx in dag.indices())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the offline heterogeneous DAG MILP with HiGHS or Gurobi."
    )
    parser.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    parser.add_argument("--ws", type=int, default=1, help="workspace id for platform presets")
    parser.add_argument("--dot", type=str, default=None, help="typed or untyped .dot DAG")
    parser.add_argument("--assign-types", choices=["alpha", "random"], default="alpha")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--typed-dot-out", type=str, default=None)

    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--fat", type=float, default=0.5)
    parser.add_argument("--regular", type=float, default=0.8)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--jump", type=int, default=2)
    parser.add_argument("--ccr", type=int, default=2)
    return parser


def _load_dag(args: argparse.Namespace):
    raw_dot: str | None = None
    if args.dot:
        dag = parse_dot(args.dot)
        raw_dot = Path(args.dot).read_text()
    else:
        raw_dot = generate_dot(
            n=args.n,
            fat=args.fat,
            regular=args.regular,
            density=args.density,
            jump=args.jump,
            ccr=args.ccr,
            seed=args.seed,
        )
        dag = parse_dot_str(raw_dot)

    if _needs_types(dag):
        assign_node_types(dag, strategy=args.assign_types, seed=args.seed)
        if args.typed_dot_out:
            typed_dot = inject_node_types_dot(raw_dot or "", dag)
            Path(args.typed_dot_out).write_text(typed_dot)

    return dag


def main() -> None:
    args = _build_parser().parse_args()
    dag = _load_dag(args)
    platform = make_workspace(args.ws)

    try:
        solution = solve_milp(
            dag,
            platform,
            MilpSolverConfig(
                solver=args.solver,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                threads=args.threads,
                seed=args.seed,
                tee=args.tee,
            ),
        )
    except Exception as exc:
        raise SystemExit(f"MILP solve failed: {exc}") from exc

    print(
        f"solver={solution.solver} status={solution.status} "
        f"termination={solution.termination_condition} has_solution={solution.has_solution}"
    )
    print(
        f"makespan={solution.makespan} bound={solution.best_bound} "
        f"gap={solution.mip_gap} runtime_sec={solution.runtime_sec}"
    )

    if solution.has_solution:
        print("schedule:")
        for task_idx, start in sorted(solution.start_times.items(), key=lambda item: item[1]):
            node_id = dag.node_id(task_idx)
            executor_id = solution.assignment.get(task_idx)
            print(f"  task={node_id} start={start:.6f} executor={executor_id}")

    if args.compare_baselines:
        baseline = run_all(dag, platform)
        print("baselines:")
        for name, makespan in baseline.items():
            print(f"  {name}={makespan:.6f}")
        if solution.makespan is not None:
            best_baseline = min(baseline.values())
            improvement = (best_baseline - solution.makespan) / best_baseline * 100.0
            print(f"milp_vs_best_baseline_pct={improvement:.6f}")

    if args.json_out:
        payload = {
            "solver": solution.solver,
            "status": solution.status,
            "termination_condition": solution.termination_condition,
            "has_solution": solution.has_solution,
            "makespan": solution.makespan,
            "best_bound": solution.best_bound,
            "mip_gap": solution.mip_gap,
            "runtime_sec": solution.runtime_sec,
            "start_times": {
                str(dag.node_id(task_idx)): start
                for task_idx, start in solution.start_times.items()
            },
            "assignment": {
                str(dag.node_id(task_idx)): executor_id
                for task_idx, executor_id in solution.assignment.items()
            },
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
