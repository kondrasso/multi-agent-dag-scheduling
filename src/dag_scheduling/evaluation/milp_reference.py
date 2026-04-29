from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from dag_scheduling.baselines.heuristics import BASELINES, run_all
from dag_scheduling.milp.solve import MilpSolverConfig, solve_milp
from dag_scheduling.protocol import make_test_corpus


def _proximity(milp_makespan: float, alg_makespan: float) -> float:
    return milp_makespan / alg_makespan * 100.0 if alg_makespan > 0 else 0.0


def evaluate_reference(
    ws: int,
    n: int,
    solver: str,
    limit: int,
    time_limit: float,
    mip_gap: float,
    threads: int | None,
    seed: int,
    tee: bool = False,
):
    corpus = make_test_corpus(n, ws)
    if limit > 0:
        corpus = corpus[:limit]

    config = MilpSolverConfig(
        solver=solver,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        seed=seed,
        tee=tee,
    )

    statuses: Counter[str] = Counter()
    milp_makespans: list[float] = []
    baseline_makespans: dict[str, list[float]] = {name: [] for name in BASELINES}
    rows: list[dict[str, float | int | str]] = []

    for instance_idx, (dag, platform) in enumerate(corpus, start=1):
        try:
            solution = solve_milp(dag, platform, config)
        except Exception as exc:
            msg = str(exc)
            statuses[f"error:{type(exc).__name__}"] += 1
            row = {
                "instance": instance_idx,
                "status": "error",
                "termination": type(exc).__name__,
                "runtime_sec": "",
                "milp_makespan": "",
                "milp_bound": "",
                "milp_gap": "",
                "error": msg,
            }
            rows.append(row)
            print(
                f"[warn] MILP solve failed in WS{ws} n={n} instance={instance_idx}: {msg}"
            )
            continue

        statuses[solution.termination_condition] += 1

        row: dict[str, float | int | str] = {
            "instance": instance_idx,
            "status": solution.status,
            "termination": solution.termination_condition,
            "runtime_sec": solution.runtime_sec if solution.runtime_sec is not None else "",
            "milp_makespan": solution.makespan if solution.makespan is not None else "",
            "milp_bound": solution.best_bound if solution.best_bound is not None else "",
            "milp_gap": solution.mip_gap if solution.mip_gap is not None else "",
        }

        if not solution.has_solution or solution.makespan is None:
            rows.append(row)
            continue

        makespans = run_all(dag, platform)
        milp_makespans.append(solution.makespan)
        for name, alg_makespan in makespans.items():
            baseline_makespans[name].append(alg_makespan)
            row[name] = alg_makespan
            row[f"{name}_proximity_pct"] = _proximity(solution.makespan, alg_makespan)
        rows.append(row)

    summary: dict[str, float | int | dict[str, int]] = {
        "instances_total": len(corpus),
        "instances_solved": len(milp_makespans),
        "milp_mean_makespan": float(np.mean(milp_makespans)) if milp_makespans else float("nan"),
        "statuses": dict(statuses),
    }
    for name, values in baseline_makespans.items():
        if not values or not milp_makespans:
            continue
        summary[f"{name}_mean_makespan"] = float(np.mean(values))
        summary[f"{name}_mean_proximity_pct"] = float(
            np.mean(
                [
                    _proximity(milp, alg)
                    for milp, alg in zip(milp_makespans, values, strict=False)
                ]
            )
        )
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare offline heuristics to a MILP reference on tractable instances."
    )
    parser.add_argument("--ws", type=int, nargs="+", default=[1])
    parser.add_argument("--n", type=int, nargs="+", default=[30])
    parser.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--mip-gap", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    csv_rows: list[dict[str, float | int | str]] = []
    for ws in args.ws:
        for n in args.n:
            summary, rows = evaluate_reference(
                ws=ws,
                n=n,
                solver=args.solver,
                limit=args.limit,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                threads=args.threads,
                seed=args.seed,
                tee=args.tee,
            )
            print(
                f"WS{ws} n={n} solver={args.solver} "
                f"solved={summary['instances_solved']}/{summary['instances_total']}"
            )
            print(
                f"  MILP mean makespan={summary['milp_mean_makespan']}"
            )
            for name in BASELINES:
                key = f"{name}_mean_proximity_pct"
                if key in summary:
                    print(
                        f"  {name} mean proximity to MILP={summary[key]:.6f}%"
                    )
            if summary["statuses"]:
                print(f"  statuses={summary['statuses']}")

            for row in rows:
                csv_rows.append({"ws": ws, "n": n, **row})

    if args.csv_out and csv_rows:
        path = Path(args.csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in csv_rows for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"saved csv -> {path}")


if __name__ == "__main__":
    main()
