"""
Offline scheduling evaluation — matches the thesis experimental protocol.

Test corpus (MCTS/NN chapters): 48 topology classes × 10 instances = 480 DAGs
per (n, ws) combination.  Results reported as mean % improvement over DONF.

Usage (single cell):
  uv run python -m dag_scheduling.evaluation.benchmark --ws 1 --n 30

Usage (full 3×3 table, with trained weights):
  uv run python -m dag_scheduling.evaluation.benchmark \\
      --ws 1 2 3 --n 30 60 90 \\
      --nn_dir results/ --mcts_dir results/

Usage (with MILP reference on a tractable prefix of each test cell):
  uv run python -m dag_scheduling.evaluation.benchmark \\
      --ws 1 2 3 --n 30 60 90 \\
      --milp_solver highs --milp_limit 5 --milp_time_limit 900 --milp_gap 0.03

Weights are looked up as {dir}/nn_ws{ws}_n{n}.npy and {dir}/mcts_ws{ws}_n{n}.npy.
If a weights file is absent for a (ws,n) cell, that algorithm is skipped for that cell.
"""

from __future__ import annotations
import argparse
import csv
import numpy as np
from pathlib import Path

from dag_scheduling.data.generator import generate
from dag_scheduling.data.augmentor import augment_random
from dag_scheduling.core.platform import make_workspace
from dag_scheduling.baselines.heuristics import run_all, BASELINES
from dag_scheduling.algorithms.nn.train import nn_schedule
from dag_scheduling.algorithms.mcts.train import mcts_schedule
from dag_scheduling.milp.solve import MilpSolverConfig, solve_milp

# 48-class topology grid: fat × density × regularity × jump × ccr
_FAT        = [0.2, 0.5]
_DENSITY    = [0.1, 0.4, 0.8]
_REGULARITY = [0.2, 0.8]
_JUMP       = [2, 4]
_CCR        = [0.2, 0.8]
N_TEST_PER_CLASS = 10   # 48 × 10 = 480 DAGs per (n, ws)

# seeds for the test corpus are offset far from training seeds
_TEST_SEED_OFFSET = 100_000


def make_test_corpus(n: int, ws: int):
    """Generate the standard 480-DAG test corpus for one (n, ws) cell."""
    platform = make_workspace(ws)
    corpus, seed = [], _TEST_SEED_OFFSET + n * 1000 + ws * 100
    for f in _FAT:
        for d in _DENSITY:
            for r in _REGULARITY:
                for j in _JUMP:
                    for c in _CCR:
                        for _ in range(N_TEST_PER_CLASS):
                            dag = generate(
                                n=n, fat=f, regular=r, density=d,
                                jump=j, ccr=int(c * 10),
                            )
                            augment_random(dag, seed=seed)
                            corpus.append((dag, platform))
                            seed += 1
    return corpus


def _pct_improvement(baseline: np.ndarray, method: np.ndarray) -> float:
    """Mean per-instance % improvement over baseline (thesis eq. Δ)."""
    delta = (baseline - method) / np.where(baseline > 0, baseline, 1.0) * 100.0
    return float(delta.mean())


def evaluate_cell(
    n: int,
    ws: int,
    nn_weights: np.ndarray | None = None,
    mcts_weights: np.ndarray | None = None,
    mcts_k: int = 5,
    mcts_iter: int = 20,
    mcts_h: int = 5,
    milp_solver: str | None = None,
    milp_limit: int = 0,
    milp_time_limit: float = 900.0,
    milp_gap: float = 0.03,
    milp_threads: int | None = None,
    milp_seed: int = 0,
    milp_tee: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Run all algorithms on the (n, ws) test corpus.

    Returns:
      - full results over the complete test corpus
      - optional MILP reference bundle over the solved tractable prefix
    """
    corpus = make_test_corpus(n, ws)

    alg_names = list(BASELINES.keys())
    if nn_weights is not None:
        alg_names.append("NN")
    if mcts_weights is not None:
        alg_names.append("MCTS")

    results: dict[str, list[float]] = {a: [] for a in alg_names}
    milp_bundle: dict[str, list[float]] | None = None
    milp_config: MilpSolverConfig | None = None
    if milp_solver is not None and milp_limit > 0:
        milp_bundle = {
            "MILP": [],
            "instance_idx": [],
        }
        for alg in alg_names:
            milp_bundle[alg] = []
        milp_config = MilpSolverConfig(
            solver=milp_solver,
            time_limit=milp_time_limit,
            mip_gap=milp_gap,
            threads=milp_threads,
            seed=milp_seed,
            tee=milp_tee,
        )
    milp_disabled = False

    for instance_idx, (dag, platform) in enumerate(corpus):
        ms = run_all(dag, platform)
        for alg, v in ms.items():
            results[alg].append(v)
        current: dict[str, float] = dict(ms)
        if nn_weights is not None:
            nn_ms = nn_schedule(dag, platform, nn_weights)
            results["NN"].append(nn_ms)
            current["NN"] = nn_ms
        if mcts_weights is not None:
            mcts_ms = mcts_schedule(
                dag, platform, mcts_weights, k=mcts_k, n_iter=mcts_iter, h=mcts_h
            )
            results["MCTS"].append(mcts_ms)
            current["MCTS"] = mcts_ms

        if (
            milp_bundle is not None
            and milp_config is not None
            and not milp_disabled
            and instance_idx < milp_limit
        ):
            try:
                solution = solve_milp(dag, platform, milp_config)
            except Exception as exc:
                msg = str(exc)
                print(
                    f"[warn] MILP solve failed in WS{ws} n={n} instance={instance_idx + 1}: {msg}"
                )
                if "size-limited license" in msg or "not available" in msg.lower():
                    milp_disabled = True
                continue

            if solution.makespan is not None:
                milp_bundle["MILP"].append(solution.makespan)
                milp_bundle["instance_idx"].append(instance_idx)
                for alg in alg_names:
                    milp_bundle[alg].append(current[alg])

    result_arrays = {a: np.array(v) for a, v in results.items()}
    if milp_bundle is None:
        return result_arrays, None
    return result_arrays, {a: np.array(v) for a, v in milp_bundle.items()}


def print_table(table: dict[tuple[int, int], dict[str, np.ndarray]],
                n_values: list[int], ws_values: list[int],
                alg_names: list[str]):
    """
    Print thesis-style table: rows = (ws, n), columns = % improvement over DONF.
    """
    col_w = 8
    header_algs = [a for a in alg_names if a != "DONF"]
    print(f"\n{'':>12}" + "".join(f"  {a:>{col_w}}" for a in header_algs))
    print("-" * (12 + (col_w + 2) * len(header_algs)))

    for ws in ws_values:
        for n in n_values:
            key = (n, ws)
            if key not in table:
                continue
            res = table[key]
            donf = res["DONF"]
            row = f"WS{ws} n={n:>3}"
            for a in header_algs:
                if a in res:
                    pct = _pct_improvement(donf, res[a])
                    row += f"  {pct:>{col_w}.2f}%"
                else:
                    row += f"  {'N/A':>{col_w}}"
            print(row)
        if ws != ws_values[-1]:
            print()


def print_milp_proximity_table(
    table: dict[tuple[int, int], dict[str, np.ndarray]],
    n_values: list[int],
    ws_values: list[int],
    alg_names: list[str],
):
    """Print thesis-style proximity-to-MILP table (higher is better)."""
    header_algs = [a for a in alg_names if a != "MILP"]
    col_w = 8
    print(f"\n{'':>12}" + "".join(f"  {a:>{col_w}}" for a in header_algs))
    print("-" * (12 + (col_w + 2) * len(header_algs)))

    for ws in ws_values:
        for n in n_values:
            key = (n, ws)
            if key not in table:
                continue
            res = table[key]
            milp = res["MILP"]
            count = len(milp)
            row = f"WS{ws} n={n:>3}"
            for a in header_algs:
                if a in res and count:
                    prox = np.mean(milp / np.where(res[a] > 0, res[a], 1.0) * 100.0)
                    row += f"  {prox:>{col_w}.2f}%"
                else:
                    row += f"  {'N/A':>{col_w}}"
            row += f"   (N={count})"
            print(row)
        if ws != ws_values[-1]:
            print()


def _build_improvement_rows(
    table: dict[tuple[int, int], dict[str, np.ndarray]],
    n_values: list[int],
    ws_values: list[int],
    alg_names: list[str],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    header_algs = [a for a in alg_names if a != "DONF"]
    for ws in ws_values:
        for n in n_values:
            key = (n, ws)
            if key not in table:
                continue
            res = table[key]
            donf = res["DONF"]
            for alg in header_algs:
                if alg not in res:
                    continue
                rows.append({
                    "workspace": ws,
                    "dag_size": n,
                    "metric": "improvement_over_donf_pct",
                    "algorithm": alg,
                    "value": _pct_improvement(donf, res[alg]),
                    "instances": int(len(donf)),
                })
    return rows


def _build_milp_rows(
    table: dict[tuple[int, int], dict[str, np.ndarray]],
    n_values: list[int],
    ws_values: list[int],
    alg_names: list[str],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    header_algs = [a for a in alg_names if a != "MILP"]
    for ws in ws_values:
        for n in n_values:
            key = (n, ws)
            if key not in table:
                continue
            res = table[key]
            milp = res["MILP"]
            count = int(len(milp))
            if count == 0:
                continue
            for alg in header_algs:
                if alg not in res:
                    continue
                prox = float(np.mean(milp / np.where(res[alg] > 0, res[alg], 1.0) * 100.0))
                rows.append({
                    "workspace": ws,
                    "dag_size": n,
                    "metric": "proximity_to_milp_pct",
                    "algorithm": alg,
                    "value": prox,
                    "instances": count,
                })
    return rows


def _write_csv(path: str, rows: list[dict[str, str | int | float]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["workspace", "dag_size", "metric", "algorithm", "value", "instances"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved csv -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ws",       type=int,  nargs="+", default=[1])
    p.add_argument("--n",        type=int,  nargs="+", default=[30])
    p.add_argument("--nn_dir",   type=str,  default=None,
                   help="directory containing nn_ws{ws}_n{n}.npy files")
    p.add_argument("--mcts_dir", type=str,  default=None,
                   help="directory containing mcts_ws{ws}_n{n}.npy files")
    p.add_argument("--k",        type=int,  default=5)
    p.add_argument("--iter",     type=int,  default=20)
    p.add_argument("--h",        type=int,  default=5)
    p.add_argument("--milp_solver", choices=["highs", "gurobi"], default=None,
                   help="optional MILP reference backend")
    p.add_argument("--milp_limit", type=int, default=0,
                   help="solve MILP only on the first N test DAGs per cell")
    p.add_argument("--milp_time_limit", type=float, default=900.0)
    p.add_argument("--milp_gap", type=float, default=0.03)
    p.add_argument("--milp_threads", type=int, default=None)
    p.add_argument("--milp_seed", type=int, default=0)
    p.add_argument("--milp_tee", action="store_true")
    p.add_argument("--summary_csv", type=str, default=None,
                   help="optional CSV for the improvement-over-DONF summary table")
    p.add_argument("--milp_summary_csv", type=str, default=None,
                   help="optional CSV for the MILP proximity summary table")
    args = p.parse_args()

    table: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    milp_table: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    all_alg_names = list(BASELINES.keys())
    nn_seen = mcts_seen = False

    for ws in args.ws:
        for n in args.n:
            print(f"Evaluating WS{ws} n={n} ({N_TEST_PER_CLASS * 48} DAGs) …", flush=True)

            nn_w = mcts_w = None
            if args.nn_dir:
                path = Path(args.nn_dir) / f"nn_ws{ws}_n{n}.npy"
                if path.exists():
                    nn_w = np.load(path).astype(np.float32)
                    nn_seen = True
            if args.mcts_dir:
                path = Path(args.mcts_dir) / f"mcts_ws{ws}_n{n}.npy"
                if path.exists():
                    mcts_w = np.load(path).astype(np.float32)
                    mcts_seen = True

            cell_results, cell_milp = evaluate_cell(
                n, ws, nn_w, mcts_w,
                mcts_k=args.k, mcts_iter=args.iter, mcts_h=args.h,
                milp_solver=args.milp_solver,
                milp_limit=args.milp_limit,
                milp_time_limit=args.milp_time_limit,
                milp_gap=args.milp_gap,
                milp_threads=args.milp_threads,
                milp_seed=args.milp_seed,
                milp_tee=args.milp_tee,
            )
            table[(n, ws)] = cell_results
            if cell_milp is not None:
                milp_table[(n, ws)] = cell_milp

    if nn_seen and "NN" not in all_alg_names:
        all_alg_names.append("NN")
    if mcts_seen and "MCTS" not in all_alg_names:
        all_alg_names.append("MCTS")

    print("\n=== Mean % improvement over DONF (positive = better than DONF) ===")
    print_table(table, args.n, args.ws, all_alg_names)
    if args.summary_csv:
        _write_csv(
            args.summary_csv,
            _build_improvement_rows(table, args.n, args.ws, all_alg_names),
        )

    if milp_table:
        milp_alg_names = ["MILP"] + all_alg_names
        print("\n=== Mean proximity to MILP reference (higher = closer to MILP) ===")
        print_milp_proximity_table(milp_table, args.n, args.ws, milp_alg_names)
        if args.milp_summary_csv:
            _write_csv(
                args.milp_summary_csv,
                _build_milp_rows(milp_table, args.n, args.ws, milp_alg_names),
            )
