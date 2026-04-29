"""
The 13 graph scheduling metrics used by the protocol.

Index  Name           Formula / description
-----  -------------  ---------------------------------------------------
0      WOD            sum_{vj in succ(vi)} 1/ID(vj)
1      WOD_2          sum_{vj in succ(vi)} [1/ID(vj) + a*sum_{vk in succ(vj)} 1/ID(vk)]
2      rank           HEFT upward rank using per-platform average w_bar and b_bar
3      C              average processing time over compatible executors
4      |pred|         in-degree
5      |succ|         out-degree
6      TW_in          sum of incoming edge data volumes
7      TW_out         sum of outgoing edge data volumes
8      N1_succ        number of type-1 (CPU) successors
9      N2_succ        number of type-2 (GPU) successors
10     N3_succ        number of type-3 (IO) successors
11     SPD            shortest directed path to any exit node (in edges)
12     LPD            longest directed path to any exit node (in edges)

rank uses:
  w_bar(vi) = avg processing time of vi over compatible executors in the platform
  b_bar(e)  = data_volume(e) / BANDWIDTH  (fixed bandwidth → no averaging needed)

WOD_2 alpha (two-hop weight) defaults to WOD2_ALPHA = 0.5.
"""

from __future__ import annotations
import numpy as np

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Platform, BANDWIDTH

N_METRICS = 13
WOD2_ALPHA = 0.5  # two-hop contribution weight

METRIC_NAMES = [
    "WOD", "WOD_2", "rank", "C",
    "in_degree", "out_degree",
    "TW_in", "TW_out",
    "N1_succ", "N2_succ", "N3_succ",
    "SPD", "LPD",
]

# Priority-rule actions: 9 metric-based + 1 no-op = 10 total.
# Each entry is (metric_index, higher_is_better)
PRIORITY_RULES: list[tuple[int, bool]] = [
    (0,  True),   # 0: maximize WOD
    (1,  True),   # 1: maximize WOD_2
    (2,  True),   # 2: maximize rank
    (3,  True),   # 3: maximize C
    (3,  False),  # 4: minimize C
    (8,  True),   # 5: maximize N1_succ (type 1 = CPU)
    (9,  True),   # 6: maximize N2_succ (type 2 = GPU)
    (10, True),   # 7: maximize N3_succ (type 3 = IO)
    (6,  False),  # 8: minimize TW_in
]
NO_OP_ACTION = 9


def compute_metrics(dag: SchedulingDAG, platform: Platform) -> np.ndarray:
    """
    Returns ndarray shape (max_idx+1, N_METRICS); row i corresponds to
    rustworkx node index i (sparse — rows for unused indices are zero).
    Values are raw (not normalised).
    """
    indices = dag.indices()
    if not indices:
        return np.zeros((0, N_METRICS), dtype=np.float64)

    max_idx = max(indices) + 1
    M = np.zeros((max_idx, N_METRICS), dtype=np.float64)

    topo = dag.topological_order()   # list of idx in topological order

    # ------------------------------------------------------------------
    # SPD and LPD: hop-count paths to any exit node
    # Computed in reverse topological order.
    # ------------------------------------------------------------------
    exit_set = set(dag.exit_nodes())
    spd: dict[int, int] = {}
    lpd: dict[int, int] = {}

    for idx in reversed(topo):
        if idx in exit_set:
            spd[idx] = 0
            lpd[idx] = 0
        else:
            succs = dag.successors(idx)
            if not succs:
                spd[idx] = 0
                lpd[idx] = 0
            else:
                spd[idx] = 1 + min(spd.get(s, 0) for s in succs)
                lpd[idx] = 1 + max(lpd.get(s, 0) for s in succs)

    # ------------------------------------------------------------------
    # Upward rank (HEFT-style):
    #   rank(vi) = w_bar(vi) + max_{vj in succ(vi)} [b_bar(i,j) + rank(vj)]
    # Computed in reverse topological order.
    # ------------------------------------------------------------------
    rank: dict[int, float] = {}

    for idx in reversed(topo):
        node_type = dag.node_type(idx)
        w_bar = platform.avg_processing_time(node_type, dag.compute_cost(idx))

        succs = dag.successors(idx)
        if not succs:
            rank[idx] = w_bar
        else:
            rank[idx] = w_bar + max(
                dag.comm_cost(idx, s) / BANDWIDTH + rank.get(s, 0.0)
                for s in succs
            )

    # ------------------------------------------------------------------
    # Per-node metrics
    # ------------------------------------------------------------------
    for idx in indices:
        node_type = dag.node_type(idx)
        succs = dag.successors(idx)
        preds = dag.predecessors(idx)

        in_deg  = len(preds)
        out_deg = len(succs)
        tw_in   = sum(dag.comm_cost(p, idx) for p in preds)
        tw_out  = sum(dag.comm_cost(idx, s) for s in succs)

        # WOD: sum 1/ID(succ) over successors
        wod = sum(
            1.0 / dag.in_degree(s)
            for s in succs if dag.in_degree(s) > 0
        )

        # WOD_2: one-hop + alpha * two-hop
        wod2 = 0.0
        for s in succs:
            s_in = dag.in_degree(s)
            hop1 = 1.0 / s_in if s_in > 0 else 0.0
            hop2 = sum(
                1.0 / dag.in_degree(k)
                for k in dag.successors(s) if dag.in_degree(k) > 0
            )
            wod2 += hop1 + WOD2_ALPHA * hop2

        # C: average processing time over compatible executors
        c = platform.avg_processing_time(node_type, dag.compute_cost(idx))

        # successor type counts
        n1 = n2 = n3 = 0
        for s in succs:
            t = dag.node_type(s)
            if t == "CPU":
                n1 += 1
            elif t == "GPU":
                n2 += 1
            elif t == "IO":
                n3 += 1

        M[idx] = [
            wod,              # 0  WOD
            wod2,             # 1  WOD_2
            rank[idx],        # 2  rank
            c,                # 3  C
            float(in_deg),    # 4  |pred|
            float(out_deg),   # 5  |succ|
            tw_in,            # 6  TW_in
            tw_out,           # 7  TW_out
            float(n1),        # 8  N1_succ (CPU)
            float(n2),        # 9  N2_succ (GPU)
            float(n3),        # 10 N3_succ (IO)
            float(spd[idx]),  # 11 SPD
            float(lpd[idx]),  # 12 LPD
        ]

    return M


def normalise(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-column min-max normalisation to [0, 1]."""
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    return (M - lo) / np.maximum(hi - lo, eps)
