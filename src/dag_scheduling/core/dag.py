"""
SchedulingDAG: thin wrapper around rustworkx.PyDiGraph.

Node payload:  {"id": int, "compute_cost": float, "alpha": float, "node_type": str | None}
Edge payload:  {"comm_cost": float}

rustworkx indices are stable (graph is read-only after construction), so they
are used directly as keys throughout the scheduler.
"""

from __future__ import annotations
from typing import Iterator
import rustworkx as rx


NODE_TYPES = ("CPU", "GPU", "IO")


class SchedulingDAG:
    def __init__(self) -> None:
        self._g: rx.PyDiGraph = rx.PyDiGraph(check_cycle=False)
        # daggen node id (1-based int) -> rustworkx index
        self._id_to_idx: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Construction helpers (used by parser)
    # ------------------------------------------------------------------

    def add_task(self, node_id: int, compute_cost: float, alpha: float,
                 node_type: str | None = None) -> int:
        idx = self._g.add_node({
            "id": node_id,
            "compute_cost": compute_cost,
            "alpha": alpha,
            "node_type": node_type,
        })
        self._id_to_idx[node_id] = idx
        return idx

    def add_dependency(self, src_id: int, dst_id: int, comm_cost: float) -> None:
        self._g.add_edge(
            self._id_to_idx[src_id],
            self._id_to_idx[dst_id],
            {"comm_cost": comm_cost},
        )

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._g)

    def indices(self) -> list[int]:
        return list(self._g.node_indices())

    def node_id(self, idx: int) -> int:
        return self._g[idx]["id"]

    def compute_cost(self, idx: int) -> float:
        return self._g[idx]["compute_cost"]

    def alpha(self, idx: int) -> float:
        return self._g[idx]["alpha"]

    def node_type(self, idx: int) -> str | None:
        return self._g[idx]["node_type"]

    def set_node_type(self, idx: int, node_type: str) -> None:
        self._g[idx]["node_type"] = node_type

    def comm_cost(self, src_idx: int, dst_idx: int) -> float:
        return self._g.get_edge_data(src_idx, dst_idx)["comm_cost"]

    def predecessors(self, idx: int) -> list[int]:
        return list(self._g.predecessor_indices(idx))

    def successors(self, idx: int) -> list[int]:
        return list(self._g.successor_indices(idx))

    def in_degree(self, idx: int) -> int:
        return self._g.in_degree(idx)

    def out_degree(self, idx: int) -> int:
        return self._g.out_degree(idx)

    def in_edges(self, idx: int) -> list[tuple[int, float]]:
        """Returns list of (src_idx, comm_cost)."""
        return [(e[0], e[2]["comm_cost"]) for e in self._g.in_edges(idx)]

    def out_edges(self, idx: int) -> list[tuple[int, float]]:
        """Returns list of (dst_idx, comm_cost)."""
        return [(e[1], e[2]["comm_cost"]) for e in self._g.out_edges(idx)]

    def topological_order(self) -> list[int]:
        return rx.topological_sort(self._g)

    def entry_nodes(self) -> list[int]:
        """Tasks with no predecessors."""
        return [i for i in self._g.node_indices() if self._g.in_degree(i) == 0]

    def exit_nodes(self) -> list[int]:
        """Tasks with no successors."""
        return [i for i in self._g.node_indices() if self._g.out_degree(i) == 0]

    def processing_time(self, idx: int) -> float:
        """Processing time = raw compute cost (no speed factor)."""
        return self._g[idx]["compute_cost"]

    def idx_from_id(self, node_id: int) -> int:
        return self._id_to_idx[node_id]

    # ------------------------------------------------------------------
    # Convenience iteration
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[tuple[int, dict]]:
        for idx in self._g.node_indices():
            yield idx, self._g[idx]
