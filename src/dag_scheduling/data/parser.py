"""
Parse daggen .dot output into a SchedulingDAG.

Expected format (daggen --dot):
    digraph G {
      1 [size="1073741824", alpha="0.10"]
      2 [size="536870912", alpha="0.05", node_type="GPU"]
      1 -> 2 [size ="536870912"]
      ...
    }

Node ids are 1-based integers. daggen can emit an edge before the target
node is declared, so we do two passes: collect nodes first, then edges.
The optional ``node_type`` attribute is used for persisted typed DAGs.
"""

from __future__ import annotations
import re
from pathlib import Path

from dag_scheduling.core.dag import SchedulingDAG


_NODE_RE = re.compile(r'^\s*(\d+)\s*\[(.+)\]\s*$')
_EDGE_RE = re.compile(
    r'^\s*(\d+)\s*->\s*(\d+)\s*\[size\s*="([^"]+)"\]'
)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]+)"')


def _parse_lines(lines) -> SchedulingDAG:
    nodes: list[tuple[int, float, float, str | None]] = []
    edges: list[tuple[int, int, float]] = []

    for line in lines:
        em = _EDGE_RE.match(line)
        if em:
            edges.append((int(em.group(1)), int(em.group(2)), float(em.group(3))))
            continue

        nm = _NODE_RE.match(line)
        if not nm:
            continue

        attrs = dict(_ATTR_RE.findall(nm.group(2)))
        if "size" not in attrs or "alpha" not in attrs:
            continue
        nodes.append((
            int(nm.group(1)),
            float(attrs["size"]),
            float(attrs["alpha"]),
            attrs.get("node_type"),
        ))

    dag = SchedulingDAG()
    for node_id, cost, alpha, node_type in nodes:
        dag.add_task(
            node_id=node_id,
            compute_cost=cost,
            alpha=alpha,
            node_type=node_type,
        )
    for src_id, dst_id, comm in edges:
        dag.add_dependency(src_id=src_id, dst_id=dst_id, comm_cost=comm)
    return dag


def parse_dot(path: str | Path) -> SchedulingDAG:
    with open(path) as f:
        return _parse_lines(f)


def parse_dot_str(content: str) -> SchedulingDAG:
    return _parse_lines(content.splitlines())
