"""
Assign node_type (CPU / GPU / IO) to each task in a SchedulingDAG.

Two strategies:
  - random:     draw uniformly from NODE_TYPES per task
  - alpha_based: use alpha value as a proxy for task characteristics
      high alpha  -> GPU  (highly parallel)
      low alpha, high compute -> IO  (data-heavy)
      otherwise   -> CPU

Also supports serialising the augmented node_type attributes back into a
.dot file string so it can be stored on disk.
"""

from __future__ import annotations
import random
import re
from pathlib import Path

from dag_scheduling.core.dag import SchedulingDAG, NODE_TYPES


def augment_random(dag: SchedulingDAG, seed: int | None = None) -> SchedulingDAG:
    rng = random.Random(seed)
    for idx in dag.indices():
        dag.set_node_type(idx, rng.choice(NODE_TYPES))
    return dag


def augment_alpha_based(dag: SchedulingDAG) -> SchedulingDAG:
    """
    Heuristic: GPU for high parallelism (alpha > 0.15),
               IO  for large data tasks (alpha < 0.05),
               CPU otherwise.
    """
    for idx in dag.indices():
        alpha = dag.alpha(idx)
        if alpha > 0.15:
            dag.set_node_type(idx, "GPU")
        elif alpha < 0.05:
            dag.set_node_type(idx, "IO")
        else:
            dag.set_node_type(idx, "CPU")
    return dag


# ------------------------------------------------------------------
# .dot serialisation with node_type attribute
# ------------------------------------------------------------------

_NODE_LINE_RE = re.compile(r'^(\s*)(\d+)(\s*\[)(.*?)(\]\s*)$')
_NODE_TYPE_RE = re.compile(r'node_type\s*=\s*"[^"]+"')


def inject_node_types_dot(dot_str: str, dag: SchedulingDAG) -> str:
    """
    Insert node_type into the .dot text for each task line, e.g.:
      1 [size="...", alpha="..."]
    becomes:
      1 [size="...", alpha="...", node_type="CPU"]
    """
    # build lookup: daggen node id -> node_type
    id_to_type: dict[int, str] = {}
    for idx in dag.indices():
        id_to_type[dag.node_id(idx)] = dag.node_type(idx) or "CPU"

    lines = []
    for line in dot_str.splitlines():
        m = _NODE_LINE_RE.match(line)
        if m and "->" not in line:
            nid = int(m.group(2))
            ntype = id_to_type.get(nid, "CPU")
            attrs = m.group(4).strip()
            if _NODE_TYPE_RE.search(attrs):
                attrs = _NODE_TYPE_RE.sub(f'node_type="{ntype}"', attrs)
            else:
                attrs = f'{attrs}, node_type="{ntype}"'
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}{attrs}{m.group(5)}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def augment_dot_file(
    path: str | Path,
    strategy: str = "random",
    seed: int | None = None,
) -> str:
    """
    Read a .dot file, augment it with node_types, overwrite in place.
    Returns the updated dot string.
    """
    from dag_scheduling.data.parser import parse_dot

    path = Path(path)
    dag = parse_dot(path)

    if strategy == "random":
        augment_random(dag, seed=seed)
    elif strategy == "alpha":
        augment_alpha_based(dag)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    dot_str = path.read_text()
    updated = inject_node_types_dot(dot_str, dag)
    path.write_text(updated)
    return updated
