"""
Executor and Platform model.

Processing time:  w_{j,i} = workload(v_j) / GFLOPS(p_i)
Communication:    b_{u,v}  = data_volume(e) / BANDWIDTH_BYTES_PER_SEC

Executor types map to protocol "type 1/2/3":
  CPU  ↔  type 1
  GPU  ↔  type 2
  IO   ↔  type 3

Workspace configs used by the offline experiments.
"""

from __future__ import annotations
from dataclasses import dataclass

# Fixed link bandwidth used throughout the communication model.
BANDWIDTH = 1085e6  # bytes/s


@dataclass
class Executor:
    id: int
    executor_type: str   # "CPU", "GPU", or "IO"
    gflops: float        # computational capability in GFLOPS

    def processing_time(self, workload: float) -> float:
        """w_{j,i} = workload / (GFLOPS * 1e9)"""
        return workload / (self.gflops * 1e9)


@dataclass
class Platform:
    executors: list[Executor]

    def compatible(self, node_type: str) -> list[Executor]:
        return [e for e in self.executors if e.executor_type == node_type]

    def by_id(self, executor_id: int) -> Executor:
        for e in self.executors:
            if e.id == executor_id:
                return e
        raise KeyError(executor_id)

    def avg_processing_time(self, node_type: str, workload: float) -> float:
        """Average processing time over all compatible executors.
        Used for HEFT-style upward rank computation (metric 2)."""
        execs = self.compatible(node_type)
        if not execs:
            return 0.0
        return sum(e.processing_time(workload) for e in execs) / len(execs)

    @property
    def ids(self) -> list[int]:
        return [e.id for e in self.executors]


# ------------------------------------------------------------------
# Workspace presets for small and large protocol grids.
#
# Small-scale (WS1-3): heterogeneous executors, one new triple added per WS
#   WS1: CPU@26, GPU@134, IO@34
#   WS2: WS1 + CPU@50, GPU@70, IO@20
#   WS3: WS2 + CPU@125, GPU@40, IO@60
#
# Large-scale (WS4-9): p executors per type, same GFLOPS pool as WS1.
#   WS4:  1 per type  (3 total),  paired DAG size n=36
#   WS5:  2 per type  (6 total),  n=114
#   WS6:  4 per type  (12 total), n=576
#   WS7:  8 per type  (24 total), n=2400
#   WS8: 16 per type  (48 total), n=9600
#   WS9: 32 per type  (96 total), n=36864
# ------------------------------------------------------------------

_SMALL_SPECS = [
    ("CPU", 26.0),
    ("GPU", 134.0),
    ("IO",  34.0),
    ("CPU", 50.0),
    ("GPU", 70.0),
    ("IO",  20.0),
    ("CPU", 125.0),
    ("GPU", 40.0),
    ("IO",  60.0),
]

# (executors_per_type, paired_n) for WS4..WS9
_LARGE_SCALE = [
    (1,   36),
    (2,  114),
    (4,  576),
    (8,  2400),
    (16, 9600),
    (32, 36864),
]

# Base GFLOPS for each type in large-scale workspaces (same as WS1)
_LARGE_GFLOPS = {"CPU": 26.0, "GPU": 134.0, "IO": 34.0}

# Paired DAG size for each large-scale workspace (indexed 4..9)
LARGE_SCALE_N = {ws: n for ws, (_, n) in enumerate(_LARGE_SCALE, start=4)}


def make_workspace(ws: int) -> Platform:
    """
    Build a Platform for workspace ws (1–9).

    WS1-3: small-scale heterogeneous.
    WS4-9: large-scale homogeneous-per-type.
    """
    if ws in (1, 2, 3):
        specs = _SMALL_SPECS[:ws * 3]
        execs = [
            Executor(id=i, executor_type=t, gflops=g)
            for i, (t, g) in enumerate(specs, start=1)
        ]
        return Platform(execs)

    if ws in range(4, 10):
        p_per_type, _ = _LARGE_SCALE[ws - 4]
        execs, eid = [], 1
        for ntype in ("CPU", "GPU", "IO"):
            g = _LARGE_GFLOPS[ntype]
            for _ in range(p_per_type):
                execs.append(Executor(id=eid, executor_type=ntype, gflops=g))
                eid += 1
        return Platform(execs)

    raise ValueError(f"ws must be 1–9, got {ws}")
