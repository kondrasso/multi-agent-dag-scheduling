"""
Thin wrapper around the daggen binary.

Generates DAGs and returns them as SchedulingDAG objects or raw .dot strings.
The daggen binary is expected at daggen/daggen relative to the project root,
or on PATH as 'daggen'.
"""

from __future__ import annotations
import subprocess
import shutil
from pathlib import Path

from dag_scheduling.data.parser import parse_dot_str
from dag_scheduling.core.dag import SchedulingDAG

# project root is two levels above this file's package
_HERE = Path(__file__).resolve().parent
_DEFAULT_BINARY = _HERE.parents[2] / "daggen" / "daggen"


def _find_binary() -> str:
    if _DEFAULT_BINARY.exists():
        return str(_DEFAULT_BINARY)
    found = shutil.which("daggen")
    if found:
        return found
    raise FileNotFoundError(
        f"daggen binary not found at {_DEFAULT_BINARY} and not on PATH. "
        "Run 'make' inside the daggen/ directory first."
    )


def generate_dot(
    n: int = 100,
    fat: float = 0.5,
    regular: float = 0.9,
    density: float = 0.5,
    jump: int = 1,
    ccr: int = 0,
    minalpha: float = 0.0,
    maxalpha: float = 0.2,
    mindata: int = 2048,
    maxdata: int = 11264,
    seed: int | None = None,
) -> str:
    """Run daggen and return the raw .dot string."""
    binary = _find_binary()
    cmd = [
        binary,
        "--dot",
        "-n", str(n),
        "--fat", str(fat),
        "--regular", str(regular),
        "--density", str(density),
        "--jump", str(jump),
        "--ccr", str(ccr),
        "--minalpha", str(minalpha),
        "--maxalpha", str(maxalpha),
        "--mindata", str(mindata),
        "--maxdata", str(maxdata),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def generate(
    n: int = 100,
    fat: float = 0.5,
    regular: float = 0.9,
    density: float = 0.5,
    jump: int = 1,
    ccr: int = 0,
    minalpha: float = 0.0,
    maxalpha: float = 0.2,
    mindata: int = 2048,
    maxdata: int = 11264,
    seed: int | None = None,
) -> SchedulingDAG:
    """Generate a DAG and return a parsed SchedulingDAG."""
    dot = generate_dot(
        n=n, fat=fat, regular=regular, density=density,
        jump=jump, ccr=ccr, minalpha=minalpha, maxalpha=maxalpha,
        mindata=mindata, maxdata=maxdata, seed=seed,
    )
    return parse_dot_str(dot)


def generate_to_file(path: str | Path, **kwargs) -> Path:
    """Generate a DAG and write the .dot file to disk."""
    dot = generate_dot(**kwargs)
    path = Path(path)
    path.write_text(dot)
    return path
