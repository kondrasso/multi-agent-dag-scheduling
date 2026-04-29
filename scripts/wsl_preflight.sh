#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not installed or not on PATH" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "repo_root=${REPO_ROOT}"
echo "utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "hostname=$(hostname)"
echo "shell=${SHELL:-unknown}"
echo "wsl_distro=${WSL_DISTRO_NAME:-}"
echo "kernel=$(uname -r)"
echo "machine=$(uname -m)"
echo

CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
echo "cpu_count=${CPU_COUNT}"
if [[ -r /proc/meminfo ]]; then
  MEM_KB=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
  SWAP_KB=$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)
  echo "mem_total_gib=$(awk -v kb="${MEM_KB}" 'BEGIN {printf "%.2f", kb/1024/1024}')"
  echo "swap_total_gib=$(awk -v kb="${SWAP_KB}" 'BEGIN {printf "%.2f", kb/1024/1024}')"
fi
echo "recommended_cpu_jobs=$(( CPU_COUNT > 2 ? CPU_COUNT - 2 : 1 ))"
echo

echo "[build_tools]"
report_tool() {
  local label=$1
  local binary=$2
  local path
  local output
  if path=$(command -v "${binary}" 2>/dev/null); then
    echo "${label}_path=${path}"
    if output=$("${binary}" --version 2>&1); then
      echo "${label}_version=${output%%$'\n'*}"
    else
      echo "${label}_version=error"
    fi
  else
    echo "${label}=not_found"
  fi
}
report_tool make make
report_tool cc cc
report_tool gcc gcc
report_tool cxx c++
report_tool gxx g++
report_tool clangxx clang++
echo

echo "[repo_assets]"
if [[ -d "${REPO_ROOT}/daggen" ]]; then
  echo "daggen_source=present"
else
  echo "daggen_source=missing"
fi
if [[ -x "${REPO_ROOT}/daggen/daggen" ]]; then
  echo "daggen_binary=${REPO_ROOT}/daggen/daggen"
elif command -v daggen >/dev/null 2>&1; then
  echo "daggen_binary=$(command -v daggen)"
else
  echo "daggen_binary=missing"
fi
if [[ -f "${REPO_ROOT}/cpp/mlvp/Makefile" ]]; then
  echo "mlvp_makefile=present"
else
  echo "mlvp_makefile=missing"
fi
echo

echo "[gpu]"
run_nvidia_smi() {
  local smi_bin=$1
  local output
  if output=$("${smi_bin}" \
    --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>&1); then
    echo "${output}"
  else
    echo "nvidia_smi=error"
    while IFS= read -r line; do
      if [[ -n "${line}" ]]; then
        echo "nvidia_smi_error=${line}"
      fi
    done <<< "${output}"
  fi
}

if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  run_nvidia_smi /usr/lib/wsl/lib/nvidia-smi
elif command -v nvidia-smi >/dev/null 2>&1; then
  run_nvidia_smi nvidia-smi
else
  echo "nvidia_smi=not_found"
fi
echo

uv run python - <<'PY'
import os
import sys
from typing import Callable

print("[python]")
print(f"python_version={sys.version.split()[0]}")
print(f"executable={sys.executable}")

def maybe_import(name: str):
    try:
        module = __import__(name)
        print(f"{name}_import=ok")
        version = getattr(module, "__version__", None)
        if version is not None:
            print(f"{name}_version={version}")
        return module
    except Exception as exc:
        print(f"{name}_import=error:{type(exc).__name__}:{exc}")
        return None

torch = maybe_import("torch")
ray = maybe_import("ray")
maybe_import("pyomo")
print()

print("[torch_cuda]")
if torch is None:
    print("torch_cuda_available=unknown")
else:
    try:
        available = bool(torch.cuda.is_available())
        print(f"torch_cuda_available={available}")
        print(f"torch_cuda_device_count={torch.cuda.device_count()}")
        if available:
            for idx in range(torch.cuda.device_count()):
                print(f"torch_cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch_cuda_error={type(exc).__name__}:{exc}")
print()

print("[repo_checks]")
try:
    from dag_scheduling.milp.solve import MilpSolverConfig, is_solver_available, solve_milp
    from dag_scheduling.core.dag import SchedulingDAG
    from dag_scheduling.core.platform import Executor, Platform

    print(f"milp_highs_available={is_solver_available('highs')}")
    print(f"milp_gurobi_available={is_solver_available('gurobi')}")

    def tiny_instance():
        dag = SchedulingDAG()
        dag.add_task(1, 1_000_000_000.0, 0.1, "CPU")
        dag.add_task(2, 1_000_000_000.0, 0.1, "CPU")
        dag.add_dependency(1, 2, 0.0)
        platform = Platform([
            Executor(id=1, executor_type="CPU", gflops=1.0),
            Executor(id=2, executor_type="CPU", gflops=1.0),
        ])
        return dag, platform

    for solver in ("highs", "gurobi"):
        if not is_solver_available(solver):
            print(f"{solver}_tiny_solve=skipped")
            continue
        dag, platform = tiny_instance()
        try:
            solution = solve_milp(
                dag,
                platform,
                MilpSolverConfig(
                    solver=solver,
                    time_limit=10.0,
                    mip_gap=0.0,
                    threads=1,
                    seed=0,
                    tee=False,
                ),
            )
            print(
                f"{solver}_tiny_solve=status:{solution.status}:"
                f"{solution.termination_condition}:makespan={solution.makespan}"
            )
        except Exception as exc:
            print(f"{solver}_tiny_solve=error:{type(exc).__name__}:{exc}")
except Exception as exc:
    print(f"repo_checks=error:{type(exc).__name__}:{exc}")

if ray is not None:
    print()
    print("[ray]")
    try:
        print(f"ray_local_mode_ok=True")
    except Exception as exc:
        print(f"ray_error={type(exc).__name__}:{exc}")
PY
