import unittest

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import BANDWIDTH, Executor, Platform
from dag_scheduling.milp.solve import MilpSolverConfig, is_solver_available, solve_milp


def _independent_cpu_tasks() -> tuple[SchedulingDAG, Platform]:
    dag = SchedulingDAG()
    dag.add_task(1, 1_000_000_000.0, 0.1, "CPU")
    dag.add_task(2, 2_000_000_000.0, 0.1, "CPU")
    platform = Platform(
        [
            Executor(id=1, executor_type="CPU", gflops=1.0),
            Executor(id=2, executor_type="GPU", gflops=100.0),
        ]
    )
    return dag, platform


def _communication_sensitive_chain() -> tuple[SchedulingDAG, Platform]:
    dag = SchedulingDAG()
    dag.add_task(1, 1_000_000_000.0, 0.1, "CPU")
    dag.add_task(2, 1_000_000_000.0, 0.1, "CPU")
    dag.add_dependency(1, 2, 10.0 * BANDWIDTH)
    platform = Platform(
        [
            Executor(id=1, executor_type="CPU", gflops=1.0),
            Executor(id=2, executor_type="CPU", gflops=1.0),
        ]
    )
    return dag, platform


class MilpSolverTests(unittest.TestCase):
    def test_highs_respects_type_eligibility(self):
        self.assertTrue(is_solver_available("highs"))
        dag, platform = _independent_cpu_tasks()

        solution = solve_milp(
            dag,
            platform,
            MilpSolverConfig(solver="highs", time_limit=30.0, mip_gap=0.0),
        )

        self.assertTrue(solution.has_solution)
        self.assertAlmostEqual(solution.makespan or -1.0, 3.0, places=6)
        self.assertEqual(solution.assignment[0], 1)
        self.assertEqual(solution.assignment[1], 1)

    def test_gurobi_matches_highs_on_chain_instance(self):
        dag, platform = _communication_sensitive_chain()

        highs = solve_milp(
            dag,
            platform,
            MilpSolverConfig(solver="highs", time_limit=30.0, mip_gap=0.0),
        )
        self.assertTrue(highs.has_solution)
        self.assertAlmostEqual(highs.makespan or -1.0, 2.0, places=6)
        self.assertEqual(highs.assignment[0], highs.assignment[1])

        if not is_solver_available("gurobi"):
            self.skipTest("gurobi backend is not available")

        gurobi = solve_milp(
            dag,
            platform,
            MilpSolverConfig(solver="gurobi", time_limit=30.0, mip_gap=0.0),
        )
        self.assertTrue(gurobi.has_solution)
        self.assertAlmostEqual(gurobi.makespan or -1.0, highs.makespan or -1.0, places=6)
        self.assertEqual(gurobi.assignment[0], gurobi.assignment[1])


if __name__ == "__main__":
    unittest.main()
