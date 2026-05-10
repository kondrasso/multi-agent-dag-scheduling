import unittest

import numpy as np

from dag_scheduling.algorithms.mcts.search import _Node, _simulate
from dag_scheduling.algorithms.nn.model import CHROMOSOME_LEN
from dag_scheduling.baselines.heuristics import _schedule_greedy
from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import Executor, Platform
from dag_scheduling.core.simulator import ScheduleState


class ReferenceFidelityTests(unittest.TestCase):
    def test_mcts_simulation_backs_up_completed_rollout_makespan(self):
        dag = SchedulingDAG()
        dag.add_task(1, 26_000_000_000.0, 0.1, "CPU")
        dag.add_task(2, 26_000_000_000.0, 0.1, "CPU")
        dag.add_dependency(1, 2, 0.0)
        platform = Platform([Executor(id=1, executor_type="CPU", gflops=26.0)])

        state = ScheduleState(dag, platform)
        node = _Node(state, parent=None, task_idx=None, candidates=list(state.ready))
        metrics = np.zeros((max(dag.indices()) + 1, 13), dtype=np.float32)
        weights = np.zeros(CHROMOSOME_LEN, dtype=np.float32)

        value = _simulate(node, metrics, weights, h=0, k=5)

        self.assertAlmostEqual(value, -2.0, places=6)

    def test_insertion_placement_fills_idle_gap_before_later_task(self):
        dag = SchedulingDAG()
        dag.add_task(1, 10_000_000_000.0, 0.1, "GPU")
        dag.add_task(2, 1_000_000_000.0, 0.1, "CPU")
        dag.add_task(3, 5_000_000_000.0, 0.1, "CPU")
        dag.add_dependency(1, 2, 0.0)
        platform = Platform(
            [
                Executor(id=1, executor_type="GPU", gflops=1.0),
                Executor(id=2, executor_type="CPU", gflops=1.0),
            ]
        )

        priority = {
            dag.idx_from_id(1): 3.0,
            dag.idx_from_id(2): 2.0,
            dag.idx_from_id(3): 1.0,
        }

        standard = _schedule_greedy(dag, platform, priority)
        insertion = _schedule_greedy(dag, platform, priority, insertion=True)

        self.assertAlmostEqual(standard, 16.0, places=6)
        self.assertAlmostEqual(insertion, 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
