import unittest

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.core.platform import make_workspace
from dag_scheduling.env.offline_env import OfflineSchedulingEnv


def make_small_typed_dag() -> SchedulingDAG:
    dag = SchedulingDAG()
    dag.add_task(1, compute_cost=10.0, alpha=0.1, node_type="CPU")
    dag.add_task(2, compute_cost=10.0, alpha=0.1, node_type="GPU")
    dag.add_task(3, compute_cost=10.0, alpha=0.1, node_type="GPU")
    dag.add_dependency(1, 3, comm_cost=5.0)
    dag.add_dependency(2, 3, comm_cost=5.0)
    return dag


class OfflineEnvTests(unittest.TestCase):
    def test_step_returns_only_active_agents_in_next_observation(self):
        env = OfflineSchedulingEnv(
            {"dag": make_small_typed_dag(), "platform": make_workspace(1)}
        )
        obs, _ = env.reset()
        self.assertEqual(sorted(obs), ["cpu", "gpu"])

        next_obs, reward, terminated, truncated, _ = env.step({"cpu": 0, "gpu": 0})

        self.assertTrue(terminated["cpu"])
        self.assertFalse(terminated["gpu"])
        self.assertFalse(terminated["__all__"])
        self.assertEqual(env.agents, ["gpu"])
        self.assertEqual(sorted(next_obs), ["gpu"])
        self.assertIn("gpu", reward)
        self.assertIn("gpu", truncated)


if __name__ == "__main__":
    unittest.main()
