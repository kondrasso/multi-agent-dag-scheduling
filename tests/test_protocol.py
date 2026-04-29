import unittest

from dag_scheduling.core.dag import SchedulingDAG
from dag_scheduling.protocol import (
    FULL_TOPOLOGIES,
    NN_TOPOLOGIES,
    TEST_PER_CLASS,
    TRAIN_PER_CLASS,
    assign_node_types,
    test_seed_offset,
)


class ProtocolTests(unittest.TestCase):
    def test_topology_grid_sizes_match_protocol(self):
        self.assertEqual(len(NN_TOPOLOGIES), 24)
        self.assertEqual(len(FULL_TOPOLOGIES), 48)
        self.assertEqual(TRAIN_PER_CLASS, 3)
        self.assertEqual(TEST_PER_CLASS, 10)

    def test_test_seed_offset_is_cell_specific(self):
        self.assertEqual(test_seed_offset(30, 1), 130100)
        self.assertNotEqual(test_seed_offset(30, 1), test_seed_offset(30, 2))
        self.assertNotEqual(test_seed_offset(30, 1), test_seed_offset(60, 1))

    def test_alpha_type_assignment_matches_thresholds(self):
        dag = SchedulingDAG()
        dag.add_task(1, compute_cost=1.0, alpha=0.01)
        dag.add_task(2, compute_cost=1.0, alpha=0.10)
        dag.add_task(3, compute_cost=1.0, alpha=0.19)

        assign_node_types(dag, strategy="alpha")

        self.assertEqual(dag.node_type(dag.idx_from_id(1)), "IO")
        self.assertEqual(dag.node_type(dag.idx_from_id(2)), "CPU")
        self.assertEqual(dag.node_type(dag.idx_from_id(3)), "GPU")


if __name__ == "__main__":
    unittest.main()
