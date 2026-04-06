import unittest

from dag_scheduling.data.augmentor import inject_node_types_dot
from dag_scheduling.data.parser import parse_dot_str


RAW_DOT = """digraph G {
1 [size="10", alpha="0.10"]
2 [size="20", alpha="0.20"]
1 -> 2 [size ="5"]
}
"""


class ParserRoundTripTests(unittest.TestCase):
    def test_parser_reads_node_type_attribute(self):
        typed_dot = """digraph G {
1 [size="10", alpha="0.10", node_type="CPU"]
2 [size="20", alpha="0.20", node_type="GPU"]
1 -> 2 [size ="5"]
}
"""
        dag = parse_dot_str(typed_dot)

        self.assertEqual(dag.node_type(dag.idx_from_id(1)), "CPU")
        self.assertEqual(dag.node_type(dag.idx_from_id(2)), "GPU")

    def test_injected_node_types_round_trip_without_duplication(self):
        dag = parse_dot_str(RAW_DOT)
        dag.set_node_type(dag.idx_from_id(1), "CPU")
        dag.set_node_type(dag.idx_from_id(2), "GPU")

        updated = inject_node_types_dot(RAW_DOT, dag)
        rewritten = inject_node_types_dot(updated, dag)
        parsed = parse_dot_str(rewritten)

        self.assertEqual(rewritten.count('node_type="CPU"'), 1)
        self.assertEqual(rewritten.count('node_type="GPU"'), 1)
        self.assertEqual(parsed.node_type(parsed.idx_from_id(1)), "CPU")
        self.assertEqual(parsed.node_type(parsed.idx_from_id(2)), "GPU")


if __name__ == "__main__":
    unittest.main()
