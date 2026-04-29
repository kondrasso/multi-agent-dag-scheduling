import unittest

from dag_scheduling.data.generator import generate_dot


class GeneratorTests(unittest.TestCase):
    def test_daggen_output_is_parseable_dot(self):
        kwargs = dict(
            n=20,
            fat=0.5,
            regular=0.8,
            density=0.4,
            jump=2,
            ccr=2,
        )
        try:
            dot = generate_dot(**kwargs)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        self.assertIn("digraph G", dot)
        self.assertIn("->", dot)


if __name__ == "__main__":
    unittest.main()
