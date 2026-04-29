import unittest

from dag_scheduling.data.generator import generate_dot


class GeneratorTests(unittest.TestCase):
    def test_seeded_daggen_output_is_reproducible(self):
        kwargs = dict(
            n=20,
            fat=0.5,
            regular=0.8,
            density=0.4,
            jump=2,
            ccr=2,
            seed=123,
        )
        try:
            first = generate_dot(**kwargs)
            second = generate_dot(**kwargs)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        def body(dot: str) -> str:
            return "\n".join(
                line for line in dot.splitlines() if not line.startswith("//")
            )

        self.assertEqual(body(first), body(second))


if __name__ == "__main__":
    unittest.main()
