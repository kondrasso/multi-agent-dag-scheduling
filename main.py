"""Small repository entrypoint for humans running ``python main.py``."""

from dag_scheduling import __version__


def main() -> None:
    print(f"dag_scheduling {__version__}")
    print("Use `uv run python -m dag_scheduling.evaluation.benchmark --help` for benchmarks.")


if __name__ == "__main__":
    main()
