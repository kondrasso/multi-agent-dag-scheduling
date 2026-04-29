.PHONY: all daggen mlvp test smoke smoke-training preflight clean distclean

SMOKE_DIR ?= $(CURDIR)/.smoke

all: daggen mlvp

daggen:
	$(MAKE) -C daggen

mlvp:
	$(MAKE) -C cpp/mlvp all

test: mlvp
	uv run python -m unittest discover -s tests
	$(MAKE) -C cpp/mlvp test

smoke: daggen test
	./cpp/mlvp/build/bin/mlvp_benchmark --generate 2 --n 20 --workspace 1 \
		--assign-types alpha --policies mlvp,donf,fifo,minmin,maxmin

smoke-training: daggen
	mkdir -p $(SMOKE_DIR)
	uv run python -m dag_scheduling.algorithms.nn.train --ws 1 --n 30 \
		--gens 1 --pop 2 --seed 101 --out $(SMOKE_DIR)/nn_smoke.npy
	uv run python -m dag_scheduling.algorithms.mcts.train --ws 1 --n 30 \
		--gens 1 --pop 2 --k 2 --n_iter 1 --h 1 --seed 102 \
		--out $(SMOKE_DIR)/mcts_smoke.npy
	uv run python -m dag_scheduling.algorithms.marl.train --ws 1 --n 30 \
		--iters 1 --out $(SMOKE_DIR)/marl_smoke --gpus 0 --log_every 1; \
		status=$$?; uv run ray stop --force; exit $$status

preflight:
	./scripts/preflight.sh

clean:
	find . \( -path ./.git -o -path ./.venv -o -path ./daggen/.git \) -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . \( -path ./.git -o -path ./.venv -o -path ./daggen/.git \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -exec rm -f {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .uv-cache build dist
	$(MAKE) -C cpp/mlvp clean
	rm -f daggen/*.o

distclean: clean
	rm -rf results cpp/mlvp/runs .smoke
	rm -f daggen/daggen gurobi.log
