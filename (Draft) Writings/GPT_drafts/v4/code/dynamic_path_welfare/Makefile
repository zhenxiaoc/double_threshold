PY ?= python
export PYTHONPATH := src
CFG ?= configs/public_fallback.yaml
JOBS ?= 6

.PHONY: help install test lint search audit estimate boundaries infer simulate robustness report all clean

help:
	@echo "targets: install test lint search audit estimate boundaries infer simulate robustness report all clean"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	-ruff check src tests
	-mypy src/path_welfare || true

search:
	$(PY) -m path_welfare.cli search-data --config configs/dataset_search.yaml

audit:
	$(PY) -m path_welfare.cli audit --config $(CFG)

estimate:
	$(PY) -m path_welfare.cli estimate --config $(CFG)

boundaries:
	$(PY) -m path_welfare.cli boundaries --config $(CFG)

infer:
	$(PY) -m path_welfare.cli infer --config $(CFG) --nboot 499 --jobs $(JOBS)

simulate:
	$(PY) -m path_welfare.cli simulate --config $(CFG) --nrep 1000 --jobs $(JOBS)

robustness:
	$(PY) -m path_welfare.cli robustness --config $(CFG)

bootstrap-mc:
	$(PY) scripts/run_bootstrap_mc.py

report:
	$(PY) -m path_welfare.cli report --config $(CFG)

all: search audit estimate boundaries infer robustness report
	@echo "Full pipeline complete. See results/ and reports/."

clean:
	rm -rf results/tables/* results/figures/* results/logs/* results/simulations/* \
	       reports/empirical_report_auto.md .pytest_cache __pycache__
