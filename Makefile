# Convenience Makefile for common experiment commands

PY=PYTHONPATH=.
PYTHON=/opt/anaconda3/envs/DoubleThreshold/bin/python
MC=$(PYTHON) experiments/monte_carlo_convergence.py
PLOT=$(PYTHON) experiments/plot_estimates.py


# Defaults
METHOD=causal_forest # t_learner or causal_forest
T_EST=linear     # linear, spline
R=500
NS=1500 3000
OUT=experiments/results
DELTA=0.9

.PHONY: run-mc plot example

run-mc:
	$(PY) $(MC) --method $(METHOD) --t-estimator $(T_EST) --R $(R) --ns $(NS) --out-dir $(OUT) --delta $(DELTA) --progress --progress-percent 10

plot:
	# Usage: make plot RESULT=experiments/results/results_t_learner_linear_n200_R100.npz VAR=tau_S_hat
	if [ -z "$(RESULT)" ]; then echo "Set RESULT=path/to/results.npz"; exit 1; fi
	$(PY) $(PLOT) --npz $(RESULT) --variable $(VAR) --normalize --bins 50

example:
	@echo "Run an example Monte Carlo (small R) and plot results"
	$(PY) $(MC) --method t_learner --t-estimator linear --R 10 --ns 200 --out-dir experiments/tmp_results --progress --progress-percent 10
	$(PY) $(PLOT) --npz experiments/tmp_results/results_t_learner_linear_n200_R10.npz --variable tau_S_hat --normalize
