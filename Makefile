# Convenience Makefile for common experiment commands

PY=PYTHONPATH=.
PYTHON=/opt/anaconda3/envs/DoubleThreshold/bin/python
MC=$(PYTHON) experiments/monte_carlo_convergence.py
PLOT=$(PYTHON) experiments/plot_estimates.py

# ============================================================================
# Simulation Configuration (main parameters)
# ============================================================================

METHOD=t_learner         # t_learner or causal_forest
T_EST=spline             # linear or spline (only used for t_learner)
DGP=2                    # 1=linear, 2=nonlinear, 3=trigonometric
R=500                    # Monte Carlo repetitions
NS=1500 3000             # Sample sizes to run
OUT=experiments/results  # Output directory
DELTA=0.9                # Discount factor for welfare

# ============================================================================
# Tuning Parameters (by estimator type)
# ============================================================================

# T-learner with spline (only relevant if T_EST=spline)
SPLINE_KNOTS=5           # Number of spline knots (default: empty for auto)
SPLINE_DEGREE=3          # Degree of spline basis (default: 3)

# Causal forest / Random forest parameters (only relevant if METHOD=causal_forest)
# Note: Causal forest uses random forests under the hood for both the main model
# and auxiliary models (for treatment/outcome estimation), so these parameters
# apply to all forest components.
CF_N_ESTIMATORS=200      # Number of trees in causal forest
CF_MAX_DEPTH=10          # Max tree depth in causal forest
CF_MIN_SAMPLES_LEAF=20   # Min samples per leaf in causal forest

.PHONY: run-mc plot example help

help:
	@echo "Available targets:"
	@echo "  make run-mc              - Run Monte Carlo simulation with current configuration"
	@echo "  make plot                - Plot results (requires RESULT= and VAR=)"
	@echo "  make example             - Run a quick example (small R)"
	@echo ""
	@echo "Configuration variables (override with: make run-mc VAR=value):"
	@echo "  METHOD, T_EST, DGP, R, NS, OUT, DELTA"
	@echo "  SPLINE_KNOTS, SPLINE_DEGREE"
	@echo "  CF_N_ESTIMATORS, CF_MAX_DEPTH, CF_MIN_SAMPLES_LEAF"
	@echo ""
	@echo "Examples:"
	@echo "  make run-mc METHOD=causal_forest CF_N_ESTIMATORS=150"
	@echo "  make run-mc T_EST=linear DGP=1 R=100"

run-mc:
	$(PY) $(MC) \
		--method $(METHOD) \
		--t-estimator $(T_EST) \
		--dgp $(DGP) \
		--R $(R) \
		--ns $(NS) \
		--out-dir $(OUT) \
		--delta $(DELTA) \
		--spline-knots $(if $(SPLINE_KNOTS),$(SPLINE_KNOTS),) \
		--spline-degree $(SPLINE_DEGREE) \
		--cf-n-estimators $(CF_N_ESTIMATORS) \
		--cf-max-depth $(CF_MAX_DEPTH) \
		--cf-min-samples-leaf $(CF_MIN_SAMPLES_LEAF) \
		--progress --progress-percent 10

plot:
	# Usage: make plot RESULT=experiments/results/results_t_learner_linear_n200_R100.npz VAR=tau_S_hat
	if [ -z "$(RESULT)" ]; then echo "Set RESULT=path/to/results.npz"; exit 1; fi
	$(PY) $(PLOT) --npz $(RESULT) --variable $(VAR) --normalize --bins 50

example:
	@echo "Run an example Monte Carlo (small R) and plot results"
	$(PY) $(MC) --method t_learner --t-estimator linear --R 10 --ns 200 --out-dir experiments/tmp_results --progress --progress-percent 10
	$(PY) $(PLOT) --npz experiments/tmp_results/results_t_learner_linear_n200_R10.npz --variable tau_S_hat --normalize
