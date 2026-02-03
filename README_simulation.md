## Data-Generating Process (DGP)

The simulation supports three DGPs: linear, nonlinear (polynomial), and trigonometric. Users can choose which DGP to use via the `--dgp` argument (1, 2, or 3).

### Linear DGP (DGP 1, Default)

- **Covariates:**
   - $X_1 \sim N(0, 1)$
   - $X_2 \sim N(0, 1)$
- **Treatment:**
   - $D \sim \text{Bernoulli}(0.5)$ (random assignment)
- **True CATEs (linear in X):**
   - Surrogate: $\tau_S = 1.0 + 0.5 X_1 - 0.3 X_2$
   - Outcome: $\tau_Y = 0.5 + 0.2 X_1 + 0.4 X_2$
- **Potential Outcomes:**
   - Surrogate: $S_0 = \mu_S + \varepsilon_S$, $S_1 = S_0 + \tau_S$
   - Outcome: $Y_0 = \mu_Y + \varepsilon_Y$, $Y_1 = Y_0 + \tau_Y$
   - $\mu_S = 2.0 + X_1 + X_2$, $\mu_Y = 1.0 + 0.5 X_1 - 0.5 X_2$
   - $\varepsilon_S, \varepsilon_Y \sim N(0, 1)$
- **Observed Data:**
   - $S = D \cdot S_1 + (1-D) \cdot S_0$
   - $Y = D \cdot Y_1 + (1-D) \cdot Y_0$

### Nonlinear DGP (DGP 2, Polynomial with Cubic Terms)

- **Covariates and Treatment:** Same as linear DGP
- **True CATEs (highly nonlinear in X with cubic and interaction terms):**
   - Surrogate: $\tau_S = 1.0 + 0.4 X_1 - 0.5 X_2 + 0.6 X_1^2 - 0.5 X_2^2 + 0.4 X_1^3 - 0.35 X_2^3 + 0.5 X_1 X_2 + 0.3 X_1^2 X_2 - 0.3 X_1 X_2^2$
   - Outcome: $\tau_Y = 0.6 + 0.3 X_1 + 0.4 X_2 - 0.5 X_1^2 + 0.4 X_2^2 - 0.35 X_1^3 + 0.3 X_2^3 - 0.4 X_1 X_2 - 0.25 X_1^2 X_2 + 0.25 X_1 X_2^2$
- **Potential Outcomes:**
   - $\mu_S = 2.0 + 0.6 X_1 + 0.5 X_2 + 0.4 X_1^2 - 0.3 X_2^2 + 0.2 X_1^3 - 0.15 X_2^3$
   - $\mu_Y = 1.5 + 0.4 X_1 - 0.35 X_2 - 0.3 X_1^2 + 0.25 X_2^2 - 0.15 X_1^3 + 0.1 X_2^3$
   - Outcomes generated analogously to linear DGP

### Trigonometric DGP (DGP 3, Sin/Cos with Higher-Order Interactions)

- **Covariates and Treatment:** Same as linear DGP
- **True CATEs (highly nonlinear in X with trigonometric and interaction terms):**
   - Surrogate: $\tau_S = 1.0 + 1.2 \sin(X_1) + 1.0 \cos(X_2) + 0.8 \sin(X_1) \cos(X_2) + 0.5 X_1^2 \sin(X_2) + 0.4 X_2^2 \cos(X_1)$
   - Outcome: $\tau_Y = 0.5 + 0.9 \cos(X_1) + 0.7 \sin(X_2) - 0.6 \sin(X_1) \cos(X_2) - 0.4 X_1^2 \sin(X_1) + 0.35 X_2^2 \cos(X_2)$
- **Potential Outcomes:**
   - $\mu_S = 2.2 + 0.9 \sin(X_1) + 0.7 \cos(X_2) + 0.4 X_1^2 \cos(X_1)$
   - $\mu_Y = 1.4 + 0.6 \cos(X_1) - 0.5 \sin(X_2) - 0.3 X_2^2 \sin(X_2)$
   - Outcomes generated analogously to linear DGP

This structure ensures that the true CATEs and welfare parameter are known for each sample, allowing precise evaluation of plugin estimators.
# Monte Carlo Simulation Procedure

This README describes the general procedure implemented in the Monte Carlo simulation for CATE (Conditional Average Treatment Effect) and welfare estimation in this project.

## Overview
The main goal of this simulation framework is to examine the performance of plugin estimators for the welfare parameter. Different CATE estimation methods (e.g., Causal Forest, T-learner) are used as inputs to the plugin formula, but the focus is on the accuracy and properties of the welfare estimate itself, not on CATE estimation per se.

## Simulation Steps

1. **Data Generation**
   - For each simulation run, synthetic data is generated using a known data-generating process (DGP).
   - The data includes covariates (X), a binary treatment indicator (D), and potential outcomes.
   - The true CATE (tau) and welfare values are known for each sample.

2. **Estimator Fitting**
   - For each run, a CATE estimator (e.g., Causal Forest or T-learner) is fit to the generated data.
   - The estimator predicts individual treatment effects (CATE estimates) for each sample.

3. **Welfare Calculation**
    - **True Welfare** ($W_{\text{true}}$) is computed once before the simulations using a large sample from the DGP, not for each run.
    - For each run, we compute **Estimated Welfare** ($W_{\text{hat}}$) using the model-estimated CATEs.
    - Both are computed as sample averages over the subset of individuals for whom both $\tau_S > 0$ and $\tau_Y > 0$ (or their estimates).
    - The formulas are:

       - True welfare (computed once):
          $$
          W_{\text{true}} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \left[ \tau_{S,i} + \delta \cdot \tau_{Y,i} \right]
          $$
          where $\mathcal{I} = \{i : \tau_{S,i} > 0,\ \tau_{Y,i} > 0\}$

       - Estimated (plugin) welfare (computed for each run):
          $$
          W_{\text{hat}} = \frac{1}{|\hat{\mathcal{I}}|} \sum_{i \in \hat{\mathcal{I}}} \left[ \hat{\tau}_{S,i} + \delta \cdot \hat{\tau}_{Y,i} \right]
          $$
          where $\hat{\mathcal{I}} = \{i : \hat{\tau}_{S,i} > 0,\ \hat{\tau}_{Y,i} > 0\}$

    - Here, $\tau_{S,i}$ and $\tau_{Y,i}$ are the true CATEs for surrogate and outcome, and $\hat{\tau}_{S,i}$, $\hat{\tau}_{Y,i}$ are the model estimates.
    - $\delta$ is a known constant from the DGP.
    - The **welfare gap** for each run is:
       $$
        	\text{Welfare Gap} = W_{\text{hat}} - W_{\text{true}}
       $$
    - This gap summarizes the estimation error in terms of practical welfare impact, using sample averages and model estimates as described above.

4. **Repetition and Aggregation**
   - Steps 1–3 are repeated for a specified number of Monte Carlo runs (R) and for different sample sizes (n).
   - For each configuration, the empirical distribution of the welfare gap and CATE estimates is collected.

5. **Result Summarization**
   - After all runs, the simulation reports summary statistics (mean, sample standard deviation) of the welfare gap for each configuration.
   - The empirical distribution of standardized estimates is visualized and compared to the standard normal distribution.

## Estimand of Interest
**Welfare Parameter (Plugin Estimator):**
   - The main estimand is the welfare parameter, not the CATE itself.
   - Different CATE estimators are used as plugin inputs, but the focus is on how well the plugin formula recovers the true welfare value.

**Welfare Gap:**
   - The difference between the plugin-estimated and true welfare, summarizing the practical impact of estimation error.

## Usage
- Configure simulation parameters (estimator, sample size, repetitions) in the Makefile or via command line.
- Run the simulation using `make run-mc`.
- Visualize results using `make plot` or view pop-up plots after simulation.

For further details on the estimators, see the code in the `estimation/` directory.
