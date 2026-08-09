"""Smoke + sanity tests for the two-band sieve-DML harm-share estimator.

Run:  PYTHONPATH=src python -m pytest tests/test_dml.py -q
"""
import numpy as np
import pytest

from harm_share.calibration import build_oracle
from harm_share.functionals import mc_truth
from harm_share.sieve_dml import harm_share_riesz_dml, harm_share_dml


@pytest.fixture(scope="module")
def sample():
    orc = build_oracle()
    theta = mc_truth(orc, n_draw=400_000)["theta_harm"]
    df = orc.sample_experiment(3000, np.random.default_rng(0))
    return df, theta


@pytest.mark.parametrize("nuisance", ["sieve", "rf", "gbr"])
def test_riesz_dml_runs_and_covers(sample, nuisance):
    df, theta = sample
    r = harm_share_riesz_dml(df, nuisance=nuisance, seed=0)
    assert 0.0 <= r.theta_dml <= 1.0
    assert r.se > 0
    assert r.ci[0] < r.ci[1]
    # a single 95% interval on a clean draw should cover the exact truth
    assert r.ci[0] - 0.02 <= theta <= r.ci[1] + 0.02


def test_aipw_dml_runs(sample):
    df, theta = sample
    r = harm_share_dml(df, nuisance="rf", K=3, debias=False)
    assert 0.0 <= r.theta_dml <= 1.0 and r.se > 0


def test_rf_scales_to_high_dim():
    """RF nuisance must run when the tensor sieve would be infeasible (d large)."""
    rng = np.random.default_rng(0)
    n, d = 1500, 12
    X = rng.standard_normal((n, d))
    W = (rng.random(n) < 0.5).astype(int)
    tauS = X[:, 0] - 0.3
    S = 2 * X[:, 0] + W * tauS + rng.standard_normal(n)
    Y = X[:, 1] + W * (X[:, 1] - 0.2) + rng.standard_normal(n)
    import pandas as pd
    df = pd.DataFrame({**{f"X{j+1}": X[:, j] for j in range(d)}, "W": W, "S": S, "Y": Y})
    r = harm_share_riesz_dml(df, nuisance="rf", n_features=100, seed=0)
    assert 0.0 <= r.theta_dml <= 1.0 and r.se > 0
