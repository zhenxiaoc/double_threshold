"""Property tests for the surrogate-harm-share study.

Run:  PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import pytest

from harm_share.calibration import build_oracle
from harm_share.functionals import mc_truth, grid_truth, analytic_derivative, fd_derivative
from harm_share.estimator import estimate_harm_share, regular_companion_welfare


@pytest.fixture(scope="module")
def orc():
    return build_oracle()


def test_quadrants_sum_to_one(orc):
    tr = mc_truth(orc, n_draw=400_000)
    total = tr["theta_pp"] + tr["theta_harm"] + tr["theta_mp"] + tr["theta_mm"]
    assert abs(total - 1.0) < 1e-9
    assert abs((tr["theta_pp"] + tr["theta_harm"]) - tr["treat_share_S"]) < 1e-9


def test_harm_quadrant_nontrivial(orc):
    tr = mc_truth(orc, n_draw=400_000)
    assert 0.05 < tr["theta_harm"] < 0.30   # real, non-degenerate harm mass
    assert tr["ate_S"] > tr["ate_Y"] > -5   # fade-out preserved from the data


def test_grid_matches_mc(orc):
    mc = mc_truth(orc, n_draw=800_000)
    gt = grid_truth(orc, n_grid=400)
    assert abs(mc["theta_harm"] - gt.theta_harm) < 0.01


def test_geometry_regular_and_transversal(orc):
    gt = grid_truth(orc, n_grid=400)
    assert gt.grad_S_on_MS > 1.0 and gt.grad_Y_on_MY > 1.0   # regular margins
    assert gt.corner_cos < 0.98                              # transversal corner


def test_two_boundary_derivative_signs(orc):
    """tau_S-only perturbation gives +D_MS; tau_Y-only gives -D_MY; both match FD."""
    zero = lambda X: np.zeros(np.atleast_2d(X).shape[0])
    one = lambda X: np.ones(np.atleast_2d(X).shape[0])
    adS = analytic_derivative(orc, one, zero, eps=0.18, n_draw=800_000)
    fdS = fd_derivative(orc, one, zero, h=0.05, n_draw=800_000)
    assert adS["D_MS"] > 0 and abs(adS["D_MY"]) < 1e-9
    assert abs(adS["Dtheta"] - fdS) < 0.0015          # +D_MS matches FD
    adY = analytic_derivative(orc, zero, one, eps=0.18, n_draw=800_000)
    fdY = fd_derivative(orc, zero, one, h=0.05, n_draw=800_000)
    assert adY["Dtheta"] < 0                           # -D_MY sign
    assert abs(adY["Dtheta"] - fdY) < 0.0015


def test_estimator_runs_and_is_finite(orc):
    rng = np.random.default_rng(0)
    df = orc.sample_experiment(1500, rng)
    est = estimate_harm_share(df, segments=2)
    assert 0.0 <= est.theta_hat <= 1.0
    assert est.se_sieve is not None and est.se_sieve > 0
    W, se = regular_companion_welfare(df, segments=2)
    assert np.isfinite(W) and se > 0
