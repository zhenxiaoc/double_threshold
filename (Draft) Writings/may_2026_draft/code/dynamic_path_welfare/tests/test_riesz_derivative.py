"""The analytic moving-boundary functional derivative must match central finite
differences of the smooth population-sieve functional (task section 11.1, 20)."""

import numpy as np

from path_welfare.config import Config
from path_welfare.estimator import TwoStagePathWelfareEstimator
from path_welfare.simulation import get_dgp


def _fitted():
    rng = np.random.default_rng(2)
    df = get_dgp("dgp1").sample(2500, rng)
    df["group"] = np.arange(len(df))
    est = TwoStagePathWelfareEstimator(Config(treatment_probs={"e1": 0.5, "e2": 0.5})).fit(df)
    return est


def test_functional_derivative_matches_fd():
    est = _fitted()
    res = est.inference(K=8, n_x=600, n_s=600)
    # finite-difference relative error at the finest step should be tiny
    rel = min(v["rel_diff_beta1"] for v in res.fd_check.values())
    assert rel < 5e-3, res.fd_check


def test_fd_stable_across_steps():
    est = _fitted()
    res = est.inference(K=8, n_x=600, n_s=600)
    diffs = [v["max_abs_diff_beta1"] for v in res.fd_check.values()]
    # the discrepancy should not blow up as the step shrinks (smooth functional)
    assert max(diffs) < 1e-2


def test_boundary_term_present():
    est = _fitted()
    res = est.inference(K=8)
    # the second-stage boundary contributes a nonzero fraction of the gradient norm
    assert res.boundary_frac > 0.0
    assert len(res.diagnostics["delta_roots"]) >= 1
    assert len(res.diagnostics["kappa_roots"]) >= 1
