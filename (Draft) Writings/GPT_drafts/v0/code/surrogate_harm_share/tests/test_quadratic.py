"""Tests for the SS quadratic debiasing and the doubly debiased estimator."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from harm_share.affine_dgp import AffineDGP
from harm_share.quadratic import (
    dd_estimate, riesz_correction_and_variance, ss_debiased_estimate, theta_of,
)


@pytest.fixture(scope="module")
def affine_df():
    dgp = AffineDGP()
    rng = np.random.default_rng(3)
    return dgp, dgp.exact_truth(), dgp.sample_experiment(3000, rng)


def test_theta_of_matches_quadrant():
    tS = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    tY = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    # {tS>=0, tY<0}: rows 0 and 4
    assert theta_of(tS, tY) == pytest.approx(2 / 5)


def test_ss_decomposition_identity(affine_df):
    _, _, df = affine_df
    ss = ss_debiased_estimate(df, segments=2, seed=0)
    # polarization: full quadratic form = margin parts + corner cross part
    assert ss.quad_full == pytest.approx(ss.quad_S + ss.quad_Y + ss.corner, abs=1e-12)
    # jackknife identity: theta_ss = 2*theta_bar - (theta_A+theta_B)/2
    d = ss.diag
    assert ss.theta_ss == pytest.approx(
        2 * d["theta_bar"] - 0.5 * (d["theta_A"] + d["theta_B"]), abs=1e-12)


def test_ss_close_to_truth_affine(affine_df):
    dgp, truth, df = affine_df
    ss = ss_debiased_estimate(df, segments=2, seed=0)
    th0 = truth["theta_harm"]
    assert abs(ss.theta_plugin - th0) < 0.12
    assert abs(ss.theta_ss - th0) < 0.12
    assert ss.se_sieve is not None and 0 < ss.se_sieve < 0.5
    lo, hi = ss.ci_ss
    assert lo < hi


def test_ss_correction_is_small_relative_to_theta(affine_df):
    _, _, df = affine_df
    ss = ss_debiased_estimate(df, segments=2, seed=0)
    # the SS correction is a second-order term: much smaller than theta itself
    assert abs(ss.theta_ss - ss.diag["theta_bar"]) < 0.5 * max(ss.theta_plugin, 0.05)


def test_dd_runs_and_orders(affine_df):
    dgp, truth, df = affine_df
    dd = dd_estimate(df, nuisance="gbr", segments=2, seed=0)
    th0 = truth["theta_harm"]
    for th in (dd.theta_cf, dd.theta_cf_riesz, dd.theta_dd):
        assert np.isfinite(th) and abs(th - th0) < 0.2
    assert dd.se_sieve is not None and dd.se_sieve > 0
    # internal consistency: theta_cf_riesz - theta_cf == riesz correction
    assert dd.theta_cf_riesz - dd.theta_cf == pytest.approx(dd.correction_riesz, abs=1e-12)


def test_riesz_correction_zero_for_own_basis_ls(affine_df):
    """LS orthogonality: with a sieve LS first stage and the SAME basis used
    for the Riesz projection, the projected first-order correction is ~0."""
    _, _, df = affine_df
    from harm_share.estimator import _Xcols, _sieve_opts, fit_cate_surface
    Xc = _Xcols(df)
    opts = _sieve_opts(len(Xc), 2)
    out_S = fit_cate_surface(df, "S", opts)
    out_Y = fit_cate_surface(df, "Y", opts)
    Xeval = df[Xc].to_numpy(float)
    tS = np.asarray(out_S["h_hat"](Xeval)).ravel()
    tY = np.asarray(out_Y["h_hat"](Xeval)).ravel()
    corr, var, _ = riesz_correction_and_variance(out_S, out_Y, Xeval, tS, tY)
    assert var > 0
    # correction is a numerical zero relative to the SE scale
    assert abs(corr) < 1e-6 * max(1.0, np.sqrt(var) * 1e3)


def test_variance_includes_empirical_measure_term(affine_df):
    """The two-band variance must include theta(1-theta)/n: theta_hat is an
    empirical average, so its sampling error has a boundary part AND the
    average's own part.  Omitting it understates the SE by ~6-12% at these n."""
    _, _, df = affine_df
    from harm_share.estimator import (
        _Xcols, _sieve_opts, fit_cate_surface, two_band_sieve_variance,
    )
    import numpy as np
    Xc = _Xcols(df)
    opts = _sieve_opts(len(Xc), 2)
    out_S = fit_cate_surface(df, "S", opts)
    out_Y = fit_cate_surface(df, "Y", opts)
    Xeval = df[Xc].to_numpy(float)
    tS = np.asarray(out_S["h_hat"](Xeval)).ravel()
    tY = np.asarray(out_Y["h_hat"](Xeval)).ravel()

    var_on, d_on = two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY,
                                           include_empirical=True)
    var_off, _ = two_band_sieve_variance(out_S, out_Y, Xeval, tS, tY,
                                         include_empirical=False)
    th = float(np.mean((tS >= 0) & (tY < 0)))
    expected = th * (1 - th) / len(df)
    assert var_on > var_off
    assert var_on - var_off == pytest.approx(expected, rel=1e-10)
    assert d_on["var_emp"] == pytest.approx(expected, rel=1e-10)
