"""The identifying representation must hold numerically: with the ORACLE optimal rules,
the IPW functional of the observed data equals V_11^* = E[Y^(1,1) 1{delta>=0} 1{kappa>=0}]."""

import numpy as np

from path_welfare.simulation import get_dgp


def _oracle_rules(dgp):
    """Build oracle g1(s)=1{kappa(s)>=0}, g2(x)=1{delta(x)>=0} from the DGP truth grid."""
    t = dgp.true_functionals()
    # delta / kappa signs from a fine grid via interpolation of the sign structure
    s_lo, s_hi = dgp._s_range()
    xg = np.linspace(dgp._x_lo(), dgp._x_hi(), 2001)
    sg = np.linspace(s_lo, s_hi, 2001)
    mu0, mu1 = dgp._mu_eff_on_grid(xg, sg)
    delta = mu1 - mu0
    V2 = np.maximum(mu0, mu1)
    dx = xg[1] - xg[0]
    kappa = np.array([
        np.trapezoid(V2 * (dgp.p(1, xg, s) - dgp.p(0, xg, s)), dx=dx) for s in sg
    ])
    g2 = lambda x: (np.interp(x, xg, delta) >= 0).astype(float)
    g1 = lambda s: (np.interp(s, sg, kappa) >= 0).astype(float)
    return g1, g2, t["V11"]


def test_ipw_oracle_equals_truth():
    dgp = get_dgp("dgp1")
    g1, g2, v11_true = _oracle_rules(dgp)
    rng = np.random.default_rng(7)
    df = dgp.sample(120_000, rng)  # large n to drive down MC error
    e1 = e2 = 0.5
    contrib = ((df["T1"] == 1) & (df["T2"] == 1)).to_numpy().astype(float) \
        * g1(df["S"].to_numpy()) * g2(df["X"].to_numpy()) * df["Y"].to_numpy() / (e1 * e2)
    v11_ipw = float(np.mean(contrib))
    assert abs(v11_ipw - v11_true) < 0.03, (v11_ipw, v11_true)
