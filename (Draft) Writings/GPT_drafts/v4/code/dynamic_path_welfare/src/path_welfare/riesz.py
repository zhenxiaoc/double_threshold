"""Sieve-Riesz inference for V_11^* with analytic moving-boundary derivatives.

Writing mu_a(x) = b_K(x)' beta_a, the (1,1) component and its pathwise derivative
w.r.t. (beta_0, beta_1) are computable by quadrature from the fixed density
estimates (m, p_a).  In the scalar case the second- and first-stage boundary terms
are *sums over roots* weighted by 1/|delta'| and 1/|kappa'| (see
``docs/theory_summary.md`` sections 4, 7).  The analytic gradient is checked against
central finite differences of the quadrature-defined population-sieve functional.

The reported variance is the sieve-Riesz variance **conditional on the estimated
densities m, p_a** (task section 11.2, option C): it is explicitly labelled as such
and is complemented by the full-refit participant bootstrap in ``bootstrap.py``, which
captures the additional uncertainty in m and p_a.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .densities import DensitySet, fit_densities
from .splines import BSplineBasis, make_basis


def _grid_roots_interp(g: np.ndarray, v: np.ndarray) -> list[float]:
    roots = []
    sign = np.sign(v)
    for i in range(len(g) - 1):
        if sign[i] == 0:
            roots.append(float(g[i]))
        elif sign[i] * sign[i + 1] < 0:
            t = v[i] / (v[i] - v[i + 1])
            roots.append(float(g[i] + t * (g[i + 1] - g[i])))
    return roots


def _region_weights(g: np.ndarray, level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid weight vectors W_plus, W_minus s.t. W_plus @ f = int_{level>=0} f dx
    and W_minus @ f = int_{level<0} f dx, with the sign-change cells split at the exact
    (linearly interpolated) crossing.  Makes region integrals smooth in the coefficients."""
    n = g.size
    Wp = np.zeros(n)
    Wm = np.zeros(n)
    for i in range(n - 1):
        a, b = level[i], level[i + 1]
        h = g[i + 1] - g[i]
        if a >= 0 and b >= 0:
            Wp[i] += 0.5 * h; Wp[i + 1] += 0.5 * h
        elif a < 0 and b < 0:
            Wm[i] += 0.5 * h; Wm[i + 1] += 0.5 * h
        else:
            t = a / (a - b)  # crossing fraction into the cell
            if a >= 0:       # [g_i, g_c] is the >=0 side
                Lp = t * h
                Wp[i] += 0.5 * Lp * (2 - t); Wp[i + 1] += 0.5 * Lp * t
                Lm = (1 - t) * h
                Wm[i] += 0.5 * Lm * (1 - t); Wm[i + 1] += 0.5 * Lm * (1 + t)
            else:            # [g_c, g_{i+1}] is the >=0 side
                Lp = (1 - t) * h
                Wp[i] += 0.5 * Lp * (1 - t); Wp[i + 1] += 0.5 * Lp * (1 + t)
                Lm = t * h
                Wm[i] += 0.5 * Lm * (2 - t); Wm[i + 1] += 0.5 * Lm * t
    return Wp, Wm


class SievePopFunctional:
    """Population-sieve functional V_11(beta0, beta1) with fixed densities, plus its
    analytic moving-boundary gradient.

    Both the inner x-integral (over ``{delta>=0}``) and the outer s-integral (over
    ``{kappa>=0}``) use exact sub-grid region weights, so ``value_pop`` is a smooth
    function of ``(beta0, beta1)`` and its finite differences converge to the analytic
    moving-boundary gradient.  A separate empirical-S estimate (``value_emp``) is used
    for the reported density-based point estimate to avoid marginal-density smoothing
    bias.
    """

    def __init__(self, basis: BSplineBasis, dens: DensitySet, s_sample: np.ndarray,
                 *, n_x=600, n_s=600, deriv_floor=1e-6):
        self.basis = basis
        self.dens = dens
        self.deriv_floor = deriv_floor
        self.s_sample = np.asarray(s_sample, dtype=float)
        self.xg = np.linspace(dens.x_lo, dens.x_hi, n_x)
        self.sg = np.linspace(dens.s_lo, dens.s_hi, n_s)
        self.dx = self.xg[1] - self.xg[0]
        self.Bx = basis.design(self.xg)                    # (n_x, K)
        w = np.full(n_x, self.dx); w[0] *= 0.5; w[-1] *= 0.5
        self.w_trap = w                                    # full trapezoid weights (x)
        self.m_s = dens.m.pdf(self.sg)                     # (n_s,)
        XX = np.broadcast_to(self.xg, (n_s, n_x))
        SS = self.sg[:, None]
        self.P1 = dens.p1.pdf(XX, SS)                      # (n_s, n_x)
        self.P0 = dens.p0.pdf(XX, SS)

    # ----------------------- pieces on the fine s-grid ----------------------- #
    def _pieces(self, beta0, beta1):
        mu0 = self.Bx @ beta0
        mu1 = self.Bx @ beta1
        delta = mu1 - mu0
        V2 = np.maximum(mu0, mu1)
        Wpx, Wmx = _region_weights(self.xg, delta)
        A1 = self.P1 @ (self.w_trap * V2)
        A0 = self.P0 @ (self.w_trap * V2)
        kappa = A1 - A0
        Wps, Wms = _region_weights(self.sg, kappa)
        G11 = self.P1 @ (Wpx * mu1)
        G10 = self.P1 @ (Wmx * mu0)
        G01 = self.P0 @ (Wpx * mu1)
        G00 = self.P0 @ (Wmx * mu0)
        return dict(mu0=mu0, mu1=mu1, delta=delta, V2=V2, Wpx=Wpx, Wmx=Wmx,
                    A1=A1, A0=A0, kappa=kappa, Wps=Wps, Wms=Wms,
                    G11=G11, G10=G10, G01=G01, G00=G00)

    def value_pop(self, beta0, beta1) -> float:
        p = self._pieces(beta0, beta1)
        return float(p["Wps"] @ (p["G11"] * self.m_s))

    def all_values(self, beta0, beta1) -> dict:
        p = self._pieces(beta0, beta1)
        m = self.m_s
        V11 = float(p["Wps"] @ (p["G11"] * m))
        V10 = float(p["Wps"] @ (p["G10"] * m))
        V01 = float(p["Wms"] @ (p["G01"] * m))
        V00 = float(p["Wms"] @ (p["G00"] * m))
        total = float(p["Wps"] @ (p["A1"] * m) + p["Wms"] @ (p["A0"] * m))
        return {"V11": V11, "V10": V10, "V01": V01, "V00": V00,
                "total": total, "sum": V11 + V10 + V01 + V00}

    def value_emp(self, beta0, beta1) -> dict:
        """Density-based estimate with the outer expectation over the empirical S sample
        (interpolating the grid functions), avoiding marginal-density smoothing bias."""
        p = self._pieces(beta0, beta1)
        G11s = np.interp(self.s_sample, self.sg, p["G11"])
        G10s = np.interp(self.s_sample, self.sg, p["G10"])
        G01s = np.interp(self.s_sample, self.sg, p["G01"])
        G00s = np.interp(self.s_sample, self.sg, p["G00"])
        kaps = np.interp(self.s_sample, self.sg, p["kappa"])
        s1 = (kaps >= 0).astype(float)
        return {"V11": float(np.mean(s1 * G11s)), "V10": float(np.mean(s1 * G10s)),
                "V01": float(np.mean((1 - s1) * G01s)), "V00": float(np.mean((1 - s1) * G00s))}

    # keep a generic .value alias for the smooth population functional (used by FD)
    value = value_pop

    # ----------------------- analytic gradient ----------------------- #
    def gradient(self, beta0, beta1):
        """Return (grad0, grad1, info) where grad_a = d V_11 / d beta_a (population)."""
        p = self._pieces(beta0, beta1)
        mu1, kappa = p["mu1"], p["kappa"]
        Wpx, Wps = p["Wpx"], p["Wps"]
        K = self.basis.dim
        grad0 = np.zeros(K)
        grad1 = np.zeros(K)

        # ---- Term I: interior (beta1 only) ----
        # H[:, s] = int_{D2+} b(x) p1(x|s) dx ; then int_{D1+} H m ds
        BW = self.Bx * Wpx[:, None]                        # (n_x, K)
        H = self.P1 @ BW                                   # (n_s, K)
        grad1 += H.T @ (Wps * self.m_s)

        # ---- Term II: second-stage boundary (roots of delta) ----
        delta_roots = _grid_roots_interp(self.xg, p["delta"])
        term2 = np.zeros(K)
        info_delta = []
        for xj in delta_roots:
            b_xj = self.basis.design(np.array([xj]))[0]
            dprime = float(self.basis.design_deriv(np.array([xj]))[0] @ (beta1 - beta0))
            denom = max(abs(dprime), self.deriv_floor)
            mu1_xj = float(b_xj @ beta1)
            p1_xj = self.dens.p1.pdf(np.full(self.sg.shape, xj), self.sg)   # (n_s,)
            w_j = float(Wps @ (p1_xj * self.m_s))          # int_{D1+} p1(x_j|s) m ds
            term2 += b_xj * mu1_xj * w_j / denom
            info_delta.append({"root": xj, "deriv": dprime,
                               "term_norm": float(np.linalg.norm(b_xj * mu1_xj * w_j / denom))})
        grad1 += term2
        grad0 += -term2

        # ---- Term III: first-stage boundary (roots of kappa) ----
        kappa_prime = np.gradient(kappa, self.sg)
        kappa_roots = _grid_roots_interp(self.sg, kappa)
        info_kappa = []
        for sk in kappa_roots:
            kp = float(np.interp(sk, self.sg, kappa_prime))
            denom = max(abs(kp), self.deriv_floor)
            p1k = self.dens.p1.pdf(self.xg, np.full(self.xg.shape, sk))
            p0k = self.dens.p0.pdf(self.xg, np.full(self.xg.shape, sk))
            rk = p1k - p0k
            G11k = float((Wpx * mu1) @ p1k)
            mk = float(self.dens.m.pdf(np.array([sk]))[0])
            vec_plus = BW.T @ rk                            # int_{D2+} b r dx  (K,)
            vec_minus = (self.Bx * p["Wmx"][:, None]).T @ rk
            wk = G11k * mk / denom
            grad1 += vec_plus * wk
            grad0 += vec_minus * wk
            info_kappa.append({"root": sk, "deriv": kp, "G11": G11k,
                               "term_norm": float(np.linalg.norm(vec_plus * wk))})

        info = {"delta_roots": delta_roots, "kappa_roots": kappa_roots,
                "term2_norm": float(np.linalg.norm(term2)),
                "info_delta": info_delta, "info_kappa": info_kappa}
        return grad0, grad1, info

    # ----------------------- finite-difference check ----------------------- #
    def fd_check(self, beta0, beta1, steps=(1e-2, 1e-3, 1e-4)):
        g0, g1, _ = self.gradient(beta0, beta1)
        out = {}
        for h in steps:
            fd0 = np.zeros_like(beta0)
            fd1 = np.zeros_like(beta1)
            for j in range(len(beta1)):
                e = np.zeros_like(beta1); e[j] = h
                fd1[j] = (self.value(beta0, beta1 + e) - self.value(beta0, beta1 - e)) / (2 * h)
                fd0[j] = (self.value(beta0 + e, beta1) - self.value(beta0 - e, beta1)) / (2 * h)
            out[h] = {
                "max_abs_diff_beta1": float(np.max(np.abs(fd1 - g1))),
                "max_abs_diff_beta0": float(np.max(np.abs(fd0 - g0))),
                "rel_diff_beta1": float(np.max(np.abs(fd1 - g1)) / (np.max(np.abs(g1)) + 1e-12)),
            }
        return out


@dataclass
class InferenceResult:
    K: int
    estimate: float
    se_conditional: float
    ci: tuple[float, float]
    tstat: float
    riesz_norm: float
    boundary_frac: float
    all_values: dict
    fd_check: dict
    variance_label: str = "sieve-Riesz variance CONDITIONAL on estimated densities (m, p_a)"
    diagnostics: dict = field(default_factory=dict)


def _fit_beta(basis, X, Y, mask, ridge=0.0):
    B = basis.design(X[mask])
    G = B.T @ B
    A = G + ridge * np.eye(B.shape[1])
    beta, *_ = np.linalg.lstsq(A, B.T @ Y[mask], rcond=None)
    Gram = G / int(mask.sum())
    return beta, Gram


def sieve_riesz_inference(
    est, *, K: int | None = None, n_x=600, n_s=600, ridge=0.0,
    fd_steps=(1e-2, 1e-3, 1e-4), alpha=0.05, densities: DensitySet | None = None,
) -> InferenceResult:
    """Full-sample sieve-Riesz inference at sieve dimension K."""
    cfg = est.config
    if K is None:
        K = cfg.inference.primary_dim
    df = est.data_
    obs = df["Y"].notna().to_numpy()
    X = df["X"].to_numpy(); Y = df["Y"].to_numpy(); T2 = df["T2"].to_numpy()
    S = df["S"].to_numpy()

    basis = make_basis(X, K, cfg.spline.degree)
    m0 = obs & (T2 == 0)
    m1 = obs & (T2 == 1)
    beta0, G0 = _fit_beta(basis, X, Y, m0, ridge)
    beta1, G1 = _fit_beta(basis, X, Y, m1, ridge)

    dens = densities if densities is not None else fit_densities(
        df, mean_dim=cfg.density.mean_dim, logsd_dim=cfg.density.logsd_dim
    )
    fn = SievePopFunctional(basis, dens, S, n_x=n_x, n_s=n_s)
    vals = fn.all_values(beta0, beta1)
    vals_emp = fn.value_emp(beta0, beta1)
    point = vals["V11"]
    grad0, grad1, ginfo = fn.gradient(beta0, beta1)

    # Riesz representers alpha_a = G_a^{-1} grad_a
    ridge_g = ridge + 1e-8
    alpha0 = np.linalg.solve(G0 + ridge_g * np.eye(K), grad0)
    alpha1 = np.linalg.solve(G1 + ridge_g * np.eye(K), grad1)

    # influence score (conditional on densities): mu-score contributions
    e2 = est.e2_ if est.e2_ is not None else float(np.mean(T2[obs]))
    n = len(df)
    B = basis.design(X)
    mu0_hat = B @ beta0
    mu1_hat = B @ beta1
    psi = np.zeros(n)
    idx1 = np.where(m1)[0]
    idx0 = np.where(m0)[0]
    psi[idx1] = (B[idx1] @ alpha1) * (Y[idx1] - mu1_hat[idx1]) / e2
    psi[idx0] = (B[idx0] @ alpha0) * (Y[idx0] - mu0_hat[idx0]) / (1 - e2)
    var = float(np.var(psi, ddof=1))
    se = float(np.sqrt(var / n))
    z = norm.ppf(1 - alpha / 2)
    ci = (point - z * se, point + z * se)

    riesz_norm = float(np.sqrt(alpha0 @ G0 @ alpha0 + alpha1 @ G1 @ alpha1))
    # fraction of the beta1 gradient norm coming from the second-stage boundary term
    boundary_frac = float(ginfo["term2_norm"] / (np.linalg.norm(grad1) + 1e-12))

    fd = fn.fd_check(beta0, beta1, steps=fd_steps)

    return InferenceResult(
        K=K, estimate=point, se_conditional=se, ci=ci,
        tstat=float(point / se) if se > 0 else np.nan,
        riesz_norm=riesz_norm, boundary_frac=boundary_frac,
        all_values=vals, fd_check=fd,
        diagnostics={
            "delta_roots": ginfo["delta_roots"], "kappa_roots": ginfo["kappa_roots"],
            "info_delta": ginfo["info_delta"], "info_kappa": ginfo["info_kappa"],
            "grad1_norm": float(np.linalg.norm(grad1)),
            "grad0_norm": float(np.linalg.norm(grad0)),
            "V11_density_empirical_outer": vals_emp["V11"],
            "all_values_density_empirical": vals_emp,
            "influence": psi,
        },
    )
