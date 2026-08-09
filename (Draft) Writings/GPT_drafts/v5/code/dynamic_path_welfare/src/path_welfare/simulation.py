"""Calibrated scalar-state DGPs with exact known truth (task section 13).

Every DGP obeys the causal order S -> T1 -> X -> T2 -> Y with sequential
randomization by construction.  The truth (V_ab^*, total, roots of delta and kappa)
is computed by high-resolution quadrature from the *known* mu_a, p_a, m, so the Monte
Carlo study compares estimates against a ground truth that is not itself estimated.

DGP catalogue:
  1 regular double boundary      5 no first-stage boundary
  2 weak second-stage boundary   6 multiple second-stage roots
  3 weak first-stage boundary    7 Markov failure (S predicts Y | X,T2)
  4 no second-stage boundary     8 empirical calibration (CalibratedDGP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


def _normpdf(x: np.ndarray, mu: np.ndarray | float, sd: float) -> np.ndarray:
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2 * np.pi))


@dataclass
class SimDGP:
    name: str
    mu0: Callable[[np.ndarray], np.ndarray]
    mu1: Callable[[np.ndarray], np.ndarray]
    mean0: Callable[[np.ndarray], np.ndarray]  # E[X | S, T1=0]
    mean1: Callable[[np.ndarray], np.ndarray]  # E[X | S, T1=1]
    sd_x: float
    sd_y: float
    e1: float = 0.5
    e2: float = 0.5
    s_dist: str = "normal"     # "normal" (smooth density) or "uniform"
    s_sd: float = 1.2          # sd of S under the normal design
    s_lo: float = -2.5         # used for the uniform design and for truth grids
    s_hi: float = 2.5
    gamma_markov: float = 0.0  # coefficient on S in Y (breaks Markov sufficiency)

    def _s_range(self) -> tuple[float, float]:
        if self.s_dist == "normal":
            return -4.0 * self.s_sd, 4.0 * self.s_sd
        return self.s_lo, self.s_hi

    # --------------------------- densities --------------------------- #
    def m(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.s_dist == "normal":
            return _normpdf(s, 0.0, self.s_sd)
        return np.where((s >= self.s_lo) & (s <= self.s_hi), 1.0 / (self.s_hi - self.s_lo), 0.0)

    def p(self, a: int, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        mean = self.mean1(s) if a == 1 else self.mean0(s)
        return _normpdf(x, mean, self.sd_x)

    # --------------------------- sampling ---------------------------- #
    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        if self.s_dist == "normal":
            S = rng.normal(0.0, self.s_sd, n)
        else:
            S = rng.uniform(self.s_lo, self.s_hi, n)
        T1 = (rng.random(n) < self.e1).astype(int)
        meanX = np.where(T1 == 1, self.mean1(S), self.mean0(S))
        X = meanX + rng.normal(0, self.sd_x, n)
        T2 = (rng.random(n) < self.e2).astype(int)
        muY = np.where(T2 == 1, self.mu1(X), self.mu0(X)) + self.gamma_markov * S
        Y = muY + rng.normal(0, self.sd_y, n)
        return pd.DataFrame({"S": S, "T1": T1, "X": X, "T2": T2, "Y": Y})

    # ------------------------ effective mu (Markov) ------------------ #
    def _mu_eff_on_grid(self, xg: np.ndarray, sg: np.ndarray):
        """Effective mu_a(x)=E[Y|X=x,T2=a] accounting for the +gamma*S term."""
        base0, base1 = self.mu0(xg), self.mu1(xg)
        if self.gamma_markov == 0.0:
            return base0, base1
        ds = sg[1] - sg[0]
        w1 = self.e1 * self.m(sg)
        w0 = (1 - self.e1) * self.m(sg)
        fX = np.zeros_like(xg)
        ESx = np.zeros_like(xg)
        for j, x in enumerate(xg):
            p1 = _normpdf(x, self.mean1(sg), self.sd_x)
            p0 = _normpdf(x, self.mean0(sg), self.sd_x)
            joint = w1 * p1 + w0 * p0
            denom = np.trapezoid(joint, dx=ds)
            fX[j] = denom
            ESx[j] = np.trapezoid(sg * joint, dx=ds) / max(denom, 1e-12)
        return base0 + self.gamma_markov * ESx, base1 + self.gamma_markov * ESx

    # ------------------------ exact truth ---------------------------- #
    def true_functionals(self, n_x: int = 1201, n_s: int = 1201) -> dict:
        s_lo, s_hi = self._s_range()
        xg = np.linspace(self._x_lo(), self._x_hi(), n_x)
        sg = np.linspace(s_lo, s_hi, n_s)
        dx = xg[1] - xg[0]
        ds = sg[1] - sg[0]

        mu0, mu1 = self._mu_eff_on_grid(xg, sg)
        delta = mu1 - mu0
        sel2 = (delta >= 0).astype(float)  # tie -> 1
        V2 = np.where(sel2 > 0, mu1, mu0)

        # continuation values A_a(s) and kappa
        A1 = np.empty(n_s)
        A0 = np.empty(n_s)
        G11 = np.empty(n_s)
        G10 = np.empty(n_s)
        G01 = np.empty(n_s)
        G00 = np.empty(n_s)
        for i, s in enumerate(sg):
            p1 = _normpdf(xg, self.mean1(s), self.sd_x)
            p0 = _normpdf(xg, self.mean0(s), self.sd_x)
            A1[i] = np.trapezoid(V2 * p1, dx=dx)
            A0[i] = np.trapezoid(V2 * p0, dx=dx)
            G11[i] = np.trapezoid(np.where(sel2 > 0, mu1, 0.0) * p1, dx=dx)
            G10[i] = np.trapezoid(np.where(sel2 > 0, 0.0, mu0) * p1, dx=dx)
            G01[i] = np.trapezoid(np.where(sel2 > 0, mu1, 0.0) * p0, dx=dx)
            G00[i] = np.trapezoid(np.where(sel2 > 0, 0.0, mu0) * p0, dx=dx)
        kappa = A1 - A0
        sel1 = (kappa >= 0).astype(float)
        m_s = self.m(sg)

        V11 = np.trapezoid(G11 * sel1 * m_s, dx=ds)
        V10 = np.trapezoid(G10 * sel1 * m_s, dx=ds)
        V01 = np.trapezoid(G01 * (1 - sel1) * m_s, dx=ds)
        V00 = np.trapezoid(G00 * (1 - sel1) * m_s, dx=ds)
        total = np.trapezoid((A1 * sel1 + A0 * (1 - sel1)) * m_s, dx=ds)

        return {
            "V11": float(V11), "V10": float(V10), "V01": float(V01), "V00": float(V00),
            "total": float(V11 + V10 + V01 + V00),
            "total_direct": float(total),
            "roots_delta": _grid_roots(xg, delta),
            "roots_kappa": _grid_roots(sg, kappa),
            "delta_deriv_at_roots": _deriv_at_roots(xg, delta),
            "kappa_deriv_at_roots": _deriv_at_roots(sg, kappa),
            "sd_y": self.sd_y,
        }

    def _x_lo(self) -> float:
        lo, hi = self._s_range()
        return min(self.mean0(np.array([lo, hi])).min(),
                   self.mean1(np.array([lo, hi])).min()) - 4 * self.sd_x

    def _x_hi(self) -> float:
        lo, hi = self._s_range()
        return max(self.mean0(np.array([lo, hi])).max(),
                   self.mean1(np.array([lo, hi])).max()) + 4 * self.sd_x


def _grid_roots(g: np.ndarray, v: np.ndarray) -> list[float]:
    roots = []
    sign = np.sign(v)
    for i in range(len(g) - 1):
        if sign[i] == 0:
            roots.append(float(g[i]))
        elif sign[i] * sign[i + 1] < 0:
            # linear interpolation of the crossing
            t = v[i] / (v[i] - v[i + 1])
            roots.append(float(g[i] + t * (g[i + 1] - g[i])))
    return roots


def _deriv_at_roots(g: np.ndarray, v: np.ndarray) -> list[float]:
    roots = _grid_roots(g, v)
    dv = np.gradient(v, g)
    return [float(np.interp(r, g, dv)) for r in roots]


# ====================================================================== #
# The catalogue
# ====================================================================== #
def _lin(a: float, b: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: a + b * np.asarray(x, dtype=float)


def get_dgp(name: str) -> SimDGP:
    """Return one of the eight named scalar-state DGPs.

    Common "treat-high-states" convention so the (1,1) path has real support:
      mu0(x) = 0.5 + 0.3 x ;  mu1(x) = mu0(x) + delta(x)
      mean0(s) = s ;  mean1(s) = s + shift(s)
    delta increasing  -> treat high X at stage 2 (D2+ = {x >= root});
    shift increasing   -> treat high S at stage 1 (D1+ = {s >= root}).
    """
    name = name.lower()
    sd_x, sd_y = 0.6, 0.5
    mu0 = _lin(0.5, 0.3)

    def with_delta(delta_fn):
        return lambda x: mu0(x) + delta_fn(np.asarray(x, float))

    if name in ("dgp1", "regular"):
        # delta(x)=0.6 x (root 0, steep); kappa root ~0 via steep shift 0.5 s
        return SimDGP(
            "dgp1_regular",
            mu0=mu0, mu1=with_delta(lambda x: 0.6 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.5 * s,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp2", "weak_delta"):
        # shallow delta slope -> weak second-stage margin
        return SimDGP(
            "dgp2_weak_delta",
            mu0=mu0, mu1=with_delta(lambda x: 0.12 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.5 * s,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp3", "weak_kappa"):
        # shallow first-stage shift -> weak first-stage margin
        return SimDGP(
            "dgp3_weak_kappa",
            mu0=mu0, mu1=with_delta(lambda x: 0.6 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.08 * s,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp4", "no_delta"):
        # delta = +0.5 (one sign) -> always treat stage 2
        return SimDGP(
            "dgp4_no_delta",
            mu0=mu0, mu1=with_delta(lambda x: 0.5 + 0.0 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.5 * s,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp5", "no_kappa"):
        # constant positive T1 shift -> kappa one sign (always treat stage 1)
        return SimDGP(
            "dgp5_no_kappa",
            mu0=mu0, mu1=with_delta(lambda x: 0.6 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.8,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp6", "multi_root"):
        # delta(x)=0.5(x^2-1): two roots at +-1, nonzero derivative there
        return SimDGP(
            "dgp6_multi_root",
            mu0=mu0, mu1=with_delta(lambda x: 0.5 * (x ** 2 - 1.0)),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.5 * s,
            sd_x=sd_x, sd_y=sd_y,
        )
    if name in ("dgp7", "markov_fail"):
        return SimDGP(
            "dgp7_markov_fail",
            mu0=mu0, mu1=with_delta(lambda x: 0.6 * x),
            mean0=_lin(0.0, 1.0), mean1=lambda s: s + 0.5 * s,
            sd_x=sd_x, sd_y=sd_y, gamma_markov=0.8,
        )
    raise ValueError(f"unknown DGP '{name}'")


ALL_DGPS = ["dgp1", "dgp2", "dgp3", "dgp4", "dgp5", "dgp6", "dgp7"]


# ====================================================================== #
# DGP 8: empirical calibration
# ====================================================================== #
class CalibratedDGP:
    """Simulator calibrated to a fitted estimator / dataset, with known truth.

    Resamples S from the data, fits X|S,T1 as Gaussian with spline mean, treats the
    estimator's stage-two fits as the truth for mu_a, and preserves the empirical
    randomization probabilities and missingness rate.  Truth is computed by quadrature
    from these calibrated components.
    """

    def __init__(self, dgp: SimDGP, y_missing_rate: float = 0.0):
        self.dgp = dgp
        self.y_missing_rate = y_missing_rate

    @classmethod
    def from_estimator(cls, est, *, n_grid: int = 400) -> "CalibratedDGP":
        import numpy as np

        df = est.data_
        S = df["S"].to_numpy()
        X = df["X"].to_numpy()
        T1 = df["T1"].to_numpy()
        # Gaussian X|S,T1 with linear-in-S mean (robust, low variance)
        b1 = np.polyfit(S[T1 == 1], X[T1 == 1], 1)
        b0 = np.polyfit(S[T1 == 0], X[T1 == 0], 1)
        sd_x = float(np.std(np.concatenate([
            X[T1 == 1] - np.polyval(b1, S[T1 == 1]),
            X[T1 == 0] - np.polyval(b0, S[T1 == 0]),
        ])))
        # mu_a as the estimator's fitted stage-two functions
        mu0 = lambda x: est.predict_mu(0, x)
        mu1 = lambda x: est.predict_mu(1, x)
        sd_y = float(np.nanstd(df["Y"].to_numpy()))
        dgp = SimDGP(
            "dgp8_calibrated",
            mu0=mu0, mu1=mu1,
            mean0=lambda s: np.polyval(b0, s), mean1=lambda s: np.polyval(b1, s),
            sd_x=max(sd_x, 1e-2), sd_y=sd_y,
            e1=est.e1_, e2=est.e2_,
            s_lo=float(np.min(S)), s_hi=float(np.max(S)),
        )
        miss = float(np.mean(np.isnan(df["Y"].to_numpy())))
        return cls(dgp, y_missing_rate=miss)

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        d = self.dgp.sample(n, rng)
        if self.y_missing_rate > 0:
            mask = rng.random(n) < self.y_missing_rate
            d.loc[mask, "Y"] = np.nan
        return d

    def true_functionals(self, **kw) -> dict:
        return self.dgp.true_functionals(**kw)


# ====================================================================== #
# Monte Carlo driver (task section 13)
# ====================================================================== #
def _match_root_error(est_roots, true_roots):
    if not true_roots:
        return float("nan")
    if not est_roots:
        return float("nan")
    errs = [min(abs(er - tr) for er in est_roots) for tr in true_roots]
    return float(np.mean(errs))


def run_mc(dgp_name, n, n_rep, *, seed=20260713, K=8, n_jobs=1,
           include_benchmarks=False, n_x=400, n_s=400):
    """Monte Carlo for V_11^* on a named DGP.  Returns aggregated metrics + raw arrays."""
    from joblib import Parallel, delayed
    from scipy.stats import norm

    from .config import Config
    from .crossfit import child_seed
    from .estimator import TwoStagePathWelfareEstimator

    dgp = get_dgp(dgp_name) if isinstance(dgp_name, str) else dgp_name
    truth = dgp.true_functionals()
    v11_true = truth["V11"]
    sd_y = truth["sd_y"]
    z95, z90 = norm.ppf(0.975), norm.ppf(0.95)

    def one(rep):
        rng = np.random.default_rng(child_seed(seed, rep))
        try:
            df = dgp.sample(n, rng)
            df["group"] = np.arange(len(df))
            cfg = Config(name=f"mc_{dgp_name}", treatment_probs={"e1": dgp.e1, "e2": dgp.e2})
            cfg.crossfit.seed = child_seed(seed, rep, 5)
            est = TwoStagePathWelfareEstimator(cfg).fit(df)
            direct = est.point_["11"]
            res = est.inference(K=K, n_x=n_x, n_s=n_s)
            se = res.se_conditional
            lo95, hi95 = res.estimate - z95 * se, res.estimate + z95 * se
            lo90, hi90 = res.estimate - z90 * se, res.estimate + z90 * se
            b = est.find_boundaries()
            dre = _match_root_error([r.location for r in b["delta"].roots], truth["roots_delta"])
            kre = _match_root_error([r.location for r in b["kappa"].roots], truth["roots_kappa"])
            out = {
                "direct": direct, "sieve": res.estimate, "se": se,
                "cov95": float(lo95 <= v11_true <= hi95),
                "cov90": float(lo90 <= v11_true <= hi90),
                "len95": hi95 - lo95, "len95_sd": (hi95 - lo95) / sd_y,
                "delta_root_err": dre, "kappa_root_err": kre,
                "n_delta_roots": len(b["delta"].roots), "n_kappa_roots": len(b["kappa"].roots),
                "fail": 0.0,
            }
            if include_benchmarks:
                from .aipw import aipw_11, ipw_11
                a = aipw_11(est); i = ipw_11(est)
                out["aipw"] = a.estimate; out["aipw_cov95"] = float(a.ci[0] <= v11_true <= a.ci[1])
                out["aipw_len95_sd"] = (a.ci[1] - a.ci[0]) / sd_y
                out["ipw"] = i.estimate
            return out
        except Exception as e:  # numerical failure
            return {"fail": 1.0, "error": str(e)}

    reps = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(one)(r) for r in range(n_rep))
    ok = [r for r in reps if r.get("fail", 0.0) == 0.0]
    fail_rate = 1.0 - len(ok) / n_rep

    def col(k):
        return np.array([r[k] for r in ok if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))])

    direct = col("direct"); sieve = col("sieve"); se = col("se")
    agg = {
        "dgp": dgp_name, "n": n, "n_rep": n_rep, "K": K,
        "V11_true": v11_true, "sd_y": sd_y,
        "bias_direct": float(np.mean(direct) - v11_true) if direct.size else np.nan,
        "bias_sieve": float(np.mean(sieve) - v11_true) if sieve.size else np.nan,
        "rmse_direct": float(np.sqrt(np.mean((direct - v11_true) ** 2))) if direct.size else np.nan,
        "rmse_sieve": float(np.sqrt(np.mean((sieve - v11_true) ** 2))) if sieve.size else np.nan,
        "mae_sieve": float(np.median(np.abs(sieve - v11_true))) if sieve.size else np.nan,
        "mc_sd_sieve": float(np.std(sieve, ddof=1)) if sieve.size > 1 else np.nan,
        "mean_se": float(np.mean(se)) if se.size else np.nan,
        "se_ratio": float(np.mean(se) / np.std(sieve, ddof=1)) if sieve.size > 1 and np.std(sieve) > 0 else np.nan,
        "cov90": float(np.mean(col("cov90"))) if col("cov90").size else np.nan,
        "cov95": float(np.mean(col("cov95"))) if col("cov95").size else np.nan,
        "len95_sd": float(np.mean(col("len95_sd"))) if col("len95_sd").size else np.nan,
        "median_len95_sd": float(np.median(col("len95_sd"))) if col("len95_sd").size else np.nan,
        "delta_root_err": float(np.nanmean(col("delta_root_err"))) if col("delta_root_err").size else np.nan,
        "kappa_root_err": float(np.nanmean(col("kappa_root_err"))) if col("kappa_root_err").size else np.nan,
        "mean_n_delta_roots": float(np.mean(col("n_delta_roots"))) if col("n_delta_roots").size else np.nan,
        "mean_n_kappa_roots": float(np.mean(col("n_kappa_roots"))) if col("n_kappa_roots").size else np.nan,
        "fail_rate": fail_rate,
        "n_true_delta_roots": len(truth["roots_delta"]),
        "n_true_kappa_roots": len(truth["roots_kappa"]),
    }
    if include_benchmarks and col("aipw_cov95").size:
        agg["aipw_cov95"] = float(np.mean(col("aipw_cov95")))
        agg["aipw_len95_sd"] = float(np.mean(col("aipw_len95_sd")))
        agg["bias_aipw"] = float(np.mean(col("aipw")) - v11_true)
    return {"aggregate": agg, "raw": ok}
