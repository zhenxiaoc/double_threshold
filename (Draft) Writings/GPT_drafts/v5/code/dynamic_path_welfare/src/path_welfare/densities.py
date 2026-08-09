"""Scalar density estimation for the theory-aligned, density-based plug-in
cross-check and for the sieve-Riesz quadrature (task sections 8.3, 11).

We estimate m(s)=f_S(s), and the transition laws p_a(x|s)=f_{X|S,T1=a}.  The
default conditional model is Gaussian with a spline conditional mean and a
(constant or spline) conditional log-sd -- a positive, exactly-normalized density.
A Nadaraya-Watson kernel conditional density is available as a robustness option.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

from .splines import fit_spline


def _normpdf(x, mu, sd):
    z = (x - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2 * np.pi))


@dataclass
class MarginalDensity:
    kde: gaussian_kde
    lo: float
    hi: float

    def pdf(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = self.kde(s)
        return np.clip(out, 0.0, None)


def fit_marginal(s: np.ndarray, bw="scott") -> MarginalDensity:
    s = np.asarray(s, dtype=float)
    s = s[~np.isnan(s)]
    kde = gaussian_kde(s, bw_method=bw)
    return MarginalDensity(kde, float(np.min(s)), float(np.max(s)))


@dataclass
class GaussianConditional:
    """p(x|s) = Normal(mean(s), sd(s)) with spline mean and (optional) spline log-sd."""

    mean_fit: object
    logsd_fit: object | None
    const_sd: float
    trim: float

    def mean(self, s: np.ndarray) -> np.ndarray:
        return self.mean_fit.predict(np.asarray(s, dtype=float))

    def sd(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.logsd_fit is None:
            return np.full(s.shape, self.const_sd)
        return np.clip(np.exp(self.logsd_fit.predict(s)), self.trim, None)

    def pdf(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        return _normpdf(np.asarray(x, dtype=float), self.mean(s), self.sd(s))


def fit_conditional_gaussian(
    x: np.ndarray, s: np.ndarray, *, mean_dim: int = 6, logsd_dim: int | None = 4,
    trim: float = 1e-2, ridge: float = 0.0,
) -> GaussianConditional:
    """Fit p(x|s) as Gaussian with spline conditional mean and log-sd."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    mean_dim = min(mean_dim, max(4, len(np.unique(s)) // 3))
    mfit = fit_spline(s, x, mean_dim, ridge=ridge)
    resid = x - mfit.predict(s)
    const_sd = float(np.std(resid))
    lfit = None
    if logsd_dim is not None and len(np.unique(s)) > logsd_dim * 3:
        logsd_dim = min(logsd_dim, max(4, len(np.unique(s)) // 4))
        log_r2 = np.log(np.clip(resid ** 2, (0.1 * const_sd) ** 2, None))
        lr = fit_spline(s, log_r2, logsd_dim, ridge=max(ridge, 1e-3))
        # sd = exp(0.5 * E[log r^2])
        lfit = _HalfScale(lr)
    return GaussianConditional(mfit, lfit, const_sd, trim)


@dataclass
class _HalfScale:
    fit: object

    def predict(self, s):
        return 0.5 * self.fit.predict(s)


@dataclass
class DensitySet:
    """m(s), p_0(x|s), p_1(x|s) bundled together with support info."""

    m: MarginalDensity
    p0: GaussianConditional
    p1: GaussianConditional
    x_lo: float
    x_hi: float
    s_lo: float
    s_hi: float

    def p(self, a: int, x, s):
        return (self.p1 if a == 1 else self.p0).pdf(x, s)

    def r(self, x, s):
        return self.p1.pdf(x, s) - self.p0.pdf(x, s)


def fit_densities(df, cfg=None, *, mean_dim=6, logsd_dim=4) -> DensitySet:
    """Fit the full density set from a canonical dataframe."""
    S = df["S"].to_numpy(); X = df["X"].to_numpy(); T1 = df["T1"].to_numpy()
    m = fit_marginal(S)
    p0 = fit_conditional_gaussian(X[T1 == 0], S[T1 == 0], mean_dim=mean_dim, logsd_dim=logsd_dim)
    p1 = fit_conditional_gaussian(X[T1 == 1], S[T1 == 1], mean_dim=mean_dim, logsd_dim=logsd_dim)
    pad = 0.05 * (np.nanmax(X) - np.nanmin(X))
    return DensitySet(
        m, p0, p1,
        x_lo=float(np.nanmin(X) - pad), x_hi=float(np.nanmax(X) + pad),
        s_lo=float(np.nanmin(S)), s_hi=float(np.nanmax(S)),
    )
