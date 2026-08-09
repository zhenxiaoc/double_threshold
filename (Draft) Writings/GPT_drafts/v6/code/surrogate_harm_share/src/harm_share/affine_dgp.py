"""Affine (Gaussian) companion DGP with EXACT truth.

tau_S(x) = aS + bS·x,  tau_Y(x) = aY + bY·x,  X ~ N(0, R).  Then (tau_S(X), tau_Y(X))
is bivariate normal, so theta and all four quadrant probabilities are EXACT orthant
probabilities (scipy), with zero grid error, and the geometry is pristine: the two
margins are straight lines meeting at a single, cleanly transversal corner.

Two uses:
  1. a zero-error VALIDATION of the grid/MC truth machinery in `functionals.py`;
  2. a clean geometry figure that shows the codimension-2 corner unambiguously
     (the calibrated KRR oracle has wiggly, multi-component margins).

Loadings are non-collinear (angle ~75 deg) -- a LINEAR T-learner would otherwise make
tau_S, tau_Y nearly parallel and collapse the two margins (a calibration pitfall we hit
with polynomial fits on the real data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


@dataclass
class AffineDGP:
    aS: float = 12.3
    bS: tuple = (13.0, 6.0)
    aY: float = 6.0
    bY: tuple = (-2.0, 12.0)
    R: tuple = ((1.0, 0.1), (0.1, 1.0))   # covariate correlation
    e: float = 0.5
    noise_S: float = 18.0
    noise_Y: float = 18.0
    # wide box: exact_truth integrates the FULL plane, so we keep truncation
    # negligible (~2e-9 tail at 6 sigma) => exact == grid == MC to grid/MC error.
    support_bound: float = 6.0
    d: int = field(default=2, init=False)

    # ------------------------------ surfaces ----------------------------- #
    def tau_S(self, X):
        X = np.atleast_2d(X)
        return self.aS + X @ np.asarray(self.bS)

    def tau_Y(self, X):
        X = np.atleast_2d(X)
        return self.aY + X @ np.asarray(self.bY)

    # ------------------------------ sampling ----------------------------- #
    def draw_X(self, n, rng):
        # rejection (truncation), matching HarmShareOracle; box is wide so this
        # is essentially the full Gaussian (no boundary pile-up from clipping).
        b = self.support_bound
        out = np.empty((0, 2))
        while len(out) < n:
            Xz = rng.multivariate_normal(np.zeros(2), np.asarray(self.R), size=2 * n)
            Xz = Xz[np.all(np.abs(Xz) <= b, axis=1)]
            out = np.vstack([out, Xz])
        return out[:n]

    def density(self, X):
        return multivariate_normal(np.zeros(2), np.asarray(self.R)).pdf(np.atleast_2d(X))

    # a KDE-like resample hook so functionals.mc_truth works unchanged
    @property
    def kde_(self):
        raise AttributeError("AffineDGP uses draw_X directly; mc_truth calls draw_X.")

    def sample_experiment(self, n, rng):
        X = self.draw_X(n, rng)
        W = (rng.random(n) < self.e).astype(int)
        muS = np.where(W == 1, self.tau_S(X), 0.0)
        muY = np.where(W == 1, self.tau_Y(X), 0.0)
        S = muS + rng.normal(0, self.noise_S, n)
        Y = muY + rng.normal(0, self.noise_Y, n)
        return pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "W": W, "S": S, "Y": Y})

    # ------------------------------ EXACT truth -------------------------- #
    def _tau_moments(self):
        R = np.asarray(self.R); bS = np.asarray(self.bS); bY = np.asarray(self.bY)
        mu = np.array([self.aS, self.aY])
        Sig = np.array([[bS @ R @ bS, bS @ R @ bY], [bY @ R @ bS, bY @ R @ bY]])
        return mu, Sig

    def exact_truth(self) -> dict:
        """Exact theta and quadrant probabilities via bivariate-normal rectangles."""
        mu, Sig = self._tau_moments()
        mvn = multivariate_normal(mean=mu, cov=Sig)

        def rect(sx, sy):
            # P(sx: tau_S in [lo,hi], sy: tau_Y in [lo,hi]) via mvn.cdf inclusion-exclusion
            xlo, xhi = (0.0, np.inf) if sx >= 0 else (-np.inf, 0.0)
            ylo, yhi = (0.0, np.inf) if sy >= 0 else (-np.inf, 0.0)
            def F(a, b):
                a = mu[0] + 8 * np.sqrt(Sig[0, 0]) if a == np.inf else a
                b = mu[1] + 8 * np.sqrt(Sig[1, 1]) if b == np.inf else b
                a = mu[0] - 8 * np.sqrt(Sig[0, 0]) if a == -np.inf else a
                b = mu[1] - 8 * np.sqrt(Sig[1, 1]) if b == -np.inf else b
                return mvn.cdf([a, b])
            return F(xhi, yhi) - F(xlo, yhi) - F(xhi, ylo) + F(xlo, ylo)

        pp = rect(+1, +1); pm = rect(+1, -1); mp = rect(-1, +1); mm = rect(-1, -1)
        # corner: solve tau_S=0, tau_Y=0
        Amat = np.array([self.bS, self.bY]); bvec = -np.array([self.aS, self.aY])
        corner = np.linalg.solve(Amat, bvec)
        bS = np.asarray(self.bS); bY = np.asarray(self.bY)
        cos = float(abs(bS @ bY) / (np.linalg.norm(bS) * np.linalg.norm(bY)))
        return {
            "theta_harm": float(pm), "theta_pp": float(pp), "theta_mp": float(mp),
            "theta_mm": float(mm), "rho": float(pm / max(pp + pm, 1e-12)),
            "treat_share_S": float(pp + pm),
            "ate_S": self.aS, "ate_Y": self.aY,
            "corner": corner.tolist(), "corner_cos": cos,
            "corner_angle_deg": float(np.degrees(np.arccos(cos))),
        }
