"""Population functionals of the harm-share DGP, with exact (quadrature / large-MC)
truth and the analytic moving-boundary derivative.

Target (double-threshold, generic form with h0=tau_S, g0=-tau_Y, phi==1):

    theta = Pr( tau_S(X) >= 0, tau_Y(X) <= 0 )
          = integral 1{tau_S>=0} 1{tau_Y<=0} f(x) dx           (the harm share)

Four sign quadrants classify the long-run CATE sign by the short-run CATE sign:
    theta_pp = Pr(tau_S>=0, tau_Y>=0)   theta_pm = Pr(tau_S>=0, tau_Y<0)  (= theta)
    theta_mp = Pr(tau_S<0,  tau_Y>=0)   theta_mm = Pr(tau_S<0,  tau_Y<0)
and the conditional harm ratio rho = theta_pm / (theta_pp + theta_pm) = P(tau_Y<0 | tau_S>=0).

Geometry.  The boundary of the harm region is M_S ∪ M_Y with
    M_S = {tau_S = 0, tau_Y < 0}  (short-run threshold, restricted to long-run losers)
    M_Y = {tau_Y = 0, tau_S > 0}  (long-run threshold, restricted to short-run winners)
meeting at the corner C = {tau_S = 0, tau_Y = 0} (codimension 2).

Moving-boundary derivative.  Perturbing tau_S -> tau_S + t*dS and tau_Y -> tau_Y + t*dY,

    Dtheta[dS, dY] =  integral_{M_S} dS f / ||grad tau_S|| dH^{d-1}
                    - integral_{M_Y} dY f / ||grad tau_Y|| dH^{d-1}.

Sign logic: raising tau_S EXPANDS {tau_S>=0} (+ term on M_S); raising tau_Y SHRINKS
{tau_Y<=0} (- term on M_Y).  By the coarea formula the 1/||grad|| weight cancels, giving
the gradient-free narrow-band identity used below:

    integral_{M_S} dS f/||grad tau_S|| dH^{d-1} = lim_{eps->0} (1/2eps) integral_{|tau_S|<eps, tau_Y<0} dS f dx.

The codim-2 corner C contributes at second order under transversality (grad tau_S, grad tau_Y
linearly independent on C), so it drops out of the first-order derivative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TruthReport:
    theta_harm: float          # theta_pm
    theta_pp: float
    theta_mp: float
    theta_mm: float
    rho: float                 # conditional harm ratio
    treat_share_S: float       # Pr(tau_S >= 0) = single-threshold companion
    ate_S: float
    ate_Y: float
    len_M_S: float             # H^{d-1}(M_S) boundary "length"
    len_M_Y: float
    grad_S_on_MS: float        # median ||grad tau_S|| on M_S
    grad_Y_on_MY: float
    corner_cos: float          # median |cos angle(grad tau_S, grad tau_Y)| near C (<1 => transversal)
    n_grid: int

    def as_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Large-Monte-Carlo truth (works for any covariate dimension d)
# --------------------------------------------------------------------------- #
def mc_truth(oracle, n_draw: int = 1_500_000, seed: int = 20260713) -> dict:
    """Population quadrant probabilities by drawing a huge sample from f_hat."""
    rng = np.random.default_rng(seed)
    Xz = oracle.draw_X(n_draw, rng)
    tS = oracle.tau_S(Xz)
    tY = oracle.tau_Y(Xz)
    pp = float(np.mean((tS >= 0) & (tY >= 0)))
    pm = float(np.mean((tS >= 0) & (tY < 0)))
    mp = float(np.mean((tS < 0) & (tY >= 0)))
    mm = float(np.mean((tS < 0) & (tY < 0)))
    return {
        "theta_harm": pm, "theta_pp": pp, "theta_mp": mp, "theta_mm": mm,
        "rho": pm / max(pp + pm, 1e-12),
        "treat_share_S": pp + pm,
        "ate_S": float(tS.mean()), "ate_Y": float(tY.mean()),
        "n_draw": n_draw,
    }


# --------------------------------------------------------------------------- #
# 2-D grid truth (exact quadrature + geometry diagnostics + figures)
# --------------------------------------------------------------------------- #
def _support_bound(oracle):
    cfg = getattr(oracle, "cfg", None)
    return cfg.support_bound if cfg is not None else getattr(oracle, "support_bound", 3.0)


def _grid2d(oracle, n_grid=400, span=None):
    if span is None:
        span = _support_bound(oracle)  # tie the grid to the sampler's truncation box
    g = np.linspace(-span, span, n_grid)
    GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel(), GY.ravel()]
    tS = oracle.tau_S(grid).reshape(GX.shape)
    tY = oracle.tau_Y(grid).reshape(GX.shape)
    f = oracle.density(grid).reshape(GX.shape)
    dx = g[1] - g[0]
    fn = f / np.trapezoid(np.trapezoid(f, dx=dx, axis=0), dx=dx)  # renormalize on grid
    return g, GX, GY, tS, tY, fn, dx


def grid_truth(oracle, n_grid=400, span=None) -> TruthReport:
    """Exact truth by quadrature for the d=2 oracle, plus geometry diagnostics.

    `span` defaults to the oracle's truncation box (`support_bound`) so the grid
    integrates the SAME distribution the sampler draws from.
    """
    assert oracle.d == 2, "grid_truth is for the 2-D oracle; use mc_truth otherwise."
    g, GX, GY, tS, tY, fn, dx = _grid2d(oracle, n_grid, span)

    def integ(mask):
        return float(np.trapezoid(np.trapezoid(mask * fn, dx=dx, axis=0), dx=dx))

    pp = integ((tS >= 0) & (tY >= 0))
    pm = integ((tS >= 0) & (tY < 0))
    mp = integ((tS < 0) & (tY >= 0))
    mm = integ((tS < 0) & (tY < 0))

    # gradients and boundary geometry
    gSx, gSy = np.gradient(tS, g, g)
    gYx, gYy = np.gradient(tY, g, g)
    nS = np.hypot(gSx, gSy)
    nY = np.hypot(gYx, gYy)
    # coarea "length" of each sub-boundary: H^{d-1}(M) = lim (1/2eps) int_{|tau|<eps, side} |grad tau| dx
    eps = 2.5 * dx * np.maximum(np.median(nS), np.median(nY))  # band ~2.5 cells wide in tau-units
    bandS = (np.abs(tS) < eps) & (tY < 0)
    bandY = (np.abs(tY) < eps) & (tS > 0)
    len_MS = float(np.sum(bandS * nS) * dx * dx / (2 * eps))
    len_MY = float(np.sum(bandY * nY) * dx * dx / (2 * eps))
    onMS = (np.abs(tS) < eps) & (tY < 0) & (fn > fn.max() * 0.02)
    onMY = (np.abs(tY) < eps) & (tS > 0) & (fn > fn.max() * 0.02)
    corner = (np.abs(tS) < 2 * eps) & (np.abs(tY) < 2 * eps) & (fn > fn.max() * 0.05)
    cos = (gSx * gYx + gSy * gYy) / (nS * nY + 1e-12)

    return TruthReport(
        theta_harm=pm, theta_pp=pp, theta_mp=mp, theta_mm=mm,
        rho=pm / max(pp + pm, 1e-12), treat_share_S=pp + pm,
        ate_S=float(np.sum(tS * fn) * dx * dx), ate_Y=float(np.sum(tY * fn) * dx * dx),
        len_M_S=len_MS, len_M_Y=len_MY,
        grad_S_on_MS=float(np.median(nS[onMS])) if onMS.any() else float("nan"),
        grad_Y_on_MY=float(np.median(nY[onMY])) if onMY.any() else float("nan"),
        corner_cos=float(np.median(np.abs(cos[corner]))) if corner.any() else float("nan"),
        n_grid=n_grid,
    )


# --------------------------------------------------------------------------- #
# Moving-boundary derivative: analytic (narrow-band) vs finite difference
# --------------------------------------------------------------------------- #
def theta_perturbed(oracle, dS_fn, dY_fn, t, n_draw=800_000, seed=7, _cache={}):
    """theta under tau_S -> tau_S + t dS, tau_Y -> tau_Y + t dY, by large MC.

    Draws are cached by (id(oracle), n_draw, seed) so finite-difference calls at
    several t reuse the same points (variance-reduced, common random numbers).
    """
    key = (id(oracle), n_draw, seed)
    if key not in _cache:
        rng = np.random.default_rng(seed)
        Xz = oracle.draw_X(n_draw, rng)
        _cache[key] = (Xz, oracle.tau_S(Xz), oracle.tau_Y(Xz))
    Xz, tS, tY = _cache[key]
    dS, dY = dS_fn(Xz), dY_fn(Xz)  # cheap; must NOT be cached (varies per call)
    return float(np.mean(((tS + t * dS) >= 0) & ((tY + t * dY) < 0)))


def analytic_derivative(oracle, dS_fn, dY_fn, eps=0.15, n_draw=800_000, seed=7):
    """Two-boundary derivative via the gradient-free coarea narrow-band identity.

    Returns D_MS (>=0-side term on M_S), D_MY (term on M_Y) and the total
    Dtheta = D_MS - D_MY.  Each boundary term is a separate, non-cancelling
    contribution -- the signature of the double threshold.
    """
    rng = np.random.default_rng(seed)
    Xz = oracle.draw_X(n_draw, rng)
    tS, tY = oracle.tau_S(Xz), oracle.tau_Y(Xz)
    dS, dY = dS_fn(Xz), dY_fn(Xz)
    D_MS = float(np.mean(((np.abs(tS) < eps) & (tY < 0)) * dS) / (2 * eps))
    D_MY = float(np.mean(((np.abs(tY) < eps) & (tS > 0)) * dY) / (2 * eps))
    return {"D_MS": D_MS, "D_MY": D_MY, "Dtheta": D_MS - D_MY, "eps": eps}


def fd_derivative(oracle, dS_fn, dY_fn, h=0.05, n_draw=800_000, seed=7):
    """Central finite difference of theta(t) at t=0 (common random numbers)."""
    thp = theta_perturbed(oracle, dS_fn, dY_fn, +h, n_draw, seed)
    thm = theta_perturbed(oracle, dS_fn, dY_fn, -h, n_draw, seed)
    return (thp - thm) / (2 * h)
