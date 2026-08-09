"""Cubic B-spline sieve basis with analytic first derivatives, ridge/OLS fitting,
and inner cross-validation over the basis dimension.

The stage-two regressions ``mu_a`` are the *primary* learner and must expose
analytic derivatives (needed for the moving-boundary decision margins), so a
black-box learner is deliberately not used here (task section 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class BSplineBasis:
    """A clamped cubic B-spline basis on a fixed knot vector.

    The clamped B-spline basis is a partition of unity, so its span already
    contains the constant function; no separate intercept column is added
    (that would make the design rank-deficient).
    """

    knots: np.ndarray
    degree: int
    lo: float
    hi: float

    @property
    def dim(self) -> int:
        return len(self.knots) - self.degree - 1

    # ------------------------------------------------------------------ #
    def _clip(self, x: np.ndarray) -> np.ndarray:
        # design_matrix requires x within the base interval; clamp to be safe.
        return np.clip(np.asarray(x, dtype=float), self.lo, self.hi)

    def design(self, x: np.ndarray) -> np.ndarray:
        """Basis values; input of shape ``S`` -> output of shape ``(*S, dim)``."""
        x = np.asarray(x, dtype=float)
        shape = x.shape
        xc = self._clip(x).ravel()
        dm = BSpline.design_matrix(xc, self.knots, self.degree, extrapolate=False)
        arr = np.asarray(dm.todense())
        return arr.reshape(*shape, self.dim)

    def design_deriv(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """``order``-th derivatives; shape ``S`` -> ``(*S, dim)``."""
        x = np.asarray(x, dtype=float)
        shape = x.shape
        xc = self._clip(x).ravel()
        K = self.dim
        out = np.empty((xc.size, K), dtype=float)
        eye = np.eye(K)
        for j in range(K):
            spl = BSpline(self.knots, eye[j], self.degree, extrapolate=False)
            out[:, j] = spl.derivative(order)(xc)
        return np.nan_to_num(out, nan=0.0).reshape(*shape, K)


def make_basis(x: np.ndarray, dim: int, degree: int = 3, pad: float = 1e-6) -> BSplineBasis:
    """Build a clamped B-spline basis of dimension ``dim`` with interior knots at
    quantiles of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo if hi > lo else 1.0
    lo -= pad * span
    hi += pad * span
    n_interior = dim - degree - 1
    if n_interior < 0:
        raise ValueError(f"dim={dim} too small for degree={degree} (need dim>=degree+1)")
    if n_interior == 0:
        interior = np.array([])
    else:
        qs = np.linspace(0, 1, n_interior + 2)[1:-1]
        interior = np.quantile(x, qs)
        # de-duplicate ties by nudging (rich-support states rarely trigger this)
        interior = _dedupe_increasing(interior, lo, hi)
    knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    return BSplineBasis(knots=knots, degree=degree, lo=lo, hi=hi)


def _dedupe_increasing(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.sort(np.asarray(v, dtype=float))
    eps = 1e-9 * (hi - lo)
    for i in range(1, len(v)):
        if v[i] <= v[i - 1]:
            v[i] = v[i - 1] + eps
    return np.clip(v, lo + eps, hi - eps)


@dataclass
class SplineFit:
    basis: BSplineBasis
    beta: np.ndarray
    ridge: float
    dim: int
    condition_number: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.basis.design(x) @ self.beta

    def predict_deriv(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        return self.basis.design_deriv(x, order) @ self.beta

    def gram(self, x: np.ndarray) -> np.ndarray:
        """Empirical sieve Gram matrix (1/n) B'B on the given x."""
        B = self.basis.design(x)
        return (B.T @ B) / len(x)


def fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    *,
    degree: int = 3,
    ridge: float = 0.0,
    basis: BSplineBasis | None = None,
) -> SplineFit:
    """Least-squares (optionally ridge) fit of ``y`` on the B-spline basis."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if basis is None:
        basis = make_basis(x, dim, degree)
    B = basis.design(x)
    G = B.T @ B
    cond = float(np.linalg.cond(G)) if G.size else np.inf
    A = G + ridge * np.eye(B.shape[1])
    beta, *_ = np.linalg.lstsq(A, B.T @ y, rcond=None)
    return SplineFit(basis=basis, beta=beta, ridge=ridge, dim=basis.dim, condition_number=cond)


def select_dim_cv(
    x: np.ndarray,
    y: np.ndarray,
    candidate_dims: list[int],
    *,
    degree: int = 3,
    ridge: float = 0.0,
    n_folds: int = 5,
    condition_number_max: float = 1e10,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """Pick the basis dimension minimizing held-out MSE (inner CV)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_folds)
    scores: dict[int, float] = {}
    for dim in candidate_dims:
        if dim > n // 2:
            continue
        errs = []
        ok = True
        for f in folds:
            te = f
            tr = np.setdiff1d(order, te, assume_unique=False)
            if tr.size < dim + 2:
                ok = False
                break
            try:
                basis = make_basis(x[tr], dim, degree)
                fit = fit_spline(x[tr], y[tr], dim, degree=degree, ridge=ridge, basis=basis)
            except Exception:
                ok = False
                break
            if fit.condition_number > condition_number_max and ridge == 0.0:
                ok = False
                break
            pred = fit.predict(x[te])
            errs.append(np.mean((y[te] - pred) ** 2))
        if ok and errs:
            scores[dim] = float(np.mean(errs))
    if not scores:
        # fall back to the smallest feasible dimension
        dim = min(d for d in candidate_dims if d <= max(4, n // 2))
        return dim, {dim: np.nan}
    best = min(scores, key=scores.get)
    return best, scores
