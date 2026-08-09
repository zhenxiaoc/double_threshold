"""Decision-boundary root finding and diagnostics.

Because ``S`` and ``X`` are scalar, the decision boundaries ``M2 = {x: delta(x)=0}``
and ``M1 = {s: kappa(s)=0}`` are finite sets of roots.  We find them with a dense
grid + sign-change detection + Brent refinement, then classify each root as
regular / weak (task section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import brentq


@dataclass
class Root:
    location: float
    quantile: float
    derivative: float
    distance_to_support_edge: float
    local_n: int
    regular: bool
    flags: list[str] = field(default_factory=list)


@dataclass
class BoundaryResult:
    name: str  # "delta" or "kappa"
    roots: list[Root]
    grid_lo: float
    grid_hi: float
    n_sign_changes: int
    has_crossing: bool

    @property
    def locations(self) -> list[float]:
        return [r.location for r in self.roots]


def find_roots(
    f: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    *,
    n_grid: int = 2001,
    dedupe_tol: float | None = None,
) -> list[float]:
    """All sign-change roots of scalar ``f`` on [lo, hi] via grid + Brent."""
    grid = np.linspace(lo, hi, n_grid)
    vals = np.asarray(f(grid), dtype=float)
    roots: list[float] = []
    # exact grid hits
    for i in range(n_grid):
        if vals[i] == 0.0:
            roots.append(float(grid[i]))
    # sign changes
    sign = np.sign(vals)
    for i in range(n_grid - 1):
        a, b = sign[i], sign[i + 1]
        if a * b < 0:
            try:
                r = brentq(lambda z: float(f(np.array([z]))[0]), grid[i], grid[i + 1], xtol=1e-10)
                roots.append(float(r))
            except (ValueError, RuntimeError):
                continue
    if dedupe_tol is None:
        dedupe_tol = 1e-6 * (hi - lo)
    return _dedupe(roots, dedupe_tol)


def _dedupe(roots: list[float], tol: float) -> list[float]:
    if not roots:
        return []
    roots = sorted(roots)
    out = [roots[0]]
    for r in roots[1:]:
        if abs(r - out[-1]) > tol:
            out.append(r)
    return out


def classify_roots(
    name: str,
    f: Callable[[np.ndarray], np.ndarray],
    fprime: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
    n_grid: int = 2001,
    interior_q: float = 0.025,
    deriv_floor: float | None = None,
    band: float | None = None,
) -> BoundaryResult:
    """Find and classify roots of ``f`` given the observed ``data`` (S or X values)."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if lo is None:
        lo = float(np.min(data))
    if hi is None:
        hi = float(np.max(data))
    grid = np.linspace(lo, hi, n_grid)
    vals = np.asarray(f(grid), dtype=float)
    sign = np.sign(vals)
    n_sign_changes = int(np.sum(sign[:-1] * sign[1:] < 0))
    locs = find_roots(f, lo, hi, n_grid=n_grid)

    span = hi - lo if hi > lo else 1.0
    q_lo, q_hi = np.quantile(data, interior_q), np.quantile(data, 1 - interior_q)
    n = data.size
    if band is None:
        # "near the boundary" window in state units; wide enough that a genuinely
        # interior, well-supported root is not spuriously flagged.
        band = 0.10 * np.std(data)
    if deriv_floor is None:
        # scale-aware floor: derivative small relative to typical slope of f
        typ = np.median(np.abs(np.diff(vals) / np.diff(grid)))
        deriv_floor = 0.05 * typ if typ > 0 else 1e-6

    roots: list[Root] = []
    for z in locs:
        d = float(fprime(np.array([z]))[0])
        quant = float(np.mean(data <= z))
        dist_edge = float(min(z - lo, hi - z) / span)
        local_n = int(np.sum(np.abs(data - z) <= band))
        flags: list[str] = []
        if z < q_lo or z > q_hi:
            flags.append("root in outer 2.5% of support")
        if abs(d) < deriv_floor:
            flags.append("small derivative (near-tangential crossing)")
        min_local = max(50, int(0.05 * n))
        if local_n < min_local:
            flags.append(f"local support {local_n} < max(50,0.05n)={min_local}")
        regular = len(flags) == 0
        roots.append(Root(z, quant, d, dist_edge, local_n, regular, flags))

    return BoundaryResult(
        name=name,
        roots=roots,
        grid_lo=lo,
        grid_hi=hi,
        n_sign_changes=n_sign_changes,
        has_crossing=len(locs) > 0,
    )


def boundary_band_counts(
    values: np.ndarray,
    contrast_at: np.ndarray,
    hs: list[float],
) -> dict[float, int]:
    """N(h) = #{ i : |contrast(state_i)| <= h } for several bandwidths h."""
    contrast_at = np.abs(np.asarray(contrast_at, dtype=float))
    return {float(h): int(np.sum(contrast_at <= h)) for h in hs}
