# spline_basis/spline_factory.py

from __future__ import annotations

from typing import Any, Dict, Callable

import numpy as np
from .prodspline import prodspline   # or from .prodspline import prodspline, depending on your layout


def build_spline_basis_from_options(
    options: Dict[str, Any],
    X_sample: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a spline feature map f: R^d -> R^K based on `prodspline` and
    an options dictionary.

    Parameters
    ----------
    options : dict
        Expected keys (minimum):
            - "J_x_degree"   : int or array-like of length d
            - "J_x_segments" : int or array-like of length d
            - "knots"        : str or object, passed to `prodspline`
            - "basis"        : str, e.g. "additive", "tensor", "glp"

        Optional keys:
            - "X_min"        : scalar or array-like of length d
                               If provided, will be combined with sample-based
                               bounds so that the final domain still covers the sample.
            - "X_max"        : scalar or array-like of length d

    X_sample : np.ndarray
        Training covariates for constructing the basis, shape (n_sample, d).

    Returns
    -------
    fmap : Callable[[np.ndarray], np.ndarray]
        A callable that takes X_new (shape (n, d) or (d,)) and returns
        the spline basis matrix of shape (n, K).
    """
    # ------------------------------------------------------------------
    # 1. Standardize X_sample and infer dimension
    # ------------------------------------------------------------------
    X_sample = np.asarray(X_sample, dtype=float)
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(-1, 1)
    elif X_sample.ndim != 2:
        raise ValueError("X_sample must be a 1D (n,) or 2D (n, d) array.")
    n_sample, d = X_sample.shape

    # ------------------------------------------------------------------
    # 2. Degree and segments per dimension
    #    - Allow scalar or length-d array-like for each
    # ------------------------------------------------------------------
    if "J_x_degree" not in options:
        raise ValueError("options must contain 'J_x_degree'.")
    if "J_x_segments" not in options:
        raise ValueError("options must contain 'J_x_segments'.")

    J_x_degree_opt = options["J_x_degree"]
    J_x_segments_opt = options["J_x_segments"]

    # Degree
    deg_arr = np.atleast_1d(J_x_degree_opt).astype(int)
    if deg_arr.size == 1:
        deg_arr = np.repeat(deg_arr[0], d)
    elif deg_arr.size != d:
        raise ValueError(
            f"J_x_degree has length {deg_arr.size}, but X_sample has d={d}."
        )

    # Segments
    seg_arr = np.atleast_1d(J_x_segments_opt).astype(int)
    if seg_arr.size == 1:
        seg_arr = np.repeat(seg_arr[0], d)
    elif seg_arr.size != d:
        raise ValueError(
            f"J_x_segments has length {seg_arr.size}, but X_sample has d={d}."
        )

    # prodspline expects a (d, 2) matrix: [degree, segments] per variable
    K_mat = np.column_stack([deg_arr, seg_arr])

    # ------------------------------------------------------------------
    # 3. Knots / basis type
    # ------------------------------------------------------------------
    knots = options.get("knots", "quantiles")
    basis = options.get("basis", "tensor")

    # ------------------------------------------------------------------
    # 4. Domain bounds: use sample-based min/max, with optional override
    # ------------------------------------------------------------------
    X_min_sample = X_sample.min(axis=0)
    X_max_sample = X_sample.max(axis=0)

    # User-specified X_min / X_max (optional)
    X_min_opt = options.get("X_min", None)
    X_max_opt = options.get("X_max", None)

    # Convert to arrays, handling scalar vs. vector cases
    if X_min_opt is None:
        X_min_arr = X_min_sample.copy()
    else:
        X_min_arr = np.atleast_1d(X_min_opt).astype(float)
        if X_min_arr.size == 1:
            X_min_arr = np.repeat(X_min_arr[0], d)
        elif X_min_arr.size != d:
            raise ValueError(
                f"X_min has length {X_min_arr.size}, but X_sample has d={d}."
            )
        # Ensure final lower bound does not exclude any training point
        X_min_arr = np.minimum(X_min_arr, X_min_sample)

    if X_max_opt is None:
        X_max_arr = X_max_sample.copy()
    else:
        X_max_arr = np.atleast_1d(X_max_opt).astype(float)
        if X_max_arr.size == 1:
            X_max_arr = np.repeat(X_max_arr[0], d)
        elif X_max_arr.size != d:
            raise ValueError(
                f"X_max has length {X_max_arr.size}, but X_sample has d={d}."
            )
        # Ensure final upper bound does not exclude any training point
        X_max_arr = np.maximum(X_max_arr, X_max_sample)

    # Sanity: avoid degenerate dimensions (constant covariates)
    spread = X_max_sample - X_min_sample
    if np.any(spread == 0.0):
        const_idx = np.where(spread == 0.0)[0]
        raise ValueError(
            "build_spline_basis_from_options: some covariates in X_sample "
            f"are constant (indices {const_idx.tolist()}). Spline basis may "
            "be ill-defined. Consider dropping or treating them as intercepts."
        )

    # ------------------------------------------------------------------
    # 5. Internal helper: ensure X_new is 2D (n, d)
    # ------------------------------------------------------------------
    def _ensure_2d(X_new: np.ndarray) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        elif X_new.ndim != 2:
            raise ValueError("X_new must be 1D (d,) or 2D (n, d).")
        if X_new.shape[1] != d:
            raise ValueError(
                f"X_new has dimension d={X_new.shape[1]}, but basis was built "
                f"for d={d}."
            )
        return X_new

    # ------------------------------------------------------------------
    # 6. Build and return the feature map
    # ------------------------------------------------------------------
    def fmap(X_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the spline basis at X_new.

        Parameters
        ----------
        X_new : np.ndarray
            Shape (n, d) or (d,).

        Returns
        -------
        P : np.ndarray
            Basis matrix of shape (n, K_total).
        """
        X_new_2d = _ensure_2d(X_new)

        # Clip evaluation points into the domain [X_min_arr, X_max_arr]
        # to avoid extrapolation causing NaNs in prodspline.
        X_new_clipped = np.clip(X_new_2d, X_min_arr, X_max_arr)

        P = prodspline(
            x=X_sample,
            xeval=X_new_clipped,
            K=K_mat,
            knots=knots,
            basis=basis,
            x_min=X_min_arr,
            x_max=X_max_arr,
        )

        P = np.asarray(P, dtype=float)

        # Defensive check: if NaNs still appear, fail loudly.
        if np.isnan(P).any():
            nan_rows = np.where(np.isnan(P).any(axis=1))[0]
            nan_cols = np.where(np.isnan(P).any(axis=0))[0]
            raise ValueError(
                "prodspline produced NaNs in the basis matrix.\n"
                f"NaN rows (first few): {nan_rows[:10]}\n"
                f"NaN cols (first few): {nan_cols[:10]}\n"
                f"Example offending X_new row: {X_new_2d[nan_rows[0]]}"
            )

        return P

    return fmap
