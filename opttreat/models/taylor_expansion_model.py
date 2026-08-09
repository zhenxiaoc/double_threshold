"""Taylor-expansion data-generating process family."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd
from scipy.special import bernoulli

from opttreat.models.model_base import ModelBase


CONSTANT_GAP_MODE = "constant_gap_multivar"
VARIABLE_GAP_MODE = "variable_gap_univariate"
SUPPORTED_EXPANSIONS = ("tan2", "sinh2", "rational", "hyperbolic", "exp_pm")


@dataclass(frozen=True)
class TaylorExpansionSpec:
    """One Taylor expansion formula specification for a fixed truncation order."""

    name: str
    K: int
    mode: str
    baseline_coeffs: np.ndarray
    degrees: np.ndarray | None = None
    effect_constant: float | None = None
    treated_coeffs: np.ndarray | None = None
    z_index: int = 0
    include_constant: bool = False


def sec2_taylor_coeffs(K: int) -> np.ndarray:
    """Return coefficients for the Taylor series of sec(z)^2 through order K."""
    if K < 0:
        raise ValueError("K must be >= 0.")

    bernoulli_numbers = bernoulli(2 * (K + 1))
    coeffs = np.zeros(K + 1, dtype=np.float64)

    for k in range(K + 1):
        n = k + 1
        b_2n = float(bernoulli_numbers[2 * n])
        tan_coeff = (
            ((-1) ** (n - 1))
            * (2 ** (2 * n))
            * (2 ** (2 * n) - 1)
            * b_2n
            / factorial(2 * n)
        )
        coeffs[k] = (2 * n - 1) * tan_coeff

    coeffs[0] = 1.0
    return coeffs


def tan2_taylor_coeffs(K: int) -> np.ndarray:
    """Return the first K nonconstant coefficients for tan(z)^2."""
    if K <= 0:
        raise ValueError("K must be positive for tan2 coefficients.")
    return sec2_taylor_coeffs(K)[1:]


def _sinh2_taylor_coeffs(K: int) -> np.ndarray:
    """Return the first K nonconstant coefficients for sinh(z)^2."""
    if K <= 0:
        raise ValueError("K must be positive for sinh2 coefficients.")
    coeffs = np.zeros(K, dtype=float)
    for k in range(1, K + 1):
        coeffs[k - 1] = (2 ** (2 * k - 1)) / factorial(2 * k)
    return coeffs


def _poly_eval(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate a univariate polynomial using Horner's method."""
    coeffs = np.asarray(coeffs, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)

    out = np.zeros_like(z)
    for coeff in coeffs[::-1]:
        out = out * z + coeff
    return out


def _coeffs_cosh(K: int) -> np.ndarray:
    """Return coefficients for cosh(z) through degree K."""
    if K < 0:
        raise ValueError("K must be >= 0.")
    coeffs = np.zeros(K + 1, dtype=float)
    for k in range(0, K + 1, 2):
        coeffs[k] = 1.0 / factorial(k)
    return coeffs


def _coeffs_sinh(K: int) -> np.ndarray:
    """Return coefficients for sinh(z) through degree K."""
    if K < 0:
        raise ValueError("K must be >= 0.")
    coeffs = np.zeros(K + 1, dtype=float)
    for k in range(1, K + 1, 2):
        coeffs[k] = 1.0 / factorial(k)
    return coeffs


def _coeffs_exp_pos(K: int) -> np.ndarray:
    """Return coefficients for exp(z) through degree K."""
    if K < 0:
        raise ValueError("K must be >= 0.")
    return np.array([1.0 / factorial(k) for k in range(K + 1)], dtype=float)


def _coeffs_exp_neg(K: int) -> np.ndarray:
    """Return coefficients for exp(-z) through degree K."""
    if K < 0:
        raise ValueError("K must be >= 0.")
    return np.array([((-1.0) ** k) / factorial(k) for k in range(K + 1)], dtype=float)


def make_taylor_expansion_spec(expansion: str, K: int) -> TaylorExpansionSpec:
    """Build the formula specification for one TaylorExpansionModel."""
    exp = expansion.lower().strip()

    if exp == "tan2":
        return TaylorExpansionSpec(
            name=exp,
            K=K,
            mode=CONSTANT_GAP_MODE,
            baseline_coeffs=tan2_taylor_coeffs(K),
            degrees=2 * np.arange(1, K + 1, dtype=int),
            effect_constant=1.0,
        )

    if exp == "sinh2":
        return TaylorExpansionSpec(
            name=exp,
            K=K,
            mode=CONSTANT_GAP_MODE,
            baseline_coeffs=_sinh2_taylor_coeffs(K),
            degrees=2 * np.arange(1, K + 1, dtype=int),
            effect_constant=1.0,
        )

    if exp == "rational":
        return TaylorExpansionSpec(
            name=exp,
            K=K,
            mode=CONSTANT_GAP_MODE,
            baseline_coeffs=np.arange(2, K + 2, dtype=float),
            degrees=np.arange(1, K + 1, dtype=int),
            effect_constant=1.0,
        )

    if exp == "hyperbolic":
        return TaylorExpansionSpec(
            name=exp,
            K=K,
            mode=VARIABLE_GAP_MODE,
            baseline_coeffs=_coeffs_sinh(K),
            treated_coeffs=_coeffs_cosh(K),
            z_index=0,
        )

    if exp == "exp_pm":
        return TaylorExpansionSpec(
            name=exp,
            K=K,
            mode=VARIABLE_GAP_MODE,
            baseline_coeffs=_coeffs_exp_neg(K),
            treated_coeffs=_coeffs_exp_pos(K),
            z_index=0,
        )

    raise ValueError(f"Unknown expansion={expansion!r}. Supported expansions: {SUPPORTED_EXPANSIONS}.")


class TaylorExpansionModel(ModelBase):
    """Taylor-style DGP with uniform supports and configurable effect formulas."""

    def __init__(
        self,
        K: int = 10,
        p: int | None = None,
        expansion: str = "tan2",
        support0: tuple[float, float] = (-0.2, 1.2),
        support: tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)

        if K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = int(K)

        self.p = int(p if p is not None else K)
        if self.p < 1:
            raise ValueError("p must be >= 1.")

        self.spec = make_taylor_expansion_spec(expansion, self.K)
        self.expansion = self.spec.name
        self.support0 = support0
        self.support = support

        required_dim = self.K if self.spec.mode == CONSTANT_GAP_MODE else self.spec.z_index + 1
        if self.p < required_dim:
            raise ValueError(f"p={self.p} is too small for {self.expansion}; it needs at least {required_dim}.")

    @property
    def feature_columns(self) -> list[str]:
        return [f"X{j}" for j in range(1, self.p + 1)]

    def _features(self, x: np.ndarray, *, name: str = "x") -> np.ndarray:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            if self.p == 1:
                X = X.reshape(-1, 1)
            elif X.shape[0] == self.p:
                X = X.reshape(1, self.p)
            else:
                X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"TaylorExpansionModel.{name} must be 1D or 2D, got shape {X.shape}.")

        if X.shape[1] != self.p:
            raise ValueError(f"{self.expansion} expects x of shape (n, {self.p}); got {X.shape}.")
        return X

    def _draw_X(self, n: int, support: tuple[float, float]) -> pd.DataFrame:
        lo, hi = support
        X = np.random.uniform(lo, hi, size=(n, self.p))
        return pd.DataFrame(X, columns=self.feature_columns)

    def rF0(self, n: int) -> pd.DataFrame:
        return self._draw_X(n, self.support0)

    def rF(self, m: int) -> pd.DataFrame:
        return self._draw_X(m, self.support)

    def inverse_CDF(self, u: np.ndarray) -> np.ndarray:
        U = self._features(u, name="u")
        lo, hi = self.support
        return lo + U * (hi - lo)

    def p0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        return self.constant(X.shape[0], 0.5)

    def _constant_gap_baseline(self, X: np.ndarray) -> np.ndarray:
        if self.spec.degrees is None:
            raise ValueError(f"{self.expansion} is missing polynomial degrees.")

        Xk = X[:, : self.K]
        Xpow = Xk ** self.spec.degrees

        if self.spec.include_constant:
            b0 = float(self.spec.baseline_coeffs[0])
            coeffs = self.spec.baseline_coeffs[1:]
        else:
            b0 = 0.0
            coeffs = self.spec.baseline_coeffs

        return b0 + Xpow @ coeffs

    def _variable_gap_z(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.spec.z_index]

    def baseline(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        if self.spec.mode == CONSTANT_GAP_MODE:
            return self._constant_gap_baseline(X)

        z = self._variable_gap_z(X)
        return _poly_eval(self.spec.baseline_coeffs, z)

    def h0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        if self.spec.mode == CONSTANT_GAP_MODE:
            if self.spec.effect_constant is None:
                raise ValueError(f"{self.expansion} is missing a constant effect.")
            return self.constant(X.shape[0], self.spec.effect_constant)

        if self.spec.treated_coeffs is None:
            raise ValueError(f"{self.expansion} is missing treated-state coefficients.")
        z = self._variable_gap_z(X)
        treated = _poly_eval(self.spec.treated_coeffs, z)
        baseline = _poly_eval(self.spec.baseline_coeffs, z)
        return treated - baseline

    def mu0(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        X = self._features(x)
        d_vec = self.as_vector(d, n=X.shape[0], name="d")
        return self.baseline(X) + d_vec * self.h0(X)

    def generate_data(self, n: int) -> pd.DataFrame:
        df = self.rF0(n)
        X = df[self.feature_columns].to_numpy()
        d = self._draw_treatment(X)
        return self._add_outcome(df, X, d)


__all__ = [
    "SUPPORTED_EXPANSIONS",
    "TaylorExpansionModel",
    "TaylorExpansionSpec",
    "make_taylor_expansion_spec",
    "sec2_taylor_coeffs",
    "tan2_taylor_coeffs",
]
