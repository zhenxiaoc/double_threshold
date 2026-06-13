"""Formula-table implementation of Chen, Chen, and Gao (2025) DGPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit

from opttreat.models.model_base import ModelBase


Formula = Callable[[np.ndarray], np.ndarray]
MuFormula = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CCGFormulaSpec:
    """One CCG 2025 model formula and support specification."""

    name: str
    dim: int
    observed_low: tuple[float, ...]
    observed_high: tuple[float, ...]
    target_low: tuple[float, ...]
    target_high: tuple[float, ...]
    propensity: Formula
    baseline: Formula
    effect: Formula
    custom_mu0: MuFormula | None = None


CCG_FORMULA_SPECS: dict[str, CCGFormulaSpec] = {
    "Model1": CCGFormulaSpec(
        "Model1",
        1,
        (-0.2,),
        (1.2,),
        (0.0,),
        (1.0,),
        lambda X: expit(1.0 - 2.0 * X[:, 0]),
        lambda X: 5.0 * np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 0]),
        lambda X: -0.4 + 2.0 * X[:, 0] ** 2,
    ),
    "Model2": CCGFormulaSpec(
        "Model2",
        1,
        (-0.2,),
        (1.2,),
        (0.0,),
        (1.0,),
        lambda X: expit(-0.5 + X[:, 0]),
        lambda X: 0.5 * np.abs(X[:, 0]),
        lambda X: 0.5 - X[:, 0] ** 2,
    ),
    "Model3": CCGFormulaSpec(
        "Model3",
        1,
        (-0.2,),
        (1.2,),
        (0.0,),
        (1.0,),
        lambda X: expit(0.5 - X[:, 0]),
        lambda X: X[:, 0] ** 2,
        lambda X: 1.0 - X[:, 0],
    ),
    "Model4": CCGFormulaSpec(
        "Model4",
        2,
        (-0.2, -0.2),
        (1.2, 1.2),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(X[:, 0] - X[:, 1]),
        lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
        * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
        lambda X: 0.5 * X[:, 0] - 0.4 * X[:, 1],
    ),
    "Model5": CCGFormulaSpec(
        "Model5",
        2,
        (-0.2, -0.2),
        (1.2, 1.2),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(X[:, 0] - X[:, 1]),
        lambda X: (1.0 - X[:, 0] * X[:, 1])
        * (3.0 + np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])),
        lambda X: 0.3 * X[:, 0] - 0.3 * X[:, 1],
    ),
    "Model6": CCGFormulaSpec(
        "Model6",
        2,
        (-0.2, -0.2),
        (1.2, 1.2),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(1.5 * X[:, 0] - 0.5 * X[:, 1]),
        lambda X: np.log(1.0 + X[:, 0] + X[:, 1]),
        lambda X: X[:, 0] - 0.7 * X[:, 1],
    ),
    "Model7": CCGFormulaSpec(
        "Model7",
        2,
        (-0.2, -0.2),
        (1.2, 1.2),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(-0.5 + X[:, 0] + 2.0 * X[:, 1]),
        lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2) * np.exp(-X[:, 0] - X[:, 1]),
        lambda X: 0.5 - X[:, 1],
    ),
    "Model8": CCGFormulaSpec(
        "Model8",
        1,
        (0.0,),
        (1.0,),
        (0.0,),
        (1.0,),
        lambda X: expit(1.0 - 2.0 * X[:, 0]),
        lambda X: 5.0 * np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 0]),
        lambda X: -0.4 + 2.0 * X[:, 0] ** 2,
    ),
    "Model9": CCGFormulaSpec(
        "Model9",
        1,
        (0.0,),
        (1.0,),
        (0.0,),
        (1.0,),
        lambda X: expit(-0.5 + X[:, 0]),
        lambda X: 0.5 * np.abs(X[:, 0]),
        lambda X: 0.5 - X[:, 0] ** 2,
    ),
    "Model10": CCGFormulaSpec(
        "Model10",
        1,
        (0.0,),
        (1.0,),
        (0.0,),
        (1.0,),
        lambda X: expit(0.5 - X[:, 0]),
        lambda X: X[:, 0] ** 2,
        lambda X: 1.0 - X[:, 0],
    ),
    "Model11": CCGFormulaSpec(
        "Model11",
        2,
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(X[:, 0] - X[:, 1]),
        lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
        * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
        lambda X: 0.5 * X[:, 0] - 0.4 * X[:, 1],
    ),
    "Model12": CCGFormulaSpec(
        "Model12",
        2,
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(X[:, 0] - X[:, 1]),
        lambda X: (1.0 - X[:, 0] * X[:, 1])
        * (3.0 + np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])),
        lambda X: 0.3 * X[:, 0] - 0.3 * X[:, 1],
    ),
    "Model13": CCGFormulaSpec(
        "Model13",
        2,
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(1.5 * X[:, 0] - 0.5 * X[:, 1]),
        lambda X: np.log(1.0 + X[:, 0] + X[:, 1]),
        lambda X: X[:, 0] - 0.7 * X[:, 1],
    ),
    "Model14": CCGFormulaSpec(
        "Model14",
        2,
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        lambda X: expit(-0.5 + X[:, 0] + 2.0 * X[:, 1]),
        lambda X: (X[:, 0] ** 2 + X[:, 1] ** 2) * np.exp(-X[:, 0] - X[:, 1]),
        lambda X: 0.5 - X[:, 1],
    ),
    "Model15": CCGFormulaSpec(
        "Model15",
        2,
        (-2.0, -2.0),
        (2.0, 2.0),
        (-1.5, -1.5),
        (1.5, 1.5),
        lambda X: expit(X[:, 0] - X[:, 1]),
        lambda X: np.zeros(X.shape[0], dtype=float),
        lambda X: (1.0 - X[:, 0] ** 2 - X[:, 1] ** 2)
        * (4.0 + np.sin(X[:, 0]) * X[:, 1] + np.cos(X[:, 1])),
    ),
}


class CCGFormulaModel(ModelBase):
    """CCG 2025 formula-table model with uniform rectangular supports."""

    def __init__(self, spec: CCGFormulaSpec | str, noise_sd: float = 1.0):
        super().__init__(noise_sd=noise_sd)
        if isinstance(spec, str):
            try:
                spec = CCG_FORMULA_SPECS[spec]
            except KeyError as exc:
                raise ValueError(f"Unknown CCG formula model {spec!r}.") from exc
        self.spec = spec
        self.name = spec.name

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def feature_columns(self) -> list[str]:
        return ["X"] if self.dim == 1 else [f"X{j}" for j in range(1, self.dim + 1)]

    def _features(self, x: np.ndarray, *, name: str = "x") -> np.ndarray:
        X = self.as_features(x, name=f"{self.spec.name}.{name}")
        if X.shape[1] != self.dim:
            raise ValueError(f"{self.spec.name} expects x of shape (n, {self.dim}).")
        return X

    def _draw_uniform(self, n: int, low: tuple[float, ...], high: tuple[float, ...]) -> pd.DataFrame:
        lower = np.asarray(low, dtype=float)
        upper = np.asarray(high, dtype=float)
        X = lower + np.random.uniform(size=(n, self.dim)) * (upper - lower)
        return pd.DataFrame(X, columns=self.feature_columns)

    def rF0(self, n: int) -> pd.DataFrame:
        return self._draw_uniform(n, self.spec.observed_low, self.spec.observed_high)

    def rF(self, m: int) -> pd.DataFrame:
        return self._draw_uniform(m, self.spec.target_low, self.spec.target_high)

    def inverse_CDF(self, u: np.ndarray) -> np.ndarray:
        U = self._features(u, name="u")
        lower = np.asarray(self.spec.target_low, dtype=float)
        upper = np.asarray(self.spec.target_high, dtype=float)
        return lower + U * (upper - lower)

    def p0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        return self.as_vector(self.spec.propensity(X), n=X.shape[0], name=f"{self.spec.name}.p0")

    def baseline(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        return self.as_vector(self.spec.baseline(X), n=X.shape[0], name=f"{self.spec.name}.baseline")

    def h0(self, x: np.ndarray) -> np.ndarray:
        X = self._features(x)
        return self.as_vector(self.spec.effect(X), n=X.shape[0], name=f"{self.spec.name}.h0")

    def mu0(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        X = self._features(x)
        d_vec = self.as_vector(d, n=X.shape[0], name="d")
        if self.spec.custom_mu0 is not None:
            return self.as_vector(self.spec.custom_mu0(X, d_vec), n=X.shape[0], name=f"{self.spec.name}.mu0")
        return self.baseline(X) + d_vec * self.h0(X)

    def generate_data(self, n: int) -> pd.DataFrame:
        df = self.rF0(n)
        X = df[self.feature_columns].to_numpy()
        d = self._draw_treatment(X)
        return self._add_outcome(df, X, d)


def make_ccg_model(name: str, **kwargs) -> CCGFormulaModel:
    """Instantiate one CCG formula model by name."""
    return CCGFormulaModel(name, **kwargs)


__all__ = ["CCGFormulaModel", "CCGFormulaSpec", "CCG_FORMULA_SPECS", "make_ccg_model"]
