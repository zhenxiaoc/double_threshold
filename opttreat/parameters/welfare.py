"""Welfare target parameters."""

from typing import Callable
import warnings
import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.data import ensure_2d_features, ensure_vector
from .base import Parameter
from opttreat.models.model_base import ModelBase


def _sobol_draw(
    dim: int,
    n: int,
    seed: int,
    transform: Callable[[np.ndarray], np.ndarray],
    *,
    scramble: bool = True,
) -> np.ndarray:
    """Draw transformed Sobol integration points."""
    engine = Sobol(d=dim, scramble=scramble, seed=seed if scramble else None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
        U = engine.random(n)
    return ensure_2d_features(transform(U), name="X_int")


def _welfare_average(h: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> float:
    """Evaluate E[max(h(X), 0)] by sample average over X."""
    h_vals = ensure_vector(h(X), n=X.shape[0], name="h(X)")
    return float(np.maximum(h_vals, 0.0).mean())


class WelfareKnownDist(Parameter):
    """
    Welfare under a known target distribution: E[max(h(X), 0)].
    """

    def evaluate(self, h: Callable[[np.ndarray], np.ndarray]) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(X).")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", lambda u: u)
        sobol_seed = int(self.options.get("sobol_seed", 1))
        scramble = bool(self.options.get("sobol_scramble", True))

        X = _sobol_draw(dim, n, sobol_seed, transform, scramble=scramble)
        return _welfare_average(h, X)

    def get_true_value(self, model: ModelBase) -> float:
        h0 = model.h0
        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", 456)))
        scramble = bool(self.options.get("sobol_scramble", True))

        X = _sobol_draw(dim, n, sobol_seed, model.inverse_CDF, scramble=scramble)
        return _welfare_average(h0, X)



class WelfareUnknownDist(Parameter):
    """
    Welfare under the empirical distribution: n^{-1} sum_i max(h(X_i), 0).
    """

    def evaluate(self, h: Callable[[np.ndarray], np.ndarray], X: np.ndarray | None = None) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h0(X).")
        if X is None:
            X = self.options.get("X")
        if X is None:
            raise ValueError("WelfareUnknownDist.evaluate requires observed X.")

        X_arr = ensure_2d_features(X, name="X")
        return _welfare_average(h, X_arr)

    def get_true_value(self, model: ModelBase) -> float:
        h0 = model.h0
        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", model.inverse_CDF)
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", 456)))
        scramble = bool(self.options.get("sobol_scramble", True))
        X = _sobol_draw(dim, n, sobol_seed, transform, scramble=scramble)
        return _welfare_average(h0, X)

