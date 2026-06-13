"""Value target parameters."""

from typing import Callable
import warnings
import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.data import ensure_2d_features, ensure_vector
from .base import Parameter


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


def _value_average(
    h: Callable[[np.ndarray], np.ndarray],
    v: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
) -> float:
    """Evaluate E[1{h(X)>0} v(X)] by sample average over X."""
    h_vals = ensure_vector(h(X), n=X.shape[0], name="h(X)")
    v_vals = ensure_vector(v(X), n=X.shape[0], name="v(X)")
    return float(((h_vals > 0.0).astype(float) * v_vals).mean())


class ValueKnownDist(Parameter):
    """
    Value under a known target distribution: E[1{h(X)>0} v(X)].
    """

    def evaluate(self, h: Callable[[np.ndarray], np.ndarray]) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(x).")

        v = self.options.get("v_func", None)
        if not callable(v):
            raise TypeError("options['v_func'] must be a callable v(x).")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", lambda u: u)
        sobol_seed = int(self.options.get("sobol_seed", 456))
        scramble = bool(self.options.get("sobol_scramble", True))

        X = _sobol_draw(dim, n, sobol_seed, transform, scramble=scramble)
        return _value_average(h, v, X)

    def get_true_value(self, model):
        if "true_value" in self.options:
            return float(self.options["true_value"])

        h0 = model.h0
        v0 = self.options.get("v_func", None)

        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        if not callable(v0):
            raise TypeError("options['v_func'] must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", 1)))
        scramble = bool(self.options.get("sobol_scramble", True))
        X = _sobol_draw(dim, n, sobol_seed, model.inverse_CDF, scramble=scramble)
        return _value_average(h0, v0, X)



class ValueUnknownDist(Parameter):
    """
    Value under the empirical distribution: n^{-1} sum_i 1{h(X_i)>0}v(X_i).
    """

    def evaluate(self, h: Callable[[np.ndarray], np.ndarray], X: np.ndarray | None = None) -> float:
        if not callable(h):
            raise TypeError("h must be a callable of the form h(x).")

        v = self.options.get("v_func", None)
        if not callable(v):
            raise TypeError("options['v_func'] must be a callable v(x).")

        if X is None:
            X = self.options.get("X")
        if X is None:
            raise ValueError("ValueUnknownDist.evaluate requires observed X.")

        X_arr = ensure_2d_features(X, name="X")
        return _value_average(h, v, X_arr)

    def get_true_value(self, model):
        h0 = model.h0
        v0 = self.options.get("v_func", None)

        if not callable(h0):
            raise TypeError("model.h0 must be callable.")

        if not callable(v0):
            raise TypeError("options['v_func'] must be callable.")

        dim = int(self.options["dim"])
        n = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", model.inverse_CDF)
        sobol_seed = int(self.options.get("true_sobol_seed", self.options.get("sobol_seed", 456)))
        scramble = bool(self.options.get("sobol_scramble", True))
        X = _sobol_draw(dim, n, sobol_seed, transform, scramble=scramble)
        return _value_average(h0, v0, X)
