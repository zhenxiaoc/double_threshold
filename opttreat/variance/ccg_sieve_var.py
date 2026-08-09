"""SieveVar formulas used by the Chen, Chen, and Gao (2025) simulations."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.config import ParameterType
from opttreat.data import ensure_2d_features, ensure_vector
from .base import VarianceEstimator


def _block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Build a dense block diagonal matrix without adding a scipy.linalg dependency."""
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]), dtype=float)
    out[: A.shape[0], : A.shape[1]] = A
    out[A.shape[0] :, A.shape[1] :] = B
    return out


class CCGSieveVariance(VarianceEstimator):
    """
    R-compatible SieveVar estimator for the CCG 2025 replication scripts.

    The implementation follows the local R scripts:
    - known welfare: Sobol pathwise derivative over the target distribution;
    - unknown welfare: common-support empirical derivative plus empirical term;
    - value known: level-set derivative with bandwidth eps and target-volume scale.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        super().__init__(name="ccg_sieve_var", options=options)

    @staticmethod
    def _param_type_str(param_type: Any) -> str:
        if isinstance(param_type, ParameterType):
            return param_type.value
        return str(param_type or "").lower()

    def _patty(self, output: dict[str, Any]) -> tuple[np.ndarray, int]:
        Psi_t = np.asarray(output["Psi_t"], dtype=float)
        Psi_c = np.asarray(output["Psi_c"], dtype=float)
        e_t = ensure_vector(output["e_t"], n=Psi_t.shape[0], name="e_t")
        e_c = ensure_vector(output["e_c"], n=Psi_c.shape[0], name="e_c")

        B = _block_diag(Psi_t, Psi_c)
        residuals = np.concatenate([e_t, e_c])
        pinv_rcond = float(self.options.get("pinv_rcond", np.sqrt(np.finfo(float).eps)))
        BBinvB = np.linalg.pinv(B.T @ B, rcond=pinv_rcond) @ B.T
        weighted = BBinvB * residuals[None, :]
        return weighted @ weighted.T, residuals.shape[0]

    @staticmethod
    def _bases(output: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_map_t = output["feature_map_t"]
        feature_map_c = output["feature_map_c"]
        Psi_t = np.asarray(feature_map_t(X), dtype=float)
        Psi_c = np.asarray(feature_map_c(X), dtype=float)
        return np.hstack([Psi_t, -Psi_c]), ensure_vector(output["h_hat"](X), n=X.shape[0], name="h_hat(X)")

    def _sobol_bun(self, output: dict[str, Any], *, value: bool) -> np.ndarray:
        dim = int(self.options["dim"])
        n_sobol = int(self.options.get("n_sobol", 40000))
        lower = np.broadcast_to(np.asarray(self.options.get("target_lower", 0.0), dtype=float), (dim,))
        upper = np.broadcast_to(np.asarray(self.options.get("target_upper", 1.0), dtype=float), (dim,))
        scramble = bool(self.options.get("sobol_scramble", False))
        seed = self.options.get("sobol_seed", None)
        chunk_size = int(self.options.get("chunk_size", n_sobol))
        engine = Sobol(d=dim, scramble=scramble, seed=None if seed is None or not scramble else int(seed))

        total = None
        seen = 0
        generated = 0
        while generated < n_sobol:
            size = min(chunk_size, n_sobol - generated)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The balance properties of Sobol")
                U = engine.random(size)
            X = lower + U * (upper - lower)
            bases, h_vals = self._bases(output, X)

            if value:
                eps = float(self.options.get("eps", 0.005))
                if eps <= 0:
                    raise ValueError("CCGSieveVariance requires eps > 0 for value parameters.")
                good = (h_vals > -eps) & (h_vals < eps)
                scale = float(self.options.get("value_scale", 1.0)) / (2.0 * eps)
            else:
                good = h_vals >= 0.0
                scale = 1.0

            contrib = bases[good].sum(axis=0) if good.any() else np.zeros(bases.shape[1])
            total = contrib if total is None else total + contrib
            seen += bases.shape[0]
            generated += size

        if total is None:
            raise RuntimeError("No Sobol points were generated for CCGSieveVariance.")
        return scale * total / seen

    def fit(self, estimator_output: dict[str, Any]) -> float:
        param_type = self._param_type_str(self.options.get("param_type"))
        patty, n_fit = self._patty(estimator_output)

        if "value" in param_type:
            Bun = self._sobol_bun(estimator_output, value=True)
            var_hat = float(Bun.T @ patty @ Bun)
        elif "unknown" in param_type:
            X_eval = ensure_2d_features(estimator_output["X_eval"], name="X_eval")
            bases, h_vals = self._bases(estimator_output, X_eval)
            good = h_vals >= 0.0
            Bun = bases[good].sum(axis=0) / X_eval.shape[0] if good.any() else np.zeros(bases.shape[1])
            welfare_vals = np.maximum(h_vals, 0.0)
            empirical = float(welfare_vals.var(ddof=0))
            var_hat = float((empirical + (Bun.T @ patty @ Bun) * n_fit) / X_eval.shape[0])
        else:
            Bun = self._sobol_bun(estimator_output, value=False)
            var_hat = float(Bun.T @ patty @ Bun)

        self.var_hat_ = max(var_hat, 0.0)
        self.se_ = float(np.sqrt(self.var_hat_))
        return self.var_hat_
