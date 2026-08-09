"""Sieve-style variance estimator for fitted treatment rules."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats.qmc import Sobol

from opttreat.data import ensure_2d_features, ensure_vector
from .base import VarianceEstimator


class SieveVariance(VarianceEstimator):
    """
    Sieve-based variance estimator for welfare/value parameters built on
    separately-estimated treated/control nuisance functions.

    Attributes:
    - name = "sieve_var"
    - options 

    """

    def __init__(self, options: Dict[str, Any] | None = None):
        super().__init__(name="sieve_var", options=options)

    # ------------------------------------------------------------------
    # Param-type helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _param_type_str(param_type: Any) -> str:
        if param_type is None:
            return ""
        if hasattr(param_type, "value"):
            return str(param_type.value).lower()
        return str(param_type).lower()

    @classmethod
    def _is_welfare(cls, param_type: Any) -> bool:
        pt = cls._param_type_str(param_type)
        return "welfare" in pt

    @classmethod
    def _is_value(cls, param_type: Any) -> bool:
        pt = cls._param_type_str(param_type)
        return "value" in pt

    @classmethod
    def _is_welfare_unknown(cls, param_type: Any) -> bool:
        pt = cls._param_type_str(param_type)
        return "welfare_unknown" in pt

    # ------------------------------------------------------------------
    # Internal: normalize v(X_int) into a vector of length n_int
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_v_vector(v_func: Any, X_int: np.ndarray) -> np.ndarray:
        """
        Return v_vec of shape (n_int,), broadcasting scalars if needed.
        """
        n_int = X_int.shape[0]

        if v_func is None:
            return np.ones(n_int, dtype=float)

        if not callable(v_func):
            raise TypeError("SieveVariance: options['v_func'] must be callable for value-type parameters.")

        return ensure_vector(v_func(X_int), n=n_int, name="v_func(X_int)")

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------
    def fit(self, estimator_output: Dict[str, Any]) -> float:
        """
        Returns estimated asymptotic variance for sqrt(n)(theta_hat - theta).

        Required estimator_output:
            - Psi_t, Psi_c
            - e_t, e_c
            - feature_map_t, feature_map_c
            - h_hat
            - alpha (or options['alpha'])

        Options:
            - dim (required)
            - n_sobol (default 1024)
            - transform (default identity)
            - sobol_seed (default 123456)
            - sobol_scramble (default True)
            - param_type (required: indicates welfare vs value + unknown distribution)
            - eps (only for value; default 0.01)
            - v_func (only for value; default v=1)
        """
        # -------------------------
        # Unpack / validate inputs
        # -------------------------
        Psi_t = np.asarray(estimator_output["Psi_t"])
        Psi_c = np.asarray(estimator_output["Psi_c"])
        e_t = np.asarray(estimator_output["e_t"]).ravel()
        e_c = np.asarray(estimator_output["e_c"]).ravel()

        feature_map_t = estimator_output["feature_map_t"]
        feature_map_c = estimator_output["feature_map_c"]
        h_hat = estimator_output["h_hat"]

        if not callable(feature_map_t):
            raise TypeError("estimator_output['feature_map_t'] must be callable.")
        if not callable(feature_map_c):
            raise TypeError("estimator_output['feature_map_c'] must be callable.")
        if not callable(h_hat):
            raise TypeError("estimator_output['h_hat'] must be callable.")

        n_t, K_t = Psi_t.shape
        n_c, K_c = Psi_c.shape

        # Ridge penalty
        alpha = estimator_output.get("alpha", self.options.get("alpha", None))
        if alpha is None:
            raise ValueError("SieveVariance requires alpha in estimator_output['alpha'] or options['alpha'].")

        # -------------------------
        # Sobol integration points
        # -------------------------
        if "dim" not in self.options:
            raise ValueError("SieveVariance.options must include 'dim'.")

        dim = int(self.options["dim"])
        n_sobol = int(self.options.get("n_sobol", 1024))
        transform = self.options.get("transform", lambda u: u)

        sobol_seed = self.options.get("sobol_seed", 789)
        scramble = bool(self.options.get("sobol_scramble", True))
        engine = Sobol(d=dim, scramble=scramble, seed=sobol_seed if scramble else None)
        U = engine.random(n_sobol)
        X_int = ensure_2d_features(transform(U), name="X_int")

        # Feature maps at integration points
        Psi_t_int = np.asarray(feature_map_t(X_int))
        Psi_c_int = np.asarray(feature_map_c(X_int))

        if Psi_t_int.shape[0] != Psi_c_int.shape[0]:
            raise ValueError("feature_map_t(X_int) and feature_map_c(X_int) must have same #rows.")

        # h at integration points
        h_int = ensure_vector(h_hat(X_int), n=Psi_t_int.shape[0], name="h_hat(X_int)")

        # bases = [Psi_t, -Psi_c]
        bases = np.hstack([Psi_t_int, -Psi_c_int])
        n_int = bases.shape[0]

        # -------------------------
        # Bun computation
        # -------------------------
        param_type = self.options.get("param_type", None)

        if self._is_welfare(param_type):
            # Welfare: 1{h >= 0}
            good = (h_int >= 0.0)
            if good.any():
                Bun = bases[good, :].sum(axis=0) / n_int
            else:
                Bun = np.zeros(K_t + K_c, dtype=float)

        elif self._is_value(param_type):
            # Value: (1/(2eps)) * 1{|h| <= eps} * v(X)
            eps = float(self.options.get("eps", 0.01))
            if eps <= 0:
                raise ValueError("SieveVariance: eps must be positive for value-type parameters.")

            good = (np.abs(h_int) <= eps)

            v_func = self.options.get("v_func", None)
            v_vec = self._eval_v_vector(v_func, X_int)  # (n_int,)

            if good.any():
                Bun = (bases[good, :] * v_vec[good, None]).sum(axis=0) / n_int
                Bun = Bun / (2.0 * eps)
            else:
                Bun = np.zeros(K_t + K_c, dtype=float)

        else:
            raise ValueError(
                "SieveVariance: options['param_type'] must indicate 'welfare' or 'value'. "
                f"Got param_type={param_type}."
            )

        Bun_t = Bun[:K_t]
        Bun_c = Bun[K_t:K_t + K_c]

        # -------------------------
        # Core sieve variance piece
        # -------------------------
        I_t = np.eye(K_t)
        I_c = np.eye(K_c)

        Sigma_t = (Psi_t * (e_t ** 2)[:, None]).T @ Psi_t
        Sigma_c = (Psi_c * (e_c ** 2)[:, None]).T @ Psi_c

        gram_t = Psi_t.T @ Psi_t + alpha * I_t
        gram_c = Psi_c.T @ Psi_c + alpha * I_c
        Patty_t = np.linalg.solve(gram_t, Sigma_t @ np.linalg.solve(gram_t, I_t))
        Patty_c = np.linalg.solve(gram_c, Sigma_c @ np.linalg.solve(gram_c, I_c))

        var_sieve = float(Bun_t.T @ Patty_t @ Bun_t) + float(Bun_c.T @ Patty_c @ Bun_c)
        var_total = var_sieve

        # -------------------------
        # Optional: welfare unknown distribution extra term
        # -------------------------
        if self._is_welfare_unknown(param_type):
            X_unknown = estimator_output.get("X_eval", estimator_output.get("X_all"))
            if X_unknown is None:
                raise ValueError(
                    "SieveVariance: welfare_unknown requires estimator_output['X_eval'] or ['X_all']."
                )

            X_all = ensure_2d_features(X_unknown, name="X_eval")
            h_vals = ensure_vector(h_hat(X_all), n=X_all.shape[0], name="h_hat(X_all)")
            welfare_vals = np.maximum(h_vals, 0.0)

            n = X_all.shape[0]
            var_empirical = float(welfare_vals.var(ddof=1))
            var_total = var_sieve + var_empirical / n

        # Store / return
        self.var_hat_ = float(var_total)
        self.se_ = float(np.sqrt(var_total))
        return float(var_total)

    def __repr__(self) -> str:
        return f"SieveVariance(name={self.name}, options={self.options})"
