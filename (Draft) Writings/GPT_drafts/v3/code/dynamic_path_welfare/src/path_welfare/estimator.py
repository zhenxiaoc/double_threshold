"""``TwoStagePathWelfareEstimator`` -- the primary plug-in estimator.

Implements the cross-fitted direct-regression plug-in of the four path components
``V_ab^*`` (and hence total welfare ``V^*``) with honest inner cross-fitting for the
stage-one pseudo-outcomes (task sections 6, 8).  The (1,1) component ``V_11^*`` is the
headline target.

Design map (all univariate splines; see ``docs/theory_summary.md``):

  mu_a(x)   = E[Y | X=x, T2=a]                    stage-two regressions
  delta(x)  = mu_1 - mu_0,  V2 = max(mu0,mu1)
  A_a(s)    = E[V2_hat(X) | S=s, T1=a]            continuation value
  kappa(s)  = A_1 - A_0
  Z1(x)     = mu1_hat(x) 1{delta_hat>=0}          treated stage-2 payoff
  Z0(x)     = mu0_hat(x) 1{delta_hat<0}           untreated stage-2 payoff  (V2 = Z1+Z0)
  G_11(s)=E[Z1|S,T1=1]  G_10(s)=E[Z0|S,T1=1]
  G_01(s)=E[Z1|S,T1=0]  G_00(s)=E[Z0|S,T1=0]
  contribution_ab_i = G_ab(S_i) 1{kappa(S_i) (>=0 if a==1 first stage else <0)}
  V_ab_hat = mean_i contribution_ab_i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import PATH_LABELS
from .boundaries import BoundaryResult, boundary_band_counts, classify_roots
from .config import Config
from .crossfit import check_no_group_leak, make_folds
from .splines import SplineFit, fit_spline, make_basis, select_dim_cv

_TIE_TO_ONE = True  # action 1 chosen when a contrast is exactly zero


@dataclass
class NuisanceSet:
    """Fitted nuisances on one training set (all univariate splines)."""

    mu0: SplineFit
    mu1: SplineFit
    A0: SplineFit
    A1: SplineFit
    G11: SplineFit
    G10: SplineFit
    G01: SplineFit
    G00: SplineFit
    dims: dict[str, int]

    def delta(self, x: np.ndarray) -> np.ndarray:
        return self.mu1.predict(x) - self.mu0.predict(x)

    def delta_deriv(self, x: np.ndarray) -> np.ndarray:
        return self.mu1.predict_deriv(x) - self.mu0.predict_deriv(x)

    def v2(self, x: np.ndarray) -> np.ndarray:
        m0, m1 = self.mu0.predict(x), self.mu1.predict(x)
        return np.maximum(m0, m1)

    def kappa(self, s: np.ndarray) -> np.ndarray:
        return self.A1.predict(s) - self.A0.predict(s)

    def kappa_deriv(self, s: np.ndarray) -> np.ndarray:
        return self.A1.predict_deriv(s) - self.A0.predict_deriv(s)


def _sel2(delta_hat: np.ndarray) -> np.ndarray:
    """Second-stage optimal indicator 1{delta>=0} with tie -> 1."""
    return (delta_hat >= 0).astype(float) if _TIE_TO_ONE else (delta_hat > 0).astype(float)


def _sel1(kappa_hat: np.ndarray) -> np.ndarray:
    return (kappa_hat >= 0).astype(float) if _TIE_TO_ONE else (kappa_hat > 0).astype(float)


class TwoStagePathWelfareEstimator:
    """Plug-in estimator for path-specific welfare components under the optimal
    two-stage regime."""

    def __init__(self, config: Config):
        self.config = config
        self.data_: pd.DataFrame | None = None
        self.full_: NuisanceSet | None = None
        self.contributions_: pd.DataFrame | None = None
        self.point_: dict[str, float] = {}
        self.fold_dims_: list[dict[str, int]] = []
        self.e1_: float | None = None
        self.e2_: float | None = None
        self._rng = np.random.default_rng(config.seed)

    # ================================================================== #
    # Fitting
    # ================================================================== #
    def fit(self, data: pd.DataFrame) -> "TwoStagePathWelfareEstimator":
        df = data.reset_index(drop=True).copy()
        self.data_ = df
        n = len(df)
        groups = df["group"].to_numpy() if "group" in df.columns else None

        # design propensities (known constants preferred; else empirical constant)
        self.e1_ = self.config.treatment_probs.e1 or float(np.mean(df["T1"]))
        self.e2_ = self.config.treatment_probs.e2 or float(np.mean(df["T2"]))

        folds = make_folds(
            n, self.config.crossfit.n_outer_folds, seed=self.config.crossfit.seed, groups=groups
        )
        if groups is not None:
            assert check_no_group_leak(folds, groups), "participant leaked across folds"

        contrib = {lbl: np.full(n, np.nan) for lbl in PATH_LABELS.values()}
        contrib["Astar"] = np.full(n, np.nan)  # A1 1{k>=0}+A0 1{k<0}  (for sum check)
        self.fold_dims_ = []
        self.folds_ = folds
        self.fold_nuis_: list[tuple[np.ndarray, np.ndarray, NuisanceSet]] = []

        for k, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
            nuis, dims = self._fit_nuisances(df, train_idx, fold_id=k)
            self.fold_dims_.append(dims)
            self.fold_nuis_.append((test_idx, train_idx, nuis))
            s_te = df["S"].to_numpy()[test_idx]
            kap = nuis.kappa(s_te)
            sel1 = _sel1(kap)
            g11 = nuis.G11.predict(s_te)
            g10 = nuis.G10.predict(s_te)
            g01 = nuis.G01.predict(s_te)
            g00 = nuis.G00.predict(s_te)
            contrib["11"][test_idx] = g11 * sel1
            contrib["10"][test_idx] = g10 * sel1
            contrib["01"][test_idx] = g01 * (1 - sel1)
            contrib["00"][test_idx] = g00 * (1 - sel1)
            a1 = nuis.A1.predict(s_te)
            a0 = nuis.A0.predict(s_te)
            contrib["Astar"][test_idx] = a1 * sel1 + a0 * (1 - sel1)

        self.contributions_ = pd.DataFrame(contrib)
        if groups is not None:
            self.contributions_["group"] = groups
        self.point_ = {lbl: float(np.nanmean(contrib[lbl])) for lbl in PATH_LABELS.values()}
        self.point_["total"] = sum(self.point_[l] for l in PATH_LABELS.values())
        self.point_["total_direct"] = float(np.nanmean(contrib["Astar"]))
        self.point_["sum_residual"] = self.point_["total"] - self.point_["total_direct"]

        # full-sample nuisances for plotting / boundaries / inference
        self.full_, _ = self._fit_nuisances(df, np.arange(n), fold_id=-1)
        return self

    # ------------------------------------------------------------------ #
    def _fit_nuisances(
        self, df: pd.DataFrame, idx: np.ndarray, *, fold_id: int
    ) -> tuple[NuisanceSet, dict[str, int]]:
        cfg = self.config
        sub = df.iloc[idx]
        S = sub["S"].to_numpy()
        X = sub["X"].to_numpy()
        Y = sub["Y"].to_numpy()
        T1 = sub["T1"].to_numpy()
        T2 = sub["T2"].to_numpy()
        groups = sub["group"].to_numpy() if "group" in sub.columns else None

        # stage-two dims via inner CV (record per fold)
        m0_idx = np.where(T2 == 0)[0]
        m1_idx = np.where(T2 == 1)[0]
        seed = int(abs(hash(("mu", fold_id))) % (2**31))
        dim0, _ = select_dim_cv(
            X[m0_idx], Y[m0_idx], cfg.spline.candidate_dims,
            ridge=cfg.spline.ridge, n_folds=cfg.spline.inner_cv_folds,
            condition_number_max=cfg.spline.condition_number_max, seed=seed,
        )
        dim1, _ = select_dim_cv(
            X[m1_idx], Y[m1_idx], cfg.spline.candidate_dims,
            ridge=cfg.spline.ridge, n_folds=cfg.spline.inner_cv_folds,
            condition_number_max=cfg.spline.condition_number_max, seed=seed + 1,
        )
        # basis on the whole training X-range so honest predictions extrapolate safely
        basis0 = make_basis(X, dim0, cfg.spline.degree)
        basis1 = make_basis(X, dim1, cfg.spline.degree)
        mu0 = fit_spline(X[m0_idx], Y[m0_idx], dim0, ridge=cfg.spline.ridge, basis=basis0)
        mu1 = fit_spline(X[m1_idx], Y[m1_idx], dim1, ridge=cfg.spline.ridge, basis=basis1)

        # honest inner cross-fitted mu predictions for ALL training rows
        mu0_hat, mu1_hat = self._honest_mu(
            X, Y, T2, dim0, dim1, basis0, basis1, groups=groups, fold_id=fold_id
        )
        delta_hat = mu1_hat - mu0_hat
        sel = _sel2(delta_hat)
        z1 = mu1_hat * sel          # treated stage-2 payoff
        z0 = mu0_hat * (1 - sel)    # untreated stage-2 payoff
        v2_hat = z1 + z0            # = max(mu0_hat, mu1_hat)

        # stage-one regressions on S
        s_dim = cfg.density.mean_dim  # reuse a moderate spline dim for S-regressions
        s_dim = min(s_dim, max(4, len(np.unique(S)) // 3))
        t1_1 = np.where(T1 == 1)[0]
        t1_0 = np.where(T1 == 0)[0]
        A1 = fit_spline(S[t1_1], v2_hat[t1_1], s_dim, ridge=cfg.spline.ridge)
        A0 = fit_spline(S[t1_0], v2_hat[t1_0], s_dim, ridge=cfg.spline.ridge)
        G11 = fit_spline(S[t1_1], z1[t1_1], s_dim, ridge=cfg.spline.ridge)
        G10 = fit_spline(S[t1_1], z0[t1_1], s_dim, ridge=cfg.spline.ridge)
        G01 = fit_spline(S[t1_0], z1[t1_0], s_dim, ridge=cfg.spline.ridge)
        G00 = fit_spline(S[t1_0], z0[t1_0], s_dim, ridge=cfg.spline.ridge)

        dims = {"mu0": dim0, "mu1": dim1, "S": s_dim}
        return NuisanceSet(mu0, mu1, A0, A1, G11, G10, G01, G00, dims), dims

    def _honest_mu(self, X, Y, T2, dim0, dim1, basis0, basis1, *, groups, fold_id):
        """Inner cross-fit: predict mu_a(X_i) without using row i's own outcome."""
        n = X.size
        mu0_hat = np.empty(n)
        mu1_hat = np.empty(n)
        inner = make_folds(
            n, self.config.crossfit.n_inner_folds,
            seed=int(abs(hash(("inner", fold_id))) % (2**31)), groups=groups,
        )
        for j, ite in enumerate(inner):
            itr = np.setdiff1d(np.arange(n), ite, assume_unique=False)
            tr0 = itr[T2[itr] == 0]
            tr1 = itr[T2[itr] == 1]
            # guard: need enough rows per arm
            if tr0.size >= dim0 + 2:
                f0 = fit_spline(X[tr0], Y[tr0], dim0, ridge=self.config.spline.ridge, basis=basis0)
                mu0_hat[ite] = f0.predict(X[ite])
            else:
                mu0_hat[ite] = np.mean(Y[T2 == 0])
            if tr1.size >= dim1 + 2:
                f1 = fit_spline(X[tr1], Y[tr1], dim1, ridge=self.config.spline.ridge, basis=basis1)
                mu1_hat[ite] = f1.predict(X[ite])
            else:
                mu1_hat[ite] = np.mean(Y[T2 == 1])
        return mu0_hat, mu1_hat

    # ================================================================== #
    # Prediction API (uses full-sample nuisances)
    # ================================================================== #
    def _full(self) -> NuisanceSet:
        if self.full_ is None:
            raise RuntimeError("call fit() first")
        return self.full_

    def predict_mu(self, a: int, x: np.ndarray) -> np.ndarray:
        f = self._full()
        return (f.mu1 if a == 1 else f.mu0).predict(np.asarray(x, dtype=float))

    def predict_delta(self, x: np.ndarray) -> np.ndarray:
        return self._full().delta(np.asarray(x, dtype=float))

    def predict_v2(self, x: np.ndarray) -> np.ndarray:
        return self._full().v2(np.asarray(x, dtype=float))

    def predict_A(self, a: int, s: np.ndarray) -> np.ndarray:
        f = self._full()
        return (f.A1 if a == 1 else f.A0).predict(np.asarray(s, dtype=float))

    def predict_kappa(self, s: np.ndarray) -> np.ndarray:
        return self._full().kappa(np.asarray(s, dtype=float))

    def predict_G(self, label: str, s: np.ndarray) -> np.ndarray:
        f = self._full()
        return getattr(f, f"G{label}").predict(np.asarray(s, dtype=float))

    def predict_G11(self, s: np.ndarray) -> np.ndarray:
        return self.predict_G("11", s)

    # ================================================================== #
    # Point estimates
    # ================================================================== #
    def estimate_path_value(self, path: tuple[int, int]) -> float:
        return self.point_[PATH_LABELS[path]]

    def estimate_all_paths(self) -> dict[str, float]:
        return dict(self.point_)

    def estimate_total_value(self) -> float:
        return self.point_["total"]

    # ================================================================== #
    # Boundaries
    # ================================================================== #
    def find_boundaries(self, n_grid: int = 2001) -> dict[str, BoundaryResult]:
        f = self._full()
        X = self.data_["X"].to_numpy()
        S = self.data_["S"].to_numpy()
        res_delta = classify_roots(
            "delta", f.delta, f.delta_deriv, X, n_grid=n_grid
        )
        res_kappa = classify_roots(
            "kappa", f.kappa, f.kappa_deriv, S, n_grid=n_grid
        )
        return {"delta": res_delta, "kappa": res_kappa}

    def boundary_diagnostics(self) -> dict[str, Any]:
        f = self._full()
        X = self.data_["X"].to_numpy()
        S = self.data_["S"].to_numpy()
        b = self.find_boundaries()
        sd_y = float(np.nanstd(self.data_["Y"].to_numpy()))
        hs_delta = [0.05 * sd_y, 0.10 * sd_y, 0.20 * sd_y]
        hs_kappa = [0.05 * sd_y, 0.10 * sd_y, 0.20 * sd_y]
        out = {
            "delta": {
                "roots": [r.__dict__ for r in b["delta"].roots],
                "has_crossing": b["delta"].has_crossing,
                "n_sign_changes": b["delta"].n_sign_changes,
                "band_counts": boundary_band_counts(X, f.delta(X), hs_delta),
            },
            "kappa": {
                "roots": [r.__dict__ for r in b["kappa"].roots],
                "has_crossing": b["kappa"].has_crossing,
                "n_sign_changes": b["kappa"].n_sign_changes,
                "band_counts": boundary_band_counts(S, f.kappa(S), hs_kappa),
            },
        }
        return out

    # ================================================================== #
    # Inference / bootstrap (delegated)
    # ================================================================== #
    def inference(self, **kwargs):
        from .riesz import sieve_riesz_inference

        return sieve_riesz_inference(self, **kwargs)

    def bootstrap(self, **kwargs):
        from .bootstrap import run_bootstrap

        return run_bootstrap(self, **kwargs)

    def simulate_calibrated_dgp(self, **kwargs):
        from .simulation import CalibratedDGP

        return CalibratedDGP.from_estimator(self, **kwargs)
