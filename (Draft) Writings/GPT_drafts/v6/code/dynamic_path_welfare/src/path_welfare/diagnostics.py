"""Identification and data-quality diagnostics (task section 7).

None of these can *prove* an untestable assumption.  Positivity and the four path
counts are design facts; sequential ignorability is supplied by randomization;
Markov sufficiency and continuous-state adequacy are *testable* and reported as
diagnostics, never asserted as "satisfied".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from .schemas import continuity_report, path_counts


# ---------------------------------------------------------------------- #
# Continuous-state diagnostics
# ---------------------------------------------------------------------- #
def continuous_state_diagnostics(df: pd.DataFrame) -> dict:
    out = {}
    for name in ("S", "X"):
        rep = continuity_report(df[name].to_numpy(), name)
        by_arm = {}
        for t1 in (0, 1):
            v = df.loc[df["T1"] == t1, name].to_numpy()
            by_arm[f"T1={t1}"] = {"n": int(v.size), "min": float(np.nanmin(v)),
                                  "max": float(np.nanmax(v))}
        out[name] = {
            "n_unique": rep.n_unique, "max_point_mass": rep.max_point_mass,
            "quantiles": rep.q, "effectively_continuous": rep.effectively_continuous,
            "genuinely_continuous": rep.genuinely_continuous, "note": rep.notes,
            "support_by_T1": by_arm,
        }
    return out


# ---------------------------------------------------------------------- #
# Positivity
# ---------------------------------------------------------------------- #
def positivity_report(df: pd.DataFrame, e1=None, e2=None, availability_col=None) -> dict:
    counts = path_counts(df)
    n = len(df)
    p_t1 = float(np.mean(df["T1"]))
    # second-stage empirical probability, overall and by availability
    if availability_col and availability_col in df.columns:
        avail = df[availability_col] == 1
        p_t2 = float(np.mean(df.loc[avail, "T2"]))
        n_stage2 = int(avail.sum())
    else:
        p_t2 = float(np.mean(df["T2"]))
        n_stage2 = n
    eff = {lbl: int(c) for lbl, c in counts.items()}
    smallest = min(p_t1, 1 - p_t1, p_t2, 1 - p_t2)
    return {
        "path_counts": counts,
        "P(T1=1)_empirical": p_t1,
        "P(T2=1)_empirical": p_t2,
        "P(T1=1)_design": e1,
        "P(T2=1)_design": e2,
        "n_entering_stage2": n_stage2,
        "path_effective_sizes": eff,
        "smallest_assignment_prob": smallest,
        "positivity_holds_at_0.10": bool(smallest >= 0.10),
        "n_missing_T1": int(df["T1"].isna().sum()),
        "n_missing_T2": int(df["T2"].isna().sum()),
    }


# ---------------------------------------------------------------------- #
# Markov sufficiency
# ---------------------------------------------------------------------- #
def markov_sufficiency(df: pd.DataFrame, *, seed=0, threshold=0.05) -> dict:
    """Compare held-out fit of the restricted model Y~f(X,T2) against the rich model
    Y~f(S,T1,X,T2).  Flags Markov sufficiency as questionable if the rich model reduces
    held-out MSE by more than ``threshold`` (default 5%)."""
    d = df.dropna(subset=["S", "X", "Y", "T1", "T2"])
    Y = d["Y"].to_numpy()
    Xr = np.column_stack([d["X"], d["T2"]])
    Xrich = np.column_stack([d["S"], d["T1"], d["X"], d["T2"]])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    def mse(Xmat):
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1,
                                              max_iter=200, random_state=seed)
        scores = cross_val_score(model, Xmat, Y, cv=kf, scoring="neg_mean_squared_error")
        return float(-np.mean(scores))

    mse_r = mse(Xr)
    mse_rich = mse(Xrich)
    var_y = float(np.var(Y))
    gain = (mse_r - mse_rich) / mse_r if mse_r > 0 else 0.0
    return {
        "mse_restricted_YXT2": mse_r,
        "mse_rich_YSXT1T2": mse_rich,
        "r2_restricted": 1 - mse_r / var_y,
        "r2_rich": 1 - mse_rich / var_y,
        "incremental_mse_reduction": gain,
        "markov_questionable": bool(gain > threshold),
        "interpretation": (
            "rich history improves held-out fit by >5%: scalar Markov restriction is "
            "questionable; scalar estimator interpretation is model-dependent"
            if gain > threshold else
            "rich history does not materially improve held-out fit at the 5% threshold "
            "(consistent with -- but does not prove -- Markov sufficiency)"
        ),
    }


# ---------------------------------------------------------------------- #
# Balance / randomization diagnostics
# ---------------------------------------------------------------------- #
def balance_checks(df: pd.DataFrame, *, seed=0) -> dict:
    """Can baseline covariates predict treatment assignment beyond chance?  Used only as
    a randomization diagnostic -- balance does not prove ignorability."""
    d = df.dropna(subset=["S", "X", "T1", "T2"])
    out = {}
    # T1 predicted by S
    auc1 = _auc_predict(d[["S"]].to_numpy(), d["T1"].to_numpy(), seed)
    out["T1_from_S_auc"] = auc1
    # T2 predicted by (S,T1,X)
    auc2 = _auc_predict(d[["S", "T1", "X"]].to_numpy(), d["T2"].to_numpy(), seed)
    out["T2_from_history_auc"] = auc2
    out["note"] = ("AUC near 0.5 is consistent with randomization; AUC>>0.5 flags a "
                   "randomization or coding problem. Balance is a diagnostic, not proof "
                   "of ignorability.")
    return out


def _auc_predict(X, y, seed):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    if len(np.unique(y)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    try:
        proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------- #
# Attrition / missingness
# ---------------------------------------------------------------------- #
def attrition_report(df: pd.DataFrame) -> dict:
    n = len(df)
    out = {"n": n}
    for col in ("S", "X", "Y"):
        out[f"missing_{col}"] = float(df[col].isna().mean())
    # missing Y by path
    by_path = {}
    for (t1, t2), lbl in (((0, 0), "00"), ((0, 1), "01"), ((1, 0), "10"), ((1, 1), "11")):
        sub = df[(df["T1"] == t1) & (df["T2"] == t2)]
        by_path[lbl] = {"n": int(len(sub)),
                        "missing_Y": float(sub["Y"].isna().mean()) if len(sub) else float("nan")}
    out["missing_Y_by_path"] = by_path
    # missing Y by baseline-state quartile
    d = df.dropna(subset=["S"]).copy()
    d["Sq"] = pd.qcut(d["S"], 4, labels=False, duplicates="drop")
    out["missing_Y_by_S_quartile"] = {
        int(q): float(g["Y"].isna().mean()) for q, g in d.groupby("Sq")
    }
    if "site" in df.columns:
        out["missing_Y_by_site"] = {
            str(s): float(g["Y"].isna().mean()) for s, g in df.groupby("site")
        }
    return out


def full_identification_audit(df, cfg, e1=None, e2=None, availability_col=None) -> dict:
    """Assemble the full identification audit dictionary."""
    return {
        "continuous_state": continuous_state_diagnostics(df),
        "positivity": positivity_report(df, e1, e2, availability_col),
        "markov_sufficiency": markov_sufficiency(df, seed=cfg.seed if cfg else 0),
        "balance": balance_checks(df, seed=cfg.seed if cfg else 0),
        "attrition": attrition_report(df),
    }
