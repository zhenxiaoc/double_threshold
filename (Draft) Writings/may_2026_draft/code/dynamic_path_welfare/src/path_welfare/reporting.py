"""Tables (task section 17) and report assembly.

Each table builder returns a pandas DataFrame; ``write_table`` saves CSV + a Markdown
copy under ``results/tables``.  Estimates are reported in original and standardized
(outcome-SD) units where applicable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import PATH_LABELS


def write_table(df: pd.DataFrame, name: str, outdir: str | Path, caption: str = "") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)
    md = [f"### {name}", ""]
    if caption:
        md.append(f"_{caption}_\n")
    md.append(df.to_markdown(index=False))
    (outdir / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    return outdir / f"{name}.csv"


# --- Table 2: sample construction --------------------------------------- #
def table_sample_construction(steps: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(steps, columns=["step", "n"])


# --- Table 3: variable definitions and timing --------------------------- #
def table_variable_defs(defs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(defs)


# --- Table 4: continuous-state diagnostics ------------------------------ #
def table_continuous_state(diag: dict) -> pd.DataFrame:
    rows = []
    for name in ("S", "X"):
        c = diag[name]
        rows.append({
            "variable": name, "n_unique": c["n_unique"],
            "max_point_mass": round(c["max_point_mass"], 4),
            "q05": round(c["quantiles"]["q05"], 3), "q50": round(c["quantiles"]["q50"], 3),
            "q95": round(c["quantiles"]["q95"], 3),
            "effectively_continuous": c["effectively_continuous"],
            "genuinely_continuous": c["genuinely_continuous"],
        })
    return pd.DataFrame(rows)


# --- Table 5: path counts & randomization probabilities ----------------- #
def table_path_counts(pos: dict) -> pd.DataFrame:
    rows = [{"path": lbl, "count": pos["path_counts"][lbl]} for lbl in PATH_LABELS.values()]
    df = pd.DataFrame(rows)
    df["P(T1=1)_emp"] = round(pos["P(T1=1)_empirical"], 3)
    df["P(T2=1)_emp"] = round(pos["P(T2=1)_empirical"], 3)
    df["P(T1=1)_design"] = pos["P(T1=1)_design"]
    df["P(T2=1)_design"] = pos["P(T2=1)_design"]
    return df


# --- Table 6: balance / randomization checks ---------------------------- #
def table_balance(bal: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"check": "AUC(T1|S)", "value": round(bal["T1_from_S_auc"], 3)},
        {"check": "AUC(T2|S,T1,X)", "value": round(bal["T2_from_history_auc"], 3)},
    ])


# --- Table 7: attrition -------------------------------------------------- #
def table_attrition(att: dict) -> pd.DataFrame:
    rows = [{"quantity": f"missing_{c}", "value": round(att[f"missing_{c}"], 4)}
            for c in ("S", "X", "Y")]
    for lbl, v in att["missing_Y_by_path"].items():
        rows.append({"quantity": f"missing_Y_path_{lbl}", "value": round(v["missing_Y"], 4)})
    return pd.DataFrame(rows)


# --- Table 8: Markov diagnostics ---------------------------------------- #
def table_markov(ms: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"quantity": "MSE restricted Y~f(X,T2)", "value": round(ms["mse_restricted_YXT2"], 4)},
        {"quantity": "MSE rich Y~f(S,T1,X,T2)", "value": round(ms["mse_rich_YSXT1T2"], 4)},
        {"quantity": "incremental MSE reduction", "value": round(ms["incremental_mse_reduction"], 4)},
        {"quantity": "markov_questionable", "value": ms["markov_questionable"]},
    ])


# --- Table 9: boundary roots -------------------------------------------- #
def table_boundary_roots(bdiag: dict) -> pd.DataFrame:
    rows = []
    for which in ("delta", "kappa"):
        for r in bdiag[which]["roots"]:
            rows.append({
                "boundary": which, "location": round(r["location"], 4),
                "quantile": round(r["quantile"], 3), "derivative": round(r["derivative"], 4),
                "local_n": r["local_n"], "regular": r["regular"],
                "flags": "; ".join(r["flags"]),
            })
    if not rows:
        rows = [{"boundary": "none", "location": np.nan, "quantile": np.nan,
                 "derivative": np.nan, "local_n": 0, "regular": False, "flags": "no crossing"}]
    return pd.DataFrame(rows)


# --- Table 10: path components ------------------------------------------ #
def table_path_components(point: dict, sd_y: float, truth: dict | None = None) -> pd.DataFrame:
    rows = []
    for lbl in PATH_LABELS.values():
        row = {"path": lbl, "V_ab (orig)": round(point[lbl], 4),
               "V_ab (SD units)": round(point[lbl] / sd_y, 4)}
        if truth:
            row["truth"] = round(truth.get(f"V{lbl}", np.nan), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Table 11: total & sum check ---------------------------------------- #
def table_total_check(point: dict, truth: dict | None = None) -> pd.DataFrame:
    rows = [
        {"quantity": "sum of components", "value": round(point["total"], 5)},
        {"quantity": "direct total (E[A*])", "value": round(point["total_direct"], 5)},
        {"quantity": "component-sum residual", "value": round(point["sum_residual"], 6)},
    ]
    if truth:
        rows.append({"quantity": "true total", "value": round(truth["total"], 5)})
    return pd.DataFrame(rows)


# --- Table 12: plug-in vs IPW vs AIPW ----------------------------------- #
def table_method_comparison(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Table 13: sieve-Riesz & bootstrap intervals ------------------------ #
def table_intervals(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Table 14: cost sensitivity ----------------------------------------- #
def table_cost(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Table 15: spline & fold sensitivity -------------------------------- #
def table_spline_sensitivity(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- Table 16: Monte Carlo ---------------------------------------------- #
def table_monte_carlo(mc_aggs: list[dict]) -> pd.DataFrame:
    keep = ["dgp", "n", "n_rep", "V11_true", "bias_sieve", "rmse_sieve", "mc_sd_sieve",
            "mean_se", "se_ratio", "cov90", "cov95", "median_len95_sd", "delta_root_err",
            "kappa_root_err", "fail_rate"]
    df = pd.DataFrame(mc_aggs)
    cols = [c for c in keep if c in df.columns]
    return df[cols].round(4)


# --- Table 17: go/no-go ------------------------------------------------- #
def table_go_no_go(assessment: dict) -> pd.DataFrame:
    return pd.DataFrame([{"criterion": k, "value": v} for k, v in assessment.items()])


def go_no_go(mc_agg: dict, *, roots_stable=True, support_ok=True) -> dict:
    """Evaluate the empirical-CI go/no-go rule (task section 14)."""
    cov = mc_agg.get("cov95", np.nan)
    fail = mc_agg.get("fail_rate", 1.0)
    length = mc_agg.get("median_len95_sd", np.inf)
    se_ratio = mc_agg.get("se_ratio", np.nan)
    usable = (cov >= 0.90 and fail < 0.05 and length <= 1.0 and roots_stable and support_ok
              and (not np.isnan(se_ratio)) and 0.8 <= se_ratio <= 1.25)
    informative = usable and length <= 0.50
    return {
        "coverage95": round(cov, 3),
        "coverage95>=0.90": bool(cov >= 0.90),
        "fail_rate<0.05": bool(fail < 0.05),
        "median_len_sd<=1.0": bool(length <= 1.0),
        "se_ratio_within_20pct": bool((not np.isnan(se_ratio)) and 0.8 <= se_ratio <= 1.25),
        "roots_stable": roots_stable,
        "support_ok": support_ok,
        "USABLE": bool(usable),
        "INFORMATIVE": bool(informative),
        "verdict": ("informative" if informative else "usable" if usable else
                    "NOT usable: sample supports point estimation but not a reliable CI "
                    "for this path-specific optimal-policy component under the maintained "
                    "nonparametric model"),
    }


def write_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
