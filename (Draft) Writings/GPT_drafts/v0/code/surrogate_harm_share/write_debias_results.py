"""Render the debias-study Monte Carlo into (i) a LaTeX table body for
two_threshold_inference.tex and (ii) a markdown summary in docs/.

Usage:  python write_debias_results.py
Reads   results/tables/debias_study.csv
Writes  results/tables/debias_table.tex, docs/debias_results.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CSV = ROOT / "results" / "tables" / "debias_study.csv"
TEX = ROOT / "results" / "tables" / "debias_table.tex"
MD = ROOT / "docs" / "debias_results.md"

LABEL = {
    "plugin": r"Plug-in (sieve)",
    "ss_margins": r"SS margins-only",
    "ss": r"SS corner-aware",
    "cf_gbr": r"Cross-fit GBR",
    "cf_riesz_gbr": r"\ \ $+$ Riesz corr.",
    "dd_gbr": r"\ \ $+$ quad.\ (DD)",
}
ORDER = ["plugin", "ss_margins", "ss", "cf_gbr", "cf_riesz_gbr", "dd_gbr"]
DGP_LABEL = {"krr": "KRR (smooth truth)",
             "wgan2ds": "WGAN scalar (rough truth)",
             "wgan2d": "WGAN full-surrogate (degenerate)",
             "affine": "Affine (exact truth)"}


def latex_table(df: pd.DataFrame) -> str:
    lines = []
    for dgp in ["krr", "wgan2ds", "wgan2d", "affine"]:
        sub = df[df.dgp == dgp]
        if sub.empty:
            continue
        th0 = sub.theta_true.iloc[0]
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{8}}{{l}}{{\emph{{{DGP_LABEL[dgp]}}}, "
            rf"$\theta_0={th0:.3f}$}}\\")
        for est in ORDER:
            g = sub[sub.est == est].sort_values("n")
            if g.empty:
                continue
            first = True
            for _, r in g.iterrows():
                name = LABEL[est] if first else ""
                lines.append(
                    rf"{name} & {int(r['n'])} & {r['bias']:+.4f} & "
                    rf"{r['mc_sd']:.4f} & {r['rmse']:.4f} & "
                    rf"{r['se_ratio']:.2f} & {r['cov95']:.3f} & "
                    rf"{r['mean_len']:.3f}\\")
                first = False
    return "\n".join(lines)


def rate_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dgp, est), g in df.groupby(["dgp", "est"]):
        g = g.sort_values("n")
        if len(g) < 3:
            continue
        logn = np.log(g["n"].to_numpy(float))
        A = np.vstack([logn, np.ones_like(logn)]).T
        s_rmse = float(np.linalg.lstsq(A, np.log(g["rmse"]), rcond=None)[0][0])
        s_sd = float(np.linalg.lstsq(A, np.log(g["mc_sd"]), rcond=None)[0][0])
        rows.append({"dgp": dgp, "est": est,
                     "rmse_slope": s_rmse, "sd_slope": s_sd})
    return pd.DataFrame(rows)


def _load():
    frames = [pd.read_csv(CSV)]
    extra = CSV.parent / "debias_study_wgan2ds.csv"
    if extra.exists():
        frames.append(pd.read_csv(extra))
    return pd.concat(frames, ignore_index=True)


def main():
    df = _load()
    TEX.write_text(latex_table(df), encoding="utf-8")
    print("wrote", TEX)

    slopes = rate_slopes(df)
    md = ["# Debias study results (auto-generated)\n"]
    md.append("Estimand: theta = Pr(tau_S>=0, tau_Y<0); nominal coverage 0.95.\n")
    for dgp in ["krr", "wgan2ds", "wgan2d", "affine"]:
        sub = df[df.dgp == dgp]
        if sub.empty:
            continue
        md.append(f"\n## {DGP_LABEL[dgp]} (theta0 = {sub.theta_true.iloc[0]:.4f})\n")
        md.append("| est | n | bias | sd | rmse | se_ratio | cov95 | len |")
        md.append("|---|---|---|---|---|---|---|---|")
        for est in ORDER:
            for _, r in sub[sub.est == est].sort_values("n").iterrows():
                md.append(
                    f"| {est} | {int(r['n'])} | {r['bias']:+.4f} | "
                    f"{r['mc_sd']:.4f} | {r['rmse']:.4f} | {r['se_ratio']:.2f} "
                    f"| {r['cov95']:.3f} | {r['mean_len']:.3f} |")
        s = slopes[slopes.dgp == dgp]
        if not s.empty:
            md.append("\nlog-log RMSE slopes: " + ", ".join(
                f"{r.est}={r.rmse_slope:.2f}" for r in s.itertuples()))
        # corner diagnostics where present
        cc = sub[sub.est == "ss"].sort_values("n")
        if "corner_mean" in cc and cc["corner_mean"].notna().any():
            md.append("\ncorner cross-term (mean over reps): " + ", ".join(
                f"n={int(r['n'])}: {r['corner_mean']:+.4f}"
                for _, r in cc.iterrows() if np.isfinite(r.get("corner_mean", np.nan))))
    MD.parent.mkdir(exist_ok=True)
    MD.write_text("\n".join(md), encoding="utf-8")
    print("wrote", MD)


if __name__ == "__main__":
    main()
