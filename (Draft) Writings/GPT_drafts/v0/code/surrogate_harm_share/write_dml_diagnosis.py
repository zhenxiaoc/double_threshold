"""Render the DML-diagnosis experiment into a LaTeX table + a projection figure.

Reads  results/logs/dml_diagnosis.json (+ the dml_diagnosis_*.csv checkpoints)
Writes results/tables/dml_diag_table.tex, results/figures/fig_dml_diagnosis.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
LOGS = ROOT / "results" / "logs"
TABLES = ROOT / "results" / "tables"
FIGDIR = ROOT / "results" / "figures"

# validated categorical palette (dataviz default, light mode)
C1, C2, C3 = "#2a78d6", "#008300", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e1"


def latex_table():
    # read the CSV checkpoint (written after section D+B2), not the end-of-run
    # JSON, so the table can be built before the E/F ladders finish
    rows = pd.read_csv(TABLES / "dml_diagnosis_main.csv").to_dict("records")
    lines = []
    order = ["tensor-sieve plug-in", "gbr K=2 riesz=sieve", "gbr K=5 riesz=sieve",
             "gbr K=10 riesz=sieve", "rf K=5 riesz=sieve", "krr K=5 riesz=sieve",
             "gbr K=5 riesz=rf", "krr K=5 riesz=rf",
             "gbr(high-cap) K=5 riesz=sieve", "gbr K=5 no correction"]
    by = {r["label"]: r for r in rows}
    for lab in order:
        r = by.get(lab)
        if r is None:
            continue
        lines.append(
            rf"{lab.replace('_','\\_')} & {r['bias']:+.4f} & {r['plugin_bias']:+.4f} "
            rf"& {r['mc_sd']:.4f} & {r['se_ratio']:.2f} & {r['cov95']:.3f}\\")
    body = "\n".join(lines)
    (TABLES / "dml_diag_table.tex").write_text(body, encoding="utf-8")
    print("wrote", TABLES / "dml_diag_table.tex")

    # inline into the paper (this document's booktabs + \input interact badly,
    # so we substitute the rows into a marker rather than \input them)
    # This tree is vendored into each version folder as <version>/code/, so the
    # paper is the sibling of that code/ folder.  NB: only the v1 source carries
    # the %%DML_DIAG_TABLE_ROWS%% marker; v2+ dropped it, so this inlining step
    # is a no-op there until the marker is re-added.
    papers = sorted(ROOT.parent.parent.glob("two_threshold_inference_v*.tex"))
    paper = papers[0] if papers else None
    if paper is not None:
        import re
        s = paper.read_text(encoding="utf-8")
        # replacement must be a FUNCTION: a string replacement would interpret
        # backslash escapes and mangle every '\\' row terminator into '\'.
        s2 = re.sub(r"%%DML_DIAG_TABLE_ROWS%%.*?(?=\n\\bottomrule)",
                    lambda _m: "%%DML_DIAG_TABLE_ROWS%%\n" + body, s, flags=re.S)
        paper.write_text(s2, encoding="utf-8")
        print("inlined dml table rows into", paper.name)


def projection_figure():
    dfp = pd.read_csv(TABLES / "dml_diagnosis_projection.csv")
    # average the four r2_{S,Y}{0,1} into one span-R2 per (learner, riesz)
    r2cols = [c for c in dfp.columns if c.startswith("r2_")]
    dfp["r2_span"] = dfp[r2cols].mean(axis=1)
    learners = ["gbr", "rf", "krr"]
    labels = {"gbr": "gradient boosting", "rf": "random forest",
              "krr": "RBF ridge (smooth)"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.patch.set_facecolor("#fcfcfb")

    # panel 1: R2 of learner error on the representer span, by basis
    ax = axes[0]; ax.set_facecolor("#fcfcfb")
    x = np.arange(len(learners))
    for k, (riesz, color) in enumerate([("sieve", C1), ("rf", C2)]):
        vals = [float(dfp[(dfp.learner == l) & (dfp.riesz == riesz)]["r2_span"].mean())
                for l in learners]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color, zorder=3,
               label=f"representer = {riesz}")
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([labels[l] for l in learners], fontsize=8)
    ax.set_ylabel("$R^2$ of first-stage error\non the representer span", fontsize=9)
    ax.set_ylim(0, 1); ax.axhline(1, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Can the representer see the learner's error?", fontsize=9, color=INK)

    # panel 2: attenuation of the CATE and plug-in boundary error
    ax = axes[1]; ax.set_facecolor("#fcfcfb")
    att = [float(dfp[dfp.learner == l]["atten_S"].mean()) for l in learners]
    err = [abs(float(dfp[dfp.learner == l]["plugin_err"].mean())) for l in learners]
    ax.bar(x - 0.19, att, width=0.36, color=C1, zorder=3, label="CATE retention (slope)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, err, width=0.36, color=C3, zorder=3, label="|plug-in boundary bias|")
    for xi, v in zip(x - 0.19, att):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color=INK)
    for xi, v in zip(x + 0.19, err):
        ax2.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([labels[l] for l in learners], fontsize=8)
    ax.set_ylabel("CATE retention $\\langle\\hat\\tau,\\tau_0\\rangle/\\|\\tau_0\\|^2$",
                  fontsize=9, color=C1)
    ax2.set_ylabel("|plug-in boundary bias|", fontsize=9, color=C3)
    ax.set_ylim(0, 1.05); ax.axhline(1, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_title("Attenuation drives the boundary bias", fontsize=9, color=INK)

    for a in list(axes) + [ax2]:
        a.spines[["top"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGDIR / "fig_dml_diagnosis.png", dpi=200)
    plt.close(fig)
    print("wrote", FIGDIR / "fig_dml_diagnosis.png")


if __name__ == "__main__":
    latex_table()
    projection_figure()
