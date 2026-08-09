"""Figures for the debias study: coverage, bias, and log-log RMSE by estimator
and sample size, small-multiple panels (DGP columns x estimator-family rows).

Reads  results/tables/debias_study.csv
Writes results/figures/fig_debias_{coverage,bias,rate}.png

Form: change-across-n comparison -> line charts, small multiples, <=3 series
per panel (sieve family / ML family), fixed categorical hue order, direct
labels, single y-axis per panel, reference lines (0.95 for coverage, 0 for
bias). Static print figures for the paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
CSV = ROOT / "results" / "tables" / "debias_study.csv"
FIGDIR = ROOT / "results" / "figures"

# validated categorical palette (dataviz default, light mode), fixed order
C1, C2, C3 = "#2a78d6", "#008300", "#e87ba4"   # blue, green, magenta
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e1"

FAMILIES = {
    "sieve first stage": [("plugin", "Plug-in", C1),
                          ("ss_margins", "SS margins-only", C2),
                          ("ss", "SS corner-aware", C3)],
    "GBR first stage": [("cf_gbr", "Cross-fit", C1),
                        ("cf_riesz_gbr", "+ Riesz", C2),
                        ("dd_gbr", "+ quad (DD)", C3)],
}
DGP_LABEL = {"krr": "KRR (smooth truth)",
             "wgan2ds": "WGAN scalar (rough truth)",
             "wgan2d": "WGAN full-surr. (degenerate)",
             "affine": "Affine (exact truth)"}


def _style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if title:
        ax.set_title(title, fontsize=9, color=INK)


def _panelgrid(df, value, ylabel, refline, fname, logy=False, logx=True,
               annotate_slope=False):
    dgps = [d for d in ["krr", "wgan2ds", "wgan2d", "affine"] if d in set(df.dgp)]
    nrow, ncol = len(FAMILIES), len(dgps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, (fam, members) in enumerate(FAMILIES.items()):
        for j, dgp in enumerate(dgps):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            sub = df[df.dgp == dgp]
            for est, label, color in members:
                g = sub[sub.est == est].sort_values("n")
                if g.empty:
                    continue
                x, y = g["n"].to_numpy(float), g[value].to_numpy(float)
                ax.plot(x, y, color=color, linewidth=2, marker="o",
                        markersize=4, zorder=3, label=label)
                # selective direct label at the last point (top row only to
                # avoid clutter; legend carries identity everywhere)
                txt = label
                if annotate_slope and len(g) >= 3:
                    A = np.vstack([np.log(x), np.ones_like(x)]).T
                    sl = np.linalg.lstsq(A, np.log(np.maximum(y, 1e-12)),
                                         rcond=None)[0][0]
                    txt = f"{label} ({sl:+.2f})"
                ax.annotate(txt, (x[-1], y[-1]), textcoords="offset points",
                            xytext=(4, 0), fontsize=7, color=color,
                            ha="left", va="center")
            if refline is not None:
                ax.axhline(refline, color=MUTED, linewidth=1,
                           linestyle="--", zorder=1)
            if logx:
                ax.set_xscale("log")
                ax.set_xticks(sorted(sub["n"].unique()))
                ax.get_xaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
            if logy:
                ax.set_yscale("log")
            _style(ax, title=(DGP_LABEL[dgp] if i == 0 else None))
            if j == 0:
                ax.set_ylabel(f"{ylabel}\n[{fam}]", fontsize=8, color=INK)
            if i == nrow - 1:
                ax.set_xlabel("n", fontsize=8, color=INK)
            # room for the direct labels
            ax.margins(x=0.28)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    FIGDIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGDIR / fname, dpi=200)
    plt.close(fig)
    print("wrote", FIGDIR / fname)


def _load():
    frames = [pd.read_csv(CSV)]
    extra = CSV.parent / "debias_study_wgan2ds.csv"
    if extra.exists():
        frames.append(pd.read_csv(extra))
    return pd.concat(frames, ignore_index=True)


def main():
    df = _load()
    _panelgrid(df, "cov95", "95% coverage", 0.95, "fig_debias_coverage.png")
    _panelgrid(df, "bias", "bias", 0.0, "fig_debias_bias.png")
    _panelgrid(df, "rmse", "RMSE", None, "fig_debias_rate.png",
               logy=True, annotate_slope=True)


if __name__ == "__main__":
    main()
