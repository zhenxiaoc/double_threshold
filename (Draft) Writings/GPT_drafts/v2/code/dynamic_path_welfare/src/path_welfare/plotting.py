"""Figures (task section 16).  Every figure carries an annotation footer identifying
the dataset, n, treatment coding, state definitions, outcome, cost, and whether it is a
primary or sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FigMeta:
    dataset: str
    n: int
    treatment_coding: str = "T1=1/T2=1 = active treatment"
    state_def: str = "S baseline state; X intermediate state"
    outcome: str = "Y"
    cost: float = 0.0
    kind: str = "primary"  # or "sensitivity"

    def footer(self) -> str:
        return (f"dataset={self.dataset} | n={self.n} | {self.treatment_coding} | "
                f"{self.state_def} | outcome={self.outcome} | cost={self.cost} | {self.kind}")


def _finish(fig, ax_or_axes, meta: FigMeta, path: Path, title: str):
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.005, meta.footer(), ha="center", fontsize=6, wrap=True, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_path_counts(est, meta, outdir):
    from . import PATH_LABELS
    counts = {lbl: int(((est.data_["T1"] == t1) & (est.data_["T2"] == t2)).sum())
              for (t1, t2), lbl in zip(PATH_LABELS.keys(), PATH_LABELS.values())}
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(counts.keys()), list(counts.values()), color="#4C72B0")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_xlabel("(T1,T2) path"); ax.set_ylabel("count")
    _finish(fig, ax, meta, Path(outdir) / "fig03_path_counts.png", "Treatment-path counts")


def fig_state_distributions(est, meta, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, name in zip(axes, ("S", "X")):
        ax.hist(est.data_[name].dropna(), bins=40, color="#55A868", alpha=0.85)
        ax.set_title(f"{name}: {est.data_[name].nunique()} unique values")
        ax.set_xlabel(name)
    _finish(fig, axes, meta, Path(outdir) / "fig04_state_distributions.png",
            "Distributions of S and X")


def fig_support_by_arm(est, meta, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, name, t in zip(axes, ("S", "X"), ("T1", "T2")):
        for val, c in ((0, "#C44E52"), (1, "#4C72B0")):
            ax.hist(est.data_.loc[est.data_[t] == val, name].dropna(), bins=30,
                    alpha=0.5, label=f"{t}={val}", color=c)
        ax.set_xlabel(name); ax.legend(fontsize=7)
    _finish(fig, axes, meta, Path(outdir) / "fig05_support_by_arm.png",
            "State support by treatment arm")


def fig_mu(est, meta, outdir, truth=None):
    x = np.linspace(*np.percentile(est.data_["X"].dropna(), [1, 99]), 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, est.predict_mu(0, x), label="mu_0(x)", color="#C44E52")
    ax.plot(x, est.predict_mu(1, x), label="mu_1(x)", color="#4C72B0")
    ax.set_xlabel("X"); ax.set_ylabel("E[Y|X,T2]"); ax.legend()
    _finish(fig, ax, meta, Path(outdir) / "fig06_mu.png", "Stage-two regressions mu_0, mu_1")


def fig_delta(est, meta, outdir):
    x = np.linspace(*np.percentile(est.data_["X"].dropna(), [1, 99]), 400)
    d = est.predict_delta(x)
    b = est.find_boundaries()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(x, d, color="#8172B3", label="delta(x)=mu_1-mu_0")
    for r in b["delta"].roots:
        ax.axvline(r.location, color="k", ls="--", lw=0.8)
        ax.text(r.location, ax.get_ylim()[1], f"root {r.location:.2f}\nd'={r.derivative:.2f}",
                fontsize=6, va="top")
    ax.plot(est.data_["X"], np.full(len(est.data_), ax.get_ylim()[0]), "|", color="0.4",
            ms=4, alpha=0.15)
    ax.set_xlabel("X"); ax.set_ylabel("delta(x)"); ax.legend(fontsize=8)
    _finish(fig, ax, meta, Path(outdir) / "fig07_delta_roots.png",
            "Second-stage contrast delta(x), roots and data rug")


def fig_A(est, meta, outdir):
    s = np.linspace(*np.percentile(est.data_["S"].dropna(), [1, 99]), 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, est.predict_A(0, s), label="A_0(s)", color="#C44E52")
    ax.plot(s, est.predict_A(1, s), label="A_1(s)", color="#4C72B0")
    ax.set_xlabel("S"); ax.set_ylabel("E[V2(X)|S,T1]"); ax.legend()
    _finish(fig, ax, meta, Path(outdir) / "fig08_A.png", "Continuation values A_0, A_1")


def fig_kappa(est, meta, outdir):
    s = np.linspace(*np.percentile(est.data_["S"].dropna(), [1, 99]), 400)
    k = est.predict_kappa(s)
    b = est.find_boundaries()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(s, k, color="#CCB974", label="kappa(s)=A_1-A_0")
    for r in b["kappa"].roots:
        ax.axvline(r.location, color="k", ls="--", lw=0.8)
        ax.text(r.location, ax.get_ylim()[1], f"root {r.location:.2f}\nk'={r.derivative:.2f}",
                fontsize=6, va="top")
    ax.plot(est.data_["S"], np.full(len(est.data_), ax.get_ylim()[0]), "|", color="0.4",
            ms=4, alpha=0.15)
    ax.set_xlabel("S"); ax.set_ylabel("kappa(s)"); ax.legend(fontsize=8)
    _finish(fig, ax, meta, Path(outdir) / "fig09_kappa_roots.png",
            "First-stage contrast kappa(s), roots and data rug")


def fig_G11(est, meta, outdir):
    s = np.linspace(*np.percentile(est.data_["S"].dropna(), [1, 99]), 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, est.predict_G11(s), color="#4C72B0", label="G_11(s)")
    ax.set_xlabel("S"); ax.set_ylabel("G_11(s)"); ax.legend()
    _finish(fig, ax, meta, Path(outdir) / "fig10_G11.png",
            "(1,1) second-stage component G_11(s)")


def fig_regions(est, meta, outdir):
    x = np.linspace(*np.percentile(est.data_["X"].dropna(), [1, 99]), 200)
    s = np.linspace(*np.percentile(est.data_["S"].dropna(), [1, 99]), 200)
    g2 = (est.predict_delta(x) >= 0).astype(int)
    g1 = (est.predict_kappa(s) >= 0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].fill_between(s, 0, g1, step="mid", color="#4C72B0", alpha=0.6)
    axes[0].set_title("first-stage: 1{kappa(s)>=0}"); axes[0].set_xlabel("S")
    axes[1].fill_between(x, 0, g2, step="mid", color="#55A868", alpha=0.6)
    axes[1].set_title("second-stage: 1{delta(x)>=0}"); axes[1].set_xlabel("X")
    _finish(fig, axes, meta, Path(outdir) / "fig11_regions.png",
            "Optimal treatment regions (tie -> 1)")


def fig_boundary_bands(est, meta, outdir):
    diag = est.boundary_diagnostics()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, lbl in zip(axes, ("delta", "kappa"), ("|delta(X_i)|<=h", "|kappa(S_i)|<=h")):
        bc = diag[key]["band_counts"]
        ax.bar([f"{h:.3f}" for h in bc], list(bc.values()), color="#8172B3")
        ax.set_title(lbl); ax.set_xlabel("band h"); ax.set_ylabel("count")
    _finish(fig, axes, meta, Path(outdir) / "fig12_boundary_bands.png",
            "Boundary-band sample sizes N(h)")


def fig_cost(cost_rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(cost_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["cost"], df["V11"], "o-", color="#4C72B0")
    ax.set_xlabel("cost c (outcome SD units)"); ax.set_ylabel("V_11(c)")
    _finish(fig, ax, meta, Path(outdir) / "fig13_cost.png", "V_11 by treatment cost")


def fig_spline_sensitivity(rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df["K"], df["estimate"], yerr=1.96 * df["se"], fmt="o-", capsize=3,
                color="#C44E52")
    ax.set_xlabel("sieve dimension K"); ax.set_ylabel("V_11 (+- 1.96 conditional SE)")
    _finish(fig, ax, meta, Path(outdir) / "fig14_spline_sensitivity.png",
            "Estimates by sieve dimension")


def fig_method_comparison(rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(df))
    ax.errorbar(df["estimate"], y, xerr=[df["estimate"] - df["lo"], df["hi"] - df["estimate"]],
                fmt="o", capsize=3, color="#4C72B0")
    ax.set_yticks(y); ax.set_yticklabels(df["method"])
    ax.set_xlabel("V_11 estimate and CI")
    _finish(fig, ax, meta, Path(outdir) / "fig15_method_comparison.png",
            "Plug-in vs IPW vs AIPW")


def fig_mc_coverage(mc_rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(mc_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.95, color="0.5", ls="--", lw=1)
    for col, c, lbl in (("cov95", "#4C72B0", "sieve conditional"),
                        ("boot_cov95", "#55A868", "participant bootstrap")):
        if col in df.columns:
            ax.plot(df["dgp"].astype(str) + "_n" + df["n"].astype(str), df[col], "o-",
                    color=c, label=lbl)
    ax.set_ylim(0, 1); ax.set_ylabel("95% coverage"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    _finish(fig, ax, meta, Path(outdir) / "fig16_mc_coverage.png", "Monte Carlo coverage")


def fig_mc_length(mc_rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(mc_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["dgp"].astype(str) + "_n" + df["n"].astype(str), df["median_len95_sd"], "o-",
            color="#8172B3")
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="1 SD (usable ceiling)")
    ax.axhline(0.5, color="0.7", ls=":", lw=1, label="0.5 SD (informative)")
    ax.set_ylabel("median 95% length (outcome SD)"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    _finish(fig, ax, meta, Path(outdir) / "fig17_mc_length.png", "Monte Carlo interval length")


def fig_root_stability(rows, meta, outdir):
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, c in (("delta", "#4C72B0"), ("kappa", "#C44E52")):
        sub = df[df["which"] == name]
        ax.plot(sub["K"], sub["root"], "o-", color=c, label=f"{name} root")
    ax.set_xlabel("sieve dimension K"); ax.set_ylabel("estimated root location"); ax.legend()
    _finish(fig, ax, meta, Path(outdir) / "fig18_root_stability.png",
            "Root-location stability across K")


def fig_missingness(est, meta, outdir):
    from .diagnostics import attrition_report
    att = attrition_report(est.data_)
    bp = att["missing_Y_by_path"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(bp.keys()), [v["missing_Y"] for v in bp.values()], color="#C44E52")
    ax.set_ylabel("missing Y fraction"); ax.set_xlabel("path")
    _finish(fig, ax, meta, Path(outdir) / "fig19_missingness.png", "Missingness by treatment path")


def fig_timing_diagram(meta, outdir):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    steps = ["S\n(baseline)", "T1\n(randomized)", "X\n(intermediate)", "T2\n(randomized)", "Y\n(outcome)"]
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((i * 2, 0), 1.6, 1, fc="#4C72B0" if i % 2 else "#55A868",
                                   alpha=0.7))
        ax.text(i * 2 + 0.8, 0.5, s, ha="center", va="center", fontsize=9, color="white")
        if i < 4:
            ax.annotate("", xy=(i * 2 + 2, 0.5), xytext=(i * 2 + 1.6, 0.5),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.2, 9.5); ax.set_ylim(-0.2, 1.2)
    _finish(fig, ax, meta, Path(outdir) / "fig01_timing.png", "Study timing: S -> T1 -> X -> T2 -> Y")


def fig_sample_flow(steps, meta, outdir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis("off")
    y = 1.0
    for label, n in steps:
        ax.text(0.5, y, f"{label}: n={n}", ha="center",
                bbox=dict(boxstyle="round", fc="#DDDDEE"))
        y -= 0.2
    _finish(fig, ax, meta, Path(outdir) / "fig02_sample_flow.png", "Sample construction flow")


def make_core_figures(est, meta, outdir, truth=None):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig_timing_diagram(meta, outdir)
    fig_path_counts(est, meta, outdir)
    fig_state_distributions(est, meta, outdir)
    fig_support_by_arm(est, meta, outdir)
    fig_mu(est, meta, outdir, truth)
    fig_delta(est, meta, outdir)
    fig_A(est, meta, outdir)
    fig_kappa(est, meta, outdir)
    fig_G11(est, meta, outdir)
    fig_regions(est, meta, outdir)
    fig_boundary_bands(est, meta, outdir)
    fig_missingness(est, meta, outdir)
