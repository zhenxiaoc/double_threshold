"""Generate figures for the surrogate-induced-harm-share study.

Usage:  PYTHONPATH=src python make_figures.py   (run after run_study.py)
Writes PNGs to results/figures.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harm_share.calibration import build_oracle, COVARIATES_2D
from harm_share.functionals import _grid2d

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN, ORANGE, GREY = "#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#4a5568"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 120})


def _load(name):
    return json.loads((LOGS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def fig_geometry(orc):
    """Two CATE surfaces, their zero level sets (M_S, M_Y), harm region, corner."""
    g, GX, GY, tS, tY, fn, dx = _grid2d(orc, n_grid=400, span=3.0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    vmax = np.quantile(np.abs(np.r_[tS.ravel(), tY.ravel()]), 0.98)
    for a, (t, lab) in zip(ax, [(tS, r"$\hat\tau_S(x)$  (short-run, 2-yr CATE)"),
                                (tY, r"$\hat\tau_Y(x)$  (long-run, 3-yr CATE)")]):
        pc = a.pcolormesh(GX, GY, t, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        cs = a.contour(GX, GY, tS, levels=[0], colors="k", linewidths=2.2)
        cy = a.contour(GX, GY, tY, levels=[0], colors=GREEN, linewidths=2.2, linestyles="--")
        a.contourf(GX, GY, ((tS >= 0) & (tY < 0)).astype(float), levels=[0.5, 1.5],
                   colors=[ORANGE], alpha=0.22)
        # density contours for support
        a.contour(GX, GY, fn, levels=np.quantile(fn[fn > 0], [0.5, 0.85]), colors="0.4",
                  linewidths=0.6, alpha=0.5)
        a.set_title(lab)
        a.set_xlabel("baseline consumption (quantile-normal)")
        a.set_ylabel("baseline asset index (quantile-normal)")
        plt.colorbar(pc, ax=a, fraction=0.046, pad=0.04)
    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color="k", lw=2.2, label=r"$M_S=\{\tau_S=0\}$"),
               Line2D([0], [0], color=GREEN, lw=2.2, ls="--", label=r"$M_Y=\{\tau_Y=0\}$"),
               Patch(facecolor=ORANGE, alpha=0.3, label=r"harm region $\{\tau_S\geq0,\tau_Y<0\}$")]
    ax[0].legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle("Two decision boundaries meet at a codimension-2 corner (Banerjee graduation calibration)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_geometry.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_geometry.png")


def fig_confusion(orc):
    """Scatter of (tau_S, tau_Y) over a population draw, colored by sign quadrant."""
    rng = np.random.default_rng(0)
    X = orc.draw_X(6000, rng)
    tS, tY = orc.tau_S(X), orc.tau_Y(X)
    tr = _load("truth.json")["mc"]
    q_pp = (tS >= 0) & (tY >= 0); q_pm = (tS >= 0) & (tY < 0)
    q_mp = (tS < 0) & (tY >= 0); q_mm = (tS < 0) & (tY < 0)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2), width_ratios=[1.25, 1])
    for m, c, lab in [(q_pp, BLUE, "++ correctly treated"),
                      (q_pm, RED, "+− HARM (help now, hurt later)"),
                      (q_mp, GREEN, "−+ withheld despite gain"),
                      (q_mm, GREY, "−− correctly untreated")]:
        ax[0].scatter(tS[m], tY[m], s=7, c=c, alpha=0.45, label=lab, edgecolors="none")
    ax[0].axhline(0, color="k", lw=1); ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel(r"$\tau_S(X)$  short-run CATE"); ax[0].set_ylabel(r"$\tau_Y(X)$  long-run CATE")
    ax[0].set_title("Sign classification of long-run by short-run effect")
    ax[0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    lo = np.quantile(np.r_[tS, tY], 0.01); hi = np.quantile(np.r_[tS, tY], 0.99)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    # confusion matrix
    M = np.array([[tr["theta_pp"], tr["theta_harm"]], [tr["theta_mp"], tr["theta_mm"]]])
    im = ax[1].imshow(M, cmap="Oranges", vmin=0, vmax=M.max())
    for (i, j), v in np.ndenumerate(M):
        ax[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=15,
                   color="white" if v > M.max() * 0.6 else "black", fontweight="bold")
    ax[1].set_xticks([0, 1], [r"$\tau_Y\geq0$", r"$\tau_Y<0$"])
    ax[1].set_yticks([0, 1], [r"$\tau_S\geq0$", r"$\tau_S<0$"])
    ax[1].set_title(f"Population confusion matrix\n"
                    f"harm share $\\theta$={tr['theta_harm']:.3f},  "
                    f"$\\rho=P(\\tau_Y{{<}}0\\,|\\,\\tau_S{{\\geq}}0)$={tr['rho']:.3f}")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_confusion.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_confusion.png")


def fig_rate():
    """log-log RMSE vs n: irregular theta & treat-share vs regular companion."""
    rate = _load("rate_experiment.json")
    cells = rate["cells"]; sl = rate["slopes"]
    n = np.array([c["n"] for c in cells], float)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # panel A: RMSE
    ax[0].loglog(n, [c["rmse"] for c in cells], "o-", color=RED,
                 label=fr"$\hat\theta$ harm share (slope {sl['theta_rmse_slope']:.2f})")
    ax[0].loglog(n, [c["treatS_rmse"] for c in cells], "s-", color=ORANGE,
                 label=fr"treat share $\Pr(\tau_S\geq0)$ (slope {sl['treatS_rmse_slope']:.2f})")
    if "W_sd_slope" in sl:
        wsd = np.array([c["W_mc_sd"] for c in cells])
        ax[0].loglog(n, wsd / wsd[0] * [c["rmse"] for c in cells][0], "^-", color=BLUE,
                     label=fr"regular companion $W_Y$ (SD slope {sl['W_sd_slope']:.2f})")
    ref = [c["rmse"] for c in cells][0] * (n / n[0]) ** (-0.5)
    ax[0].loglog(n, ref, "k--", lw=1, label=r"$n^{-1/2}$ reference")
    ax[0].set_xlabel("n"); ax[0].set_ylabel("RMSE / SD")
    ax[0].set_title("Convergence: threshold functionals vs regular companion\n"
                    "(fixed K, smooth oracle: near root-n; see text)")
    ax[0].legend(fontsize=8.5); ax[0].grid(True, which="both", alpha=0.25)
    # panel B: bias vs sd decomposition
    ax[1].loglog(n, [abs(c["bias"]) for c in cells], "o-", color=RED, label=r"$|$bias$|$ of $\hat\theta$")
    ax[1].loglog(n, [c["mc_sd"] for c in cells], "s-", color=BLUE, label=r"MC-SD of $\hat\theta$")
    ax[1].loglog(n, [c["mean_se"] for c in cells], "^--", color=GREEN, label="two-band sieve SE")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("value")
    ax[1].set_title("Bias vs SD, and the two-band SE\n(SE tracks MC-SD; bias small on smooth oracle)")
    ax[1].legend(fontsize=9); ax[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_rate.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_rate.png")


def fig_inference():
    """se_ratio, coverage vs n; the two-band sieve SE validation."""
    rate = _load("rate_experiment.json")["cells"]
    n = np.array([c["n"] for c in rate], float)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax[0].plot(n, [c["se_ratio"] for c in rate], "o-", color=BLUE)
    ax[0].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.3)
    ax[0].set_xlabel("n"); ax[0].set_ylabel("mean sieve SE / MC-SD")
    ax[0].set_title("Two-band sieve-Riesz SE vs Monte-Carlo SD\n(new derivation; ratio near 1 validates it)")
    ax[0].grid(True, alpha=0.25)
    ax[1].plot(n, [c["cov95_sieve"] for c in rate], "o-", color=RED, label="sieve interval (fixed K)")
    try:
        rg = _load("rate_experiment_growK.json")["cells"]
        ax[1].plot([c["n"] for c in rg], [c["cov95_sieve"] for c in rg], "D-", color=ORANGE,
                   label="sieve interval (growing K)")
    except FileNotFoundError:
        pass
    try:
        boot = _load("bootstrap_coverage.json")
        bn = [b["n"] for b in boot]; bc = [b["cov95_boot"] for b in boot]
        ax[1].plot(bn, bc, "s-", color=GREEN, label="full-refit bootstrap")
    except FileNotFoundError:
        pass
    ax[1].axhline(0.95, color="k", ls="--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_ylim(0.5, 1.0)
    ax[1].set_xlabel("n"); ax[1].set_ylabel("95% CI coverage")
    ax[1].set_title("Coverage of the two-band interval\n(near-nominal at fixed K on the smooth oracle)")
    ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_inference.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_inference.png")


def fig_affine_geometry():
    """Clean companion geometry: affine DGP, straight margins, single crisp corner."""
    from harm_share.affine_dgp import AffineDGP
    dgp = AffineDGP()
    ex = dgp.exact_truth()
    g = np.linspace(-3.2, 3.2, 400)
    GX, GY = np.meshgrid(g, g)
    grid = np.c_[GX.ravel(), GY.ravel()]
    tS = dgp.tau_S(grid).reshape(GX.shape)
    tY = dgp.tau_Y(grid).reshape(GX.shape)
    f = dgp.density(grid).reshape(GX.shape)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.contourf(GX, GY, ((tS >= 0) & (tY < 0)).astype(float), levels=[0.5, 1.5],
                colors=[ORANGE], alpha=0.30)
    ax.contour(GX, GY, tS, levels=[0], colors="k", linewidths=2.4)
    ax.contour(GX, GY, tY, levels=[0], colors=GREEN, linewidths=2.4, linestyles="--")
    ax.contour(GX, GY, f, levels=np.quantile(f, [0.5, 0.8, 0.95]), colors="0.5",
               linewidths=0.7, alpha=0.6)
    cx, cy = ex["corner"]
    ax.plot([cx], [cy], "o", color=RED, ms=11, mec="k", zorder=5)
    ax.annotate(f"corner C\n({cx:.2f},{cy:.2f})\nangle {ex['corner_angle_deg']:.0f}°",
                (cx, cy), (cx - 2.6, cy + 1.1), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="k"))
    # quadrant labels
    ax.text(1.8, 1.8, "++\ncorrectly\ntreated", ha="center", fontsize=9, color=BLUE)
    ax.text(1.8, -2.2, "+−\nHARM", ha="center", fontsize=10, color=RED, fontweight="bold")
    ax.text(-2.3, 1.8, "−+\nwithheld", ha="center", fontsize=9, color=GREEN)
    ax.text(-2.3, -2.2, "−−\nuntreated", ha="center", fontsize=9, color=GREY)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([0], [0], color="k", lw=2.4, label=r"$M_S=\{\tau_S=0\}$"),
                       Line2D([0], [0], color=GREEN, lw=2.4, ls="--", label=r"$M_Y=\{\tau_Y=0\}$"),
                       Patch(facecolor=ORANGE, alpha=0.4, label=r"harm $\{\tau_S\geq0,\tau_Y<0\}$")],
              loc="lower right", fontsize=9)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("baseline consumption (z)"); ax.set_ylabel("baseline asset index (z)")
    ax.set_title(f"Affine DGP: two straight margins meet at one transversal corner\n"
                 fr"exact $\theta$={ex['theta_harm']:.3f}, $\rho$={ex['rho']:.3f} "
                 "(bivariate-normal orthant, zero grid error)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1b_affine_geometry.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1b_affine_geometry.png")


def main():
    print("Building oracle for geometry figures ...")
    orc = build_oracle(covariates=COVARIATES_2D)
    fig_geometry(orc)
    fig_affine_geometry()
    fig_confusion(orc)
    fig_rate()
    fig_inference()
    print("Figures written to results/figures.")


if __name__ == "__main__":
    main()
