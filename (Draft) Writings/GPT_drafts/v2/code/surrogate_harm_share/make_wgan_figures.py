"""Validation of the WGAN calibration against the real data -- the analog of
Chen & Ritzwoller's appendix validation exhibits (their Fig. "Validation":
real-vs-generated means / variances / correlations across all variables; and
Fig. "Comparison of True and Generated Long-Term Outcome Distributions":
long-term-outcome histograms by treatment).

CR report that "the joint distributions of the true and generated data match
remarkably closely", including when moments are conditioned on treatment, and
that the generated long-term outcome histograms have "similarly shaped supports
and right tails". This reproduces those checks for the faithful full-covariate
calibration (all 20 baseline covariates + short/long consumption), in ORIGINAL
units, and writes:

    results/figures/wgan_validation.png   (means / variances / correlations)
    results/figures/wgan_histograms.png   (S, Y by treatment: real vs generated)
    results/logs/wgan_validation.json     (the underlying numbers)

Usage:  PYTHONPATH=src python make_wgan_figures.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harm_share.calibration import load_graduation, TREATMENT
from harm_share.wgan_calibration import WGANOracle

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
FIGS.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "results" / "wgan" / "wgan_oracle.npz"


def assemble(orc, df, n=200_000, seed=0):
    """Real and generated data on a common variable set, in ORIGINAL units."""
    cov = list(orc.cfg.covariates)
    d = len(cov)
    S = df[orc.cfg.s_col].to_numpy(float)
    Y = df[orc.cfg.y_col].to_numpy(float)
    W = df[TREATMENT].to_numpy(float)
    Xr = df[cov].to_numpy(float)
    keep = np.isfinite(S) & np.isfinite(Y)
    Xr = np.where(np.isfinite(Xr), Xr, np.nanmean(Xr, axis=0))       # mean-impute a few NaNs
    real = {"X": Xr[keep], "S": S[keep], "Y": Y[keep], "W": W[keep]}

    gen = orc.sample_experiment(n, np.random.default_rng(seed))
    Xz = gen[[f"X{j+1}" for j in range(orc.d)]].to_numpy(float)      # model matrix (d_model cols)
    genX = orc.raw_covariates(Xz)                                    # decode to original 20 units
    generated = {"X": genX, "S": gen["S"].to_numpy(), "Y": gen["Y"].to_numpy(),
                 "W": gen["W"].to_numpy()}
    names = cov + ["S(2yr cons)", "Y(3yr cons)"]
    return real, generated, names, d


def assemble_surrogates(orc, df, n=100_000, seed=1):
    """Real vs generated FULL 21-dim short-term outcome vector (original units).

    This is the exhibit for the full-surrogate cascade: GAN2 must reproduce the
    joint distribution of ALL two-year outcomes, since Y | S(full), X, W conditions
    on every one of them.  Returns (names, real (n_real,21), gen (n,21))."""
    s_cols = list(getattr(orc, "s_cols", [orc.cfg.s_col]))
    Sr = df[s_cols].to_numpy(float)
    Sr = np.where(np.isfinite(Sr), Sr, np.nanmean(Sr, axis=0))
    rng = np.random.default_rng(seed)
    Xz = orc.draw_X(n, rng)
    Wc = (rng.random(n) < orc.cfg.e).astype(float).reshape(-1, 1)
    Smodel = orc.genS.sample(np.column_stack([Xz, Wc]), n, rng)
    Sg = orc.raw_surrogates(Smodel)                                 # (n, 21) original units
    return s_cols, Sr, Sg


def stack(dat, d):
    return np.column_stack([dat["X"], dat["S"].reshape(-1, 1), dat["Y"].reshape(-1, 1)])


def fig_validation(Mr, Mg, Vr, Vg, Cr, Cg, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    # Means (symlog: means span index ~0 to loans ~1000s)
    a = ax[0]
    a.scatter(Mr, Mg, c="tab:blue", s=32, alpha=0.8, zorder=3)
    lim = [min(Mr.min(), Mg.min()), max(Mr.max(), Mg.max())]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xscale("symlog"); a.set_yscale("symlog")
    a.set_xlabel("real"); a.set_ylabel("generated"); a.set_title("Means (symlog)")
    # Variances (log)
    a = ax[1]
    pos = (Vr > 0) & (Vg > 0)
    a.scatter(Vr[pos], Vg[pos], c="tab:blue", s=32, alpha=0.8, zorder=3)
    lim = [min(Vr[pos].min(), Vg[pos].min()), max(Vr[pos].max(), Vg[pos].max())]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("real"); a.set_ylabel("generated"); a.set_title("Variances (log)")
    # Correlations (all pairs)
    a = ax[2]
    iu = np.triu_indices_from(Cr, k=1)
    a.scatter(Cr[iu], Cg[iu], c="tab:green", s=22, alpha=0.6, zorder=3)
    a.plot([-1, 1], [-1, 1], "k--", lw=1)
    a.set_xlim(-1, 1); a.set_ylim(-1, 1)
    a.set_xlabel("real"); a.set_ylabel("generated")
    a.set_title(f"Correlations ({len(iu[0])} pairs)")
    fig.suptitle("WGAN calibration validation: real vs generated moments, all "
                 f"{Mr.size} variables (45° = perfect match)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_histograms(real, gen, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, (key, name) in enumerate([("S", "Short-term (2-yr) consumption"),
                                     ("Y", "Long-term (3-yr) consumption")]):
        a = ax[i]
        hi = np.quantile(real[key], 0.98)
        bins = np.linspace(0, hi, 45)
        for w, color in [(1, "tab:blue"), (0, "tab:green")]:
            a.hist(real[key][real["W"] == w], bins=bins, density=True, histtype="step",
                   color=color, lw=1.8, label=f"real  W={w}")
            a.hist(gen[key][gen["W"] == w], bins=bins, density=True, histtype="step",
                   color=color, lw=1.4, ls="--", label=f"gen  W={w}")
        a.set_title(name); a.set_xlabel("per-capita monthly consumption")
        a.legend(fontsize=8)
    ax[0].set_ylabel("density")
    fig.suptitle("Real vs generated outcome distributions by treatment "
                 "(CR Fig. 'Comparison of True and Generated ... Distributions')", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_surrogates(names, Sr, Sg, path):
    """Real vs generated moments of the full 21-dim short-term outcome vector."""
    Mr, Mg = Sr.mean(0), Sg.mean(0)
    Vr, Vg = Sr.var(0), Sg.var(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cr, Cg = np.corrcoef(Sr, rowvar=False), np.corrcoef(Sg, rowvar=False)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    a = ax[0]
    a.scatter(Mr, Mg, c="tab:purple", s=32, alpha=0.8, zorder=3)
    lim = [min(Mr.min(), Mg.min()), max(Mr.max(), Mg.max())]
    a.plot(lim, lim, "k--", lw=1); a.set_xscale("symlog"); a.set_yscale("symlog")
    a.set_xlabel("real"); a.set_ylabel("generated"); a.set_title("Surrogate means (symlog)")
    a = ax[1]
    pos = (Vr > 0) & (Vg > 0)
    a.scatter(Vr[pos], Vg[pos], c="tab:purple", s=32, alpha=0.8, zorder=3)
    lim = [min(Vr[pos].min(), Vg[pos].min()), max(Vr[pos].max(), Vg[pos].max())]
    a.plot(lim, lim, "k--", lw=1); a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("real"); a.set_ylabel("generated"); a.set_title("Surrogate variances (log)")
    a = ax[2]
    iu = np.triu_indices_from(Cr, k=1)
    a.scatter(Cr[iu], Cg[iu], c="tab:green", s=22, alpha=0.6, zorder=3)
    a.plot([-1, 1], [-1, 1], "k--", lw=1); a.set_xlim(-1, 1); a.set_ylim(-1, 1)
    a.set_xlabel("real"); a.set_ylabel("generated")
    a.set_title(f"Surrogate correlations ({len(iu[0])} pairs)")
    fig.suptitle(f"Full short-term outcome vector: real vs generated, all {len(names)} "
                 "surrogates (GAN2 = S | X, W)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return float(np.nanmean(np.abs(Cr[iu] - Cg[iu]))), Mr, Mg, Vr ** .5, Vg ** .5


def main():
    if not NPZ.exists():
        raise SystemExit("Train first:  PYTHONPATH=src python train_wgan.py")
    df = load_graduation()
    orc = WGANOracle.load(NPZ, df)
    real, gen, names, d = assemble(orc, df)

    Ar, Ag = stack(real, d), stack(gen, d)
    Mr, Mg = Ar.mean(0), Ag.mean(0)
    Vr, Vg = Ar.var(0), Ag.var(0)
    with np.errstate(invalid="ignore", divide="ignore"):  # some censored vars collapse to a constant
        Cr, Cg = np.corrcoef(Ar, rowvar=False), np.corrcoef(Ag, rowvar=False)

    fig_validation(Mr, Mg, Vr, Vg, Cr, Cg, FIGS / "wgan_validation.png")
    fig_histograms(real, gen, FIGS / "wgan_histograms.png")

    # full 21-dim surrogate vector validation (the full-surrogate cascade exhibit)
    s_names, Sr, Sg = assemble_surrogates(orc, df)
    surr_corr, SMr, SMg, SSr, SSg = fig_surrogates(s_names, Sr, Sg, FIGS / "wgan_surrogates.png")

    iu = np.triu_indices_from(Cr, k=1)
    summary = {
        "n_variables": int(Mr.size),
        "mean_abs_rel_err_mean": float(np.mean(np.abs(Mg - Mr) / (np.abs(Mr) + 1e-6))),
        "corr_match_mean_abs": float(np.nanmean(np.abs(Cr[iu] - Cg[iu]))),
        "corr_match_max_abs": float(np.nanmax(np.abs(Cr[iu] - Cg[iu]))),
        "per_variable": {names[j]: {"mean_real": float(Mr[j]), "mean_gen": float(Mg[j]),
                                    "sd_real": float(Vr[j] ** .5), "sd_gen": float(Vg[j] ** .5)}
                         for j in range(Mr.size)},
        "surrogate_vector": {
            "n_surrogates": int(len(s_names)),
            "corr_match_mean_abs": surr_corr,
            "per_surrogate": {s_names[j]: {"mean_real": float(SMr[j]), "mean_gen": float(SMg[j]),
                                           "sd_real": float(SSr[j]), "sd_gen": float(SSg[j])}
                              for j in range(len(s_names))},
        },
    }
    (LOGS / "wgan_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull short-term vector ({len(s_names)} surrogates): correlation match "
          f"mean|Δ|={surr_corr:.3f}")

    print(f"Real vs generated across {Mr.size} variables (original units):")
    print(f"{'variable':26s} {'mean r/g':>24s} {'sd r/g':>24s}")
    for j, nm in enumerate(names):
        print(f"  {nm:24s} {Mr[j]:10.2f} /{Mg[j]:10.2f}   {Vr[j]**.5:10.2f} /{Vg[j]**.5:10.2f}")
    print(f"\ncorrelation match: mean|Δ|={summary['corr_match_mean_abs']:.3f}  "
          f"max|Δ|={summary['corr_match_max_abs']:.3f}")
    print(f"wrote {FIGS/'wgan_validation.png'}, {FIGS/'wgan_histograms.png'}, "
          f"{LOGS/'wgan_validation.json'}")


if __name__ == "__main__":
    main()
