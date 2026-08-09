"""Train the Chen & Ritzwoller 3-WGAN cascade on the graduation data (the SECOND
calibration), cache it, and report fidelity diagnostics.

Usage:  PYTHONPATH=src python train_wgan.py [--retrain]
Writes: results/wgan/wgan_oracle.npz   (trained generators + population truth)
        results/logs/wgan_calibration.json  (fidelity + truth diagnostics)
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from harm_share.calibration import (
    COVARIATES_2D, COVARIATES_CR_FULL, load_graduation,
)
from harm_share.wgan_backend import WGANSpec
from harm_share.wgan_calibration import WGANOracle, build_wgan_oracle

ROOT = Path(__file__).resolve().parent
WGAN_DIR = ROOT / "results" / "wgan"
LOGS = ROOT / "results" / "logs"
WGAN_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
NPZ = WGAN_DIR / "wgan_oracle.npz"


def raw_fidelity(df, orc, n=100_000, seed=0):
    """Compact real-vs-generated fidelity across ALL covariates + S + Y, in
    ORIGINAL units (the CR validation criterion; full exhibit in make_wgan_figures)."""
    cov = list(orc.cfg.covariates)
    S = df[orc.cfg.s_col].to_numpy(float); Y = df[orc.cfg.y_col].to_numpy(float)
    Xr = df[cov].to_numpy(float)
    keep = np.isfinite(S) & np.isfinite(Y)
    Xr = np.where(np.isfinite(Xr), Xr, np.nanmean(Xr, axis=0))
    Ar = np.column_stack([Xr[keep], S[keep], Y[keep]])
    gen = orc.sample_experiment(n, np.random.default_rng(seed))
    genX = orc.raw_covariates(gen[[f"X{j+1}" for j in range(orc.d)]].to_numpy(float))
    Ag = np.column_stack([genX, gen["S"].to_numpy(), gen["Y"].to_numpy()])
    names = cov + ["S", "Y"]
    Mr, Mg = Ar.mean(0), Ag.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):  # censored vars can collapse to a constant
        Cr, Cg = np.corrcoef(Ar, rowvar=False), np.corrcoef(Ag, rowvar=False)
    iu = np.triu_indices_from(Cr, k=1)
    return names, Mr, Mg, Ar.std(0), Ag.std(0), float(np.nanmean(np.abs(Cr[iu]-Cg[iu])))


def main():
    retrain = "--retrain" in sys.argv
    two_d = "--2d" in sys.argv
    # --scalar: scalar-surrogate cascade (S = the threshold surrogate only, not
    # the full 21-vector).  Conditioning Y on the FULL surrogate vector flattens
    # tau_S (single effective boundary / degenerate at d=2); the scalar cascade
    # keeps BOTH margins binding (theta ~ 0.28) and its ReLU generators give a
    # rough (s~1) truth -- the DGP used by the debias study's rough-truth arm.
    scalar = "--scalar" in sys.argv
    covs = COVARIATES_2D if two_d else COVARIATES_CR_FULL
    # distinct caches: the faithful full-covariate calibration (dataset replication)
    # and the 2-covariate variant used by the low-dim inference study.
    if two_d and scalar:
        npz = WGAN_DIR / "wgan_oracle_2d_scalar.npz"
    else:
        npz = WGAN_DIR / ("wgan_oracle_2d.npz" if two_d else "wgan_oracle.npz")
    df = load_graduation()

    if npz.exists() and not retrain:
        print(f"[load] {npz} (pass --retrain to refit)")
        orc = WGANOracle.load(npz, df)
    else:
        print(f"[train] 3-WGAN cascade on {len(covs)} covariates "
              f"(GAN1 X, GAN2 S|X,W, GAN3 Y|S,X,W) ...")
        t0 = time.time()
        # Epoch schedule.  CR's Table is 30000/30000/5000, but on THIS low-dim
        # slice (20 covariates, scalar S/Y on 854 rows) 30000 epochs mode-collapses
        # -- variances shrink and the CATE vanishes (ATE_S -> 0, treat-share -> 0.5).
        # CR avoid this because their GAN generates ~40 variables jointly, which
        # resists collapse.  Here ~8000/6000 is the sweet spot; `--cr30k` reproduces
        # the (worse) CR-length experiment for the record.
        ex, es, ey = (30000, 30000, 5000) if "--cr30k" in sys.argv else (8000, 6000, 5000)
        # --epochs ex,es,ey overrides (e.g. longer S/Y training grows CATE
        # heterogeneity until tau_S crosses zero -> both margins bind)
        for a in sys.argv:
            if a.startswith("--epochs="):
                ex, es, ey = (int(t) for t in a.split("=", 1)[1].split(","))
        specs = dict(
            spec_X=WGANSpec(epochs=ex, critic_steps=15, gp_factor=20.0,
                            critic_dropout=0.0, generator_dropout=0.1, seed=1),
            spec_S=WGANSpec(epochs=es, critic_steps=15, gp_factor=20.0,
                            critic_dropout=0.1, generator_dropout=0.1, seed=2),
            spec_Y=WGANSpec(epochs=ey, critic_steps=15, gp_factor=20.0,
                            critic_dropout=0.1, generator_dropout=0.1, seed=3),
        )
        sb = 6.5 if len(covs) > 3 else 5.5
        extra = {"full_surrogate": False} if scalar else {}
        orc = build_wgan_oracle(covariates=covs, verbose=True, support_bound=sb,
                                n_pop=100_000, tau_M=300, pop_chunk=8000,
                                **extra, **specs)
        print(f"[train] done in {time.time()-t0:.0f}s; saving -> {npz}")
        orc.save(npz)

    print("\n[truth] WGAN population truth:")
    tr = orc.truth()
    print(f"   theta_harm={tr['theta_harm']:.4f}  treatS={tr['treat_share_S']:.3f}  "
          f"ATE_S={tr['ate_S']:.2f}  ATE_Y={tr['ate_Y']:.2f}  W_Y={tr['W_Y']:.2f}")

    print(f"\n[fidelity] real vs generated, all {orc.d + 2} variables (original units):")
    names, Mr, Mg, Sr, Sg, corr_match = raw_fidelity(df, orc)
    for j, nm in enumerate(names):
        flag = "  <-- censored/hard" if abs(Mg[j] - Mr[j]) > 0.75 * (abs(Mr[j]) + 1) else ""
        print(f"   {nm:24s} mean {Mr[j]:10.2f}/{Mg[j]:10.2f}  sd {Sr[j]:9.2f}/{Sg[j]:9.2f}{flag}")
    print(f"   correlation match across all variables: mean|Δ|={corr_match:.3f}")

    diag = {"truth": tr, "covariates": list(orc.cfg.covariates),
            "fidelity": {names[j]: {"mean_real": float(Mr[j]), "mean_gen": float(Mg[j]),
                                    "sd_real": float(Sr[j]), "sd_gen": float(Sg[j])}
                         for j in range(len(names))},
            "corr_match_mean_abs": corr_match, "train_hist": getattr(orc, "hist", None)}
    log_name = ("wgan_calibration_2d_scalar.json" if (two_d and scalar)
                else "wgan_calibration_2d.json" if two_d
                else "wgan_calibration.json")
    (LOGS / log_name).write_text(json.dumps(diag, indent=2, default=float), encoding="utf-8")
    print(f"\n[done] wrote {LOGS/log_name};  "
          f"run make_wgan_figures.py for the full validation exhibit.")


if __name__ == "__main__":
    main()
