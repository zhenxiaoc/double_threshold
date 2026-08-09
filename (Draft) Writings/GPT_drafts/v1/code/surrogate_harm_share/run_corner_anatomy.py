"""Corner anatomy: empirical validation of the order claim in the two-threshold
theory --- the corner cross diagonal E[Delta_g(x*) Delta_h(x*)] at the corner
points x* in {tau_S=0} cap {tau_Y=0} scales like rho_SY * K/n (same order as
the margin diagonals), and the SS jackknife's cross (corner) correction tracks
the implied plug-in corner bias.

Outputs results/logs/corner_anatomy.json and a console table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import fsolve

sys.path.insert(0, str(Path(__file__).parent / "src"))

from harm_share.calibration import build_oracle                     # noqa: E402
from harm_share.estimator import _sieve_opts, fit_cate_surface      # noqa: E402
from harm_share.quadratic import ss_debiased_estimate               # noqa: E402

N_GRID = [2000, 4000, 8000]
N_REP = 300
JOBS = 4
SEED = 20260719
SEG = 2


def find_corners(oracle, bound=3.0, n_scan=241):
    """Locate corner points: joint roots of (tau_S, tau_Y) inside the support."""
    xs = np.linspace(-bound, bound, n_scan)
    XX, YY = np.meshgrid(xs, xs)
    P = np.column_stack([XX.ravel(), YY.ravel()])
    tS = oracle.tau_S(P).reshape(XX.shape)
    tY = oracle.tau_Y(P).reshape(XX.shape)
    # cells where both surfaces change sign
    cand = []
    sS = np.sign(tS)
    sY = np.sign(tY)
    for i in range(n_scan - 1):
        for j in range(n_scan - 1):
            blockS = sS[i:i + 2, j:j + 2]
            blockY = sY[i:i + 2, j:j + 2]
            if blockS.max() > 0 > blockS.min() and blockY.max() > 0 > blockY.min():
                cand.append((0.5 * (xs[j] + xs[j + 1]), 0.5 * (xs[i] + xs[i + 1])))
    # refine with fsolve and dedupe
    corners = []
    for x0 in cand:
        fun = lambda p: np.array(
            [np.asarray(oracle.tau_S(np.atleast_2d(p))).ravel()[0],
             np.asarray(oracle.tau_Y(np.atleast_2d(p))).ravel()[0]])
        sol, info, ier, _ = fsolve(fun, np.array(x0), full_output=True)
        if ier != 1 or np.abs(fun(sol)).max() > 1e-6:
            continue
        if np.abs(sol).max() > bound:
            continue
        if all(np.linalg.norm(sol - c) > 1e-3 for c in corners):
            corners.append(sol)
    return np.array(corners)


def corner_geometry(oracle, corners, h=1e-4):
    """Gradients, angle, and density weight at each corner point."""
    geo = []
    for x in corners:
        def grad(fun):
            g = np.zeros(2)
            for k in range(2):
                e = np.zeros(2); e[k] = h
                g[k] = (np.asarray(fun(np.atleast_2d(x + e))).ravel()[0]
                        - np.asarray(fun(np.atleast_2d(x - e))).ravel()[0]) / (2 * h)
            return g
        gS, gY = grad(oracle.tau_S), grad(oracle.tau_Y)
        cos = float(gS @ gY / (np.linalg.norm(gS) * np.linalg.norm(gY)))
        geo.append({
            "x": x.tolist(),
            "norm_gS": float(np.linalg.norm(gS)),
            "norm_gY": float(np.linalg.norm(gY)),
            "cos": cos, "sin": float(np.sqrt(max(1 - cos ** 2, 1e-12))),
            "f": float(np.asarray(oracle.density(np.atleast_2d(x))).ravel()[0]),
        })
    return geo


def residual_correlation(oracle, n=200_000, seed=11):
    rng = np.random.default_rng(seed)
    df = oracle.sample_experiment(n, rng)
    X = df[[c for c in df.columns if c.startswith("X")]].to_numpy(float)
    W = df["W"].to_numpy().astype(int)
    muS = np.where(W == 1, oracle.mu_S(X, 1), oracle.mu_S(X, 0))
    muY = np.where(W == 1, oracle.mu_Y(X, 1), oracle.mu_Y(X, 0))
    eS = df["S"].to_numpy(float) - muS
    eY = df["Y"].to_numpy(float) - muY
    return float(np.corrcoef(eS, eY)[0, 1])


def one_rep(oracle, corners, n, rep):
    rng = np.random.default_rng(
        int(np.random.SeedSequence([SEED, n, rep, 5]).generate_state(1)[0]))
    df = oracle.sample_experiment(n, rng)
    opts = _sieve_opts(2, SEG)
    out_S = fit_cate_surface(df, "S", opts)
    out_Y = fit_cate_surface(df, "Y", opts)
    dg = (np.asarray(out_S["h_hat"](corners)).ravel()
          - oracle.tau_S(corners).ravel())
    dh = (np.asarray(out_Y["h_hat"](corners)).ravel()
          - oracle.tau_Y(corners).ravel())
    ss = ss_debiased_estimate(df, segments=SEG, with_sieve_se=False,
                              seed=int(np.random.SeedSequence(
                                  [SEED, n, rep, 6]).generate_state(1)[0]))
    return dg, dh, ss.corner / 8.0, ss.quad_S / 8.0, ss.quad_Y / 8.0


def main():
    t_all = time.time()
    oracle = build_oracle()
    corners = find_corners(oracle)
    geo = corner_geometry(oracle, corners)
    rho = residual_correlation(oracle)
    print(f"{len(corners)} corner point(s); rho_SY = {rho:.3f}")
    for g in geo:
        print(f"  x*={np.round(g['x'],3)} |gS|={g['norm_gS']:.1f} "
              f"|gY|={g['norm_gY']:.1f} cos={g['cos']:.3f} f={g['f']:.4f}")

    out = {"rho_SY": rho, "corners": geo, "cells": []}
    for n in N_GRID:
        t0 = time.time()
        res = Parallel(n_jobs=JOBS, prefer="threads")(
            delayed(one_rep)(oracle, corners, n, r) for r in range(N_REP))
        DG = np.stack([r[0] for r in res])          # (reps, n_corner)
        DH = np.stack([r[1] for r in res])
        corr_ss = np.array([r[2] for r in res])     # SS corner correction /8
        cross = (DG * DH).mean(axis=0)              # E[dg*dh] per corner
        cov = np.array([np.cov(DG[:, k], DH[:, k])[0, 1]
                        for k in range(DG.shape[1])])
        corrcoef = np.array([np.corrcoef(DG[:, k], DH[:, k])[0, 1]
                             for k in range(DG.shape[1])])
        # theoretical plug-in corner bias: sum_x* E[dg dh] w / (|gS||gY||sin|)
        bias_corner = float(sum(
            cross[k] * g["f"] / (g["norm_gS"] * g["norm_gY"] * g["sin"])
            for k, g in enumerate(geo)))
        cell = {
            "n": n, "n_rep": N_REP,
            "mean_dgdh": cross.tolist(),
            "cov_dgdh": cov.tolist(),
            "corr_dgdh": corrcoef.tolist(),
            "var_dg": DG.var(axis=0, ddof=1).tolist(),
            "var_dh": DH.var(axis=0, ddof=1).tolist(),
            "bias_corner_theory": bias_corner,
            "ss_corner_corr_mean": float(corr_ss.mean()),
            "ss_corner_corr_sd": float(corr_ss.std(ddof=1)),
        }
        out["cells"].append(cell)
        print(f"n={n} ({time.time()-t0:.0f}s): E[dg*dh]={np.round(cross,4)} "
              f"corr={np.round(corrcoef,3)} | plug-in corner bias "
              f"{bias_corner:+.5f} vs SS corner corr {corr_ss.mean():+.5f}"
              f" (sd {corr_ss.std(ddof=1):.5f})")

    path = Path(__file__).parent / "results" / "logs" / "corner_anatomy.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"done in {time.time()-t_all:.0f}s ->", path)


if __name__ == "__main__":
    main()
