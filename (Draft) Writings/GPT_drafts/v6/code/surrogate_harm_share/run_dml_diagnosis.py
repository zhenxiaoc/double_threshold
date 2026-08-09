"""Why does sieve-DML undercover?  A decomposition experiment.

(warnings silenced; sklearn convergence chatter is not informative here)

The two-band sieve-Riesz interval is valid for the SIEVE plug-in (coverage
0.93-0.95) but undercovers badly with a boosted-tree first stage on the SAME
DGP and the SAME samples.  This script separates the candidate mechanisms:

  A. FOLDS      -- K in {2,5,10}: nuisance trained on (K-1)/K of the sample.
  B. LEARNER    -- gbr / rf (tree, piecewise-constant) vs krr / mlp (smooth)
                   vs the paper's tensor-sieve plug-in (reference).
  C. RIESZ BASIS-- tensor B-spline vs random features for the projection.
  D. CAPACITY   -- default vs high-capacity GBR (does less shrinkage help?).
  E. SNR        -- noise_scale in {0.17,0.34,0.68,1.0}: is it "the dataset"?
  F. SAMPLE SIZE-- n in {2000,...,16000} for the two extremes.

plus PROJECTION DIAGNOSTICS against the oracle arm means: the R^2 of the
learner's error on the Riesz span (the sieve-approximation condition), the
attenuation of the CATE, and the fraction of the plug-in's boundary bias the
correction actually removes.

Usage:  python run_dml_diagnosis.py [--quick]
Writes: results/logs/dml_diagnosis.json, results/tables/dml_diagnosis*.csv
"""
from __future__ import annotations

# pin BLAS/OpenMP to one thread BEFORE numpy/sklearn import: joblib runs
# JOBS worker threads and each sklearn fit would otherwise spawn its own
# BLAS pool, oversubscribing the machine by an order of magnitude.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from harm_share.calibration import build_oracle, OracleConfig      # noqa: E402
from harm_share.functionals import mc_truth                        # noqa: E402
from harm_share.cf_dml import cf_riesz_dml, projection_diagnostics  # noqa: E402
from harm_share.estimator import estimate_harm_share               # noqa: E402

QUICK = "--quick" in sys.argv
N_REP = 300 if not QUICK else 20
N_MAIN = 4000
JOBS = 16
SEED = 20260720
Z = 1.959964

ROOT = Path(__file__).parent
LOGS = ROOT / "results" / "logs"
TABLES = ROOT / "results" / "tables"


def _seed(*ix):
    return int(np.random.SeedSequence([SEED, *ix]).generate_state(1)[0])


# --------------------------------------------------------------------------- #
# One Monte Carlo cell
# --------------------------------------------------------------------------- #
def mc_cell(oracle, th_true, n, *, kind, label, n_rep=N_REP, **kw):
    """kind='cf' -> cf_riesz_dml with kw; kind='sieve' -> tensor-sieve plug-in."""
    def one(rep):
        rng = np.random.default_rng(_seed(n, rep))
        df = oracle.sample_experiment(n, rng)
        if kind == "sieve":
            e = estimate_harm_share(df, segments=kw.get("segments", 2),
                                    delta=kw.get("delta", 0.08))
            return {"theta": e.theta_hat, "se": e.se_sieve,
                    "cov": float(e.ci_sieve[0] <= th_true <= e.ci_sieve[1]),
                    "corr": 0.0, "plugin": e.theta_hat}
        r = cf_riesz_dml(df, seed=_seed(n, rep, 3), **kw)
        return {"theta": r.theta, "se": r.se,
                "cov": float(r.ci[0] <= th_true <= r.ci[1]),
                "corr": r.correction, "plugin": r.theta_plugin}

    t0 = time.time()
    rows = Parallel(n_jobs=JOBS, prefer="threads")(
        delayed(one)(r) for r in range(n_rep))
    th = np.array([r["theta"] for r in rows])
    pl = np.array([r["plugin"] for r in rows])
    se = np.array([r["se"] for r in rows])
    sd = th.std(ddof=1)
    out = {
        "label": label, "n": n, "n_rep": n_rep, "theta_true": th_true,
        "bias": float(th.mean() - th_true),
        "plugin_bias": float(pl.mean() - th_true),
        "mc_sd": float(sd), "plugin_sd": float(pl.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean((th - th_true) ** 2))),
        "mean_se": float(se.mean()),
        "se_ratio": float(se.mean() / sd) if sd > 0 else np.nan,
        "cov95": float(np.mean([r["cov"] for r in rows])),
        "mean_corr": float(np.mean([r["corr"] for r in rows])),
        "bias_over_se": float(abs(th.mean() - th_true) / se.mean()) if se.mean() > 0 else np.nan,
        "secs": round(time.time() - t0, 1),
    }
    out.update({k: v for k, v in kw.items() if isinstance(v, (str, int, float))})
    print(f"  {label:34s} bias={out['bias']:+.4f} (plug {out['plugin_bias']:+.4f}) "
          f"sd={out['mc_sd']:.4f} se/sd={out['se_ratio']:.2f} "
          f"|b|/se={out['bias_over_se']:.2f} cov={out['cov95']:.3f}  [{out['secs']}s]")
    return out


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    oracle = build_oracle()
    truth = mc_truth(oracle, n_draw=1_500_000, seed=7)
    th_true = truth["theta_harm"]
    print(f"KRR truth theta_0 = {th_true:.4f}\n")

    cells, proj = [], []

    # ---------------- projection diagnostics (the mechanism) ---------------- #
    print("[proj] learner-error projection on the Riesz span (oracle truth):")
    for learner in ["gbr", "rf", "krr"]:
        for riesz in ["sieve", "rf"]:
            reps = []
            for rep in range(5 if not QUICK else 2):
                rng = np.random.default_rng(_seed(N_MAIN, rep, 11))
                df = oracle.sample_experiment(N_MAIN, rng)
                reps.append(projection_diagnostics(
                    df, oracle, learner=learner, K=5, riesz=riesz,
                    seed=_seed(rep, 12)))
            agg = {k: float(np.nanmean([r[k] for r in reps])) for k in reps[0]}
            agg.update({"learner": learner, "riesz": riesz, "n": N_MAIN})
            proj.append(agg)
            r2 = np.mean([agg[f"r2_{o}{w}"] for o in "SY" for w in (0, 1)])
            print(f"  {learner:4s} riesz={riesz:5s} R2(err on span)={r2:.3f} "
                  f"atten_S={agg['atten_S']:.3f} atten_Y={agg['atten_Y']:.3f} "
                  f"plugin_err={agg['plugin_err']:+.4f} "
                  f"frac_bias_removed={agg['frac_removed']:+.2f}")

    # ---------------- A. reference: tensor-sieve plug-in -------------------- #
    print("\n[A] reference (paper's tensor-sieve plug-in):")
    cells.append(mc_cell(oracle, th_true, N_MAIN, kind="sieve",
                         label="tensor-sieve plug-in", segments=2, delta=0.08))

    # ---------------- B. folds ---------------------------------------------- #
    print("\n[B] cross-fitting folds (learner=gbr, riesz=sieve):")
    for K in [2, 5, 10]:
        cells.append(mc_cell(oracle, th_true, N_MAIN, kind="cf",
                             label=f"gbr K={K} riesz=sieve",
                             learner="gbr", K=K, riesz="sieve"))

    # ---------------- C. learner -------------------------------------------- #
    print("\n[C] learner (K=5, riesz=sieve):")
    for learner in ["rf", "krr"]:
        cells.append(mc_cell(oracle, th_true, N_MAIN, kind="cf",
                             label=f"{learner} K=5 riesz=sieve",
                             learner=learner, K=5, riesz="sieve"))

    # ---------------- D. Riesz basis + capacity ----------------------------- #
    print("\n[D] Riesz basis and learner capacity (K=5):")
    for learner in ["gbr", "krr"]:
        cells.append(mc_cell(oracle, th_true, N_MAIN, kind="cf",
                             label=f"{learner} K=5 riesz=rf",
                             learner=learner, K=5, riesz="rf", n_features=400))
    cells.append(mc_cell(oracle, th_true, N_MAIN, kind="cf",
                         label="gbr(high-cap) K=5 riesz=sieve",
                         learner="gbr", K=5, riesz="sieve", capacity="high"))
    cells.append(mc_cell(oracle, th_true, N_MAIN, kind="cf",
                         label="gbr K=5 no correction",
                         learner="gbr", K=5, riesz="sieve", correct=False))

    # ---- decisive test: does the correction ABSORB or ADD variance? -------- #
    # Theory: the correction cancels only the in-span part of the first-stage
    # error; the orthogonal part survives and, by cross-fitting, is INDEPENDENT
    # of the score.  Variances then ADD:
    #     Var(DML) - Var(plug-in)  ==  sigma_n^2/n   (ratio -> 1)
    # For a matched sieve first stage the correction is ~0 and the ratio is ~0.
    print("\n[B2] does the correction absorb or ADD variance? "
          "(theory: ratio -> 1 when the error is out-of-span)")
    var_test = []
    for learner in ["gbr", "rf", "krr"]:
        def one(rep):
            rng = np.random.default_rng(_seed(N_MAIN, rep, 77))
            df = oracle.sample_experiment(N_MAIN, rng)
            r = cf_riesz_dml(df, learner=learner, K=5, riesz="sieve",
                             seed=_seed(N_MAIN, rep, 78))
            return r.theta_plugin, r.theta, r.se
        rows = np.array(Parallel(n_jobs=JOBS, prefer="threads")(
            delayed(one)(r) for r in range(N_REP)))
        v_pl, v_dml = rows[:, 0].var(ddof=1), rows[:, 1].var(ddof=1)
        se2 = float(np.mean(rows[:, 2] ** 2))
        rec = {"learner": learner, "var_plugin": float(v_pl),
               "var_dml": float(v_dml), "mean_se2": se2,
               "added_var_ratio": float((v_dml - v_pl) / se2)}
        var_test.append(rec)
        print(f"  {learner:4s}: Var(plug)={v_pl:.3e} Var(DML)={v_dml:.3e} "
              f"se^2={se2:.3e}  (Var(DML)-Var(plug))/se^2 = "
              f"{rec['added_var_ratio']:+.2f}")

    pd.DataFrame(cells).to_csv(TABLES / "dml_diagnosis_main.csv", index=False)
    pd.DataFrame(proj).to_csv(TABLES / "dml_diagnosis_projection.csv", index=False)
    pd.DataFrame(var_test).to_csv(TABLES / "dml_diagnosis_vartest.csv", index=False)

    # ---------------- E. is it the DGP?  SNR ladder ------------------------- #
    print("\n[E] signal-to-noise ladder (is it 'the dataset'?):")
    snr_cells = []
    for ns in [0.17, 0.34, 0.68, 1.0]:
        orc = build_oracle(noise_scale=ns)
        tr = mc_truth(orc, n_draw=800_000, seed=7)["theta_harm"]
        print(f" noise_scale={ns} (theta_0={tr:.4f}):")
        for kind, label, kw in [
            ("sieve", "tensor-sieve plug-in", {"segments": 2, "delta": 0.08}),
            ("cf", "gbr K=5", {"learner": "gbr", "K": 5, "riesz": "sieve"}),
            ("cf", "krr K=5", {"learner": "krr", "K": 5, "riesz": "sieve"}),
        ]:
            c = mc_cell(orc, tr, N_MAIN, kind=kind, label=f"{label} ns={ns}", **kw)
            c["noise_scale"] = ns
            snr_cells.append(c)
        pd.DataFrame(snr_cells).to_csv(TABLES / "dml_diagnosis_snr.csv", index=False)

    # ---------------- F. sample size --------------------------------------- #
    print("\n[F] sample size (does the bias wash out?):")
    n_cells = []
    for n in ([2000, 8000, 16000] if not QUICK else [2000]):
        for kind, label, kw in [
            ("sieve", "tensor-sieve plug-in", {"segments": 2, "delta": 0.08}),
            ("cf", "gbr K=5", {"learner": "gbr", "K": 5, "riesz": "sieve"}),
            ("cf", "krr K=5", {"learner": "krr", "K": 5, "riesz": "sieve"}),
        ]:
            c = mc_cell(oracle, th_true, n, kind=kind, label=f"{label} n={n}", **kw)
            n_cells.append(c)
        pd.DataFrame(n_cells).to_csv(TABLES / "dml_diagnosis_n.csv", index=False)

    (LOGS / "dml_diagnosis.json").write_text(json.dumps(
        {"truth": th_true, "projection": proj, "main": cells,
         "var_test": var_test, "snr": snr_cells, "n": n_cells}, indent=1))
    print("\ndone ->", TABLES / "dml_diagnosis_main.csv")


if __name__ == "__main__":
    main()
