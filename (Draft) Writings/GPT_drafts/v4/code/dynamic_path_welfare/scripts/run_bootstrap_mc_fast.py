"""Faster participant-bootstrap coverage MC using a FIXED sieve dimension (K=8, no per-fit
CV) -- an undersmoothed fixed-K inference demonstration for the go/no-go.  Writes
results/simulations/bootstrap_mc.json.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "src")
from path_welfare.bootstrap import participant_bootstrap
from path_welfare.config import Config
from path_welfare.crossfit import child_seed
from path_welfare.estimator import TwoStagePathWelfareEstimator
from path_welfare.simulation import get_dgp


def cov(dgp_name, n, n_datasets, n_boot, seed=4242, jobs=6):
    dgp = get_dgp(dgp_name); truth = dgp.true_functionals()
    v11, sdy = truth["V11"], truth["sd_y"]

    def one(r):
        rng = np.random.default_rng(child_seed(seed, r))
        df = dgp.sample(n, rng); df["group"] = np.arange(n)
        cfg = Config(treatment_probs={"e1": dgp.e1, "e2": dgp.e2})
        cfg.spline.candidate_dims = [8]      # fixed K -> no inner CV (fast)
        cfg.crossfit.seed = child_seed(seed, r, 2)
        est = TwoStagePathWelfareEstimator(cfg).fit(df)
        bs = participant_bootstrap(est, n_rep=n_boot, seed=child_seed(seed, r, 3), n_jobs=1)
        lo90, hi90 = np.percentile(bs.reps, [5, 95])
        return (float(bs.ci[0] <= v11 <= bs.ci[1]), float(lo90 <= v11 <= hi90),
                (bs.ci[1] - bs.ci[0]) / sdy, est.point_["11"])

    res = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(r) for r in range(n_datasets))
    a = np.array(res, dtype=float)
    return {"dgp": dgp_name, "n": n, "n_datasets": n_datasets, "n_boot": n_boot, "K": 8,
            "V11_true": v11, "boot_cov95": float(a[:, 0].mean()),
            "boot_cov90": float(a[:, 1].mean()),
            "boot_median_len95_sd": float(np.median(a[:, 2])),
            "bias": float(a[:, 3].mean() - v11)}


if __name__ == "__main__":
    t = time.time(); out = []
    for dg, n, nd, nb in [("dgp1", 1500, 40, 79), ("dgp1", 1000, 40, 79),
                          ("dgp3", 1500, 30, 79), ("dgp6", 1500, 30, 79)]:
        r = cov(dg, n, nd, nb)
        out.append(r)
        print(f"{dg} n={n}: boot_cov95={r['boot_cov95']:.2f} boot_cov90={r['boot_cov90']:.2f} "
              f"len_sd={r['boot_median_len95_sd']:.2f} bias={r['bias']:.4f}", flush=True)
    Path("results/simulations").mkdir(parents=True, exist_ok=True)
    Path("results/simulations/bootstrap_mc.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"elapsed {time.time()-t:.0f}s", flush=True)
