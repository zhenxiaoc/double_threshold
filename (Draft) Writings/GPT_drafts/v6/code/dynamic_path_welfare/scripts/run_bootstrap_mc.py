"""Focused Monte Carlo for the participant-bootstrap interval (the recommended,
complete-variance interval).  Complements the conditional-sieve MC in table16.

Run: PYTHONPATH=src python scripts/run_bootstrap_mc.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "src")
from path_welfare.bootstrap import participant_bootstrap  # noqa: E402
from path_welfare.config import Config  # noqa: E402
from path_welfare.crossfit import child_seed  # noqa: E402
from path_welfare.estimator import TwoStagePathWelfareEstimator  # noqa: E402
from path_welfare.simulation import get_dgp  # noqa: E402


def bootstrap_coverage(dgp_name, n, n_datasets, n_boot, seed=4242, jobs=6):
    dgp = get_dgp(dgp_name)
    truth = dgp.true_functionals()
    v11, sdy = truth["V11"], truth["sd_y"]

    def one(r):
        rng = np.random.default_rng(child_seed(seed, r))
        df = dgp.sample(n, rng); df["group"] = np.arange(n)
        cfg = Config(treatment_probs={"e1": dgp.e1, "e2": dgp.e2})
        cfg.crossfit.seed = child_seed(seed, r, 2)
        est = TwoStagePathWelfareEstimator(cfg).fit(df)
        bs = participant_bootstrap(est, n_rep=n_boot, seed=child_seed(seed, r, 3), n_jobs=1)
        lo90 = np.percentile(bs.reps, 5); hi90 = np.percentile(bs.reps, 95)
        return {"cov95": float(bs.ci[0] <= v11 <= bs.ci[1]),
                "cov90": float(lo90 <= v11 <= hi90),
                "len95_sd": (bs.ci[1] - bs.ci[0]) / sdy,
                "point": est.point_["11"]}

    reps = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(r) for r in range(n_datasets))
    cov95 = float(np.mean([x["cov95"] for x in reps]))
    cov90 = float(np.mean([x["cov90"] for x in reps]))
    med_len = float(np.median([x["len95_sd"] for x in reps]))
    return {"dgp": dgp_name, "n": n, "n_datasets": n_datasets, "n_boot": n_boot,
            "V11_true": v11, "boot_cov95": cov95, "boot_cov90": cov90,
            "boot_median_len95_sd": med_len,
            "bias": float(np.mean([x["point"] for x in reps]) - v11)}


if __name__ == "__main__":
    t = time.time()
    out = []
    for dg, n in [("dgp1", 1500), ("dgp1", 1000), ("dgp3", 1500), ("dgp6", 1500)]:
        r = bootstrap_coverage(dg, n, n_datasets=60, n_boot=99)
        out.append(r)
        print(f"{dg} n={n}: boot_cov95={r['boot_cov95']:.2f} boot_cov90={r['boot_cov90']:.2f} "
              f"median_len_sd={r['boot_median_len95_sd']:.2f} bias={r['bias']:.4f}")
    Path("results/simulations").mkdir(parents=True, exist_ok=True)
    Path("results/simulations/bootstrap_mc.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"elapsed {time.time()-t:.0f}s -> results/simulations/bootstrap_mc.json")
