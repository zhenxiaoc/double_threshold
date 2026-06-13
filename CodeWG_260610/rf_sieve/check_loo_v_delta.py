"""Quick check: is LOO-V's high variance driven by the D^2V central-difference
step delta0 (noisy for indicator functionals)? Cell: dense d50 n4000 share95."""

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from rf_sieve_lib import (make_dgp, generate, compute_truth, draw_feature_map,
                          fit_both_arms, rf_inference, F_V, loo_debias)

REPS, DIM, N, K = 60, 50, 4000, 200
DELTAS = (0.05, 0.2, 0.5)
dgp = make_dgp("dense", DIM, shift=-0.40)
_, V_true = compute_truth(dgp)
X_sobol = Sobol(d=DIM, scramble=False).random(8192)
tau_sobol = dgp["tau"](X_sobol)

rows = []
for rep in range(REPS):
    rng = np.random.default_rng(151 + 1000 * rep)
    data = generate(dgp, N, rng)
    psi = draw_feature_map(DIM, K, rng, 1.5)
    fits = fit_both_arms(psi, data)
    if fits is None:
        continue
    res = rf_inference(fits[0], fits[1], psi, X_sobol, tau_sobol)
    row = {"rep": rep, "V_plug": res["V_hat"]}
    for d0 in DELTAS:
        row[f"loo_d{d0}"] = loo_debias(F_V, fits[0], fits[1], psi, X_sobol,
                                       res["V_hat"], n_total=N, delta0=d0, rng=rng)
    rows.append(row)

df = pd.DataFrame(rows)
out = {"V_true": V_true, "plug_bias": df["V_plug"].mean() - V_true,
       "plug_sd": df["V_plug"].std()}
for d0 in DELTAS:
    out[f"loo_d{d0}_bias"] = df[f"loo_d{d0}"].mean() - V_true
    out[f"loo_d{d0}_sd"] = df[f"loo_d{d0}"].std()
print(pd.Series(out).to_string())
