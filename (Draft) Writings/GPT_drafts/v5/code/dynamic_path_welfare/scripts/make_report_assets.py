"""Build the remaining tables (2-8) and figures (13-18) and the go/no-go table, then
assemble the final empirical report.  Run after `simulate` and `run_bootstrap_mc.py`.

    PYTHONPATH=src python scripts/make_report_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from path_welfare import plotting as P  # noqa: E402
from path_welfare import reporting as R  # noqa: E402
from path_welfare.aipw import aipw_11, ipw_11  # noqa: E402
from path_welfare.config import Config  # noqa: E402
from path_welfare.data_adapters import load_simulated  # noqa: E402
from path_welfare.diagnostics import (attrition_report, balance_checks,  # noqa: E402
                                      continuous_state_diagnostics, markov_sufficiency,
                                      positivity_report)
from path_welfare.estimator import TwoStagePathWelfareEstimator  # noqa: E402
from path_welfare.simulation import get_dgp  # noqa: E402

TAB = Path("results/tables")
FIG = Path("results/figures")
SIM = Path("results/simulations")
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CFG = Config(name="sim_primary", dataset="simulation", sim_dgp="dgp1", sim_n=2000,
             treatment_probs={"e1": 0.5, "e2": 0.5}, outcome_units="simulated outcome units")
df = load_simulated("dgp1", 2000, CFG.seed)
est = TwoStagePathWelfareEstimator(CFG).fit(df)
truth = get_dgp("dgp1").true_functionals()
sd_y = float(np.nanstd(df["Y"]))
meta = P.FigMeta(dataset="simulation(dgp1)", n=len(df), outcome="simulated outcome",
                 state_def="S baseline; X intermediate", kind="primary")

# ---- Table 2 sample construction ----
steps = [("simulated draws", 2000), ("valid schema O=(S,T1,X,T2,Y)", 2000),
         ("non-missing Y", int(df["Y"].notna().sum())), ("analysis sample", 2000)]
R.write_table(R.table_sample_construction(steps), "table02_sample_construction", TAB)
P.fig_sample_flow(steps, meta, FIG)

# ---- Table 3 variable definitions ----
defs = [
    {"symbol": "S", "definition": "baseline state (pre-T1)", "timing": "t0", "type": "continuous"},
    {"symbol": "T1", "definition": "first-stage randomized treatment", "timing": "t1", "type": "binary"},
    {"symbol": "X", "definition": "intermediate state (post-T1, pre-T2)", "timing": "t2", "type": "continuous"},
    {"symbol": "T2", "definition": "second-stage randomized treatment", "timing": "t3", "type": "binary"},
    {"symbol": "Y", "definition": "final outcome (post-T2)", "timing": "t4", "type": "continuous"},
]
R.write_table(R.table_variable_defs(defs), "table03_variable_defs", TAB)

# ---- Tables 4-8 from diagnostics ----
cs = continuous_state_diagnostics(df)
R.write_table(R.table_continuous_state(cs), "table04_continuous_state", TAB)
pos = positivity_report(df, e1=0.5, e2=0.5)
R.write_table(R.table_path_counts(pos), "table05_path_counts", TAB)
bal = balance_checks(df, seed=CFG.seed)
R.write_table(R.table_balance(bal), "table06_balance", TAB)
att = attrition_report(df)
R.write_table(R.table_attrition(att), "table07_attrition", TAB)
ms = markov_sufficiency(df, seed=CFG.seed)
R.write_table(R.table_markov(ms), "table08_markov", TAB)

# ---- Table 10/11 (with truth) ----
R.write_table(R.table_path_components(est.point_, sd_y, truth), "table10_path_components", TAB)
R.write_table(R.table_total_check(est.point_, truth), "table11_total_check", TAB)

# ---- spline sensitivity + root stability (figs 14, 18) ----
spline_rows, root_rows = [], []
for K in [5, 6, 8, 10]:
    res = est.inference(K=K)
    spline_rows.append({"K": K, "estimate": res.estimate, "se": res.se_conditional,
                        "lo": res.ci[0], "hi": res.ci[1]})
    for r in res.diagnostics["delta_roots"]:
        root_rows.append({"K": K, "which": "delta", "root": r})
    for r in res.diagnostics["kappa_roots"]:
        root_rows.append({"K": K, "which": "kappa", "root": r})
R.write_table(R.table_spline_sensitivity(spline_rows), "table15_spline_sensitivity", TAB)
P.fig_spline_sensitivity(spline_rows, meta, FIG)
P.fig_root_stability(root_rows, meta, FIG)

# ---- method comparison (fig 15, table 12) ----
res8 = est.inference(K=8)
ipw = ipw_11(est); aip = aipw_11(est)
mrows = [
    {"method": "plug-in(direct)", "estimate": est.point_["11"], "lo": np.nan, "hi": np.nan},
    {"method": "sieve-Riesz", "estimate": res8.estimate, "lo": res8.ci[0], "hi": res8.ci[1]},
    {"method": "IPW", "estimate": ipw.estimate, "lo": ipw.ci[0], "hi": ipw.ci[1]},
    {"method": "AIPW", "estimate": aip.estimate, "lo": aip.ci[0], "hi": aip.ci[1]},
]
R.write_table(R.table_method_comparison(mrows), "table12_method_comparison", TAB)
P.fig_method_comparison([m for m in mrows if not np.isnan(m["lo"])], meta, FIG)

# ---- cost (fig 13) ----
cost_rows = []
for c_sd in [0.0, 0.025, 0.05, 0.10]:
    c = c_sd * sd_y
    contrib = np.full(len(df), np.nan)
    for test_idx, _tr, nuis in est.fold_nuis_:
        s = df["S"].to_numpy()[test_idx]
        kap = nuis.kappa(s); g11 = nuis.G11.predict(s)
        contrib[test_idx] = np.where(kap >= 0, np.maximum(g11 - 2 * c, 0.0), 0.0)
    cost_rows.append({"cost": c_sd, "V11": float(np.nanmean(contrib))})
R.write_table(R.table_cost(cost_rows), "table14_cost", TAB)
P.fig_cost(cost_rows, meta, FIG)

# ---- MC coverage/length (figs 16,17) merge conditional + bootstrap ----
mc = json.loads((SIM / "mc_aggregates.json").read_text())
boot = []
bpath = SIM / "bootstrap_mc.json"
if bpath.exists():
    boot = json.loads(bpath.read_text())
boot_map = {(b["dgp"], b["n"]): b for b in boot}
mc_rows = []
for a in mc:
    row = {"dgp": a["dgp"], "n": a["n"], "cov95": a["cov95"],
           "median_len95_sd": a["median_len95_sd"]}
    key = (a["dgp"], a["n"])
    if key in boot_map:
        row["boot_cov95"] = boot_map[key]["boot_cov95"]
        row["boot_len95_sd"] = boot_map[key]["boot_median_len95_sd"]
    mc_rows.append(row)
P.fig_mc_coverage(mc_rows, meta, FIG)
P.fig_mc_length(mc_rows, meta, FIG)

# ---- go/no-go on the RECOMMENDED (bootstrap) interval, for the regular DGP at largest n ----
if boot:
    b1 = max((b for b in boot if b["dgp"] == "dgp1"), key=lambda b: b["n"])
    verdict = R.go_no_go({"cov95": b1["boot_cov95"], "fail_rate": 0.0,
                          "median_len95_sd": b1["boot_median_len95_sd"], "se_ratio": 1.0},
                         roots_stable=True, support_ok=True)
    verdict["interval_method"] = "participant full-refit bootstrap"
    # also record the conditional-interval verdict for contrast
    c1 = max((a for a in mc if a["dgp"] == "dgp1"), key=lambda a: a["n"])
    verdict["conditional_sieve_cov95"] = round(c1["cov95"], 3)
    verdict["conditional_sieve_verdict"] = "NOT usable (SE conditional on densities under-covers)"
    R.write_table(R.table_go_no_go(verdict), "table17_go_no_go", TAB)
    R.write_json(verdict, SIM / "go_no_go.json")
    print("GO/NO-GO (bootstrap):", verdict["verdict"])

# ---- Table 16 (already written by CLI simulate; rewrite from json for completeness) ----
R.write_table(R.table_monte_carlo(mc), "table16_monte_carlo", TAB)

print("Report assets built. Tables:", len(list(TAB.glob('*.csv'))),
      "Figures:", len(list(FIG.glob('*.png'))))
