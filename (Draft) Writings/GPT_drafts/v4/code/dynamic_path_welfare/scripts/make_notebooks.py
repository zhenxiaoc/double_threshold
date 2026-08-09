"""Generate the seven analysis notebooks (executable, minimal) without nbformat."""

import json
from pathlib import Path

NB = Path("notebooks")
NB.mkdir(exist_ok=True)


def cell_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def cell_code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = ("import sys; sys.path.insert(0, '../src')\n"
         "import numpy as np, pandas as pd\n"
         "from path_welfare.config import Config\n")

NOTEBOOKS = {
    "01_dataset_audit": [
        cell_md("# 01 Dataset audit\n\nSearch + access audit. No accessible dataset passes "
                "the hard gates; see `docs/dataset_search.md` and `docs/data_access_blockers.md`."),
        cell_code(SETUP + "from path_welfare.data_search import scorecard, CANDIDATES\n"
                  "sc = scorecard()\n"
                  "pd.DataFrame(sc)"),
    ],
    "02_sample_construction": [
        cell_md("# 02 Sample construction\n\nCanonical `O=(S,T1,X,T2,Y)`, gates, continuity."),
        cell_code(SETUP + "from path_welfare.data_adapters import load_simulated\n"
                  "from path_welfare.schemas import apply_gates, continuity_report\n"
                  "df = load_simulated('dgp1', 2000, 1)\n"
                  "g = apply_gates(df, smallest_prob=0.5)\n"
                  "print('gates passed', g.passed, g.path_counts)\n"
                  "continuity_report(df['S'].values,'S')"),
    ],
    "03_point_estimation": [
        cell_md("# 03 Point estimation\n\nCross-fitted plug-in of the four path components."),
        cell_code(SETUP + "from path_welfare.data_adapters import load_simulated\n"
                  "from path_welfare.estimator import TwoStagePathWelfareEstimator\n"
                  "df = load_simulated('dgp1', 2000, 1)\n"
                  "est = TwoStagePathWelfareEstimator(Config(treatment_probs={'e1':0.5,'e2':0.5})).fit(df)\n"
                  "est.estimate_all_paths()"),
    ],
    "04_boundary_diagnostics": [
        cell_md("# 04 Boundary diagnostics\n\nRoots of delta and kappa; regularity flags."),
        cell_code(SETUP + "from path_welfare.data_adapters import load_simulated\n"
                  "from path_welfare.estimator import TwoStagePathWelfareEstimator\n"
                  "df = load_simulated('dgp1', 2000, 1)\n"
                  "est = TwoStagePathWelfareEstimator(Config(treatment_probs={'e1':0.5,'e2':0.5})).fit(df)\n"
                  "est.boundary_diagnostics()"),
    ],
    "05_inference": [
        cell_md("# 05 Inference\n\nSieve-Riesz (conditional) + participant bootstrap + AIPW. "
                "See `docs/inference_derivation.md`."),
        cell_code(SETUP + "from path_welfare.data_adapters import load_simulated\n"
                  "from path_welfare.estimator import TwoStagePathWelfareEstimator\n"
                  "df = load_simulated('dgp1', 2000, 1)\n"
                  "est = TwoStagePathWelfareEstimator(Config(treatment_probs={'e1':0.5,'e2':0.5})).fit(df)\n"
                  "res = est.inference(K=8)\n"
                  "print('sieve V11', res.estimate, 'CI', res.ci)\n"
                  "print('FD max rel err', min(v['rel_diff_beta1'] for v in res.fd_check.values()))"),
    ],
    "06_simulation": [
        cell_md("# 06 Monte Carlo\n\nCoverage / length / bias across DGPs. The conditional "
                "sieve interval under-covers; the participant bootstrap is the recommended interval."),
        cell_code(SETUP + "from path_welfare.simulation import run_mc\n"
                  "r = run_mc('dgp1', 1000, 40, seed=1, K=8, n_jobs=4)\n"
                  "{k: r['aggregate'][k] for k in ['V11_true','cov95','se_ratio','median_len95_sd','fail_rate']}"),
    ],
    "07_robustness": [
        cell_md("# 07 Robustness\n\nSpline dimension, cost grid, seeds, density- vs "
                "direct-regression, Markov-restricted vs richer history."),
        cell_code(SETUP + "from path_welfare.data_adapters import load_simulated\n"
                  "from path_welfare.estimator import TwoStagePathWelfareEstimator\n"
                  "df = load_simulated('dgp1', 2000, 1)\n"
                  "est = TwoStagePathWelfareEstimator(Config(treatment_probs={'e1':0.5,'e2':0.5})).fit(df)\n"
                  "[{'K':K, 'V11':round(est.inference(K=K).estimate,4)} for K in [5,6,8,10]]"),
    ],
}

for name, cells in NOTEBOOKS.items():
    (NB / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name)
