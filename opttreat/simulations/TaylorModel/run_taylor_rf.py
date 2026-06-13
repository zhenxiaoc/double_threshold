"""Runnable TaylorExpansionModel random-feature welfare simulations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, ParameterType
from opttreat.models import TaylorExpansionModel
from opttreat.simulations.simulation_engine import (
    SimulationRunConfig,
    SimulationSpec,
    run_simulation_specs,
    write_simulation_outputs,
)


EXPANSIONS = ("tan2", "sinh2", "rational", "hyperbolic", "exp_pm")
K_VALUES = (4, 7, 10)
REPLICATIONS = 1500
N_VALUES = (1500, 3000, 6000)
SEED = 2025
THETA_SOBOL = 5000
N_FEATURES = 100
ACTIVATION = "exp"
RFG_TYPE = "iid_sphere"
ALPHA = 1e-5
SHARE_FEATURES = True
PROGRESS_EVERY = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

PAPER_REPLICATIONS = 1500
PAPER_N_VALUES = (1500, 3000, 6000)
PAPER_THETA_SOBOL = 5000
PAPER_N_FEATURES = 100
PAPER_ALPHA = 1e-5


def estimator_config(random_state: int = SEED) -> EstimatorConfig:
    return EstimatorConfig(
        method="rf_ridge",
        options={
            "rfg_type": RFG_TYPE,
            "activation": ACTIVATION,
            "share_features": SHARE_FEATURES,
            "n_features": N_FEATURES,
            "random_state": random_state,
            "alpha": ALPHA,
        },
    )


def build_specs() -> list[SimulationSpec]:
    specs = []
    for k in K_VALUES:
        p = k
        for expansion in EXPANSIONS:
            specs.append(
                SimulationSpec(
                    label=f"{expansion}_K{k}_p{p}",
                    model_factory=lambda expansion=expansion, k=k, p=p: TaylorExpansionModel(
                        K=k,
                        p=p,
                        expansion=expansion,
                    ),
                    parameter_config=parameter_config_for_dim(p),
                    estimator_config=estimator_config(),
                    variance_config=None,
                )
            )
    return specs


def parameter_config_for_dim(dim: int) -> ParameterConfig:
    return ParameterConfig(
        param_type=ParameterType.WELFARE_KNOWN_DIST,
        options={
            "dim": int(dim),
            "n_sobol": THETA_SOBOL,
            "sobol_scramble": False,
            "transform": lambda U: U,
        },
    )


def _add_taylor_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["spec"].str.extract(r"(?P<expansion>.+)_K(?P<K>\d+)_p(?P<p>\d+)")
    out.insert(0, "expansion", parts["expansion"])
    out.insert(1, "K", parts["K"].astype(int))
    out.insert(2, "p", parts["p"].astype(int))
    return out


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = SimulationRunConfig(REPLICATIONS, N_VALUES, SEED, progress_every=PROGRESS_EVERY)
    summary, draws = run_simulation_specs(build_specs(), run_config)
    summary = _add_taylor_columns(summary)
    draws = _add_taylor_columns(draws)
    for df in (summary, draws):
        df["rfg_type"] = RFG_TYPE
        df["activation"] = ACTIVATION
        df["n_features"] = N_FEATURES
        df["alpha"] = ALPHA

    paths = write_simulation_outputs(
        output_dir=OUTPUT_DIR,
        stem="TaylorModel_rf",
        summary=summary,
        draws=draws,
        run_config=run_config,
        suffix_parts=(f"nf{N_FEATURES}", f"K{'_'.join(str(k) for k in K_VALUES)}"),
        settings={
            "expansions": ", ".join(EXPANSIONS),
            "K values": ", ".join(str(k) for k in K_VALUES),
            "p": "p = K",
            "sample sizes": ", ".join(str(n) for n in N_VALUES),
            "replications": REPLICATIONS,
            "progress every": PROGRESS_EVERY,
            "random features": N_FEATURES,
            "random feature generator": RFG_TYPE,
            "activation": ACTIVATION,
            "ridge alpha": ALPHA,
            "parameter": "known-distribution welfare",
            "inference": "none",
        },
        notes=[
            "This workflow is estimation-only; no standard errors, confidence intervals, or coverage columns are computed.",
            "The rational expansion becomes difficult as K=p increases.",
        ],
    )

    print(summary.to_string(index=False))
    print(f"Wrote summary to {paths['summary']}")
    print(f"Wrote draws to {paths['draws']}")
    print(f"Wrote report to {paths['report']}")
    return summary, draws


if __name__ == "__main__":
    main()
