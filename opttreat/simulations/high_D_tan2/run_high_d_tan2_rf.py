"""Runnable high-dimensional tan2 random-feature welfare simulations."""

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


TAYLOR_DIMS = (3, 7, 10)
EXPANSION = "tan2"
REPLICATIONS = 2
N_VALUES = (120,)
SEED = 2025
PROGRESS_EVERY = 100
THETA_SOBOL = 128
N_FEATURES = 40
ACTIVATION = "exp"
RFG_TYPE = "iid_sphere"
ALPHA = 1e-3
SHARE_FEATURES = True
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

PAPER_REPLICATIONS = 2000
PAPER_N_VALUES = (1500, 3000, 6000)
PAPER_THETA_SOBOL = 5000
PAPER_N_FEATURES = 500
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


def parameter_config(dim: int) -> ParameterConfig:
    return ParameterConfig(
        param_type=ParameterType.WELFARE_KNOWN_DIST,
        options={
            "dim": dim,
            "n_sobol": THETA_SOBOL,
            "sobol_scramble": False,
            "transform": lambda U: U,
        },
    )


def build_specs() -> list[SimulationSpec]:
    specs = []
    for dim in TAYLOR_DIMS:
        specs.append(
            SimulationSpec(
                label=f"{EXPANSION}_K{dim}_p{dim}",
                model_factory=lambda dim=dim: TaylorExpansionModel(K=dim, p=dim, expansion=EXPANSION),
                parameter_config=parameter_config(dim),
                estimator_config=estimator_config(),
                variance_config=None,
            )
        )
    return specs


def _add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["spec"].str.extract(r"(?P<expansion>.+)_K(?P<K>\d+)_p(?P<p>\d+)")
    out.insert(0, "expansion", parts["expansion"])
    out.insert(1, "K", parts["K"].astype(int))
    out.insert(2, "p", parts["p"].astype(int))
    out["rfg_type"] = RFG_TYPE
    out["activation"] = ACTIVATION
    out["n_features"] = N_FEATURES
    out["alpha"] = ALPHA
    return out


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = SimulationRunConfig(REPLICATIONS, N_VALUES, SEED, progress_every=PROGRESS_EVERY)
    summary, draws = run_simulation_specs(build_specs(), run_config)
    summary = _add_design_columns(summary)
    draws = _add_design_columns(draws)

    paths = write_simulation_outputs(
        output_dir=OUTPUT_DIR,
        stem="high_D_tan2_rf",
        summary=summary,
        draws=draws,
        run_config=run_config,
        suffix_parts=(f"nf{N_FEATURES}", f"K{'_'.join(str(dim) for dim in TAYLOR_DIMS)}"),
        settings={
            "expansion": EXPANSION,
            "K values": ", ".join(str(dim) for dim in TAYLOR_DIMS),
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
            "The numbered Model99-Model102 aliases are not used; these are explicit TaylorExpansionModel tan2 specifications.",
        ],
    )

    print(summary.to_string(index=False))
    print(f"Wrote summary to {paths['summary']}")
    print(f"Wrote draws to {paths['draws']}")
    print(f"Wrote report to {paths['report']}")
    return summary, draws


if __name__ == "__main__":
    main()
