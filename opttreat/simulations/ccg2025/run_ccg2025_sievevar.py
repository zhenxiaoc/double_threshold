"""Runnable CCG 2025 SieveVar simulation with explicit top-level settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, ParameterType, VarianceConfig
from opttreat.models import get_model
from opttreat.simulations.simulation_engine import (
    SimulationRunConfig,
    SimulationSpec,
    run_simulation_specs,
    write_simulation_outputs,
)


MODEL_NAMES = ("Model1", "Model8", "Model15")
REPLICATIONS = 1
N_VALUES = (90,)
SEED = 2025
THETA_SOBOL = 512
VARIANCE_SOBOL = 32
VALUE_VARIANCE_SOBOL = 64
CHUNK_SIZE = 32
PROGRESS_EVERY = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

PAPER_REPLICATIONS = 2000
PAPER_N_VALUES = (1500, 3000, 6000)
PAPER_THETA_SOBOL = 5000
PAPER_VARIANCE_SOBOL = 40000
PAPER_VALUE_VARIANCE_SOBOL = 1_000_000
PAPER_CHUNK_SIZE = 50_000

SPLINE_DEGREE = 3
SPLINE_BASIS = "tensor"
SPLINE_KNOTS = "uniform"
PINV_RCOND = float(np.sqrt(np.finfo(float).eps))
VALUE_SCALE = 9.0
VALUE_EPS = 0.005


@dataclass(frozen=True)
class CCGDesign:
    """One CCG 2025 simulation design and tuning row."""

    model_name: str
    parameter_type: ParameterType
    dim: int
    target_lower: tuple[float, ...]
    target_upper: tuple[float, ...]
    j_segments_c: int
    j_segments_t: int
    theorem: str
    eps: float = VALUE_EPS
    value_scale: float = 1.0

    @property
    def label(self) -> str:
        return self.model_name

    def model_factory(self):
        return lambda: get_model(self.model_name)

    def target_transform(self):
        lower = np.asarray(self.target_lower, dtype=float)
        upper = np.asarray(self.target_upper, dtype=float)
        return lambda U: lower + np.asarray(U, dtype=float) * (upper - lower)

    def estimator_config(self) -> EstimatorConfig:
        return EstimatorConfig(
            method="sieve",
            options={
                "solver": "pinv",
                "share_features": False,
                "J_x_degree": SPLINE_DEGREE,
                "J_x_segments_c": self.j_segments_c,
                "J_x_segments_t": self.j_segments_t,
                "knots": SPLINE_KNOTS,
                "basis": SPLINE_BASIS,
                "X_min": None,
                "X_max": None,
                "pinv_rcond": PINV_RCOND,
            },
        )

    def parameter_config(self) -> ParameterConfig:
        options = {
            "dim": self.dim,
            "n_sobol": THETA_SOBOL,
            "sobol_scramble": False,
            "transform": self.target_transform(),
        }
        if self.parameter_type == ParameterType.VALUE_KNOWN_DIST:
            options["v_func"] = lambda X: np.full(np.asarray(X).shape[0], self.value_scale)
            options["true_value"] = np.pi
        if self.parameter_type in (ParameterType.WELFARE_UNKNOWN_DIST, ParameterType.VALUE_UNKNOWN_DIST):
            options["common_support"] = True
        return ParameterConfig(self.parameter_type, options)

    def variance_config(self) -> VarianceConfig:
        n_sobol = VALUE_VARIANCE_SOBOL if self.parameter_type == ParameterType.VALUE_KNOWN_DIST else VARIANCE_SOBOL
        return VarianceConfig(
            method="ccg_sieve_var",
            options={
                "param_type": self.parameter_type,
                "dim": self.dim,
                "n_sobol": n_sobol,
                "target_lower": self.target_lower,
                "target_upper": self.target_upper,
                "sobol_scramble": False,
                "eps": self.eps,
                "value_scale": self.value_scale,
                "chunk_size": CHUNK_SIZE,
                "pinv_rcond": PINV_RCOND,
            },
        )

    def simulation_spec(self) -> SimulationSpec:
        return SimulationSpec(
            label=self.label,
            model_factory=self.model_factory(),
            parameter_config=self.parameter_config(),
            estimator_config=self.estimator_config(),
            variance_config=self.variance_config(),
        )


ALL_CCG_DESIGNS = (
    CCGDesign("Model1", ParameterType.WELFARE_KNOWN_DIST, 1, (0.0,), (1.0,), 16, 16, "Theorem 1"),
    CCGDesign("Model2", ParameterType.WELFARE_KNOWN_DIST, 1, (0.0,), (1.0,), 1, 1, "Theorem 1"),
    CCGDesign("Model3", ParameterType.WELFARE_KNOWN_DIST, 1, (0.0,), (1.0,), 1, 1, "Theorem 1"),
    CCGDesign("Model4", ParameterType.WELFARE_KNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 1"),
    CCGDesign("Model5", ParameterType.WELFARE_KNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 1"),
    CCGDesign("Model6", ParameterType.WELFARE_KNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 4, 1, "Theorem 1"),
    CCGDesign("Model7", ParameterType.WELFARE_KNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 4, 1, "Theorem 1"),
    CCGDesign("Model8", ParameterType.WELFARE_UNKNOWN_DIST, 1, (0.0,), (1.0,), 8, 8, "Theorem 2"),
    CCGDesign("Model9", ParameterType.WELFARE_UNKNOWN_DIST, 1, (0.0,), (1.0,), 1, 4, "Theorem 2"),
    CCGDesign("Model10", ParameterType.WELFARE_UNKNOWN_DIST, 1, (0.0,), (1.0,), 4, 1, "Theorem 2"),
    CCGDesign("Model11", ParameterType.WELFARE_UNKNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 2"),
    CCGDesign("Model12", ParameterType.WELFARE_UNKNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 2"),
    CCGDesign("Model13", ParameterType.WELFARE_UNKNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 2"),
    CCGDesign("Model14", ParameterType.WELFARE_UNKNOWN_DIST, 2, (0.0, 0.0), (1.0, 1.0), 1, 1, "Theorem 2"),
    CCGDesign(
        "Model15",
        ParameterType.VALUE_KNOWN_DIST,
        2,
        (-1.5, -1.5),
        (1.5, 1.5),
        1,
        4,
        "Theorem 3",
        eps=VALUE_EPS,
        value_scale=VALUE_SCALE,
    ),
)


def selected_designs() -> list[CCGDesign]:
    selected = set(MODEL_NAMES)
    return [design for design in ALL_CCG_DESIGNS if design.model_name in selected]


def build_specs() -> list[SimulationSpec]:
    return [design.simulation_spec() for design in selected_designs()]


def _add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    designs = {design.model_name: design for design in ALL_CCG_DESIGNS}
    out = df.copy()
    out["model"] = out["spec"]
    out["theorem"] = out["model"].map(lambda name: designs[name].theorem)
    cols = ["spec", "model", "theorem"] + [col for col in out.columns if col not in {"spec", "model", "theorem"}]
    return out[cols]


def _model_suffix_part() -> str:
    model_numbers = [int(design.model_name.replace("Model", "")) for design in selected_designs()]
    if model_numbers == list(range(1, 16)):
        return "M1_15"
    return "M" + "_".join(str(number) for number in model_numbers)


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = SimulationRunConfig(
        replications=REPLICATIONS,
        n_values=N_VALUES,
        seed=SEED,
        progress_every=PROGRESS_EVERY,
    )
    summary, draws = run_simulation_specs(build_specs(), run_config)
    summary = _add_design_columns(summary)
    draws = _add_design_columns(draws)

    paths = write_simulation_outputs(
        output_dir=OUTPUT_DIR,
        stem="ccg2025_sievevar",
        summary=summary,
        draws=draws,
        run_config=run_config,
        suffix_parts=(_model_suffix_part(),),
        settings={
            "models": ", ".join(design.model_name for design in selected_designs()),
            "sample sizes": ", ".join(str(n) for n in N_VALUES),
            "replications": REPLICATIONS,
            "progress every": PROGRESS_EVERY,
            "theta Sobol": THETA_SOBOL,
            "welfare variance Sobol": VARIANCE_SOBOL,
            "value variance Sobol": VALUE_VARIANCE_SOBOL,
            "spline degree": SPLINE_DEGREE,
            "spline basis": SPLINE_BASIS,
            "spline knots": SPLINE_KNOTS,
            "solver": "pinv",
            "pinv rcond": PINV_RCOND,
        },
        notes=[
            "CCG paper-specific tuning lives in this simulation layer, not in the model classes.",
            "Coverage columns are reported only because variance_config is provided.",
        ],
    )

    print(summary.to_string(index=False))
    print(f"Wrote summary to {paths['summary']}")
    print(f"Wrote draws to {paths['draws']}")
    print(f"Wrote report to {paths['report']}")
    return summary, draws


if __name__ == "__main__":
    main()
