from __future__ import annotations

import numpy as np

from opttreat.config import EstimatorConfig, ParameterType, VarianceConfig
from opttreat.data import split_treated_control
from opttreat.estimation import get_estimator
from opttreat.models import Model4
from opttreat.simulations.ccg2025.run_ccg2025_sievevar import main as ccg_main
from opttreat.simulations.simulation_engine import SimulationRunConfig, SimulationSpec, run_simulation_specs
from opttreat.config import ParameterConfig
from opttreat.models import Model1, Model8, Model15
from opttreat.variance import get_variance_estimator


def _rf_output():
    np.random.seed(123)
    parsed = split_treated_control(Model4().generate_data(90))
    estimator = get_estimator(
        EstimatorConfig(
            method="rf_ridge",
            options={
                "rfg_type": "iid_sphere",
                "activation": "sigmoid",
                "share_features": True,
                "n_features": 14,
                "random_state": 123,
                "alpha": 1e-3,
            },
        )
    )
    return estimator.fit(parsed)


def test_sieve_variance_welfare_known_and_unknown() -> None:
    output = _rf_output()

    for param_type in [ParameterType.WELFARE_KNOWN_DIST, ParameterType.WELFARE_UNKNOWN_DIST]:
        variance = get_variance_estimator(
            VarianceConfig(
                method="sieve",
                options={
                    "alpha": 1e-3,
                    "dim": 2,
                    "n_sobol": 16,
                    "sobol_seed": 321,
                    "param_type": param_type,
                },
            )
        )
        var_hat = variance.fit(output)
        assert np.isfinite(var_hat)
        assert var_hat >= 0.0


def test_sieve_variance_value_known() -> None:
    output = _rf_output()
    variance = get_variance_estimator(
        VarianceConfig(
            method="sieve",
            options={
                "alpha": 1e-3,
                "dim": 2,
                "n_sobol": 16,
                "sobol_seed": 321,
                "param_type": ParameterType.VALUE_KNOWN_DIST,
                "eps": 0.1,
                "v_func": lambda X: np.ones(X.shape[0]),
            },
        )
    )

    var_hat = variance.fit(output)
    assert np.isfinite(var_hat)
    assert var_hat >= 0.0


def test_simulation_engine_known_unknown_and_value_smoke() -> None:
    estimator_config = EstimatorConfig(
        method="sieve",
        options={
            "solver": "pinv",
            "share_features": False,
            "J_x_degree": 2,
            "J_x_segments_t": 2,
            "J_x_segments_c": 2,
            "knots": "uniform",
            "basis": "tensor",
            "pinv_rcond": float(np.sqrt(np.finfo(float).eps)),
        },
    )
    specs = [
        SimulationSpec(
            "known",
            Model1,
            ParameterConfig(
                ParameterType.WELFARE_KNOWN_DIST,
                {"dim": 1, "n_sobol": 16, "sobol_scramble": False, "transform": lambda U: U},
            ),
            estimator_config,
            None,
        ),
        SimulationSpec(
            "unknown",
            Model8,
            ParameterConfig(ParameterType.WELFARE_UNKNOWN_DIST, {"dim": 1, "common_support": True}),
            estimator_config,
            None,
        ),
        SimulationSpec(
            "value",
            Model15,
            ParameterConfig(
                ParameterType.VALUE_KNOWN_DIST,
                {
                    "dim": 2,
                    "n_sobol": 32,
                    "sobol_scramble": False,
                    "transform": lambda U: -1.5 + 3.0 * U,
                    "v_func": lambda X: np.full(X.shape[0], 9.0),
                },
            ),
            estimator_config,
            None,
        ),
    ]

    summary, draws = run_simulation_specs(specs, SimulationRunConfig(replications=1, n_values=(90,), seed=2025))
    assert summary.shape[0] == 3
    assert draws.shape[0] == 3
    assert np.isfinite(summary[["W_true", "W_mean", "bias", "sd"]].to_numpy()).all()


def test_ccg_sievevar_runnable_smoke() -> None:
    summary, draws = ccg_main()

    assert summary.shape[0] == 3
    assert draws.shape[0] == 3
    assert set(summary["model"]) == {"Model1", "Model8", "Model15"}
    assert np.isfinite(summary[["W_true", "bias", "se", "coverage"]].to_numpy()).all()
    assert draws.loc[draws["model"] == "Model15", "W_hat"].between(0.0, 9.0).all()
