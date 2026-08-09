from __future__ import annotations

import numpy as np

from opttreat.config import ParameterConfig, ParameterType
from opttreat.models import TaylorExpansionModel
from opttreat.parameters import get_parameter


def test_welfare_known_and_unknown_evaluate() -> None:
    h = lambda X: X[:, 0] - 0.5
    known = get_parameter(
        ParameterConfig(
            ParameterType.WELFARE_KNOWN_DIST,
            {"dim": 1, "n_sobol": 8, "sobol_seed": 1},
        )
    )
    unknown = get_parameter(ParameterConfig(ParameterType.WELFARE_UNKNOWN_DIST, {"dim": 1}))
    X = np.array([[0.25], [0.75]])

    assert known.evaluate(h) >= 0.0
    assert unknown.evaluate(h, X) == 0.125


def test_value_unknown_uses_indicator_not_relu_weight() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    h = lambda Z: np.array([2.0, -2.0, 0.5])
    v = lambda Z: np.ones(Z.shape[0]) * 3.0
    parameter = get_parameter(
        ParameterConfig(ParameterType.VALUE_UNKNOWN_DIST, {"dim": 1, "X": X, "v_func": v})
    )

    assert parameter.evaluate(h) == 2.0


def test_value_true_value_broadcasts_scalar_h_and_v() -> None:
    parameter = get_parameter(
        ParameterConfig(
            ParameterType.VALUE_KNOWN_DIST,
            {
                "dim": 1,
                "n_sobol": 8,
                "sobol_seed": 2,
                "v_func": lambda X: 2.0,
            },
        )
    )

    assert parameter.get_true_value(TaylorExpansionModel(K=1, p=1, expansion="tan2")) == 2.0
