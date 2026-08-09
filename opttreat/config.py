"""Public configuration objects used by OptTreat factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(Enum):
    """Supported target parameter families."""

    WELFARE_KNOWN_DIST = "welfare_known"
    WELFARE_UNKNOWN_DIST = "welfare_unknown"
    VALUE_KNOWN_DIST = "value_known"
    VALUE_UNKNOWN_DIST = "value_unknown"


@dataclass
class EstimatorConfig:
    """Factory input for first-stage estimators.

    ``method`` names the estimator implementation, such as ``"sieve"`` or
    ``"rf_ridge"``. ``options`` are passed to that estimator unchanged.
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class VarianceConfig:
    """Factory input for variance estimators.

    ``method`` names the variance estimator implementation. ``options`` are
    passed to that estimator unchanged.
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterConfig:
    """Factory input for target parameters.

    ``param_type`` selects welfare/value and known/unknown distribution logic.
    ``options`` are passed to the parameter implementation unchanged.
    """

    param_type: ParameterType
    options: dict[str, Any] = field(default_factory=dict)
