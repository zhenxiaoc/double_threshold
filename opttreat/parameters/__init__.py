# parameters/__init__.py

from __future__ import annotations

from typing import Any

from .base import Parameter
from .welfare import WelfareKnownDist, WelfareUnknownDist
from .value import ValueKnownDist, ValueUnknownDist

from opttreat.config import ParameterConfig, ParameterType

__all__ = [
    "Parameter",
    "WelfareKnownDist",
    "WelfareUnknownDist",
    "ValueKnownDist",
    "ValueUnknownDist",
    "get_parameter",
]


def get_parameter(cfg: ParameterConfig) -> Parameter:
    """
    Factory that turns a ParameterConfig into a concrete Parameter object.
    """
    pt = cfg.param_type

    if pt == ParameterType.WELFARE_KNOWN_DIST:
        return WelfareKnownDist(
            name="welfare_known",
            options=cfg.options,
        )

    if pt == ParameterType.WELFARE_UNKNOWN_DIST:
        return WelfareUnknownDist(
            name="welfare_unknown",
            options=cfg.options,
        )

    if pt == ParameterType.VALUE_KNOWN_DIST:
        return ValueKnownDist(
            name="value_known",
            options=cfg.options,
        )

    if pt == ParameterType.VALUE_UNKNOWN_DIST:
        return ValueUnknownDist(
            name="value_unknown",
            options=cfg.options,
        )

    raise ValueError(f"Unknown ParameterType: {pt!r}")
