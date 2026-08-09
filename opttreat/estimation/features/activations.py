# random_features/activations.py

from __future__ import annotations
import numpy as np
from typing import Any, Callable


def get_activation(act_opt: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Convert an activation option into a callable.

    Parameters
    ----------
    act_opt : {"cos", "relu"} or callable

    Returns
    -------
    activation : callable
        A function f(z) applied elementwise to the feature matrix.
    """
    if callable(act_opt):
        return act_opt

    if act_opt is None or act_opt == "cos":
        return np.cos

    if act_opt == "relu":
        return lambda z: np.maximum(z, 0.0)
    
    if act_opt == "sigmoid":
        return lambda z: 1 / (1 + np.exp(-z))
    
    if act_opt == "tanh":
        return lambda z: np.tanh(z)

    if act_opt == "exp":
        return lambda z: np.exp(z)

    raise ValueError(f"Unsupported activation option: {act_opt!r}")
