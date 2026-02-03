import os
import sys
import numpy as np

# Ensure project root is on sys.path when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dgp.dgp import generate_data
from estimation.t_learner import fit_t_learner


def test_t_learner_shape_and_reasonable_values():
    df, _ = generate_data(n=200, seed=123)
    t = fit_t_learner(df, "Y", estimator="linear")

    X = df[["X1", "X2"]].values
    tau_hat = t.effect(X)

    assert tau_hat.shape == (len(df),)
    assert np.all(np.isfinite(tau_hat))
