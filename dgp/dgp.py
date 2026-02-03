import numpy as np
import pandas as pd

def generate_data(n=10000, delta=0.9, seed=None):
    rng = np.random.default_rng(seed)

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n)

    # True CATEs
    tau_S = 1.0 + 0.5 * X1 - 0.3 * X2
    tau_Y = 0.5 + 0.2 * X1 + 0.4 * X2

    mu_S = 2.0 + X1 + X2
    mu_Y = 1.0 + 0.5 * X1 - 0.5 * X2

    S0 = mu_S + rng.normal(0, 1, n)
    S1 = S0 + tau_S

    Y0 = mu_Y + rng.normal(0, 1, n)
    Y1 = Y0 + tau_Y

    S = D * S1 + (1 - D) * S0
    Y = D * Y1 + (1 - D) * Y0

    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "D": D,
        "S": S,
        "Y": Y,
        "tau_S": tau_S,
        "tau_Y": tau_Y
    })

    return df, delta