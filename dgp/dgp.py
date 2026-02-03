import numpy as np
import pandas as pd

def _dgp_linear(n, rng, delta):
    """Linear DGP."""
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n)

    # True CATEs (linear in X)
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

    return df


def _dgp_nonlinear(n, rng, delta):
    """Nonlinear DGP with strong polynomial and cubic terms."""
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n)

    # True CATEs (highly nonlinear in X with cubic terms and strong interactions)
    tau_S = 1.0 + 0.4 * X1 - 0.5 * X2 + 0.6 * (X1 ** 2) - 0.5 * (X2 ** 2) + 0.4 * (X1 ** 3) - 0.35 * (X2 ** 3) + 0.5 * X1 * X2 + 0.3 * (X1 ** 2) * X2 - 0.3 * X1 * (X2 ** 2)
    tau_Y = 0.6 + 0.3 * X1 + 0.4 * X2 - 0.5 * (X1 ** 2) + 0.4 * (X2 ** 2) - 0.35 * (X1 ** 3) + 0.3 * (X2 ** 3) - 0.4 * X1 * X2 - 0.25 * (X1 ** 2) * X2 + 0.25 * X1 * (X2 ** 2)

    mu_S = 2.0 + 0.6 * X1 + 0.5 * X2 + 0.4 * (X1 ** 2) - 0.3 * (X2 ** 2) + 0.2 * (X1 ** 3) - 0.15 * (X2 ** 3)
    mu_Y = 1.5 + 0.4 * X1 - 0.35 * X2 - 0.3 * (X1 ** 2) + 0.25 * (X2 ** 2) - 0.15 * (X1 ** 3) + 0.1 * (X2 ** 3)

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

    return df


def _dgp_trigonometric(n, rng, delta):
    """Nonlinear DGP with strong trigonometric (sin and cos) functions and higher-order interactions."""
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n)

    # True CATEs (highly nonlinear with sin/cos and complex interactions)
    tau_S = 1.0 + 1.2 * np.sin(X1) + 1.0 * np.cos(X2) + 0.8 * np.sin(X1) * np.cos(X2) + 0.5 * (X1 ** 2) * np.sin(X2) + 0.4 * (X2 ** 2) * np.cos(X1)
    tau_Y = 0.5 + 0.9 * np.cos(X1) + 0.7 * np.sin(X2) - 0.6 * np.sin(X1) * np.cos(X2) - 0.4 * (X1 ** 2) * np.sin(X1) + 0.35 * (X2 ** 2) * np.cos(X2)

    mu_S = 2.2 + 0.9 * np.sin(X1) + 0.7 * np.cos(X2) + 0.4 * (X1 ** 2) * np.cos(X1)
    mu_Y = 1.4 + 0.6 * np.cos(X1) - 0.5 * np.sin(X2) - 0.3 * (X2 ** 2) * np.sin(X2)

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

    return df


def generate_data(n=10000, delta=0.9, seed=None, dgp="linear"):
    """Generate synthetic data using the specified DGP.
    
    Parameters:
    - n: sample size
    - delta: discount factor
    - seed: random seed
    - dgp: "linear", "nonlinear", or "trigonometric"
    """
    if dgp not in ["linear", "nonlinear", "trigonometric"]:
        raise ValueError(f"Unknown DGP: {dgp}. Choose 'linear', 'nonlinear', or 'trigonometric'.")
    
    rng = np.random.default_rng(seed)
    
    if dgp == "linear":
        df = _dgp_linear(n, rng, delta)
    elif dgp == "nonlinear":
        df = _dgp_nonlinear(n, rng, delta)
    else:  # trigonometric
        df = _dgp_trigonometric(n, rng, delta)
    
    return df, delta