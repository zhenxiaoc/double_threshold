"""T-learner implementation for CATE estimation.

Provides a simple T-learner that fits separate outcome models for treated
and control groups. Supported base learners: 'linear' (LinearRegression) and
'spline' (B-spline series followed by LinearRegression).
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
import numpy as np


class TLearnerEstimator:
    """Wrapper that exposes an `effect(X)` method returning E[Y|X,D=1]-E[Y|X,D=0]."""

    def __init__(self, model_treated, model_control):
        self.model_treated = model_treated
        self.model_control = model_control

    def effect(self, X):
        return self.model_treated.predict(X) - self.model_control.predict(X)


def _is_binary_treatment(T):
    uniq = np.unique(T)
    if len(uniq) != 2:
        return False
    try:
        u = np.sort(uniq).astype(float)
    except Exception:
        return False
    return np.allclose(u, [0.0, 1.0])


def fit_t_learner(df, outcome, estimator="linear", spline_knots=None, spline_degree=3, random_state=None):
    """random_state parameter accepted for API consistency (currently unused).

    Kept for forward-compatibility so callers can pass per-run seeds.
    """
    """Fit a T-learner using separate models for treated and control groups.

    Parameters
    - df: pandas.DataFrame with columns `X1`, `X2`, `D` and the outcome
    - outcome: name of outcome column (string)
    - estimator: 'linear' or 'spline' (B-spline series followed by linear regression)
    - spline_knots, spline_degree: parameters for the B-spline basis when estimator='spline'.
      These are only required and used when ``estimator='spline'``; otherwise they are ignored.

    Returns
    - TLearnerEstimator with an `effect(X)` method.
    """
    X = df[["X1", "X2"]].values
    T = df["D"].values
    Y = df[outcome].values

    if not _is_binary_treatment(T):
        raise ValueError("T-learner requires a binary treatment variable 'D' with values {0,1}.")

    # Validate spline parameters only when needed
    if estimator == "spline":
        if spline_knots is None:
            spline_knots = 5
        if not (isinstance(spline_knots, int) and spline_knots >= 2):
            raise ValueError("spline_knots must be an integer >= 2 when estimator='spline'.")
        if not (isinstance(spline_degree, int) and 1 <= spline_degree <= 5):
            raise ValueError("spline_degree must be an integer between 1 and 5 when estimator='spline'.")
    else:
        # If user supplied spline params but chose a different estimator, warn and ignore
        if spline_knots is not None:
            import warnings
            warnings.warn("spline_knots parameter is ignored when estimator!='spline'", UserWarning)

    X_treated = df.loc[df["D"] == 1, ["X1", "X2"]].values
    Y_treated = df.loc[df["D"] == 1, outcome].values
    X_control = df.loc[df["D"] == 0, ["X1", "X2"]].values
    Y_control = df.loc[df["D"] == 0, outcome].values

    def _make_model(kind):
        if kind == "linear":
            return LinearRegression()
        elif kind == "spline":
            transformer = SplineTransformer(n_knots=spline_knots, degree=spline_degree, include_bias=False)
            return make_pipeline(transformer, LinearRegression())
        else:
            raise ValueError("estimator must be 'linear' or 'spline'")

    model_treated = _make_model(estimator)
    model_control = _make_model(estimator)

    model_treated.fit(X_treated, Y_treated)
    model_control.fit(X_control, Y_control)

    return TLearnerEstimator(model_treated, model_control)
