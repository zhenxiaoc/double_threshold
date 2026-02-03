from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
import warnings
from .t_learner import fit_t_learner


def _is_binary_treatment(T):
    """Return True if treatment appears binary with values {0, 1}."""
    uniq = np.unique(T)
    if len(uniq) != 2:
        return False
    try:
        u = np.sort(uniq).astype(float)
    except Exception:
        return False
    return np.allclose(u, [0.0, 1.0])


def fit_causal_forest(
    df, outcome, random_state=None,
    rf_regressor_params=None, rf_classifier_params=None, cf_params=None
):
    """Fit a causal forest. Accepts optional `random_state` and tuning parameter dicts.

    `random_state` is forwarded to the underlying random forest models and the
    `CausalForestDML` estimator. If provided, small offsets are used for different
    components to avoid identical RNG streams.
    rf_regressor_params, rf_classifier_params, cf_params: dicts of keyword args for
    RandomForestRegressor, RandomForestClassifier, and CausalForestDML respectively.
    """
    X = df[["X1", "X2"]].values
    T = df["D"].values
    Y = df[outcome].values

    # Ensure treatment is binary in this simulation design
    if not _is_binary_treatment(T):
        raise ValueError(
            "Treatment variable 'D' does not appear to be binary (expected values {0,1})."
        )

    # Detect whether treatment is discrete (e.g., binary or low-cardinality integer values).
    uniq = np.unique(T)
    is_discrete = False
    # Consider discrete when values are integer-valued and there are few unique values
    if np.issubdtype(T.dtype, np.integer) or len(uniq) <= 10:
        if np.allclose(uniq, np.round(uniq)):
            is_discrete = True

    # Use reproducible random states when provided

    rs_model_y = None if random_state is None else int(random_state)
    rs_model_t = None if random_state is None else int(random_state) + 1
    rs_cf = None if random_state is None else int(random_state) + 2

    rf_regressor_params = rf_regressor_params or {}
    rf_classifier_params = rf_classifier_params or {}
    cf_params = cf_params or {}

    # Set defaults if not provided, and always use n_jobs=-1 for speed
    rf_regressor_defaults = dict(n_estimators=100, min_samples_leaf=10, max_depth=10, n_jobs=-1)
    rf_classifier_defaults = dict(n_estimators=100, min_samples_leaf=10, max_depth=10, n_jobs=-1)
    cf_defaults = dict(n_estimators=200, min_samples_leaf=20, n_jobs=-1)
    rf_regressor_defaults.update(rf_regressor_params)
    rf_classifier_defaults.update(rf_classifier_params)
    cf_defaults.update(cf_params)

    if is_discrete:
        model_t = RandomForestClassifier(random_state=rs_model_t, **rf_classifier_defaults)
    else:
        model_t = RandomForestRegressor(random_state=rs_model_t, **rf_regressor_defaults)

    cf = CausalForestDML(
        model_y=RandomForestRegressor(random_state=rs_model_y, **rf_regressor_defaults),
        model_t=model_t,
        discrete_treatment=is_discrete,
        random_state=rs_cf,
        **cf_defaults
    )

    cf.fit(Y, T, X=X)
    return cf
