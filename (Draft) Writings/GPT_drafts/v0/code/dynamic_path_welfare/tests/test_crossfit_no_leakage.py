import numpy as np

from path_welfare.crossfit import check_no_group_leak, make_folds


def test_no_group_leak():
    n = 600
    groups = np.repeat(np.arange(120), 5)  # 5 rows per participant
    folds = make_folds(n, 5, seed=3, groups=groups)
    assert check_no_group_leak(folds, groups)


def test_folds_partition():
    folds = make_folds(500, 5, seed=1)
    allidx = np.concatenate(folds)
    assert np.array_equal(np.sort(allidx), np.arange(500))


def test_no_observation_trains_own_nuisance(sample_df, cfg):
    """The honest inner cross-fit must not use a row's own outcome to predict its own mu.
    We check this by perturbing one training row's Y and confirming its OWN honest mu
    prediction is (near) unchanged, while an in-sample fit would move."""
    from path_welfare.estimator import TwoStagePathWelfareEstimator

    est = TwoStagePathWelfareEstimator(cfg)
    df = sample_df.copy()
    X = df["X"].to_numpy(); Y = df["Y"].to_numpy(); T2 = df["T2"].to_numpy()
    groups = df["group"].to_numpy()
    # honest predictions on the full sample
    dim0 = dim1 = 6
    from path_welfare.splines import make_basis
    b0 = make_basis(X, dim0); b1 = make_basis(X, dim1)
    mu0a, mu1a = est._honest_mu(X, Y, T2, dim0, dim1, b0, b1, groups=groups, fold_id=0)
    # perturb one row's Y hugely
    i = np.where(T2 == 1)[0][0]
    Y2 = Y.copy(); Y2[i] += 1000.0
    mu0b, mu1b = est._honest_mu(X, Y2, T2, dim0, dim1, b0, b1, groups=groups, fold_id=0)
    # row i's own honest mu1 prediction must be essentially unaffected by its own Y
    assert abs(mu1a[i] - mu1b[i]) < 1.0  # << 1000
