"""AIPW augmentation terms must have mean ~0 under a known DGP with known
probabilities (Neyman orthogonality)."""

import numpy as np

from path_welfare.aipw import aipw_11, augmentation_mean, ipw_11
from path_welfare.config import Config
from path_welfare.estimator import TwoStagePathWelfareEstimator
from path_welfare.simulation import get_dgp


def _fit(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    df = get_dgp("dgp1").sample(n, rng)
    df["group"] = np.arange(len(df))
    return TwoStagePathWelfareEstimator(Config(treatment_probs={"e1": 0.5, "e2": 0.5})).fit(df)


def test_augmentation_mean_near_zero():
    est = _fit()
    aug = augmentation_mean(est)
    sd_y = float(np.nanstd(est.data_["Y"]))
    # both augmentation terms should be small relative to the outcome scale
    assert abs(aug["aug_b_mean"]) < 0.05 * sd_y
    assert abs(aug["aug_c_mean"]) < 0.05 * sd_y


def test_ipw_and_aipw_agree_roughly():
    est = _fit()
    i = ipw_11(est); a = aipw_11(est)
    # IPW and AIPW estimate the same fixed-policy (1,1) contribution; they should be close
    assert abs(i.estimate - a.estimate) < 0.1


def test_aipw_close_to_truth():
    est = _fit()
    truth = get_dgp("dgp1").true_functionals()["V11"]
    a = aipw_11(est)
    # the learned policy's (1,1) value is close to the optimal one at large n
    assert abs(a.estimate - truth) < 0.08
