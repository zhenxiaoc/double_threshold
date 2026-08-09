import numpy as np

from path_welfare.estimator import TwoStagePathWelfareEstimator


def test_components_sum_to_total(sample_df, cfg):
    est = TwoStagePathWelfareEstimator(cfg).fit(sample_df)
    p = est.estimate_all_paths()
    s = p["11"] + p["10"] + p["01"] + p["00"]
    assert abs(s - p["total"]) < 1e-9
    # the cross-fitted component sum equals the direct total up to fold discrepancy
    assert abs(p["total"] - p["total_direct"]) < 1e-6


def test_true_components_sum(dgp1):
    t = dgp1.true_functionals()
    s = t["V11"] + t["V10"] + t["V01"] + t["V00"]
    assert abs(s - t["total"]) < 1e-6
    assert abs(t["total"] - t["total_direct"]) < 1e-3


def test_plugin_close_to_truth(dgp1):
    truth = dgp1.true_functionals()
    from path_welfare.config import Config
    rng = np.random.default_rng(11)
    df = dgp1.sample(4000, rng)
    df["group"] = np.arange(len(df))
    est = TwoStagePathWelfareEstimator(Config(treatment_probs={"e1": 0.5, "e2": 0.5})).fit(df)
    p = est.estimate_all_paths()
    assert abs(p["11"] - truth["V11"]) < 0.06  # within a few % of truth at n=4000
