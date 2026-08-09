import numpy as np

from path_welfare.diagnostics import positivity_report


def test_design_propensities_recorded(sample_df):
    pos = positivity_report(sample_df, e1=0.5, e2=0.5)
    assert pos["P(T1=1)_design"] == 0.5
    assert pos["P(T2=1)_design"] == 0.5
    assert abs(pos["P(T1=1)_empirical"] - 0.5) < 0.05
    assert pos["positivity_holds_at_0.10"]


def test_empirical_matches_design(sample_df):
    # under the 0.5 design, empirical must be close to 0.5
    assert abs(np.mean(sample_df["T1"]) - 0.5) < 0.05
    assert abs(np.mean(sample_df["T2"]) - 0.5) < 0.05


def test_estimator_stores_design_probs(sample_df, cfg):
    from path_welfare.estimator import TwoStagePathWelfareEstimator
    est = TwoStagePathWelfareEstimator(cfg).fit(sample_df)
    assert est.e1_ == 0.5
    assert est.e2_ == 0.5
