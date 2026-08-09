import numpy as np

from path_welfare.config import Config
from path_welfare.estimator import TwoStagePathWelfareEstimator
from path_welfare.simulation import get_dgp


def _run(seed_data=0, seed_cf=20260713):
    rng = np.random.default_rng(seed_data)
    df = get_dgp("dgp1").sample(1500, rng)
    df["group"] = np.arange(len(df))
    cfg = Config(treatment_probs={"e1": 0.5, "e2": 0.5})
    cfg.crossfit.seed = seed_cf
    return TwoStagePathWelfareEstimator(cfg).fit(df)


def test_same_seed_same_estimate():
    a = _run()
    b = _run()
    assert a.point_["11"] == b.point_["11"]
    assert a.point_["total"] == b.point_["total"]


def test_same_seed_same_inference():
    a = _run().inference(K=8)
    b = _run().inference(K=8)
    assert a.estimate == b.estimate
    assert a.se_conditional == b.se_conditional


def test_config_hash_stable():
    c1 = Config(name="x", seed=1)
    c2 = Config(name="x", seed=1)
    assert c1.hash() == c2.hash()
    assert Config(name="x", seed=2).hash() != c1.hash()
