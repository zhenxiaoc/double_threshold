"""Property tests for the WGAN (second) calibration.

These train small GANs, so they are slower than the primary-oracle tests; run:
    PYTHONPATH=src python -m pytest tests/test_wgan.py -q
"""
import numpy as np
import pytest

from harm_share.wgan_backend import WGANSpec, train_conditional_wgan, NumpyGenerator
from harm_share.wgan_calibration import WGANConfig, build_wgan_oracle, _CovariateEncoder
from harm_share.calibration import COVARIATES_CR_FULL, BINARY_COVARIATES, load_graduation


# --------------------------- covariate encoder --------------------------- #
def test_covariate_encoder_roundtrip_and_kinds():
    """Encoder must classify (binary/hurdle/cont) and decode exactly."""
    df = load_graduation()
    Xr = df[COVARIATES_CR_FULL].to_numpy(float)
    Xr = np.where(np.isfinite(Xr), Xr, np.nanmean(Xr, axis=0))
    enc = _CovariateEncoder(BINARY_COVARIATES, hurdle_frac=0.40).fit(Xr, COVARIATES_CR_FULL)
    # 5 binaries + 6 heavily-censored (>40% zero) monetary vars
    kinds = {COVARIATES_CR_FULL[j]: enc.kind[j] for j in range(len(COVARIATES_CR_FULL))}
    assert sum(k == "binary" for k in kinds.values()) == 5
    assert sum(k == "hurdle" for k in kinds.values()) == 6
    Xrec = enc.decode(enc.encode(Xr))
    assert np.max(np.abs(Xrec - Xr) / (np.abs(Xr) + 1)) < 1e-6      # exact roundtrip
    # binaries decode to {0,1}; hurdle vars preserve their zero mass
    b = COVARIATES_CR_FULL.index("fs_enoughfood_bsl")
    assert set(np.unique(Xrec[:, b])) <= {0.0, 1.0}
    assert np.allclose(enc.from_state(enc.state()).decode(enc.encode(Xr)), Xrec)


def test_categorical_backend_numpy_matches_torch():
    """The softmax categorical head must replay exactly in NumPy."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 2))
    b = (rng.random(400) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    ng, hist = train_conditional_wgan(X, None, WGANSpec(epochs=60, critic_steps=3, seed=0),
                                      cat=b.reshape(-1, 1), cat_cards=[2])
    assert hist["numpy_torch_max_err"] < 1e-4
    assert ng.cat_cards == (2,) and ng.p_cont == 2
    assert ng.sample(None, 10, rng).shape == (10, 3)               # 2 cont + 1 category


# --------------------------- backend unit tests --------------------------- #
def test_numpy_matches_torch_forward():
    """The exported NumPy generator must replay the torch generator exactly."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 2))
    Y = (X[:, :1] * 1.5 - 0.5) + 0.3 * rng.standard_normal((300, 1))
    spec = WGANSpec(epochs=60, critic_steps=3, seed=0)
    ng, hist = train_conditional_wgan(Y, X, spec)
    assert hist["numpy_torch_max_err"] < 1e-4


def test_generator_state_roundtrip():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 2))
    Y = rng.standard_normal((200, 1))
    ng, _ = train_conditional_wgan(Y, X, WGANSpec(epochs=40, critic_steps=2))
    st = ng.to_state("g")
    ng2 = NumpyGenerator.from_state("g", st)
    z = rng.standard_normal((10, ng.noise_dim))
    assert np.allclose(ng.generate(X[:10], z), ng2.generate(X[:10], z))


# ------------------------------ oracle tests ------------------------------ #
@pytest.fixture(scope="module")
def small_oracle():
    """A deliberately tiny/fast WGAN oracle (not for inference quality)."""
    fast = dict(
        spec_X=WGANSpec(epochs=400, critic_steps=5, seed=1),
        spec_S=WGANSpec(epochs=400, critic_steps=5, seed=2),
        spec_Y=WGANSpec(epochs=300, critic_steps=5, seed=3),
        n_pop=20_000, tau_M=150, pop_chunk=4000,
    )
    return build_wgan_oracle(verbose=False, **fast)


def test_oracle_interface(small_oracle):
    orc = small_oracle
    rng = np.random.default_rng(0)
    df = orc.sample_experiment(500, rng)
    assert list(df.columns) == ["X1", "X2", "W", "S", "Y"]
    assert set(np.unique(df["W"])) <= {0, 1}
    assert np.isfinite(df[["S", "Y"]].to_numpy()).all()
    tS, tY = orc.true_cates(df.iloc[:100])
    assert tS.shape == (100,) and tY.shape == (100,)


def test_truth_quadrants_sum_to_one(small_oracle):
    tr = small_oracle.truth()
    total = tr["theta_pp"] + tr["theta_harm"] + tr["theta_mp"] + tr["theta_mm"]
    assert abs(total - 1.0) < 1e-9
    assert abs((tr["theta_pp"] + tr["theta_harm"]) - tr["treat_share_S"]) < 1e-9
    assert 0.0 <= tr["theta_harm"] <= 1.0


def test_save_load_roundtrip(small_oracle, tmp_path):
    from harm_share.wgan_calibration import WGANOracle
    p = tmp_path / "orc.npz"
    small_oracle.save(p)
    orc2 = WGANOracle.load(p)
    # same population truth after reload
    t1, t2 = small_oracle.truth(), orc2.truth()
    assert abs(t1["theta_harm"] - t2["theta_harm"]) < 1e-9
    # sampler reproducible and well-formed
    df = orc2.sample_experiment(50, np.random.default_rng(3))
    assert len(df) == 50
