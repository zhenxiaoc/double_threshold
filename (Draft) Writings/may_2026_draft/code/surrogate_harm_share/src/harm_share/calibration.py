"""Calibrated oracle population for the surrogate-induced harm share.

We calibrate a generative model to the Banerjee et al. (2015) "graduation" RCT
(the Pakistan/Sindh subset shipped with the Chen & Ritzwoller `longterm` R
package, `data/graduation.rda`, 854 households, randomized treatment).  The
object we build is an *oracle population*: known short- and long-run CATE
surfaces

    tau_S(x) = E[S(1) - S(0) | X = x]   (short-run, 2-year, effect)
    tau_Y(x) = E[Y(1) - Y(0) | X = x]   (long-run, 3-year, effect)

plus a covariate density f(x) and a sampler for finite experiments, so that the
population truth of every functional (harm share, the four sign quadrants, the
two decision boundaries) is *computable, not estimated*.

Relationship to Chen & Ritzwoller's GAN calibration (their Appendix D.2/D.3).
CR fit three cascaded GANs: (1) the marginal of X, (2) S | X, W, (3) Y | S, X, W
(with the `wgan` package of Athey et al. 2021), then take 10^7 draws as the
population -- a job they report costs ~60 CPU-years on a cluster.  Because our
target theta is a functional of the conditional *means* (through tau_S, tau_Y)
and of f(x), the GAN's role -- realistic conditional *shape* -- does not change
the population truth; it only affects finite-sample estimation noise.  We
therefore reproduce the SAME cascade structure

        X  ->  (S(0), S(1))  ->  (Y(0), Y(1))

with smooth kernel-ridge conditional means and nonparametric residual
resampling in place of the WGANs.  This runs in CPU-seconds and yields an exact
truth.  `gan_calibration.py` provides an optional drop-in WGAN backend (GPU)
that swaps only the conditional sampler; the oracle surfaces and hence the truth
are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import QuantileTransformer

# --------------------------------------------------------------------------- #
# Column contract of the graduation data (see longterm/man/graduation.Rd)
# --------------------------------------------------------------------------- #
TREATMENT = "treatment"
# short-run S = 2-year per-capita monthly total consumption; long-run Y = 3-year.
S_COL = "ctotal_pcmonth_end"
Y_COL = "ctotal_pcmonth_fup"
# economically interpretable baseline covariates (all pre-treatment, *_bsl).
COVARIATES_2D = ["ctotal_pcmonth_bsl", "asset_index_bsl"]
COVARIATES_3D = ["ctotal_pcmonth_bsl", "asset_index_bsl", "index_foodsecurity_bsl"]

# Chen & Ritzwoller's FULL baseline covariate set (App. §Data): all 20 pre-treatment
# variables across five categories -- consumption, food security, assets, finance,
# income/revenue.  Used by the faithful high-dimensional WGAN calibration.
COVARIATES_CR_FULL = [
    # consumption (per-capita monthly)
    "ctotal_pcmonth_bsl", "cnonfood_pcmonth_bsl", "cfood_pcmonth_bsl", "cdurable_pcmonth_bsl",
    # food security (index + 5 binary indicators)
    "index_foodsecurity_bsl", "fs_enoughfood_bsl", "fs_adultskip_bsl", "fs_wholeday_bsl",
    "fs_childskip_bsl", "fs_twomeals_bsl",
    # assets (three indices)
    "asset_index_bsl", "asset_prod_index_bsl", "asset_hh_index_bsl",
    # finance (loans + savings; heavily censored at 0)
    "loan_totalamt_bsl", "loan_informalamt_bsl", "loan_formalamt_bsl", "sav_depositamt_bsl",
    # income / revenue
    "iagri_month_bsl", "ranimals_month_bsl", "percep_econ_bsl",
]
# the five binary (categorical) food-security indicators within COVARIATES_CR_FULL
BINARY_COVARIATES = {
    "fs_enoughfood_bsl", "fs_adultskip_bsl", "fs_wholeday_bsl", "fs_childskip_bsl", "fs_twomeals_bsl",
}

# Chen & Ritzwoller's FULL short-term outcome set (App. §Data): the 21 two-year
# (Endline-1, *_end) measurements.  CR use this whole vector as S_i; to forecast
# the long-run Y well one must condition on all of them (a single surrogate is not
# a sufficient statistic).  The faithful WGAN generates this 21-vector as S | X, W
# and forecasts Y | S(full), X, W.  The harm-share threshold rule 1{tau_S >= 0}
# still acts on the scalar short-run welfare target `S_COL` (total consumption),
# which is kept in RAW units as the leading column so tau_S needs no back-transform.
S_COLS_FULL = [
    # consumption (per-capita monthly) -- ctotal is the threshold surrogate S_COL
    "ctotal_pcmonth_end", "cnonfood_pcmonth_end", "cfood_pcmonth_end", "cdurable_pcmonth_end",
    # food security (index + 5 binary indicators)
    "index_foodsecurity_end", "fs_enoughfood_end", "fs_adultskip_end", "fs_wholeday_end",
    "fs_childskip_end", "fs_twomeals_end",
    # assets (three indices)
    "asset_index_end", "asset_prod_index_end", "asset_hh_index_end",
    # finance (loans + savings; heavily censored at 0)
    "loan_totalamt_end", "loan_informalamt_end", "loan_formalamt_end", "sav_depositamt_end",
    # income / revenue
    "iagri_month_end", "ibusiness_month_end", "ranimals_month_end", "percep_econ_end",
]
# the five binary food-security indicators within S_COLS_FULL (softmax heads)
BINARY_S = {
    "fs_enoughfood_end", "fs_adultskip_end", "fs_wholeday_end", "fs_childskip_end", "fs_twomeals_end",
}

_DEFAULT_RDA = (
    Path(__file__).resolve().parents[3]
    / "longterm-main" / "longterm-main" / "data" / "graduation.rda"
)


def load_graduation(rda_path: str | Path | None = None) -> pd.DataFrame:
    """Load the graduation RCT from the .rda shipped with the longterm package."""
    import pyreadr

    path = Path(rda_path) if rda_path is not None else _DEFAULT_RDA
    if not path.exists():
        raise FileNotFoundError(
            f"graduation.rda not found at {path}. Point `rda_path` at "
            "longterm-main/longterm-main/data/graduation.rda."
        )
    res = pyreadr.read_r(str(path))
    return res["graduation"]


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
@dataclass
class OracleConfig:
    covariates: tuple[str, ...] = tuple(COVARIATES_2D)
    s_col: str = S_COL
    y_col: str = Y_COL
    # KRR (RBF) smoothing.  Tuned so that (i) the harm quadrant retains ~0.15
    # mass, (ii) both zero level sets have non-vanishing gradient, and (iii) the
    # two gradients are transversal at the corner (see docs/design.md).
    gamma_s: float = 0.5
    alpha_s: float = 0.5
    gamma_y: float = 0.4
    alpha_y: float = 0.6
    e: float = 0.5                    # randomization probability P(W=1)
    kde_bw: float = 0.35             # covariate-density KDE bandwidth (Scott factor)
    sl_coupling: float = 0.6         # optional short->long residual coupling (fidelity only)
    n_quantiles: int = 400
    support_bound: float = 3.0       # clip covariate draws/grid to [-b, b]^d (clean sieve knots)
    # Outcome-noise scale.  The oracle CATE surfaces (and hence the population
    # truth) are FIXED; noise_scale multiplies the resampled residuals so we can
    # study a clean SNR~1 regime (noise_scale ~ 0.34) and the realistic
    # low-SNR regime (noise_scale = 1.0) on the SAME truth.  See docs/design.md.
    noise_scale: float = 0.34


class HarmShareOracle:
    """Fitted oracle: smooth CATE surfaces + covariate density + cascade sampler.

    All covariates are pushed through a Gaussian quantile transform (fit on the
    real data) so the support is well-behaved and outlier-robust; every method
    below operates in that transformed z-space, which is the space in which the
    population truth is defined.
    """

    def __init__(self, cfg: OracleConfig | None = None):
        self.cfg = cfg or OracleConfig()
        self.d = len(self.cfg.covariates)

    # ------------------------------ fitting ------------------------------ #
    def fit(self, df: pd.DataFrame) -> "HarmShareOracle":
        cfg = self.cfg
        W = df[TREATMENT].to_numpy().astype(int)
        S = df[cfg.s_col].to_numpy(float)
        Y = df[cfg.y_col].to_numpy(float)
        Xr = df[list(cfg.covariates)].to_numpy(float)

        self.qt_ = QuantileTransformer(
            output_distribution="normal", n_quantiles=cfg.n_quantiles, random_state=0
        ).fit(Xr)
        Xz = self.qt_.transform(Xr)

        keep = np.isfinite(S) & np.isfinite(Y) & np.all(np.isfinite(Xz), axis=1)
        Xz, W, S, Y = Xz[keep], W[keep], S[keep], Y[keep]

        self._muS = self._fit_arm(Xz, W, S, cfg.gamma_s, cfg.alpha_s)
        self._muY = self._fit_arm(Xz, W, Y, cfg.gamma_y, cfg.alpha_y)

        # residual pools for the cascade sampler (nonparametric shapes)
        self.res_S_ = {w: S[W == w] - self._muS[w].predict(Xz[W == w]) for w in (0, 1)}
        self.res_Y_ = {w: Y[W == w] - self._muY[w].predict(Xz[W == w]) for w in (0, 1)}

        # covariate density in z-space (Gaussian KDE) and an empirical z-pool
        from scipy.stats import gaussian_kde

        self.Xz_ = Xz
        self.kde_ = gaussian_kde(Xz.T, bw_method=cfg.kde_bw)
        self.sd_S_ = float(np.nanstd(S))
        self.sd_Y_ = float(np.nanstd(Y))
        self.ate_S_ = float(self.tau_S(Xz).mean())
        self.ate_Y_ = float(self.tau_Y(Xz).mean())
        return self

    @staticmethod
    def _fit_arm(Xz, W, out, gamma, alpha):
        return {
            w: KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Xz[W == w], out[W == w])
            for w in (0, 1)
        }

    # --------------------------- oracle surfaces ------------------------- #
    def mu_S(self, xz, w):
        return self._muS[w].predict(np.atleast_2d(xz))

    def mu_Y(self, xz, w):
        return self._muY[w].predict(np.atleast_2d(xz))

    def tau_S(self, xz):
        xz = np.atleast_2d(xz)
        return self._muS[1].predict(xz) - self._muS[0].predict(xz)

    def tau_Y(self, xz):
        xz = np.atleast_2d(xz)
        return self._muY[1].predict(xz) - self._muY[0].predict(xz)

    def density(self, xz):
        return self.kde_(np.atleast_2d(xz).T)

    def draw_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw covariates from f_hat (KDE) TRUNCATED to the [-b, b]^d box.

        Rejection sampling (not clipping): clipping would pile mass on the box
        boundary and distort the density; truncation keeps the sampled law equal
        to the KDE renormalized on the box -- exactly the density `grid_truth`
        integrates, so the MC and grid truths agree.
        """
        b = self.cfg.support_bound
        out = np.empty((0, self.d))
        while len(out) < n:
            Xz = self.kde_.resample(2 * n, seed=rng.integers(1 << 31)).T
            Xz = Xz[np.all(np.abs(Xz) <= b, axis=1)]
            out = np.vstack([out, Xz])
        return out[:n]

    # ------------------------------ sampler ------------------------------ #
    def sample_experiment(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw an experimental sample from the calibrated cascade.

        X ~ f_hat (KDE draw)  ->  W ~ Bernoulli(e)  ->  S = mu_S,W(X)+eps_S,
        Y = mu_Y,W(X) + coupling*(S-mu_S,W(X)) + eps_Y, with eps resampled from
        the fitted residual pools.  The short->long coupling induces a realistic
        S-Y correlation WITHOUT changing tau_Y(x) (the added term is mean-zero
        given X, W).  Only the realized potential outcome for the drawn W is
        returned (as in a real experiment).
        """
        cfg = self.cfg
        Xz = self.draw_X(n, rng)
        W = (rng.random(n) < cfg.e).astype(int)
        S = np.empty(n)
        Y = np.empty(n)
        for w in (0, 1):
            m = W == w
            if not m.any():
                continue
            muS = self._muS[w].predict(Xz[m])
            muY = self._muY[w].predict(Xz[m])
            eS = cfg.noise_scale * rng.choice(self.res_S_[w], size=m.sum(), replace=True)
            eY = cfg.noise_scale * rng.choice(self.res_Y_[w], size=m.sum(), replace=True)
            S[m] = muS + eS
            Y[m] = muY + cfg.sl_coupling * eS + eY
        cols = {f"X{j+1}": Xz[:, j] for j in range(self.d)}
        cols.update({"W": W, "S": S, "Y": Y})
        return pd.DataFrame(cols)

    def true_cates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xz = df[[f"X{j+1}" for j in range(self.d)]].to_numpy(float)
        return self.tau_S(Xz), self.tau_Y(Xz)


def build_oracle(covariates=COVARIATES_2D, rda_path=None, **cfg_kw) -> HarmShareOracle:
    df = load_graduation(rda_path)
    cfg = OracleConfig(covariates=tuple(covariates), **cfg_kw)
    return HarmShareOracle(cfg).fit(df)
