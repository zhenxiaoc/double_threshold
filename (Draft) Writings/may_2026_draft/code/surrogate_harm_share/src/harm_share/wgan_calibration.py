"""Second calibration: a *truthful* WGAN population for the harm share.

Where the primary calibration (`calibration.py`) replaces Chen & Ritzwoller's
three WGANs with smooth kernel-ridge conditional means (so the population truth
is closed-form and exact), this module does what CR actually do: it **trains the
three cascaded Wasserstein GANs** on the graduation data and **takes a large
sample from the generative model as the population**.

    GAN1:  X            (marginal of pre-treatment covariates, in z-space)
    GAN2:  S | X, W     (the FULL short-term outcome VECTOR given covariates, W)
    GAN3:  Y | S, X, W  (long-term given the full short-term vector, covariates, W)

Following Chen & Ritzwoller, S is the whole 21-dimensional vector of two-year
(Endline-1) measurements, not a single surrogate: to forecast the three-year
outcome Y well one must condition on enough surrogates (one number is not a
sufficient statistic for the long run).  The harm-share estimand still needs a
*scalar* short-run outcome for its rule 1{tau_S(X) >= 0}, so the threshold
surrogate (total consumption, `s_col`) is carried as RAW column 0 of the S vector
-- tau_S reads straight off it -- while the other 20 surrogates are encoded
(continuous / binary-softmax / hurdle) and serve only as conditioning richness for
GAN3.  Set `full_surrogate=False` to fall back to the scalar-S cascade.

Generation cascade for one unit (CR App. D.3):

    X ~ GAN1
    S(0), S(1) ~ GAN2( X, W=0 ), GAN2( X, W=1 )      # full surrogate vectors
    Y(0), Y(1) ~ GAN3( S(0), X, 0 ), GAN3( S(1), X, 1 )

Conditioning on treatment.  As in CR, GAN2/GAN3 condition on the binary treatment
W by feeding it as a context input to a single generator (GAN2 = S | X, W; GAN3 =
Y | S, X, W).  The CATE is then read off by contrasting the generator at W=1 vs
W=0.  This recovers the average effect well (generated E[S|W=1]-E[S|W=0] matches
the data) and, with enough training, a growing share of the CATE *heterogeneity*
-- though on 854 households the short-run heterogeneity is learned only slowly, so
the short-run threshold barely binds (see train_wgan.py diagnostics).

Because our estimand θ = Pr(τ_S(X) ≥ 0, τ_Y(X) ≤ 0) is a functional of the two
*conditional-mean* CATEs, the population truth here is NOT a closed form: it is a
property of the trained generators.  We compute it exactly (to controllable
Monte-Carlo precision) by

    τ_S(x) = E_U[ G_S(x,1,U)[0] − G_S(x,0,U)[0] ],          # raw threshold column
    τ_Y(x) = E_{U,V}[ G_Y(G_S(x,1,U),x,1,V) − G_Y(G_S(x,0,U),x,0,V) ],

evaluated by conditional Monte Carlo over the generator noise with **common
random numbers** (same U,V across the two arms — a large variance reduction for
the difference), on a large fixed population of X draws.  The truth is thus the
harm share of the *WGAN's own* conditional means, computed rather than closed
form, and the finite-sample experiments are genuine cascade draws with the
GAN's realistic (heteroskedastic, skewed, heavy-tailed) conditional shapes.

Notes on faithfulness / cost.
  * Tuning follows CR's Table exactly where they state it (batch 256, lr 1e-4 for
    both networks, gradient penalty 20, dropout 0.1); see `wgan_backend.WGANSpec`.
  * CR run 30000 epochs (X, S) / 5000 (Y) on a cluster (~60 CPU-years for the full
    high-dimensional problem).  Here d=2 covariates and 1-D S,Y on 854 rows is a
    far smaller problem; we train for `epochs` chosen so the generated joint
    distribution matches the data (means / variances / correlations align on the
    45-degree line; see `make_wgan_figures.py`), which is minutes on one GPU.  The
    treatment CONTRASTS (ATEs) come out with the right sign and order of magnitude
    but attenuated -- an honest reflection of learning a subtle CATE from 854 rows.
    This is a *compute* deviation, not a *procedure* deviation.
  * Covariates live in the same Gaussian quantile z-space as the primary oracle,
    so the sieve knots and the two calibrations are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .calibration import (
    BINARY_COVARIATES, BINARY_S, COVARIATES_2D, COVARIATES_CR_FULL, S_COL, S_COLS_FULL, Y_COL,
    TREATMENT, load_graduation,
)
from .wgan_backend import NumpyGenerator, WGANSpec, train_conditional_wgan


# --------------------------------------------------------------------------- #
# Covariate preprocessing: a signed-log transform that tames the heavy right
# tails / zero-spikes of the monetary and asset covariates (loans, savings,
# consumption, asset indices have skew 3-18).  Without it the Gaussian quantile
# transform's inverse amplifies tail errors and the generated monetary variables
# blow up.  signed_log1p handles negatives (e.g. agricultural income) too.
# --------------------------------------------------------------------------- #
def _signed_log1p(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return np.sign(a) * np.log1p(np.abs(a))


def _signed_expm1(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return np.sign(a) * np.expm1(np.abs(a))


# --------------------------------------------------------------------------- #
# Covariate encoder: ds-wgan-faithful mixed representation of the 20 covariates.
#   * continuous, well-behaved -> quantile-normal z
#   * continuous, heavy-tailed (|skew|>thr) -> signed-log1p then quantile-normal z
#   * binary -> a categorical (softmax) column, value = category code
#   * heavily censored (>hurdle_frac zeros) -> a HURDLE: a categorical indicator
#     {x!=0} (softmax) + a continuous signed-log positive-part column (zeros filled
#     with the nonzero-mean so the continuous column carries no zero spike).
# GAN1 generates the model matrix [continuous z ... | category indices ...]; the
# encoder decodes it back to the 20 covariates in original units.
# --------------------------------------------------------------------------- #
class _CovariateEncoder:
    def __init__(self, binary_set, hurdle_frac=0.40, log_skew_threshold=2.0):
        self.binary_set = set(binary_set)
        self.hurdle_frac = hurdle_frac
        self.log_skew_threshold = log_skew_threshold

    def fit(self, Xr: np.ndarray, names: list[str]) -> "_CovariateEncoder":
        from scipy.stats import skew
        d = Xr.shape[1]
        self.names = list(names)
        self.kind = []            # per original col: 'cont' | 'binary' | 'hurdle'
        self.cont_log = []        # per original col: signed-log the continuous part?
        self.hurdle_fill = {}     # col idx -> fill value for zero rows (logval)
        cont_blocks, cat_blocks = [], []   # (orig_col, 'value'|'logval'|'indicator')
        for j in range(d):
            x = Xr[:, j]
            xf = x[np.isfinite(x)]
            if names[j] in self.binary_set:
                self.kind.append("binary"); self.cont_log.append(False)
                cat_blocks.append((j, "value"))
            elif np.mean(xf == 0) > self.hurdle_frac:
                self.kind.append("hurdle"); self.cont_log.append(True)
                lv = _signed_log1p(x)
                self.hurdle_fill[j] = float(np.mean(lv[xf != 0])) if np.any(xf != 0) else 0.0
                cat_blocks.append((j, "indicator"))
                cont_blocks.append((j, "logval"))
            else:
                use_log = np.isfinite(skew(xf)) and abs(skew(xf)) > self.log_skew_threshold
                self.kind.append("cont"); self.cont_log.append(bool(use_log))
                cont_blocks.append((j, "value"))
        self.cont_blocks = cont_blocks
        self.cat_blocks = cat_blocks
        self.cat_cards = [2] * len(cat_blocks)
        # build the continuous model matrix and fit the quantile transform on it
        M = self._cont_matrix(Xr)
        self.qt_ = QuantileTransformer(output_distribution="normal",
                                       n_quantiles=min(400, M.shape[0]), random_state=0).fit(M)
        self.p_cont = M.shape[1]
        self.n_cat = len(cat_blocks)
        self.d_model = self.p_cont + self.n_cat
        return self

    def _cont_matrix(self, Xr):
        cols = []
        for (j, role) in self.cont_blocks:
            x = Xr[:, j].astype(float)
            if role == "logval":                       # hurdle positive-part
                v = _signed_log1p(x)
                v = np.where(np.isfinite(x) & (x != 0), v, self.hurdle_fill[j])
            else:                                       # plain continuous
                v = _signed_log1p(x) if self.cont_log[j] else x
            cols.append(v.reshape(-1, 1))
        return np.concatenate(cols, axis=1)

    def _cat_matrix(self, Xr):
        cols = []
        for (j, role) in self.cat_blocks:
            x = Xr[:, j].astype(float)
            code = (x != 0).astype(int) if role == "indicator" else np.rint(x).astype(int)
            cols.append(np.clip(code, 0, 1).reshape(-1, 1))
        return np.concatenate(cols, axis=1) if cols else np.zeros((len(Xr), 0), int)

    def encode(self, Xr: np.ndarray):
        """Xr (n, d_orig) -> model matrix (n, d_model) = [cont z | category indices]."""
        Xr = np.atleast_2d(np.asarray(Xr, float))
        cont_z = self.qt_.transform(self._cont_matrix(Xr))
        cat = self._cat_matrix(Xr).astype(float)
        return np.concatenate([cont_z, cat], axis=1)

    def decode(self, model: np.ndarray) -> np.ndarray:
        """model matrix (n, d_model) -> covariates (n, d_orig) in ORIGINAL units."""
        model = np.atleast_2d(np.asarray(model, float))
        cont = self.qt_.inverse_transform(model[:, :self.p_cont])
        cat = model[:, self.p_cont:]
        d = len(self.kind)
        out = np.zeros((model.shape[0], d))
        cont_of = {j: c for c, (j, _) in enumerate(self.cont_blocks)}
        cat_of = {j: k for k, (j, _) in enumerate(self.cat_blocks)}
        for j in range(d):
            if self.kind[j] == "binary":
                out[:, j] = np.rint(cat[:, cat_of[j]])
            elif self.kind[j] == "hurdle":
                val = _signed_expm1(cont[:, cont_of[j]])
                out[:, j] = np.where(cat[:, cat_of[j]] > 0.5, val, 0.0)
            else:
                v = cont[:, cont_of[j]]
                out[:, j] = _signed_expm1(v) if self.cont_log[j] else v
        return out

    def state(self) -> dict:
        return {
            "names": np.array(self.names), "kind": np.array(self.kind),
            "cont_log": np.array(self.cont_log, dtype=int),
            "cont_blocks": np.array([[j, 0 if r == "value" else 1] for j, r in self.cont_blocks]),
            "cat_blocks": np.array([[j, 0 if r == "value" else 1] for j, r in self.cat_blocks]),
            "hurdle_fill_keys": np.array(list(self.hurdle_fill.keys()), dtype=int),
            "hurdle_fill_vals": np.array(list(self.hurdle_fill.values()), dtype=float),
            "qt_quantiles": self.qt_.quantiles_, "qt_references": self.qt_.references_,
            "meta": np.array([self.hurdle_frac, self.log_skew_threshold, self.p_cont, self.n_cat]),
        }

    @classmethod
    def from_state(cls, st) -> "_CovariateEncoder":
        enc = cls(binary_set=set())
        enc.names = [str(x) for x in st["names"]]
        enc.kind = [str(x) for x in st["kind"]]
        enc.cont_log = [bool(x) for x in st["cont_log"]]
        enc.cont_blocks = [(int(j), "value" if r == 0 else "logval") for j, r in st["cont_blocks"]]
        enc.cat_blocks = [(int(j), "value" if r == 0 else "indicator") for j, r in st["cat_blocks"]]
        enc.hurdle_fill = {int(k): float(v) for k, v in zip(st["hurdle_fill_keys"], st["hurdle_fill_vals"])}
        enc.cat_cards = [2] * len(enc.cat_blocks)
        hf, lst, pc, nc = st["meta"]
        enc.hurdle_frac, enc.log_skew_threshold = float(hf), float(lst)
        enc.p_cont, enc.n_cat = int(pc), int(nc)
        enc.d_model = enc.p_cont + enc.n_cat
        qt = QuantileTransformer(output_distribution="normal")
        qt.quantiles_ = st["qt_quantiles"]; qt.references_ = st["qt_references"]
        qt.n_quantiles_ = qt.quantiles_.shape[0]; qt.n_features_in_ = qt.quantiles_.shape[1]
        enc.qt_ = qt
        return enc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class WGANConfig:
    covariates: tuple[str, ...] = tuple(COVARIATES_2D)
    s_col: str = S_COL                    # scalar short-run welfare target the rule 1{tau_S>=0} acts on
    y_col: str = Y_COL
    # CR's full 21-dim short-term outcome vector.  When `full_surrogate` is True the
    # generator models S | X, W as this whole vector (so Y | S(full), X, W conditions
    # on ALL surrogates -- a much better long-run forecast than one surrogate); the
    # harm-share threshold still uses only the scalar `s_col` (kept raw as column 0).
    s_cols: tuple[str, ...] = tuple(S_COLS_FULL)
    full_surrogate: bool = True
    e: float = 0.5                        # randomization probability P(W=1)
    n_quantiles: int = 400
    # covariates with |skewness| above this get a signed-log1p transform before
    # the quantile transform (tames heavy monetary/asset tails); see helpers above
    log_skew_threshold: float = 2.0
    # covariates with a zero-fraction above this get a HURDLE representation (a
    # softmax nonzero-indicator + a continuous positive-part) instead of being
    # forced through a single continuous head (ds-wgan-faithful for spike-and-slab)
    hurdle_frac: float = 0.40
    # Truncation box for draw_X.  The graduation covariates map (via the Gaussian
    # quantile transform) to a spiky z-space with heavy clipping tails at +/-5.2
    # (asset_index has many ties); GAN1 reproduces that range, so we set the box
    # WIDE (5.5) to keep the sampled X faithful to the real covariate law -- a
    # tight box would drop ~16% of the mass and distort both the X spread and the
    # average CATE (the tails carry a large part of the treatment effect).
    support_bound: float = 5.5
    # per-GAN training specs (epochs are the main knob; see module docstring).
    # gp_factor 20, batch 256, lr 1e-4 = CR Table; critic_steps 15, Adam default
    # betas, ReLU, one-sided GP, output clamp = ds-wgan defaults (wgan_backend).
    # S/Y trained long: the WGAN learns the CATE *heterogeneity* only slowly (the
    # short-run threshold barely binds at few epochs), so more steps are needed
    # for a non-degenerate two-boundary geometry -- verified in train_wgan.py.
    spec_X: WGANSpec = field(default_factory=lambda: WGANSpec(
        epochs=5000, critic_steps=15, gp_factor=20.0,
        critic_dropout=0.0, generator_dropout=0.1, seed=1))
    spec_S: WGANSpec = field(default_factory=lambda: WGANSpec(
        epochs=6000, critic_steps=15, gp_factor=20.0,
        critic_dropout=0.1, generator_dropout=0.1, seed=2))
    spec_Y: WGANSpec = field(default_factory=lambda: WGANSpec(
        epochs=5000, critic_steps=15, gp_factor=20.0,
        critic_dropout=0.1, generator_dropout=0.1, seed=3))
    # truth precompute (GPU-accelerated when available)
    n_pop: int = 200_000                  # population size (theta MC error ~ 1/sqrt)
    tau_M: int = 500                      # noise draws per point for the CATE means (CRN)
    pop_chunk: int = 8000                 # eval-batch for CRN (memory: chunk*tau_M rows)
    seed: int = 20260713


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
class WGANOracle:
    """Trained 3-GAN cascade exposing the same interface as `HarmShareOracle`.

    Exposes: draw_X, tau_S, tau_Y, density, sample_experiment, true_cates, plus
    a precomputed population (`X_pop`, `tauS_pop`, `tauY_pop`) from which the
    exact WGAN truth is read.
    """

    def __init__(self, cfg: WGANConfig | None = None):
        self.cfg = cfg or WGANConfig()
        self.d = len(self.cfg.covariates)
        self.genX: NumpyGenerator | None = None
        self.genS: NumpyGenerator | None = None      # S(full) | X, W  (W a context input)
        self.genY: NumpyGenerator | None = None      # Y | S(full), X, W
        self.encS: _CovariateEncoder | None = None   # encoder for the non-threshold surrogates
        self.s_cols = [self.cfg.s_col]
        self.s_cat_cards: list = []
        self.p_cont_S = 1                            # threshold surrogate only, until fit
        self.n_groups_S = 0

    # ------------------------------ training ----------------------------- #
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "WGANOracle":
        cfg = self.cfg
        W = df[TREATMENT].to_numpy().astype(int)
        Y = df[cfg.y_col].to_numpy(float)
        Xr = df[list(cfg.covariates)].to_numpy(float)
        Xr = np.where(np.isfinite(Xr), Xr, np.nanmean(Xr, axis=0))   # mean-impute, as CR

        # short-term outcome: CR's full 21-vector S | X, W (or just the scalar
        # threshold surrogate when full_surrogate is off).  The THRESHOLD surrogate
        # (cfg.s_col, total consumption) is column 0 and kept in RAW units, so the
        # policy rule 1{tau_S>=0} and tau_S read straight off it with no back-transform;
        # the OTHER surrogates are encoded (continuous / binary-softmax / hurdle) purely
        # as conditioning richness for the long-run forecast Y | S(full), X, W.
        s_cols = list(cfg.s_cols) if cfg.full_surrogate else [cfg.s_col]
        other_cols = [c for c in s_cols if c != cfg.s_col]
        self.s_cols = [cfg.s_col] + other_cols
        S_thr = df[cfg.s_col].to_numpy(float)
        S_other = (df[other_cols].to_numpy(float).reshape(len(df), -1)
                   if other_cols else np.zeros((len(df), 0)))
        if other_cols:
            S_other = np.where(np.isfinite(S_other), S_other, np.nanmean(S_other, axis=0))
        keep = np.isfinite(S_thr) & np.isfinite(Y)
        Xr, W, Y, S_thr, S_other = Xr[keep], W[keep], Y[keep], S_thr[keep], S_other[keep]

        # ds-wgan-faithful covariate encoding: continuous / binary-softmax / hurdle
        self.enc = _CovariateEncoder(BINARY_COVARIATES, hurdle_frac=cfg.hurdle_frac,
                                     log_skew_threshold=cfg.log_skew_threshold).fit(Xr, list(cfg.covariates))
        self.d = self.enc.d_model
        Xmodel = self.enc.encode(Xr)                          # (n, d_model) = [cont z | cat idx]
        Xcont = Xmodel[:, :self.enc.p_cont]
        Xcat = np.rint(Xmodel[:, self.enc.p_cont:]).astype(int)

        # encode the OTHER surrogates (threshold surrogate stays raw)
        if other_cols:
            self.encS = _CovariateEncoder(BINARY_S, hurdle_frac=cfg.hurdle_frac,
                                          log_skew_threshold=cfg.log_skew_threshold).fit(S_other, other_cols)
            Somodel = self.encS.encode(S_other)              # [other cont z | other cat idx]
            So_cont = Somodel[:, :self.encS.p_cont]
            So_cat = np.rint(Somodel[:, self.encS.p_cont:]).astype(int)
            self.s_cat_cards = list(self.encS.cat_cards)
        else:
            self.encS = None
            So_cont = np.zeros((len(S_thr), 0))
            So_cat = np.zeros((len(S_thr), 0), int)
            self.s_cat_cards = []
        # GAN2 output = [ S_thr(raw) | other cont z ]  continuous  +  other categorical groups
        Scont = np.column_stack([S_thr.reshape(-1, 1), So_cont])
        self.p_cont_S = Scont.shape[1]
        self.n_groups_S = len(self.s_cat_cards)
        # the FULL S model matrix (what GAN3 conditions on) = [continuous | cat indices]
        Smodel_full = np.column_stack([Scont, So_cat.astype(float)]) if self.n_groups_S else Scont
        self.sd_S_ = float(S_thr.std())
        self.sd_Y_ = float(Y.std())
        self.hist = {}

        lg = (lambda sp: max(1, sp.epochs // 5)) if verbose else (lambda sp: 0)
        if verbose:
            print(f"    [GAN1] X  ({self.enc.p_cont} continuous + {self.enc.n_cat} categorical) ...")
        self.genX, self.hist["X"] = train_conditional_wgan(
            Xcont, None, cfg.spec_X, log_every=lg(cfg.spec_X),
            cat=Xcat, cat_cards=self.enc.cat_cards)

        # GAN2 = S(full) | X, W ; GAN3 = Y | S(full), X, W  (both condition on Xmodel + W)
        if verbose:
            print(f"    [GAN2] S | X, W  ({self.p_cont_S} continuous + {self.n_groups_S} categorical, "
                  f"n={len(S_thr)}) ...")
        ctxS = np.column_stack([Xmodel, W.astype(float)])
        self.genS, self.hist["S"] = train_conditional_wgan(
            Scont, ctxS, cfg.spec_S, log_every=lg(cfg.spec_S),
            cat=(So_cat if self.n_groups_S else None),
            cat_cards=(self.s_cat_cards if self.n_groups_S else None))
        if verbose:
            print(f"    [GAN3] Y | S({Smodel_full.shape[1]}-dim), X, W  (n={len(Y)}) ...")
        ctxY = np.column_stack([Smodel_full, Xmodel, W.astype(float)])
        self.genY, self.hist["Y"] = train_conditional_wgan(
            Y.reshape(-1, 1), ctxY, cfg.spec_Y, log_every=lg(cfg.spec_Y))

        # covariate density (for the d<=2 grid-geometry diagnostics only)
        if self.d <= 2:
            from scipy.stats import gaussian_kde
            rng = np.random.default_rng(cfg.seed)
            self._kde = gaussian_kde(self.genX.sample(None, 200_000, rng)[:, :self.d].T, bw_method=0.15)
        else:
            self._kde = None

        self._precompute_population(verbose=verbose)
        return self

    # --------------------------- CATE via CRN ---------------------------- #
    def _cate(self, Xeval: np.ndarray, M: int, rng: np.random.Generator):
        """(τ_S, τ_Y) at Xeval by conditional MC over generator noise (CRN).

        Common random numbers: the SAME noise U (for S) and V (for Y) are used
        for the W=1 and W=0 arms, so the differences S(1)-S(0), Y(1)-Y(0) have
        far lower Monte-Carlo variance than independent draws.  Runs on the GPU
        when available (a big speedup for the population truth precompute); the
        NumPy path is the portable fallback and gives identical results.
        """
        gpu = self._ensure_gpu()
        Xeval = np.atleast_2d(np.asarray(Xeval, float))
        Nc = Xeval.shape[0]
        nzS = self.genS.noise_dim
        nzY = self.genY.noise_dim
        ng = self.n_groups_S
        tauS = np.empty(Nc)
        tauY = np.empty(Nc)
        chunk = max(1, min(self.cfg.pop_chunk, int(4_000_000 // max(M, 1))))
        for a in range(0, Nc, chunk):
            Xc = Xeval[a:a + chunk]
            m = Xc.shape[0]
            Xt = np.repeat(Xc, M, axis=0)                       # (m*M, d)
            U = rng.standard_normal((m * M, nzS))               # S noise, shared across arms (CRN)
            V = rng.standard_normal((m * M, nzY))               # Y noise, shared across arms (CRN)
            catU = rng.random((m * M, ng)) if ng else None      # S categorical uniforms, shared (CRN)
            if gpu:
                dS, dY = self._cate_chunk_gpu(Xt, U, V, catU, gpu)
            else:
                one, zero = np.ones((m * M, 1)), np.zeros((m * M, 1))
                # full S model row [threshold(raw) | other cont z | other cat idx]
                S1 = self.genS.generate_full(np.concatenate([Xt, one], axis=1), U, catU)
                S0 = self.genS.generate_full(np.concatenate([Xt, zero], axis=1), U, catU)
                Y1 = self.genY.generate(np.concatenate([S1, Xt, one], axis=1), V)
                Y0 = self.genY.generate(np.concatenate([S0, Xt, zero], axis=1), V)
                dS = (S1[:, 0] - S0[:, 0])                       # tau_S off raw threshold column
                dY = (Y1 - Y0).ravel()
            tauS[a:a + chunk] = dS.reshape(m, M).mean(axis=1)
            tauY[a:a + chunk] = dY.reshape(m, M).mean(axis=1)
        return tauS, tauY

    # ------------------------ GPU-accelerated forward -------------------- #
    def _ensure_gpu(self):
        """Cache torch-GPU copies of the (exported) generator weights, or False."""
        if getattr(self, "_gpu", None) is not None:
            return self._gpu
        try:
            import torch
            if not torch.cuda.is_available():
                self._gpu = False
                return False
        except Exception:
            self._gpu = False
            return False
        self._torch = torch
        self._dev = torch.device("cuda")

        def pack(ng):
            t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self._dev)
            d = {"W": [(t(W), t(b)) for (W, b) in ng.weights],
                 "bounds": t(ng.bounds), "p_cont": int(ng.p_cont),
                 "cat_cards": tuple(int(c) for c in ng.cat_cards),
                 "out_mean": t(ng.out.mean), "out_scale": t(ng.out.scale),
                 "ctx_mean": t(ng.ctx.mean) if ng.ctx is not None else None,
                 "ctx_scale": t(ng.ctx.scale) if ng.ctx is not None else None}
            return d
        self._gpu = {"S": pack(self.genS), "Y": pack(self.genY)}
        return self._gpu

    def _graw(self, pk, ctx_t, noise_t):
        """Raw generator output (pre-clamp) on the GPU: MLP over (noise, std-context)."""
        torch = self._torch
        if pk["ctx_mean"] is not None:
            h = torch.cat([noise_t, (ctx_t - pk["ctx_mean"]) / pk["ctx_scale"]], dim=1)
        else:
            h = noise_t
        for (W, b) in pk["W"][:-1]:
            h = torch.relu(h @ W.t() + b)
        Wl, bl = pk["W"][-1]
        return h @ Wl.t() + bl

    def _gfwd(self, pk, ctx_t, noise_t):
        """Continuous-only generator forward (Y): clamp + de-standardize."""
        torch = self._torch
        o = self._graw(pk, ctx_t, noise_t)[:, :pk["p_cont"]]
        o = torch.min(torch.max(o, pk["bounds"][0:1]), pk["bounds"][1:2])
        return o * pk["out_scale"] + pk["out_mean"]

    def _gfwd_full(self, pk, ctx_t, noise_t, cat_u_t):
        """Full mixed generator forward (S): continuous block + categorical indices.

        Continuous block clamped/de-standardized; each softmax group sampled by
        inverse-CDF from the shared uniforms `cat_u_t` (same across arms -> CRN)."""
        torch = self._torch
        raw = self._graw(pk, ctx_t, noise_t)
        pc = pk["p_cont"]
        cont = torch.min(torch.max(raw[:, :pc], pk["bounds"][0:1]), pk["bounds"][1:2])
        cont = cont * pk["out_scale"] + pk["out_mean"]
        cards = pk["cat_cards"]
        if not cards:
            return cont
        cols, off = [cont], pc
        for g, k in enumerate(cards):
            p = torch.softmax(raw[:, off:off + k], dim=1); off += k
            idx = (cat_u_t[:, g:g + 1] > torch.cumsum(p, dim=1)).sum(dim=1, keepdim=True).to(cont.dtype)
            cols.append(idx)
        return torch.cat(cols, dim=1)

    def _cate_chunk_gpu(self, Xt, U, V, catU, gpu):
        torch = self._torch
        with torch.no_grad():
            Xg = torch.tensor(Xt, dtype=torch.float32, device=self._dev)
            Ug = torch.tensor(U, dtype=torch.float32, device=self._dev)
            Vg = torch.tensor(V, dtype=torch.float32, device=self._dev)
            catUg = (torch.tensor(catU, dtype=torch.float32, device=self._dev)
                     if catU is not None else None)
            one = torch.ones(Xg.shape[0], 1, device=self._dev)
            zero = torch.zeros(Xg.shape[0], 1, device=self._dev)
            S1 = self._gfwd_full(gpu["S"], torch.cat([Xg, one], 1), Ug, catUg)
            S0 = self._gfwd_full(gpu["S"], torch.cat([Xg, zero], 1), Ug, catUg)
            Y1 = self._gfwd(gpu["Y"], torch.cat([S1, Xg, one], 1), Vg)
            Y0 = self._gfwd(gpu["Y"], torch.cat([S0, Xg, zero], 1), Vg)
            dS = (S1[:, 0:1] - S0[:, 0:1]).ravel().cpu().numpy()   # tau_S off raw threshold col
            dY = (Y1 - Y0).ravel().cpu().numpy()
            return dS, dY

    def tau_S(self, xz):
        rng = np.random.default_rng(self.cfg.seed + 101)
        return self._cate(xz, self.cfg.tau_M, rng)[0]

    def tau_Y(self, xz):
        rng = np.random.default_rng(self.cfg.seed + 101)
        return self._cate(xz, self.cfg.tau_M, rng)[1]

    def density(self, xz):
        return self._kde(np.atleast_2d(np.asarray(xz, float)).T)

    def raw_covariates(self, Xmodel: np.ndarray) -> np.ndarray:
        """Map generated model-covariates back to the 20 covariates in ORIGINAL
        units (inverse quantile transform / signed-log; hurdle + binary decode)."""
        return self.enc.decode(np.atleast_2d(np.asarray(Xmodel, float)))

    def raw_surrogates(self, Smodel: np.ndarray) -> np.ndarray:
        """Map a generated S model row [threshold(raw) | other cont z | other cat idx]
        back to the full short-term vector (order = self.s_cols) in ORIGINAL units.
        The threshold surrogate is column 0 (already raw); the rest are decoded."""
        Smodel = np.atleast_2d(np.asarray(Smodel, float))
        thr = Smodel[:, 0:1]
        if self.encS is None:
            return thr
        other_model = np.column_stack([Smodel[:, 1:self.p_cont_S],           # other cont z
                                       Smodel[:, self.p_cont_S:]])           # other cat idx
        other_raw = self.encS.decode(other_model)
        return np.column_stack([thr, other_raw])

    # ------------------------------ sampling ----------------------------- #
    def draw_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        b = self.cfg.support_bound
        pc = self.enc.p_cont                       # truncate only the continuous block
        out = np.empty((0, self.d))
        while len(out) < n:
            Xm = self.genX.sample(None, 2 * n, rng)
            Xm = Xm[np.all(np.abs(Xm[:, :pc]) <= b, axis=1)]
            out = np.vstack([out, Xm])
        return out[:n]

    def sample_experiment(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw a finite experiment from the full trained cascade.

        X ~ GAN1 (truncated to the box), W ~ Bernoulli(e), S ~ GAN2(X,W),
        Y ~ GAN3(S,X,W).  Only the realized potential outcome for the drawn W is
        returned, exactly as in a real experiment.  All conditional *shapes* are
        the GAN's own (heteroskedastic / skewed), which is what makes this a
        harder, more realistic estimation target than the primary oracle.
        """
        cfg = self.cfg
        Xz = self.draw_X(n, rng)
        W = (rng.random(n) < cfg.e).astype(int)
        ctxS = np.column_stack([Xz, W.astype(float)])
        # genS.sample returns the full S model row [threshold(raw) | other z | other cat idx];
        # column 0 is the scalar short-run outcome the harm-share rule acts on.
        Smodel = self.genS.sample(ctxS, n, rng)
        Smodel = np.atleast_2d(Smodel)
        S = Smodel[:, 0]
        ctxY = np.column_stack([Smodel, Xz, W.astype(float)])
        Y = self.genY.sample(ctxY, n, rng).ravel()
        cols = {f"X{j+1}": Xz[:, j] for j in range(self.d)}
        cols.update({"W": W, "S": S, "Y": Y})
        return pd.DataFrame(cols)

    def true_cates(self, df: pd.DataFrame):
        Xz = df[[f"X{j+1}" for j in range(self.d)]].to_numpy(float)
        rng = np.random.default_rng(self.cfg.seed + 101)
        return self._cate(Xz, self.cfg.tau_M, rng)

    # ------------------------- population truth --------------------------- #
    def _precompute_population(self, verbose: bool = True):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + 7)
        if verbose:
            print(f"    [truth] population {cfg.n_pop:,} draws x M={cfg.tau_M} CRN noise ...")
        self.X_pop = self.draw_X(cfg.n_pop, rng)
        self.tauS_pop, self.tauY_pop = self._cate(self.X_pop, cfg.tau_M, rng)

    def truth(self) -> dict:
        """Exact WGAN population truth read off the precomputed population."""
        tS, tY = self.tauS_pop, self.tauY_pop
        pp = float(np.mean((tS >= 0) & (tY >= 0)))
        pm = float(np.mean((tS >= 0) & (tY < 0)))
        mp = float(np.mean((tS < 0) & (tY >= 0)))
        mm = float(np.mean((tS < 0) & (tY < 0)))
        return {
            "theta_harm": pm, "theta_pp": pp, "theta_mp": mp, "theta_mm": mm,
            "rho": pm / max(pp + pm, 1e-12),
            "treat_share_S": pp + pm,
            "ate_S": float(tS.mean()), "ate_Y": float(tY.mean()),
            "W_Y": float(np.mean(np.maximum(tY, 0.0))),   # regular companion truth
            "n_draw": int(self.cfg.n_pop),
            "tau_M": int(self.cfg.tau_M),
        }

    # ---------------------------- persistence ---------------------------- #
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st = {}
        st.update(self.genX.to_state("X"))
        st.update(self.genS.to_state("S"))
        st.update(self.genY.to_state("Y"))
        for k, v in self.enc.state().items():           # covariate encoder
            st[f"enc/{k}"] = v
        st["has_encS"] = np.array(1 if self.encS is not None else 0)
        if self.encS is not None:                       # surrogate encoder
            for k, v in self.encS.state().items():
                st[f"encS/{k}"] = v
        st["meta"] = np.array([self.d, self.cfg.n_pop, self.cfg.tau_M,
                               self.cfg.support_bound, self.cfg.e], dtype=float)
        st["s_meta"] = np.array([self.p_cont_S, self.n_groups_S], dtype=int)
        st["s_cols"] = np.array(list(self.s_cols))
        st["X_pop"] = self.X_pop
        st["tauS_pop"] = self.tauS_pop
        st["tauY_pop"] = self.tauY_pop
        st["covariates"] = np.array(list(self.cfg.covariates))
        np.savez_compressed(path, **st)

    @classmethod
    def load(cls, path: str | Path, df: pd.DataFrame | None = None) -> "WGANOracle":
        st = np.load(path, allow_pickle=False)
        covariates = tuple(str(c) for c in st["covariates"])
        cfg = WGANConfig(covariates=covariates)
        d, n_pop, tau_M, sb, e = st["meta"]
        cfg.n_pop, cfg.tau_M, cfg.support_bound, cfg.e = int(n_pop), int(tau_M), float(sb), float(e)
        orc = cls(cfg)
        orc.genX = NumpyGenerator.from_state("X", st)
        orc.genS = NumpyGenerator.from_state("S", st)
        orc.genY = NumpyGenerator.from_state("Y", st)
        enc_state = {k[len("enc/"):]: st[k] for k in st.files if k.startswith("enc/")}
        orc.enc = _CovariateEncoder.from_state(enc_state)
        orc.d = orc.enc.d_model
        # surrogate encoder + full-S geometry
        if "has_encS" in st.files and int(st["has_encS"]):
            encS_state = {k[len("encS/"):]: st[k] for k in st.files if k.startswith("encS/")}
            orc.encS = _CovariateEncoder.from_state(encS_state)
        else:
            orc.encS = None
        if "s_meta" in st.files:
            orc.p_cont_S, orc.n_groups_S = (int(x) for x in st["s_meta"])
        else:                                            # legacy scalar-S cache
            orc.p_cont_S, orc.n_groups_S = 1, 0
        orc.s_cols = [str(c) for c in st["s_cols"]] if "s_cols" in st.files else [cfg.s_col]
        orc.s_cat_cards = list(orc.encS.cat_cards) if orc.encS is not None else []
        orc.X_pop = st["X_pop"]
        orc.tauS_pop = st["tauS_pop"]
        orc.tauY_pop = st["tauY_pop"]
        orc._kde = None
        return orc


def build_wgan_oracle(covariates=COVARIATES_2D, rda_path=None, verbose=True,
                      **cfg_kw) -> WGANOracle:
    df = load_graduation(rda_path)
    cfg = WGANConfig(covariates=tuple(covariates), **cfg_kw)
    return WGANOracle(cfg).fit(df, verbose=verbose)
