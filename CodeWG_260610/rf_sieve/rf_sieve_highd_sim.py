"""High-dimensional welfare/value simulations with a random-feature (shallow NN) linear sieve.

Exploration script for the Chen-Chen-Gao project (drafted in ClaudeWS).

Idea
----
A shallow neural network with *random* first-layer weights and trained second
layer is a linear series estimator: the features

    psi_k(x) = act( gamma * (w_k' x + b_k) ),   (w_k, b_k) iid on the unit sphere,

are fixed conditional on the random draw, and the second stage is OLS on
(psi_1, ..., psi_K).  This makes the paper's entire sieve inference machinery
applicable verbatim (stacked treated/control regression; sieve variance
Bun' Patty Bun with the indicator "bun" for the welfare functional W and the
eps-band "bun" for the value functional V), while scaling to covariate
dimensions d_x where tensor-product B-splines are infeasible.

This script is self-contained (numpy/scipy/pandas only) but mirrors the
conventions of `OptTreat/Python Codes/opttreat`:
  - features: iid (w,b) on the sphere S^{d_x} (cf. estimation/features/iid_sphere.py),
    activations cos / relu / tanh (cf. features/activations.py);
  - variance: block-diagonal "patty" (B'B)^- B' diag(e^2) B (B'B)^- and Sobol
    "bun" derivative vector (cf. variance/ccg_sieve_var.py), except the second
    stage here is OLS (pinv), not ridge, to match the linear-sieve theory;
  - outputs: summary/draws CSV + md report with the shared naming convention.

Estimands (target distribution F = Uniform[0,1]^{d_x}, known):
  W = E_F[ max(tau(X), 0) ]          (welfare, root-n regular)
  V = P_F( tau(X) >= 0 )             (value/share, irregular, sieve t-stat)

Optional: leave-one-out (LOO) debiased estimator of V per Theorem 5 of the
draft, with D^2 V computed by second-order central differences on Sobol points
and the closed-form OLS leave-one-out residual e_i / (1 - H_ii).

Run:  python rf_sieve_highd_sim.py        (smoke settings below)
For research runs, edit the CONFIG block (PAPER_* suggestions provided).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

# =====================================================================
# CONFIG (smoke-size defaults; PAPER_* values suggested for real runs)
# =====================================================================
DIMS = (3, 5)               # covariate dimensions d_x          | PAPER: (3, 5, 10, 20)
N_VALUES = (500, 1000)      # sample sizes                      | PAPER: (1500, 3000, 6000)
K_FEATURES = (25, 50)       # number of random features per arm | PAPER: (50, 100, 200, 400)
REPLICATIONS = 50           # Monte Carlo replications          | PAPER: 1000-2000
ACTIVATION = "cos"          # "cos", "relu", "tanh"
GAMMA = 3.0                 # feature scale: act(gamma * (w'x + b))
SHARE_FEATURES = True       # one feature draw shared by both arms
M_SOBOL = 4096              # Sobol points for functionals/derivatives | PAPER: 32768
M_SOBOL_TRUTH = 2**15       # Sobol points for the true W, V           | PAPER: 2**17
IOTA = 0.01                 # eps = IOTA * SD(h_hat at Sobol points) for the V-band
TAU_SCALE = 3.0             # CATE signal scale; with sd(eps)=1, TAU_SCALE=3 gives a
                            # signal-to-noise ratio at which the ReLU (Jensen) bias of
                            # W(h_hat) is modest at the smoke sample sizes; TAU_SCALE=1
                            # demonstrates the bias problem (see README)
HETEROSKEDASTIC = False     # True: sd(eps|D=1)=1.25, sd(eps|D=0)=0.75 (sieve var is robust)
RUN_LOO_DEBIAS = False      # LOO debiased V (Theorem 5); slower, O(M*n) per arm
LOO_DELTA0 = 0.05           # central-difference step scale for D^2 V
SEED = 2026
PROGRESS_EVERY = 25
MIN_ARM_SLACK = 10          # require n_arm >= K + 1 + MIN_ARM_SLACK, else skip draw
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
Z975 = 1.959963984540054


# =====================================================================
# DGP: ridge-function (Barron-type) CATE in d_x dimensions
# =====================================================================
# Single-index ("ridge") components are exactly the function class shallow
# networks approximate well, and they keep the level set {tau = 0} a smooth,
# curved (d_x - 1)-manifold with nonvanishing gradient.
@dataclass(frozen=True)
class HighDimDGP:
    dim: int

    def _indices(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        a1 = np.array([(-1.0) ** j for j in range(self.dim)]) / np.sqrt(self.dim)
        a2 = np.ones(self.dim) / np.sqrt(self.dim)
        m1 = 0.5 * a1.sum()
        m2 = 0.5 * a2.sum()
        return a1, a2, m1, m2

    @staticmethod
    def _sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def tau(self, X: np.ndarray) -> np.ndarray:
        """CATE: sum of two sigmoid ridge functions minus a constant (Barron-type)."""
        a1, a2, m1, m2 = self._indices()
        raw = self._sig(3.0 * (X @ a1 - m1)) + 0.5 * self._sig(3.0 * (X @ a2 - m2)) - 0.70
        return TAU_SCALE * raw

    def baseline(self, X: np.ndarray) -> np.ndarray:
        a1, a2, m1, m2 = self._indices()
        return 2.0 * np.tanh(2.0 * (X @ a2 - m2)) + np.sin(2.0 * np.pi * (X @ a1 - m1))

    def propensity(self, X: np.ndarray) -> np.ndarray:
        a1, a2, m1, m2 = self._indices()
        return self._sig(1.2 * (X @ a1 - m1) + 0.4 * (X @ a2 - m2))

    def generate(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        X = rng.uniform(size=(n, self.dim))
        D = (rng.uniform(size=n) < self.propensity(X)).astype(float)
        mu = self.baseline(X) + D * self.tau(X)
        if HETEROSKEDASTIC:
            sd = np.where(D == 1.0, 1.25, 0.75)
        else:
            sd = 1.0
        Y = mu + sd * rng.normal(size=n)
        return {"X": X, "D": D, "Y": Y}

    def truth(self, m_sobol: int) -> tuple[float, float]:
        Xs = Sobol(d=self.dim, scramble=False).random(m_sobol)
        t = self.tau(Xs)
        return float(np.maximum(t, 0.0).mean()), float((t >= 0.0).mean())


# =====================================================================
# Random-feature (shallow NN) linear sieve
# =====================================================================
def _activation(name: str):
    if name == "cos":
        return np.cos
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh
    raise ValueError(f"unsupported activation: {name!r}")


def draw_feature_map(dim: int, K: int, rng: np.random.Generator):
    """(w_k, b_k) iid uniform on the unit sphere S^{dim} (joint draw), plus intercept.

    Returns psi: X (m, dim) -> (m, K + 1) with first column = 1.
    """
    Z = rng.normal(size=(K, dim + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    W, b = Z[:, :dim], Z[:, dim]
    act = _activation(ACTIVATION)

    def psi(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feats = act(GAMMA * (X @ W.T + b[None, :]))
        return np.hstack([np.ones((X.shape[0], 1)), feats])

    return psi


@dataclass
class ArmFit:
    """OLS fit of one treatment arm on the random-feature design."""
    beta: np.ndarray          # (K+1,)
    resid: np.ndarray         # (n_arm,)
    Psi: np.ndarray           # (n_arm, K+1)
    MtM_pinv: np.ndarray      # pinv(Psi' Psi)

    @property
    def patty(self) -> np.ndarray:
        """(Psi'Psi)^- Psi' diag(e^2) Psi (Psi'Psi)^-  -- the OLS sandwich block."""
        A = self.MtM_pinv @ (self.Psi.T * self.resid[None, :])
        return A @ A.T


def fit_arm(psi, X: np.ndarray, Y: np.ndarray) -> ArmFit:
    Psi = psi(X)
    MtM_pinv = np.linalg.pinv(Psi.T @ Psi, rcond=1e-10)
    beta = MtM_pinv @ (Psi.T @ Y)
    resid = Y - Psi @ beta
    return ArmFit(beta=beta, resid=resid, Psi=Psi, MtM_pinv=MtM_pinv)


# =====================================================================
# Functionals, sieve variance, and (optional) LOO debiasing
# =====================================================================
def estimate_and_infer(
    fit_t: ArmFit,
    fit_c: ArmFit,
    psi,
    X_sobol: np.ndarray,
) -> dict[str, float]:
    """Plug-in W, V over Sobol points + sieve variances (Bun' Patty Bun)."""
    Psi_s = psi(X_sobol)                       # (M, K+1)
    h_hat = Psi_s @ (fit_t.beta - fit_c.beta)  # CATE at Sobol points
    M = X_sobol.shape[0]

    W_hat = float(np.maximum(h_hat, 0.0).mean())
    V_hat = float((h_hat >= 0.0).mean())

    P_t, P_c = fit_t.patty, fit_c.patty

    # ---- welfare bun: (1/M) sum 1{h >= 0} psi(x_j); var = b'P_t b + b'P_c b
    pos = h_hat >= 0.0
    bun_W = Psi_s[pos].sum(axis=0) / M if pos.any() else np.zeros(Psi_s.shape[1])
    var_W = float(bun_W @ P_t @ bun_W + bun_W @ P_c @ bun_W)

    # ---- value bun: eps-band derivative (1/(2 eps M)) sum 1{|h| < eps} psi(x_j)
    eps = IOTA * float(h_hat.std())
    band = np.abs(h_hat) < eps
    n_band = int(band.sum())
    if n_band > 0 and eps > 0:
        bun_V = Psi_s[band].sum(axis=0) / (2.0 * eps * M)
        var_V = float(bun_V @ P_t @ bun_V + bun_V @ P_c @ bun_V)
    else:
        var_V = np.nan

    return {
        "W_hat": W_hat, "W_se": float(np.sqrt(max(var_W, 0.0))),
        "V_hat": V_hat, "V_se": float(np.sqrt(max(var_V, 0.0))) if np.isfinite(var_V) else np.nan,
        "eps_band": eps, "n_band": n_band,
        "share_pos": float(pos.mean()),
    }


def loo_debias_V(
    fit_t: ArmFit,
    fit_c: ArmFit,
    psi,
    X_sobol: np.ndarray,
    V_plugin: float,
    n_total: int,
    chunk: int = 256,
) -> float:
    """LOO debiased value estimator (Theorem 5):

        V_loo = V(h_hat) - (1 / (2 n^2)) * sum_i D2V(h_hat)[s_i, s_i] * (e_i^{(-i)})^2,

    s_i(x) = n * psi(x)'(Psi_a'Psi_a)^- psi(X_i) with sign +/- for treated/control,
    e_i^{(-i)} = e_i / (1 - H_ii), and D2V via central differences on Sobol points.
    """
    Psi_s = psi(X_sobol)
    h_hat = Psi_s @ (fit_t.beta - fit_c.beta)
    delta = LOO_DELTA0 * float(h_hat.std())
    M = X_sobol.shape[0]
    total = 0.0

    for fit in (fit_t, fit_c):   # sign of s_i cancels in the quadratic form
        n_a = fit.Psi.shape[0]
        for start in range(0, n_a, chunk):
            stop = min(start + chunk, n_a)
            block = fit.Psi[start:stop]                            # (b, K+1)
            H_diag = np.einsum("ij,jk,ik->i", block, fit.MtM_pinv, block)
            H_diag = np.clip(H_diag, 0.0, 0.999)
            e_loo = fit.resid[start:stop] / (1.0 - H_diag)
            S = n_total * (Psi_s @ (fit.MtM_pinv @ block.T))       # (M, b): s_i at Sobol pts
            # normalize each direction to unit sd for a stable difference step
            scale = S.std(axis=0)
            scale[scale == 0.0] = 1.0
            U = S / scale[None, :]
            V_plus = (h_hat[:, None] + delta * U >= 0.0).mean(axis=0)
            V_minus = (h_hat[:, None] - delta * U >= 0.0).mean(axis=0)
            d2v = (V_plus - 2.0 * V_plugin + V_minus) / delta**2 * scale**2
            total += float((d2v * e_loo**2).sum())

    return V_plugin - total / (2.0 * n_total**2)


# =====================================================================
# Monte Carlo driver
# =====================================================================
def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    rows = []
    for dim in DIMS:
        dgp = HighDimDGP(dim)
        W_true, V_true = dgp.truth(M_SOBOL_TRUTH)
        X_sobol = Sobol(d=dim, scramble=False).random(M_SOBOL)
        print(f"[dim={dim}] W_true={W_true:.4f}  V_true={V_true:.4f}")

        for n in N_VALUES:
            for K in K_FEATURES:
                for rep in range(REPLICATIONS):
                    rng = np.random.default_rng(SEED + 100003 * rep + 7919 * dim + n + K)
                    data = dgp.generate(n, rng)
                    Dt = data["D"] == 1.0
                    n_t, n_c = int(Dt.sum()), int((~Dt).sum())
                    if min(n_t, n_c) < K + 1 + MIN_ARM_SLACK:
                        continue   # design too rich for this arm; skip draw

                    if SHARE_FEATURES:
                        psi = draw_feature_map(dim, K, rng)
                        psi_t = psi_c = psi
                    else:
                        psi_t = draw_feature_map(dim, K, rng)
                        psi_c = draw_feature_map(dim, K, rng)
                        psi = psi_t   # functionals evaluated with shared map only
                    if not SHARE_FEATURES:
                        raise NotImplementedError(
                            "Per-arm feature maps require evaluating h_hat with both maps; "
                            "keep SHARE_FEATURES=True for this exploration script."
                        )

                    fit_t = fit_arm(psi, data["X"][Dt], data["Y"][Dt])
                    fit_c = fit_arm(psi, data["X"][~Dt], data["Y"][~Dt])
                    res = estimate_and_infer(fit_t, fit_c, psi, X_sobol)

                    if RUN_LOO_DEBIAS:
                        res["V_loo"] = loo_debias_V(fit_t, fit_c, psi, X_sobol,
                                                    res["V_hat"], n_total=n)
                    else:
                        res["V_loo"] = np.nan

                    rows.append({
                        "dim": dim, "n": n, "K": K, "rep": rep,
                        "n_t": n_t, "n_c": n_c,
                        "W_true": W_true, "V_true": V_true, **res,
                    })
                    done = len(rows)
                    if done % PROGRESS_EVERY == 0:
                        print(f"  ... {done} draws done ({time.time() - t0:.0f}s)")

    draws = pd.DataFrame(rows)
    draws["W_dev"] = draws["W_hat"] - draws["W_true"]
    draws["V_dev"] = draws["V_hat"] - draws["V_true"]
    draws["n_varV"] = draws["n"] * draws["V_se"] ** 2
    draws["W_cover"] = (np.abs(draws["W_dev"]) <= Z975 * draws["W_se"]).astype(float)
    draws["V_cover"] = (np.abs(draws["V_dev"]) <= Z975 * draws["V_se"]).astype(float)
    if RUN_LOO_DEBIAS:
        draws["V_loo_dev"] = draws["V_loo"] - draws["V_true"]
        draws["V_loo_cover"] = (np.abs(draws["V_loo_dev"]) <= Z975 * draws["V_se"]).astype(float)

    g = draws.groupby(["dim", "n", "K"])
    summary = pd.DataFrame({
        "W_true": g["W_true"].first(),
        "W_bias": g["W_dev"].mean(),
        "W_sd": g["W_hat"].std(),
        "W_se_mean": g["W_se"].mean(),
        "W_cover": g["W_cover"].mean(),
        "V_true": g["V_true"].first(),
        "V_bias": g["V_dev"].mean(),
        "V_sd": g["V_hat"].std(),
        "V_se_mean": g["V_se"].mean(),
        "V_cover": g["V_cover"].mean(),
        # growth diagnostic: n * Var_hat(V) tracks sigma_{V,n}^2; its scaling in K
        # is the empirical analog of the K^{1/d_x} sieve-Riesz growth for splines
        "n_varV_mean": g["n_varV"].mean(),
        "eps_mean": g["eps_band"].mean(),
        "n_band_mean": g["n_band"].mean(),
        "draws": g.size(),
    }).reset_index()
    if RUN_LOO_DEBIAS:
        summary["V_loo_bias"] = g["V_loo_dev"].mean().values
        summary["V_loo_cover"] = g["V_loo_cover"].mean().values
    return summary, draws


def write_outputs(summary: pd.DataFrame, draws: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"n{'_'.join(map(str, N_VALUES))}_rep{REPLICATIONS}"
        f"_nf{'_'.join(map(str, K_FEATURES))}_d{'_'.join(map(str, DIMS))}_{ACTIVATION}"
    )
    s_path = OUTPUT_DIR / f"rf_sieve_highd_summary_{suffix}.csv"
    d_path = OUTPUT_DIR / f"rf_sieve_highd_draws_{suffix}.csv"
    r_path = OUTPUT_DIR / f"rf_sieve_highd_results_{suffix}.md"
    summary.to_csv(s_path, index=False)
    draws.to_csv(d_path, index=False)
    with open(r_path, "w", encoding="utf-8") as f:
        f.write("# RF-sieve high-dimensional simulation\n\n")
        f.write(f"- DGP: ridge-function CATE (HighDimDGP), dims = {DIMS}\n")
        f.write(f"- first stage: OLS on {ACTIVATION} random features, gamma = {GAMMA}, "
                f"shared across arms = {SHARE_FEATURES}\n")
        f.write(f"- K (features per arm) = {K_FEATURES}; n = {N_VALUES}; reps = {REPLICATIONS}\n")
        f.write(f"- Sobol points: functionals {M_SOBOL}, truth {M_SOBOL_TRUTH}; "
                f"iota = {IOTA}; heteroskedastic = {HETEROSKEDASTIC}; "
                f"LOO debias = {RUN_LOO_DEBIAS}\n\n")
        try:
            f.write(summary.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:   # tabulate not installed
            f.write("```\n" + summary.to_string(index=False) + "\n```")
        f.write("\n")
    print(f"\nWrote {s_path}\nWrote {d_path}\nWrote {r_path}")


if __name__ == "__main__":
    summary_df, draws_df = run()
    pd.set_option("display.width", 200)
    print()
    print(summary_df.to_string(index=False))
    write_outputs(summary_df, draws_df)
