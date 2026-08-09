"""Fixed-policy IPW and sequential AIPW benchmarks for the (1,1) contribution
(task section 9).

These EVALUATE the learned fixed policy (g1, g2) on held-out data.  That is a
different object from the population optimal-path component ``V_11^*``: the AIPW
interval is a useful benchmark but does not by itself solve moving-boundary inference
for ``V_11^*`` (see ``docs/aipw_derivation.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class BenchmarkResult:
    method: str
    estimate: float
    se: float
    ci: tuple[float, float]
    n_eff: int
    note: str = ""


def _g1(nuis, s):
    return (nuis.kappa(s) >= 0).astype(float)


def _g2(nuis, x):
    return (nuis.delta(x) >= 0).astype(float)


def ipw_11(est, *, e1: float | None = None, e2: float | None = None) -> BenchmarkResult:
    """Cross-fitted IPW of the (1,1) contribution of the learned policy:

        mean[ 1{T1=1} 1{T2=1} g1(S) g2(X) Y / (e1 e2) ].
    """
    df = est.data_
    e1 = e1 if e1 is not None else est.e1_
    e2 = e2 if e2 is not None else est.e2_
    n = len(df)
    scores = np.zeros(n)
    S = df["S"].to_numpy(); X = df["X"].to_numpy(); Y = df["Y"].to_numpy()
    T1 = df["T1"].to_numpy(); T2 = df["T2"].to_numpy()
    for test_idx, _train_idx, nuis in est.fold_nuis_:
        s, x = S[test_idx], X[test_idx]
        g1, g2 = _g1(nuis, s), _g2(nuis, x)
        y = np.nan_to_num(Y[test_idx], nan=0.0)
        obs = (T1[test_idx] == 1) & (T2[test_idx] == 1)
        scores[test_idx] = obs.astype(float) * g1 * g2 * y / (e1 * e2)
    est_val = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n))
    z = norm.ppf(0.975)
    return BenchmarkResult(
        "IPW(1,1) fixed-policy", est_val, se, (est_val - z * se, est_val + z * se),
        n_eff=int(np.sum(scores != 0)),
        note="evaluates the learned fixed policy; uses known design probabilities",
    )


def aipw_11(est, *, e1: float | None = None, e2: float | None = None) -> BenchmarkResult:
    """Cross-fitted sequential AIPW of the (1,1) contribution of the learned policy.

    score = g1(S) M1(S)
          + g1(S) 1{T1=1}/e1 (g2(X) Q2 - M1(S))
          + g1(S) 1{T1=1}/e1 g2(X) 1{T2=1}/e2 (Y - Q2),
    with Q2(H2)=mu1(X) (Markov) and M1(S)=E[g2(X) mu1(X)|S,T1=1]=G11(S).
    """
    df = est.data_
    e1 = e1 if e1 is not None else est.e1_
    e2 = e2 if e2 is not None else est.e2_
    n = len(df)
    scores = np.zeros(n)
    S = df["S"].to_numpy(); X = df["X"].to_numpy(); Y = df["Y"].to_numpy()
    T1 = df["T1"].to_numpy(); T2 = df["T2"].to_numpy()
    for test_idx, _train_idx, nuis in est.fold_nuis_:
        s, x = S[test_idx], X[test_idx]
        g1, g2 = _g1(nuis, s), _g2(nuis, x)
        q2 = nuis.mu1.predict(x)                 # Q2(H2)=mu1(X) under Markov
        m1 = nuis.G11.predict(s)                 # M1(S)=E[g2 mu1|S,T1=1]=G11(S)
        t1 = (T1[test_idx] == 1).astype(float)
        t2 = (T2[test_idx] == 1).astype(float)
        y = np.nan_to_num(Y[test_idx], nan=0.0)
        obs_y = (~np.isnan(Y[test_idx])).astype(float)
        term_a = g1 * m1
        term_b = g1 * t1 / e1 * (g2 * q2 - m1)
        term_c = g1 * t1 / e1 * g2 * t2 / e2 * obs_y * (y - q2)
        scores[test_idx] = term_a + term_b + term_c
    est_val = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n))
    z = norm.ppf(0.975)
    return BenchmarkResult(
        "AIPW(1,1) fixed-policy", est_val, se, (est_val - z * se, est_val + z * se),
        n_eff=n,
        note="doubly-robust evaluation of the learned fixed policy; NOT moving-boundary "
             "inference for V_11^*",
    )


def augmentation_mean(est, *, e1=None, e2=None) -> dict[str, float]:
    """Return the sample means of the two AIPW augmentation terms.

    Under a correct DGP with known probabilities these have mean ~0 (Neyman
    orthogonality); ``tests/test_aipw_score.py`` checks this.
    """
    df = est.data_
    e1 = e1 if e1 is not None else est.e1_
    e2 = e2 if e2 is not None else est.e2_
    S = df["S"].to_numpy(); X = df["X"].to_numpy(); Y = df["Y"].to_numpy()
    T1 = df["T1"].to_numpy(); T2 = df["T2"].to_numpy()
    b_terms = []
    c_terms = []
    for test_idx, _train_idx, nuis in est.fold_nuis_:
        s, x = S[test_idx], X[test_idx]
        g1, g2 = _g1(nuis, s), _g2(nuis, x)
        q2 = nuis.mu1.predict(x); m1 = nuis.G11.predict(s)
        t1 = (T1[test_idx] == 1).astype(float); t2 = (T2[test_idx] == 1).astype(float)
        y = np.nan_to_num(Y[test_idx], nan=0.0)
        obs_y = (~np.isnan(Y[test_idx])).astype(float)
        b_terms.append(g1 * t1 / e1 * (g2 * q2 - m1))
        c_terms.append(g1 * t1 / e1 * g2 * t2 / e2 * obs_y * (y - q2))
    return {
        "aug_b_mean": float(np.mean(np.concatenate(b_terms))),
        "aug_c_mean": float(np.mean(np.concatenate(c_terms))),
    }
