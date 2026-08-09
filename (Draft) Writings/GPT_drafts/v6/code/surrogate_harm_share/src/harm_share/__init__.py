"""harm_share: calibrated simulation of the surrogate-induced harm share,
a double-threshold value functional theta = Pr(tau_S(X) >= 0, tau_Y(X) <= 0).

Companion to Zhenxiao Chen's JMP on inference for policy value under
admissibility-constrained optimal treatment rules.
"""
from .calibration import build_oracle, HarmShareOracle, OracleConfig, load_graduation
from .functionals import mc_truth, grid_truth, analytic_derivative, fd_derivative, TruthReport
from .estimator import estimate_harm_share, bootstrap_ci, regular_companion_welfare
from .simulation import run_mc, rate_experiment, bootstrap_coverage
from .affine_dgp import AffineDGP
from .wgan_calibration import build_wgan_oracle, WGANOracle, WGANConfig

__all__ = [
    "build_oracle", "HarmShareOracle", "OracleConfig", "load_graduation",
    "mc_truth", "grid_truth", "analytic_derivative", "fd_derivative", "TruthReport",
    "estimate_harm_share", "bootstrap_ci", "regular_companion_welfare",
    "run_mc", "rate_experiment", "bootstrap_coverage", "AffineDGP",
    "build_wgan_oracle", "WGANOracle", "WGANConfig",
]
