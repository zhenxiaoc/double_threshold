from .base import VarianceEstimator
from .ccg_sieve_var import CCGSieveVariance
from .sieve_var import SieveVariance
from opttreat.config import VarianceConfig

__all__ = [
    "VarianceEstimator",
    "CCGSieveVariance",
    "SieveVariance",
    "get_variance_estimator",
]


def get_variance_estimator(cfg: VarianceConfig | None) -> VarianceEstimator | None:
    """
    Factory that turns a VarianceConfig into a concrete VarianceEstimator.

    If cfg is None, returns None (no variance estimator).
    """
    if cfg is None:
        return None

    method = cfg.method.lower()

    if method in ("sieve", "sieve_var"):
        # SieveVariance.__init__ only takes `options`
        return SieveVariance(options=cfg.options)

    if method in ("ccg_sieve", "ccg_sieve_var", "ccg2025_sieve_var"):
        return CCGSieveVariance(options=cfg.options)

    raise ValueError(f"Unknown variance method: {cfg.method!r}")
