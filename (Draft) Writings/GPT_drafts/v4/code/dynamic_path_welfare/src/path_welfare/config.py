"""Configuration schema (pydantic) and YAML loading with provenance hashing.

Every run records the config hash, package versions and seeds so that results are
reproducible and auditable (task sections 18-19).
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SplineConfig(BaseModel):
    """Cubic B-spline sieve settings for the stage-two regressions."""

    degree: int = 3
    candidate_dims: list[int] = Field(default_factory=lambda: [4, 5, 6, 8, 10, 12])
    ridge: float = 0.0  # 0 => plain least squares
    inner_cv_folds: int = 5
    condition_number_max: float = 1e10

    @field_validator("candidate_dims")
    @classmethod
    def _sorted_unique(cls, v: list[int]) -> list[int]:
        v = sorted(set(int(x) for x in v))
        if any(d < 4 for d in v):
            raise ValueError("cubic B-spline dimension must be >= 4")
        return v


class CrossFitConfig(BaseModel):
    n_outer_folds: int = 5
    n_inner_folds: int = 5
    seed: int = 20260713
    group_col: str | None = None  # keep a participant's rows in one fold


class DensityConfig(BaseModel):
    """Conditional-density cross-check settings (section 8.3)."""

    method: Literal["gaussian_spline", "kernel"] = "gaussian_spline"
    mean_dim: int = 6
    logsd_dim: int = 4
    kde_bandwidth: Literal["scott", "silverman"] | float = "scott"
    trim_quantile: float = 0.01  # denominator trimming for stability
    n_quad_x: int = 200
    n_quad_s: int = 200


class InferenceConfig(BaseModel):
    sieve_dims: list[int] = Field(default_factory=lambda: [5, 6, 8, 10])
    primary_dim: int = 8  # chosen BEFORE seeing the final point estimate (from simulation)
    fd_step_sizes: list[float] = Field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    n_quad_x: int = 400
    n_quad_s: int = 400
    trim_prob: float = 0.10  # minimum admissible randomization probability


class BootstrapConfig(BaseModel):
    n_boot_dev: int = 99
    n_boot_final: int = 499
    n_subsample_m: list[int] = Field(default_factory=lambda: [])  # filled from n if empty
    seed: int = 909090
    n_jobs: int = 1


class CostConfig(BaseModel):
    grid_sd_units: list[float] = Field(default_factory=lambda: [0.0, 0.025, 0.05, 0.10])


class VariableMap(BaseModel):
    """Maps dataset columns to the canonical O = (S,T1,X,T2,Y)."""

    S: str = "S"
    T1: str = "T1"
    X: str = "X"
    T2: str = "T2"
    Y: str = "Y"
    group: str | None = None
    site: str | None = None
    weight: str | None = None


class TreatmentProbs(BaseModel):
    """Known / design randomization probabilities. None => must be reconstructed."""

    e1: float | None = None  # P(T1=1 | S)  (constant if known by design)
    e2: float | None = None  # P(T2=1 | history)


class TransformConfig(BaseModel):
    S: Literal["identity", "log", "log1p", "zscore", "rank"] = "identity"
    X: Literal["identity", "log", "log1p", "zscore", "rank"] = "identity"


class Config(BaseModel):
    """Top-level run configuration."""

    name: str = "unnamed"
    dataset: str = "simulation"
    outcome_units: str = "outcome units"
    seed: int = 20260713
    sim_dgp: str = "dgp1"   # which named DGP when dataset is a simulation
    sim_n: int = 2000       # n when dataset is a simulation
    variables: VariableMap = Field(default_factory=VariableMap)
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    treatment_probs: TreatmentProbs = Field(default_factory=TreatmentProbs)
    spline: SplineConfig = Field(default_factory=SplineConfig)
    crossfit: CrossFitConfig = Field(default_factory=CrossFitConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    availability_col: str | None = None  # second-stage availability indicator (SMARTs)
    results_dir: str = "results"

    # ------------------------------------------------------------------ #
    def hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def provenance(self) -> dict[str, Any]:
        import numpy
        import pandas
        import scipy
        import sklearn

        return {
            "config_hash": self.hash(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "seed": self.seed,
        }


def load_config(path: str | Path) -> Config:
    """Load a YAML config into a validated :class:`Config`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
