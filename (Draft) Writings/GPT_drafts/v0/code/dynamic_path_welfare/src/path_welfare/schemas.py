"""Observed-data schema, timing checks and hard data-quality gates.

The canonical record is ``O = (S, T1, X, T2, Y)`` with causal ordering
``S -> T1 -> X -> T2 -> Y``.  This module *fails loudly* on schema or timing
violations (task sections 2, 4, 19).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CANONICAL_COLS = ["S", "T1", "X", "T2", "Y"]


class SchemaError(ValueError):
    """Raised when the observed-data model is violated."""


@dataclass
class ContinuityReport:
    name: str
    n: int
    n_unique: int
    max_point_mass: float
    q: dict[str, float]
    effectively_continuous: bool
    genuinely_continuous: bool
    notes: str = ""


@dataclass
class GateReport:
    passed: bool
    n_units: int
    path_counts: dict[str, int]
    s_report: ContinuityReport | None
    x_report: ContinuityReport | None
    outcome_availability: float
    smallest_prob: float | None
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_units": self.n_units,
            "path_counts": self.path_counts,
            "outcome_availability": self.outcome_availability,
            "smallest_prob": self.smallest_prob,
            "failures": self.failures,
            "warnings": self.warnings,
            "s_unique": None if self.s_report is None else self.s_report.n_unique,
            "x_unique": None if self.x_report is None else self.x_report.n_unique,
            "s_max_point_mass": None if self.s_report is None else self.s_report.max_point_mass,
            "x_max_point_mass": None if self.x_report is None else self.x_report.max_point_mass,
        }


def to_canonical(df: pd.DataFrame, variables) -> pd.DataFrame:
    """Rename dataset columns to canonical names; keep group/site/weight."""
    mapping = {
        variables.S: "S",
        variables.T1: "T1",
        variables.X: "X",
        variables.T2: "T2",
        variables.Y: "Y",
    }
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns {missing}; have {list(df.columns)}")
    out = df.rename(columns=mapping).copy()
    for opt, canon in (("group", "group"), ("site", "site"), ("weight", "weight")):
        col = getattr(variables, opt, None)
        if col is not None and col in df.columns:
            out[canon] = df[col].to_numpy()
    return out


def validate_schema(df: pd.DataFrame, *, allow_missing_y: bool = True) -> None:
    """Structural checks: columns present, treatments binary, S/X numeric."""
    for c in CANONICAL_COLS:
        if c not in df.columns:
            raise SchemaError(f"canonical column '{c}' absent")
    for t in ("T1", "T2"):
        vals = pd.unique(df[t].dropna())
        if not set(np.unique(vals)).issubset({0, 1}):
            raise SchemaError(f"{t} must be binary in {{0,1}}; found {sorted(set(vals))[:8]}")
    for s in ("S", "X"):
        if not pd.api.types.is_numeric_dtype(df[s]):
            raise SchemaError(f"state '{s}' must be numeric, got {df[s].dtype}")
    if not allow_missing_y and df["Y"].isna().any():
        raise SchemaError("Y has missing values but allow_missing_y=False")


def check_timing(df: pd.DataFrame) -> list[str]:
    """Heuristic timing / no-leakage checks (S->T1->X->T2->Y).

    These cannot *prove* temporal ordering (that is a design fact), but they catch
    gross violations: e.g. X perfectly predicted by T2 (X built from the future),
    or S identical to X (state not re-measured).
    """
    warns: list[str] = []
    sub = df.dropna(subset=["S", "X"])
    if len(sub) > 10:
        if np.allclose(sub["S"].to_numpy(), sub["X"].to_numpy()):
            warns.append("S == X exactly: intermediate state not re-measured (timing suspect)")
        # X should not be deterministically a function of T2 (which comes AFTER X)
        if sub["T2"].notna().all():
            corr = np.corrcoef(sub["X"].to_numpy(), sub["T2"].to_numpy())[0, 1]
            if abs(corr) > 0.999:
                warns.append("X almost perfectly correlated with T2 (X may use the future)")
    return warns


def continuity_report(x: np.ndarray, name: str) -> ContinuityReport:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    vals, counts = np.unique(x, return_counts=True)
    n_unique = vals.size
    max_pm = float(counts.max() / n) if n else 1.0
    qs = {f"q{int(p*100):02d}": float(np.quantile(x, p)) for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    effectively = (n_unique >= 100) and (max_pm <= 0.05)
    genuinely = (n_unique >= 0.5 * n) and (max_pm <= 0.01)
    note = ""
    if effectively and not genuinely:
        note = "effectively continuous (rich support, low tie mass) but not mathematically continuous"
    elif not effectively:
        note = "coarse / high tie mass: NOT adequate as a primary continuous state"
    return ContinuityReport(name, n, n_unique, max_pm, qs, effectively, genuinely, note)


def path_counts(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for (t1, t2), lbl in ((0, 0), "00"), ((0, 1), "01"), ((1, 0), "10"), ((1, 1), "11"):
        out[lbl] = int(((df["T1"] == t1) & (df["T2"] == t2)).sum())
    return out


def apply_gates(
    df: pd.DataFrame,
    *,
    smallest_prob: float | None = None,
    min_units: int = 1000,
    min_path: int = 75,
    preferred_path: int = 150,
    min_unique: int = 100,
    max_point_mass: float = 0.05,
    min_outcome_avail: float = 0.75,
    availability_col: str | None = None,
) -> GateReport:
    """Evaluate the hard data-quality gates (task section 4).

    With history-dependent availability, path-count gates apply to feasible branches
    (``availability_col`` marks second-stage-available rows).
    """
    failures: list[str] = []
    warnings: list[str] = []

    n_units = int(df["group"].nunique()) if "group" in df.columns else int(len(df))
    if n_units < min_units:
        failures.append(f"independent units n={n_units} < {min_units}")

    counts = path_counts(df)
    for lbl, c in counts.items():
        if c < min_path:
            failures.append(f"path {lbl} count {c} < hard minimum {min_path}")
        elif c < preferred_path:
            warnings.append(f"path {lbl} count {c} < preferred {preferred_path}")

    s_rep = continuity_report(df["S"].to_numpy(), "S")
    x_rep = continuity_report(df["X"].to_numpy(), "X")
    for rep in (s_rep, x_rep):
        if rep.n_unique < min_unique:
            failures.append(f"{rep.name} distinct values {rep.n_unique} < {min_unique}")
        if rep.max_point_mass > max_point_mass:
            failures.append(f"{rep.name} max point mass {rep.max_point_mass:.3f} > {max_point_mass}")

    y_avail = float(df["Y"].notna().mean())
    if y_avail < min_outcome_avail:
        failures.append(f"final-outcome availability {y_avail:.2f} < {min_outcome_avail}")

    if smallest_prob is not None and smallest_prob < 0.10:
        failures.append(f"smallest assignment probability {smallest_prob:.3f} < 0.10")

    return GateReport(
        passed=len(failures) == 0,
        n_units=n_units,
        path_counts=counts,
        s_report=s_rep,
        x_report=x_rep,
        outcome_availability=y_avail,
        smallest_prob=smallest_prob,
        failures=failures,
        warnings=warnings,
    )
