"""Dataset adapters producing the canonical O = (S, T1, X, T2, Y) frame.

Includes the fully-implemented simulated adapter, a generic CSV adapter, and
best-effort public-data adapters (Project STAR, HeartSteps V1) for the clearly
labelled proof-of-concept / software validation.  Restricted microdata is never
bundled; adapters read from ``data/raw`` which is git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import to_canonical, validate_schema


def load_simulated(dgp_name: str, n: int, seed: int) -> pd.DataFrame:
    from .simulation import get_dgp

    rng = np.random.default_rng(seed)
    df = get_dgp(dgp_name).sample(n, rng)
    df["group"] = np.arange(len(df))  # one independent unit per row
    return df


def load_generic_csv(path: str | Path, variables) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = to_canonical(df, variables)
    validate_schema(df)
    return df


def load_project_star(path: str | Path) -> pd.DataFrame:
    """ILLUSTRATIVE proof-of-concept only (task 3.4).

    Follows Sakaguchi (2026)'s two-stage reduction of Project STAR:
      S  = a baseline (kindergarten free-lunch / prior) score proxy,
      T1 = kindergarten small-class assignment (1) vs regular/aide (0),
      X  = kindergarten test percentile,
      T2 = grade-1 small-class assignment (1) vs regular/aide (0),
      Y  = grade-1 test percentile.

    WARNING: STAR's grade-1 class type is NOT independently re-randomized for
    continuing students, and test scores are finely-supported, NOT genuinely
    continuous.  This adapter is for software illustration, not a valid main
    empirical application.  Column names vary across STAR releases; adjust the map.
    """
    df = pd.read_csv(path)
    raise NotImplementedError(
        "Project STAR column mapping is release-specific; provide a VariableMap and use "
        "load_generic_csv, or fill in the STAR-specific column names here. This adapter is "
        "intentionally a stub to avoid fabricating a mapping."
    )


def load_heartsteps(suggestions_csv: str | Path = "data/raw/hs_suggestions.csv") -> pd.DataFrame:
    """SOFTWARE VALIDATION ONLY (task 3.4): public HeartSteps V1 MRT (N=37 participants,
    github.com/klasnja/HeartStepsV1).  Builds ONE adjacent-decision triplet per participant
    so the independent unit is the participant (n=37 -- far below the n>=1000 gate).

    Mapping (two consecutive available, randomized decision points d, d+1):
      S  = log(1+jbsteps30pre) at d     (steps before the first suggestion)
      T1 = send at d                     (1 = activity suggestion sent)
      X  = log(1+jbsteps30pre) at d+1    (steps before the second suggestion; post-T1)
      T2 = send at d+1
      Y  = log(1+jbsteps30) at d+1       (steps after the second suggestion)

    Step counts are heavily zero-inflated, so this is 'effectively continuous' at best.
    N=37 means this exercises the software; it CANNOT support inference.
    """
    d = pd.read_csv(suggestions_csv, low_memory=False)
    # The randomized treatment is send.active (activity suggestion vs not) among AVAILABLE
    # decisions; the raw `send` flag is the realized message and is degenerate after filtering.
    d["send_i"] = d["send.active"].map({True: 1, False: 0, "True": 1, "False": 0})
    d = d[(d["avail"] == True) & (d["is.randomized"] == True) & d["send_i"].notna()]  # noqa: E712
    d = d.sort_values(["user.index", "decision.index"])
    rng = np.random.default_rng(20260713)  # seeded: pair chosen by availability, NOT by treatment
    rows = []
    for uid, g in d.groupby("user.index"):
        g = g.reset_index(drop=True)
        # all consecutive pairs with the needed step fields present, then pick one at random
        valid = [i for i in range(len(g) - 1)
                 if pd.notna(g.loc[i, "jbsteps30pre"]) and pd.notna(g.loc[i + 1, "jbsteps30pre"])
                 and pd.notna(g.loc[i + 1, "jbsteps30"])]
        if valid:
            i = int(rng.choice(valid))
            a, b = g.loc[i], g.loc[i + 1]
            rows.append({
                "S": np.log1p(float(a["jbsteps30pre"])),
                "T1": int(a["send_i"]),
                "X": np.log1p(float(b["jbsteps30pre"])),
                "T2": int(b["send_i"]),
                "Y": np.log1p(float(b["jbsteps30"])),
                "group": int(uid),
            })
    out = pd.DataFrame(rows)
    validate_schema(out)
    return out
