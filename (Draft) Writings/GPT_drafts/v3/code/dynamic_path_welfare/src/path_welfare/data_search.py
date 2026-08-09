"""Dataset discovery + access-audit catalogue writer (task section 3).

The catalogue below records the *result* of the July-2026 access audit (see
``docs/dataset_search.md``).  It does not itself hit the network; the CLI
``search-data`` command additionally scans local directories for candidate files.
"""

from __future__ import annotations

import csv
from pathlib import Path

# One row per candidate.  access_state in {PUBLIC, GATED, NOT_FOUND, UNVERIFIED}.
CANDIDATES: list[dict] = [
    {
        "study": "Ida/Ishihara/Ito/Kido/Kitagawa/Sakaguchi/Sasaki energy rebate (Dynamic Targeting; Choosing Who Chooses)",
        "domain": "energy / household electricity",
        "design": "fully-crossed 2x2 two-period RCT (both periods independently randomized)",
        "unit": "household",
        "n_units": 2400, "n_stage2": 2400,
        "arms_stage1": "rebate vs none", "arms_stage2": "rebate vs none",
        "both_randomized": True, "stage2_response_dependent": False,
        "S": "pre-experiment peak kWh (log)", "X": "period-1 peak kWh (log)", "Y": "period-2 peak kWh (neg)",
        "S_continuous": "genuine", "X_continuous": "genuine",
        "path_counts": "625/606/581/588 (UU/UT/TU/TT)",
        "rand_probs": "~0.5 per period (design-known)",
        "attrition": "low (smart-meter panel)", "clustering": "household/region",
        "interference": "low", "access_state": "GATED",
        "license": "CC-BY-4.0 (code only)", "repo": "Zenodo 10.5281/zenodo.17074824; NBER w32561",
        "suitability": "PERFECT design but microdata restricted/non-redistributable",
        "include": "shortlist-for-request", "reason": "only design matching the 2x2 estimator; data not public",
    },
    {
        "study": "Intern Health Study 2018 MRT",
        "domain": "mHealth / physician wellbeing",
        "design": "micro-randomized trial (weekly notification randomization ~26 wks)",
        "unit": "intern", "n_units": 1500, "n_stage2": None,
        "arms_stage1": "notify vs none", "arms_stage2": "notify vs none",
        "both_randomized": True, "stage2_response_dependent": False,
        "S": "mean sleep wk w-1", "X": "mean sleep wk w", "Y": "mean sleep wk w+1",
        "S_continuous": "genuine", "X_continuous": "genuine",
        "path_counts": "unknown (data gated)", "rand_probs": "known (weekly)",
        "attrition": "moderate", "clustering": "institution/specialty",
        "interference": "possible (hospital)", "access_state": "GATED",
        "license": "UMich DUA", "repo": "openICPSR 129225 (survey); wearable/assignment gated",
        "suitability": "continuous states but MRT not 2x2; micro-data DUA-gated",
        "include": "no", "reason": "intervention+wearable logs behind UMich DUA; MRT structure",
    },
    {
        "study": "STAR*D / CATIE / STEP-BD (NIMH SMARTs)", "domain": "psychiatry",
        "design": "equipoise-stratified / multi-phase; stage-2 for non-responders",
        "unit": "patient", "n_units": 4000, "n_stage2": None,
        "arms_stage1": "multi", "arms_stage2": "multi",
        "both_randomized": False, "stage2_response_dependent": True,
        "S": "QIDS/PANSS", "X": "QIDS/PANSS", "Y": "remission/discontinuation",
        "S_continuous": "partial", "X_continuous": "partial",
        "path_counts": "n/a", "rand_probs": "partial",
        "attrition": "high", "clustering": "site", "interference": "low",
        "access_state": "GATED", "license": "NDA Data Use Certification",
        "repo": "NIMH NDA collection 2148 (STAR*D)",
        "suitability": "gated AND non-responder-only stage-2 breaks global positivity",
        "include": "no", "reason": "controlled access + design mismatch (positivity fails globally)",
    },
    {
        "study": "ADAPT-2 (meth) / ADAPT-R (HIV) / BestFIT (weight)", "domain": "clinical SMART",
        "design": "SMART; stage-2 re-randomizes non-responders only",
        "unit": "patient", "n_units": 468, "n_stage2": None,
        "arms_stage1": "2-3", "arms_stage2": "2",
        "both_randomized": False, "stage2_response_dependent": True,
        "S": "baseline BMI / clinical", "X": "% change / clinical", "Y": "clinical outcome",
        "S_continuous": "genuine (BestFIT BMI)", "X_continuous": "genuine",
        "path_counts": "n/a", "rand_probs": "known",
        "attrition": "moderate", "clustering": "site", "interference": "low",
        "access_state": "GATED", "license": "NIDA/registration DUA",
        "repo": "NIDA Data Share CTN-0068 (ADAPT-2)",
        "suitability": "n<1000 and non-responder-only stage-2 (branch positivity only)",
        "include": "no", "reason": "small n and availability-dependent second stage",
    },
    {
        "study": "Drink Less MRT", "domain": "alcohol mHealth",
        "design": "MRT, 1 decision/day, 3-arm notification", "unit": "user",
        "n_units": 566, "n_stage2": None,
        "arms_stage1": "none/standard/new (3-arm)", "arms_stage2": "same",
        "both_randomized": True, "stage2_response_dependent": False,
        "S": "engagement", "X": "engagement", "Y": "app engagement",
        "S_continuous": "semi", "X_continuous": "semi",
        "path_counts": "n/a (3-arm MRT)", "rand_probs": "0.40/0.30/0.30",
        "attrition": "moderate", "clustering": "none", "interference": "low",
        "access_state": "PUBLIC", "license": "none set (OSF)", "repo": "OSF osf.io/w3szp",
        "suitability": "public but MRT, 3-arm treatment, n<1000",
        "include": "public-example-only", "reason": "public but fails n>=1000 and binary-2x2 gates",
    },
    {
        "study": "HeartSteps V1", "domain": "activity mHealth",
        "design": "MRT, up to 5 decisions/day, P(suggest)=0.6", "unit": "participant",
        "n_units": 37, "n_stage2": None,
        "arms_stage1": "suggest vs none", "arms_stage2": "suggest vs none",
        "both_randomized": True, "stage2_response_dependent": False,
        "S": "step count", "X": "step count", "Y": "30-min steps",
        "S_continuous": "effective", "X_continuous": "effective",
        "path_counts": "n/a", "rand_probs": "0.6",
        "attrition": "low", "clustering": "none", "interference": "low",
        "access_state": "PUBLIC", "license": "repo", "repo": "github.com/klasnja/HeartStepsV1",
        "suitability": "public + clean randomization but N=37 (software validation only)",
        "include": "software-validation", "reason": "public but far below n>=1000",
    },
    {
        "study": "Project STAR", "domain": "education",
        "design": "K assignment to 3 class types; not a clean binary 2-stage randomization",
        "unit": "student", "n_units": 7000, "n_stage2": None,
        "arms_stage1": "small/regular/aide", "arms_stage2": "not independently re-randomized",
        "both_randomized": False, "stage2_response_dependent": False,
        "S": "prior score", "X": "K test score", "Y": "grade-1 test score",
        "S_continuous": "finely-supported (NOT genuine)", "X_continuous": "finely-supported",
        "path_counts": "n/a", "rand_probs": "partial",
        "attrition": "moderate", "clustering": "school", "interference": "classroom",
        "access_state": "PUBLIC", "license": "Harvard Dataverse", "repo": "hdl:1902.1/10766",
        "suitability": "public but not a binary 2x2; scores finely-supported not continuous",
        "include": "optional-comparison", "reason": "used by Sakaguchi (2026) as illustrative DTR only",
    },
]

_FIELDS = list(CANDIDATES[0].keys())


def write_candidates_csv(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        for row in CANDIDATES:
            w.writerow(row)
    return path


def score_candidate(c: dict) -> dict:
    """Selection scorecard (task section 5).  Only candidates that pass the hard gates
    should be scored; gated/failing candidates are scored 0 on access."""
    seq = 30 if (c["both_randomized"] and not c["stage2_response_dependent"]) else (
        12 if c["both_randomized"] else 0)
    cont = 20 if (c["S_continuous"] == "genuine" and c["X_continuous"] == "genuine") else (
        10 if "genuine" in (c["S_continuous"] + c["X_continuous"]) else 4)
    sample = 20 if (c["n_units"] or 0) >= 1000 else (8 if (c["n_units"] or 0) >= 500 else 0)
    access = 15 if c["access_state"] == "PUBLIC" else 0
    lowint = 10 if c["interference"] == "low" and c["attrition"] in ("low", "moderate") else 5
    interp = 5 if c["S_continuous"] == "genuine" else 3
    total = seq + cont + sample + access + lowint + interp
    return {"study": c["study"], "sequential_id": seq, "continuous": cont,
            "sample_support": sample, "access": access, "low_interference": lowint,
            "interpretability": interp, "total": total, "access_state": c["access_state"]}


def scorecard(path: str | Path | None = None) -> list[dict]:
    rows = [score_candidate(c) for c in CANDIDATES]
    rows.sort(key=lambda r: r["total"], reverse=True)
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def scan_local(root: str | Path) -> list[str]:
    """Scan local directories for candidate microdata files (never opens restricted data)."""
    root = Path(root)
    hits = []
    for pat in ("*.csv", "*.dta", "*.rds", "*.RData", "*.parquet", "*.sav"):
        hits += [str(p) for p in root.rglob(pat)]
    return sorted(hits)
