# Public Proof-of-Concept: HeartSteps V1 (SOFTWARE VALIDATION ONLY)

**This is NOT a main empirical application.** HeartSteps V1 is a public micro-randomized
trial with **N = 37 participants** — far below the `n ≥ 1000` gate. It is used here only to
demonstrate that the pipeline runs on **real randomized data** and that the hard
data-quality gates **correctly reject** an unsuitable dataset (task §3.4, §4).

## Source
Public repository `github.com/klasnja/HeartStepsV1`, file `data_files/suggestions.csv`
(downloaded to `data/raw/hs_suggestions.csv`, which is git-ignored — never committed).

## Two-stage reduction (one adjacent-decision triplet per participant)
For each user, a **seeded random** pair of consecutive *available, randomized* decision
points `(d, d+1)` is chosen (selection depends only on availability, never on treatment
or outcome):

| canonical | HeartSteps field | meaning |
|---|---|---|
| `S`  | `log(1 + jbsteps30pre)` at `d`   | Jawbone steps in the 30 min before the first decision |
| `T1` | `send.active` at `d`             | 1 = activity suggestion randomized to be sent |
| `X`  | `log(1 + jbsteps30pre)` at `d+1` | steps before the second decision (post-`T1`) |
| `T2` | `send.active` at `d+1`           | 1 = activity suggestion at the second decision |
| `Y`  | `log(1 + jbsteps30)` at `d+1`    | steps in the 30 min after the second decision |

The independent unit is the **participant** (n = 37). The randomized treatment is
`send.active` (the raw `send` flag is the *realized* message and is degenerate after
filtering to available+randomized decisions — a coding subtlety the adapter handles).

## Result (the gates do their job)
- **n = 37 independent users** — fails `n ≥ 1000`.
- Path counts `(00,01,10,11) = (8, 6, 8, 15)` — all four paths present but far below the
  75/150 minimums.
- `P(T1=1) = 0.62`, `P(T2=1) = 0.57` — genuine randomization variation.
- `S`, `X` have 21 distinct values with **max point mass ≈ 0.35–0.40** (step-count zeros
  during sedentary windows) → **effectively continuous at best, not genuinely continuous**.
- **`apply_gates` returns `passed = False`** with 10 documented failures.

Machine-readable summary: `results/simulations/heartsteps_poc.json`.

## Interpretation
The pipeline ingests real randomized MRT microdata, constructs the canonical
`O = (S,T1,X,T2,Y)`, and the gates correctly refuse to certify it for the irregular
inference target `V_11^*`. Point estimation and inference are **not** reported for
HeartSteps because N = 37 cannot support them — reporting a confidence interval here would
violate the project's own go/no-go rule. This is the intended outcome of a software
validation, and it corroborates the audit conclusion that **no accessible public dataset
meets the requirements** (see `docs/data_access_blockers.md`).
