# Data-Access Blockers and Data-Request Instructions

## Status: the main empirical data requirement is UNRESOLVED

As of the July-2026 audit (`docs/dataset_search.md`), **no dataset that is downloadable
without a data-use agreement passes the hard gates** (n ≥ 1000 independent units, both
stages randomized, all four `(T1,T2)` paths, genuinely/effectively continuous scalar `S`
and `X`, known randomization probabilities). Consequently:

* The full estimator, boundary diagnostics, sieve-Riesz + bootstrap inference, and the
  Monte Carlo study are complete and run on **calibrated simulated data with known truth**.
* A **public proof-of-concept** (HeartSteps V1 / Drink Less / Project STAR) can exercise
  the pipeline on real randomized data, but each fails a hard gate and is **not** a valid
  main application (labelled accordingly).
* **This project does not claim a completed valid main empirical application.**

## The strongest inaccessible candidate: the energy-rebate experiment

The only design that cleanly matches the estimator (fully-crossed 2×2, n = 2,400,
continuous kWh `S/X/Y`, ~0.5 known propensities, all four paths ≈ 600 each) is the
**Ida–Ishihara–Ito–Kido–Kitagawa–Sakaguchi–Sasaki** energy-rebate field experiment. Its
household smart-meter microdata is **proprietary and non-redistributable**: the Zenodo
replication package (`10.5281/zenodo.17074824`) ships code plus non-restricted partial
data only, and the Econometrica data-availability statement records a publication
exemption because the authors do not have the right to republish the raw utility data.

### Precise data-request instructions
1. **Contact the corresponding authors** of "Dynamic Targeting: Experimental Evidence from
   Energy Rebate Programs" (NBER w32561) / "Choosing Who Chooses" (Econometrica 94(1),
   2026) — e.g. Koichiro Ito, Takanori Ida, Shusaku Sasaki, Takunori Sakaguchi.
2. **State the exact variables needed** at the household level:
   `household_id, region, period1_rebate (T1), period2_rebate (T2),
    log pre-experiment peak kWh (S), log period-1 peak kWh (X), period-2 peak kWh (Y)`,
   plus the exact per-period randomization probabilities and any stratification.
3. **Offer the standard safeguards:** sign the utility's / Ministry of the Environment's
   DUA; keep the microdata only under `data/raw/` (git-ignored, never committed); run
   entirely locally; publish only aggregate estimates and figures.
4. **Map to the pipeline:** place the unit-level file at `data/raw/energy.csv` with the
   columns in `configs/energy.yaml`, then run
   `python -m path_welfare.cli audit --config configs/energy.yaml` (gates + identification),
   followed by `estimate`, `boundaries`, `infer`, `robustness`, `report`.
5. **Fallback if only aggregate/partial data is released:** use it to *calibrate* DGP 8
   (`simulation.CalibratedDGP`) — fit `X|S,T1` and `Y|X,T2`, preserve the empirical
   propensities and missingness — and report simulation-calibrated results, clearly
   labelled as calibrated rather than a direct empirical application.

## Alternative gated routes (documented, not pursued here)
* **Intern Health Study 2018 MRT** — apply for the University of Michigan DUA (openICPSR
  129225; `intern_health@med.umich.edu`). Note it is an MRT reduced to a two-week slice,
  not a native 2×2, and Apple SensorKit streams cannot be shared.
* **NIMH SMARTs (STAR\*D, CATIE, STEP-BD)** — apply to the NIMH Data Archive with a Data
  Use Certification; accept that non-responder-only second-stage randomization breaks
  global positivity, so only a **branch-specific, availability-relative** path parameter is
  identified (use `configs/smart.yaml` and `availability_col`).

## What "resolved" would require
A signed DUA (or author-mediated access) delivering unit-level assignment logs + continuous
states + outcomes for a fully-crossed two-stage randomized experiment with n ≥ 1000. Until
then, the empirical claims in this project are explicitly **simulation-based**, and the
main empirical data requirement is stated as **unresolved**.
