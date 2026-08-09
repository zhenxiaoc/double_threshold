# Identification (narrative)

This document assesses each identifying assumption **separately** and uses a strict
language key, never asserting that an untestable assumption is "satisfied":

* **design** — supplied by randomized design;
* **protocol** — documented by the trial protocol;
* **diagnostic** — supported (or contradicted) by a data diagnostic;
* **maintained** — a modelling restriction imposed, not verified;
* **unresolved**.

The machine-checkable version for whatever dataset is loaded is written by
`python -m path_welfare.cli audit` to `results/identification_audit.{json,md}`.

## 7.1 Consistency and treatment definition
Determine whether "treatment" means randomized assignment, offer, delivery, receipt, or
adherence. **Primary analysis uses randomized assignment** (intention-to-treat style)
unless a valid noncompliance analysis is implemented. For the energy-rebate design,
`T = 1` is the *assigned* rebate; for the Intern Health Study, `T = 1` is *assignment to
any notification* — a randomized mixture of categories (run category-specific sensitivity
analyses). Status: **design / protocol**.

## 7.2 Sequential ignorability
Two-stage randomization implies, *by design*:
`T1 ⟂ (X^(τ1), Y^(τ1,π2)) | S` and `T2 ⟂ Y^(τ1,τ2) | S,T1,X` (+ availability). This is a
**design** fact when both stages are randomized; it is not proven by covariate balance.
Balance and treatment-prediction AUCs are reported only as **diagnostics** (`AUC(T1|S)`,
`AUC(T2|S,T1,X)` near 0.5 is consistent with randomization; far from 0.5 flags a problem).
Status: **design** (given randomization) + **diagnostic** (balance).

## 7.3 Positivity
Use **known design probabilities** where available. Report all four path counts,
stage probabilities, availability-status probabilities, path-specific effective sample
sizes, the smallest assignment probability, missing assignments, and deviations from the
planned randomization. Do **not** ML-estimate propensities when the design is known.
For SMARTs with non-responder-only second-stage randomization, global positivity **fails**;
use *availability-relative* positivity on the feasible branch. Status: **design** for
fully-crossed 2×2; **unresolved / branch-only** for availability-dependent designs.

## 7.4 Markov sufficiency
`E[Y|S,T1,X,T2] = E[Y|X,T2]`. **Randomization does not imply this** — it is a
**maintained** modelling restriction. Diagnostic: compare held-out MSE of the restricted
model `Y ~ f(X,T2)` against the rich model `Y ~ f(S,T1,X,T2)`; flag as *questionable* if
the rich model reduces held-out MSE by > 5% (`diagnostics.markov_sufficiency`). If
questionable: retain the scalar estimator as the theorem-aligned restricted model, label
its interpretation model-dependent, and add a richer-history robustness specification. Do
**not** claim the scalar Markov assumption is empirically verified. Status: **maintained**
(+ **diagnostic**).

## 7.5 Attrition and missingness
Report missing `S, X, Y` by `T1`, `T2`, path, baseline-state quantiles, time, and site.
Run complete-case, inverse-probability-of-observation, and a delta-adjustment sensitivity
for missing `Y`. Status: **diagnostic**.

## 7.6 Interference
Identify plausible household / workplace / clinic / classroom / notification /
general-equilibrium spillovers. Use the **participant (or household) as the minimum
resampling unit**; provide site-clustered sensitivity where site identifiers exist. The
energy-rebate design has less direct interference than classroom/network experiments.
Status: **maintained** (SUTVA) + **diagnostic** (clustered resampling).

## 7.7 Continuous-state diagnostics
For `S` and `X` report unique-value count, maximum point mass, quantiles, density, support
gaps, support by arm and by path, common-support region, tails, and transformations.
Transformations are **prespecified** (log for positive right-skewed, z-score for
presentation; monotone rank only as sensitivity). No winsorizing in the primary analysis
unless a documented measurement-error rule requires it. `S`/`X` are labelled
*genuinely continuous* vs *effectively continuous* (rich support, low tie mass, meaningful
metric — e.g. step counts) vs *finely-supported* (e.g. Project STAR scores — **not**
continuous). Status: **diagnostic**.

## Summary of statuses for the maintained runnable dataset
The pipeline currently runs on a **calibrated simulation** (no accessible real dataset
passes the hard gates — see `docs/data_access_blockers.md`). On the simulation, sequential
ignorability and positivity hold **by construction (design)**, Markov sufficiency is
**true by construction** for DGPs 1–6 and **deliberately violated** in DGP 7 (where the
diagnostic correctly flags it). For a real dataset, the audit CLI recomputes every status
above from the actual data.
