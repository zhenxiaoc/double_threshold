# Datasets for a larger calibration

The current calibration uses the Banerjee et al. (2015) graduation RCT
(`graduation.rda`, **n = 854**). That is not the cause of the sieve-DML
coverage failure (see `dml_coverage_diagnosis.md`), but it *is* a real
constraint: it starves the generative model and caps how large a simulated `n`
can be drawn without extrapolating past the support of the real data.

Requirements for this estimand, `theta = Pr(tau_S >= 0, tau_Y <= 0)`:
randomized binary treatment; **both** a short-run `S` and a longer-run `Y`
measured on the **same units**, ideally the same continuous variable at two
horizons; n >= 5,000 (10,000+ preferred); a few strong continuous covariates so
a tensor sieve stays feasible; public microdata.

## Recommended

### 1. National Job Corps Study (NJCS), 1994-1998 — n = 15,386

- **S / Y**: self-reported **weekly earnings for each of 208 weeks** after
  random assignment (interviews at 12/30/48 months). Natural pair: week ~52
  vs week 208. Same variable, same scale, two horizons.
- **Access**: public-use file. openICPSR replication package for Schochet,
  Burghardt & McConnell (AER 2008), <https://doi.org/10.3886/E113269V1> —
  contains actual SAS microdata (`baseline.sas7bdat`, follow-ups), free ICPSR
  login only.
- **Why**: the only public candidate with a dense within-person earnings panel
  on ~15k individually randomized units — ~18x the current calibration, so
  per-arm generators have far more to fit and simulated n of 20,000+ stays
  inside the support. Individual randomization with documented
  treatment-subsampling probabilities, so propensities are known by
  construction. Covariates give 2-3 strong continuous axes (prior-year
  earnings, age, prior weeks worked).
- **Caveat, and why it is a feature**: Job Corps runs the *opposite* way —
  negative short-run effects (residential lock-in), positive long-run effects —
  so `theta` is the smaller quadrant (~0.10-0.25). A small, near-boundary
  `theta` is exactly where a plug-in estimator of a non-smooth functional is
  most stressed. Flip the inequality to get the complementary quadrant from the
  same frozen DGP.
- **Must handle**: carry the treatment-subsampling design weights; model the
  point mass at zero earnings with a hurdle/two-part generator (a plain GAN
  smears it across the origin); earnings are self-reported.

### 2. National JTPA Study (NJS), 1987-1989 — n = 20,601 randomized

- **S**: monthly/quarterly self-reported earnings over the first 18 months.
  **Y**: annual earnings in years 3-5 from **SSA administrative records**
  (13,699 matched).
- **Access**: free, no login, no fee — W.E. Upjohn Institute ERDC,
  <https://www.upjohn.org/data-tools/employment-research-data-center/national-jtpa-study>
  (Stata `.dta` and SAS).
- **Why this one is not optional**: (a) the **sign pattern is natively right** —
  positive 18-month gains that faded to insignificance by year 5, with outright
  negative 18-month impacts for male youth, so genuine short-run-gain /
  long-run-harm mass exists and `theta` is substantial without relabeling;
  (b) **comparability** — JTPA is Chen-Chen-Gao's own empirical application and
  Kitagawa-Tetenov's workhorse, so a JTPA-calibrated DGP plugs the simulation
  directly into the literature the estimator lives in; (c) longest public
  horizon (5-year administrative earnings).
- **Must handle**: long-run sample is 13,699 not 20,601 and SSA match failure is
  non-random (a second selection layer — say so); SSA earnings are **top-coded**
  at the taxable maximum, so use a censor-aware likelihood or trim; assignment
  ratios vary by site/service-strategy stratum, so condition on the
  randomization block; S is survey-reported while Y is administrative; merging
  the 14 data directories is genuine work.

**Recommendation: do both.** Job Corps for the panel depth and n; JTPA for the
right sign pattern and literature comparability. Each covers the other's weakness.

## Fallback

**NEWWS** (n ~ 44,000, 20 quarters of UI earnings, free no-registration zip at
<https://aspe.hhs.gov/national-evaluation-welfare-work-strategies-restricted-access-public-use-data>).
Tempting for raw n, but the public file **rounds quarterly earnings to the
nearest $100**, discretizing the outcome onto a grid — a continuous generative
model and a density-based sieve both misbehave, and modelling it as
interval-censored contaminates the "frozen truth". Also 11 stacked
site-by-model experiments rather than one DGP.

## Ruled out (with reasons)

| Dataset | Why not |
|---|---|
| Connecticut Jobs First (n=4,803) | Most theoretically apt (time limit creates short-run-gain/long-run-harm), but below the size floor |
| WIA Gold Standard (n=35,665) | Restricted-use file only |
| HPOG 1.0 (n=13,717) | Follow-up files restricted-access at ICPSR |
| GAIN | Not public — Opportunity Insights' repository ships *simulated* data |
| Self-Sufficiency Project | Statistics Canada will never release public-use microdata |
| PROGRESA (n~24,000) | Randomized at the **locality** level, only 506 clusters — effective n is ~506, trading a small clean n for a large fake one; controls phased in |
| Project STAR (n=11,601) | Public long-run outcome is essentially **binary**; continuous earnings behind an IRS restriction; three-armed treatment |
| ERA (n~45,000 pooled) | 16 heterogeneous experiments, none over 7,000; thin covariates |
| Karlan-List (n=50,083) | No long-run outcome on the same units — `theta` undefined |
