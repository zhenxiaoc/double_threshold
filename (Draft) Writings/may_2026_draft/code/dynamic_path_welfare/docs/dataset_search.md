# Dataset Search and Access Audit (July 2026)

Access states: **PUBLIC** = downloadable now, no DUA; **GATED** = DUA / controlled access
/ application; **NOT_FOUND / UNVERIFIED**. Registration + click-through DUA is treated as
GATED. The machine-readable catalogue is `results/dataset_candidates.csv`; the scorecard
(hard-gate passers only) is `results/tables/table01_scorecard.csv`.

## Requirement
Two sequentially **randomized** binary treatments (`T1` then `T2`), scalar continuous
baseline `S` and intermediate `X`, scalar outcome `Y`, `n ≥ 1000` independent units, all
four `(T1,T2)` paths supported, known randomization probabilities.

## Candidate A — Energy-rebate experiment (Ida, Ishihara, Ito, Kido, Kitagawa, Sakaguchi, Sasaki)
"Dynamic Targeting" (NBER w32561, 2024) and "Choosing Who Chooses" (Econometrica 94(1),
2026), same 7-author team, same field experiment.
* **Design (primary-source verified):** two-period sequential RCT, **fully-crossed 2×2**,
  both periods independently randomized. Four groups (U,U)=625, (U,T)=606, (T,U)=581,
  (T,T)=588 → **2,400 households**. Global positivity holds; ~0.5 marginal treatment
  probability per period (design-known). Unit = household (Kansai/Chubu, Japan; Ministry
  of the Environment). Treatment = peak-time rebate.
* **S / X / Y:** pre-experiment peak kWh / post-period-1 peak kWh / period-2 peak kWh — all
  **genuinely continuous** (30-min smart-meter data).
* **Access: GATED.** Replication package Zenodo `10.5281/zenodo.17074824` is classified
  *Software*, `Replication_package.zip` **3.6 MB**, CC-BY-4.0 — code + partial/non-restricted
  data only. Econometrica data-availability statement: the authors received an exemption
  because the underlying household consumption data is proprietary and non-redistributable.
* **Verdict:** design is essentially perfect for the 2×2 estimator, **but the microdata is
  not public.** This is the shortlisted target for a data request (`docs/data_access_blockers.md`).

## Candidate B — Intern Health Study 2018 MRT (Sen lab, Michigan; NeCamp et al. 2020)
Weekly micro-randomized notification trial, ~1,500–2,000 interns, genuinely continuous
Fitbit sleep/step/mood states, known weekly randomization probability. **Access: GATED** —
openICPSR 129225 (survey portion possibly public after registration); intervention-assignment
logs + wearable streams behind a **University of Michigan DUA** (SensorKit not shareable;
some cohorts via NDA). It is an **MRT, not a native 2×2**. Verdict: continuous states, but
micro-data gated and MRT structure — not confirmed public.

## Candidate C — Medical SMARTs / MRTs
Structural note: in **ADAPT-R, BestFIT, STAR\*D, CATIE, STEP-BD, ADAPT-2** the second-stage
randomization is **restricted to non-responders** (availability-dependent), so **global
positivity fails** independent of access.
* **STAR\*D / CATIE / STEP-BD:** **GATED** via NIMH Data Archive (Data Use Certification);
  equipoise / multi-phase, not binary 2×2.
* **ADAPT-2** (meth, NIDA-CTN-0068): **GATED-lite** (registration + click-through DUA at
  NIDA Data Share); n=403 (< 1000); SPCD non-responder second stage.
* **ADAPT-R** (HIV, Kenya): n≈1,809 but **3-arm** stage 1 and non-responder stage 2; **no
  public repository found**.
* **BestFIT** (weight loss): continuous BMI/percent-loss but n=468 (< 1000) and
  non-responder second stage; no public deposit found.

## Candidate D — Public proof-of-concept fallbacks
* **Drink Less MRT** (OSF `osf.io/w3szp`, `public: true`): **PUBLIC** but an MRT with a
  **3-arm** daily notification (0.40/0.30/0.30) and n≈566 — fails `n ≥ 1000` and binary-2×2.
* **HeartSteps V1** (`github.com/klasnja/HeartStepsV1`): **PUBLIC**, clean randomization
  (P=0.6), continuous steps, but **N=37** — software validation only.
* **Project STAR** (Harvard Dataverse `hdl:1902.1/10766`): **PUBLIC**, ~7,000 students, but
  **not a binary 2-stage randomization** (K assignment to 3 class types; grade-1 type not
  independently re-randomized for continuing students) and test scores are **finely-supported,
  not genuinely continuous**. Sakaguchi (2026) uses it as an *illustrative* two-stage DTR
  (K → grade 1) — usable only as an optional comparison, never the primary application.

## Selection scorecard (see `results/tables/table01_scorecard.csv`)
The energy-rebate design scores highest on sequential-design + continuity + sample support
+ interpretability, but **scores 0 on data access** (GATED), so it cannot be selected as
the runnable main application. Every genuinely PUBLIC candidate fails at least one hard
gate (n, binary-2×2, or continuous intermediate state).

## Conclusion
**No accessible dataset passes the main hard gates.** Per task §4/§14, the project
therefore: (1) does **not** claim a completed valid main empirical application; (2) runs
the full estimator + inference pipeline on **calibrated simulated data with known truth**;
(3) documents a clearly-labelled public proof-of-concept option (HeartSteps V1 / Drink
Less / Project STAR, each with its stated limitation); (4) provides
`docs/data_access_blockers.md` with a precise data request for the strongest inaccessible
candidate (the energy-rebate experiment).
