# Identification Audit

Dataset: **simulation**  (config: sim_primary)

## Positivity (design-based)
- Path counts: {'00': 503, '01': 516, '10': 527, '11': 454}
- P(T1=1) empirical=0.490 design=0.5
- P(T2=1) empirical=0.485 design=0.5
- Smallest assignment prob: 0.485 (>=0.10: True)

## Markov sufficiency (testable modelling restriction)
- Held-out MSE restricted Y~f(X,T2): 0.2580
- Held-out MSE rich Y~f(S,T1,X,T2): 0.2682
- Incremental MSE reduction: -4.0%
- **Markov questionable: False** -- rich history does not materially improve held-out fit at the 5% threshold (consistent with -- but does not prove -- Markov sufficiency)

## Randomization / balance diagnostics
- AUC(T1 | S) = 0.509
- AUC(T2 | S,T1,X) = 0.521
- AUC near 0.5 is consistent with randomization; AUC>>0.5 flags a randomization or coding problem. Balance is a diagnostic, not proof of ignorability.

## Attrition / missingness
- Missing S/X/Y: 0.000 / 0.000 / 0.000
- Missing Y by path: 00:0.000, 01:0.000, 10:0.000, 11:0.000

## Continuous-state diagnostics
- S: unique=2000, max point mass=0.001, effectively_continuous=True, genuinely_continuous=True
- X: unique=2000, max point mass=0.001, effectively_continuous=True, genuinely_continuous=True

_Language key: design-based (positivity, sequential ignorability from randomization) vs. testable modelling restriction (Markov) vs. diagnostic (balance, continuity)._