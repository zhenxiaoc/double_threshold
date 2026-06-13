# CCG 2025 SieveVar Full-Replication Check

This check compares the Python CCG 2025 SieveVar replication against the
reported CCG 2025 appendix SieveVar tables at `n = 1500` using `rep = 2000`.

The full run used the paper Sobol counts and paper spline settings:

- M1-M14: welfare SieveVar with 40,000 Sobol points for variance.
- M15: value SieveVar with 1,000,000 Sobol points for variance.
- `solver="pinv"` with `rcond=sqrt(machine epsilon)`, matching the tolerance
  scale of R `MASS::ginv`.

The first full Python run exposed one numerical mismatch in M7. Python's default
`np.linalg.pinv` tolerance is much smaller than R `ginv`, retaining near-singular
spline directions and creating rare extreme outlier draws. OptTreat now uses the
R-compatible tolerance in the CCG `pinv` workflow. M7 was rerun at full
`rep = 2000` after that fix.

| Model | Truth R/Python | Bias R/Python | SD R/Python | SE R/Python | SD(SE) R/Python | Coverage R/Python |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.3857/0.3857 | 0.0152/0.0133 | 0.0469/0.0462 | 0.0467/0.0466 | 0.0035/0.0034 | 0.9370/0.9390 |
| M2 | 0.2358/0.2358 | 0.0039/0.0025 | 0.0481/0.0487 | 0.0488/0.0488 | 0.0029/0.0028 | 0.9480/0.9495 |
| M3 | 0.5001/0.5001 | 0.0038/0.0024 | 0.0580/0.0577 | 0.0581/0.0581 | 0.0023/0.0024 | 0.9560/0.9565 |
| M4 | 0.1033/0.1033 | 0.0185/0.0186 | 0.0478/0.0474 | 0.0482/0.0485 | 0.0088/0.0086 | 0.9400/0.9375 |
| M5 | 0.0499/0.0499 | 0.0282/0.0277 | 0.0418/0.0414 | 0.0431/0.0425 | 0.0110/0.0113 | 0.9310/0.9250 |
| M6 | 0.2315/0.2315 | 0.0268/0.0249 | 0.0559/0.0577 | 0.0552/0.0550 | 0.0057/0.0057 | 0.9260/0.9140 |
| M7 | 0.1250/0.1250 | 0.0462/0.0467 | 0.0505/0.0496 | 0.0502/0.0501 | 0.0087/0.0085 | 0.8920/0.8830 |
| M8 | 0.3857/0.3857 | 0.0068/0.0060 | 0.0414/0.0417 | 0.0417/0.0417 | 0.0026/0.0026 | 0.9475/0.9490 |
| M9 | 0.2358/0.2358 | 0.0042/0.0017 | 0.0425/0.0436 | 0.0431/0.0431 | 0.0026/0.0026 | 0.9555/0.9430 |
| M10 | 0.5001/0.5001 | 0.0067/0.0040 | 0.0511/0.0528 | 0.0511/0.0512 | 0.0020/0.0020 | 0.9480/0.9450 |
| M11 | 0.1033/0.1033 | 0.0307/0.0295 | 0.0365/0.0354 | 0.0379/0.0379 | 0.0041/0.0040 | 0.9065/0.9180 |
| M12 | 0.0499/0.0499 | 0.0418/0.0413 | 0.0316/0.0314 | 0.0344/0.0343 | 0.0050/0.0050 | 0.8410/0.8530 |
| M13 | 0.2315/0.2315 | 0.0177/0.0165 | 0.0432/0.0434 | 0.0438/0.0438 | 0.0030/0.0030 | 0.9420/0.9415 |
| M14 | 0.1250/0.1250 | 0.0251/0.0238 | 0.0382/0.0377 | 0.0386/0.0385 | 0.0050/0.0048 | 0.9230/0.9260 |
| M15 | 3.1416/3.1416 | 0.0076/0.0120 | 0.0710/0.0748 | 0.0711/0.0717 | 0.0092/0.0130 | 0.9420/0.9400 |

Mean absolute differences across the 15 rows:

| Metric | Mean absolute difference | Max absolute difference |
| --- | ---: | ---: |
| True value | 0.000026 | 0.000047 |
| Bias | 0.001477 | 0.004391 |
| SD | 0.000921 | 0.003750 |
| SE | 0.000146 | 0.000578 |
| SD(SE) | 0.000344 | 0.003780 |
| Coverage | 0.005300 | 0.012500 |

Conclusion: after matching R's generalized-inverse tolerance, the Python
workflow reproduces the CCG 2025 SieveVar reports at full `rep = 2000`,
`n = 1500` to the reported Monte Carlo precision.

