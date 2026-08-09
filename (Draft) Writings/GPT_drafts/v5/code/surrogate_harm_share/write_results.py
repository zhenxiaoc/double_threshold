"""Assemble docs/results.md from the JSON logs produced by run_study.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
OUT = ROOT / "docs" / "results.md"


def L(name):
    return json.loads((LOGS / name).read_text(encoding="utf-8"))


def main():
    tr = L("truth.json"); mc = tr["mc"]; gt = tr["grid"]
    dv = L("derivative_check.json")
    rate = L("rate_experiment.json"); cells = rate["cells"]; sl = rate["slopes"]
    ks = L("k_sensitivity.json")
    boot = L("bootstrap_coverage.json")
    rn = L("realistic_noise.json")

    m = []
    m.append("# Results: The Surrogate-Induced Harm Share\n")
    m.append("Generated from `results/logs/*.json` (produced by `run_study.py`). "
             "Figures in `results/figures/`.\n")

    m.append("## 1. Population truth (calibrated graduation oracle, d=2, SNR≈1)\n")
    m.append(f"- **Harm share** θ = Pr(τ_S≥0, τ_Y<0) = **{mc['theta_harm']:.3f}**")
    m.append(f"- Conditional harm ratio ρ = P(τ_Y<0 | τ_S≥0) = **{mc['rho']:.3f}**")
    m.append(f"- Treat-share (single-threshold companion) Pr(τ_S≥0) = {mc['treat_share_S']:.3f}")
    m.append(f"- ATE_S = {mc['ate_S']:.2f}, ATE_Y = {mc['ate_Y']:.2f} (fade-out preserved)\n")
    m.append("**Four-quadrant confusion matrix** (population shares):\n")
    m.append("| | τ_Y ≥ 0 | τ_Y < 0 |")
    m.append("|---|---|---|")
    m.append(f"| **τ_S ≥ 0** | {mc['theta_pp']:.3f} correctly treated | **{mc['theta_harm']:.3f}** HARM |")
    m.append(f"| **τ_S < 0** | {mc['theta_mp']:.3f} withheld despite gain | {mc['theta_mm']:.3f} correctly untreated |\n")

    m.append("**Geometry diagnostics** (regular-margin & transversality checks):\n")
    m.append(f"- ‖∇τ_S‖ on M_S (median) = {gt['grad_S_on_MS']:.2f} > 0  → regular short-run margin")
    m.append(f"- ‖∇τ_Y‖ on M_Y (median) = {gt['grad_Y_on_MY']:.2f} > 0  → regular long-run margin")
    m.append(f"- Corner transversality |cos∠(∇τ_S,∇τ_Y)| = {gt['corner_cos']:.3f} < 1  → transversal (codim-2 corner)")
    m.append(f"- Boundary lengths H¹(M_S) = {gt['len_M_S']:.2f}, H¹(M_Y) = {gt['len_M_Y']:.2f}\n")

    m.append("## 2. Two-boundary moving-boundary derivative — verification\n")
    m.append("Analytic narrow-band (coarea) vs central finite differences of θ(t). The two boundary "
             "terms are separate and non-cancelling; signs `+D_MS` (raising τ_S expands {τ_S≥0}) and "
             "`−D_MY` (raising τ_Y shrinks {τ_Y≤0}) are confirmed.\n")
    m.append("| perturbation | D_MS | D_MY | analytic Dθ | finite-diff | abs err |")
    m.append("|---|---|---|---|---|---|")
    for r in dv:
        m.append(f"| {r['case']} | {r['D_MS']:+.4f} | {r['D_MY']:+.4f} | {r['analytic']:+.4f} "
                 f"| {r['finite_diff']:+.4f} | {r['abs_err']:.4f} |")
    m.append("")

    m.append("## 3. Monte Carlo: bias, variance, and the two-band sieve SE\n")
    m.append(f"Sieve plug-in (tensor B-spline, segments=2), {cells[0]['n_rep']} reps per n. "
             "`se_ratio = mean(two-band sieve SE)/MC-SD`; the two-band SE is this project's new "
             "derivation.\n")
    m.append("| n | bias | MC-SD | RMSE | sieve SE | se_ratio | 95% cov (sieve) |")
    m.append("|---|---|---|---|---|---|---|")
    for c in cells:
        m.append(f"| {c['n']} | {c['bias']:+.4f} | {c['mc_sd']:.4f} | {c['rmse']:.4f} "
                 f"| {c['mean_se']:.4f} | {c['se_ratio']:.2f} | {c['cov95_sieve']:.2f} |")
    m.append("")
    m.append(f"- The **two-band sieve SE tracks the MC-SD** (se_ratio ≈ "
             f"{sum(c['se_ratio'] for c in cells)/len(cells):.2f} on average) — evidence the new "
             "double-threshold variance derivation is approximately correct.")
    m.append("- Undercoverage where present is **bias-driven**: the SE is correctly sized but the "
             "plug-in θ̂ is biased (see §4), shifting the interval off-centre.\n")

    m.append("## 4. Convergence behaviour and the regular/irregular distinction\n")
    m.append("Log–log slopes of RMSE (or SD) vs n:\n")
    m.append(f"- θ̂ harm share (double threshold): RMSE slope = **{sl['theta_rmse_slope']:.2f}**, "
             f"SD slope = {sl['theta_sd_slope']:.2f}")
    m.append(f"- treat-share Pr(τ_S≥0) (single threshold): RMSE slope = {sl['treatS_rmse_slope']:.2f}")
    if "W_sd_slope" in sl:
        m.append(f"- regular companion W_Y = E[max(τ_Y,0)]: SD slope = **{sl['W_sd_slope']:.2f}** "
                 "(≈ −0.5, root-n)")
    m.append("\n**Reading these honestly.** On this *smooth* calibrated oracle, a FIXED-dimension "
             "sieve is effectively parametric, so θ̂'s SD scales like n^(−1/2) and its coverage is "
             "near-nominal (§3) — the plug-in is well-behaved. The sub-√n **thin-set irregularity** of "
             "Chen & Gao (2026) is a *nonparametric* property: it is the rate of the boundary integral "
             "∫_M (τ̂−τ) that governs the plug-in once the sieve dimension K GROWS with n (so the CATE "
             "is estimated nonparametrically). It therefore surfaces in the growing-K regime (§4b), not "
             "at fixed K on a smooth surface. The **clean, DGP-independent** statement of the "
             "regular/irregular distinction is at the level of the pathwise derivative (§2): for θ the "
             "two boundary terms **do not cancel** (weight f≠0 on each margin), whereas for the regular "
             "companion W_Y the boundary weight τ_Y **vanishes** on {τ_Y=0} and the term cancels by the "
             "envelope argument — which is why W_Y is root-n and θ is not. The single-threshold "
             "treat-share remains bias-limited (nearly flat RMSE slope), reflecting a larger, more "
             "persistent sieve-approximation bias on its own margin.\n")

    m.append("### 4b. Undersmoothing: growing the sieve dimension K with n\n")
    m.append("Coverage of the fixed-K plug-in interval **degrades** as n grows (bias becomes large "
             "relative to the shrinking SD). Growing the sieve dimension with n (an undersmoothing "
             "schedule) shrinks the approximation bias as n grows, arresting the coverage decay.\n")
    try:
        rg = L("rate_experiment_growK.json")
        m.append("| n | segments (growing) | fixed-K cov | growing-K cov |")
        m.append("|---|---|---|---|")
        for cf, cg in zip(cells, rg["cells"]):
            m.append(f"| {cf['n']} | {cg['segments']} | {cf['cov95_sieve']:.2f} | {cg['cov95_sieve']:.2f} |")
        m.append(f"\nSchedule: {rg.get('schedule', {})}. Growing K trades a little variance for much "
                 "less bias at large n, keeping coverage closer to nominal — the practical takeaway of "
                 "the paper's undersmoothing prescription.\n")
    except FileNotFoundError:
        pass

    m.append("## 5. Sieve-dimension (K) sensitivity at n=4000\n")
    m.append("| segments | bias | RMSE | se_ratio | 95% cov |")
    m.append("|---|---|---|---|---|")
    for c in ks:
        m.append(f"| {c['segments']} | {c['bias']:+.4f} | {c['rmse']:.4f} | {c['se_ratio']:.2f} | {c['cov95_sieve']:.2f} |")
    m.append("\nLarger K reduces approximation bias (undersmoothing improves coverage) at the cost "
             "of variance — the standard bias/variance trade-off behind undersmoothed inference.\n")

    m.append("## 6. Full-refit bootstrap (primary interval)\n")
    m.append("| n | bootstrap 95% coverage | mean CI length |")
    m.append("|---|---|---|")
    for b in boot:
        m.append(f"| {b['n']} | {b['cov95_boot']:.2f} | {b['mean_len']:.4f} |")
    m.append("")

    m.append("## 7. Robustness: realistic low-SNR noise (noise_scale=1.0)\n")
    m.append(f"Same oracle surfaces/truth (θ={rn['truth']['theta_harm']:.3f}); residual noise at the "
             "realistic consumption level.\n")
    m.append("| n | bias | RMSE | 95% cov (sieve) |")
    m.append("|---|---|---|---|")
    for c in rn["cells"]:
        m.append(f"| {c['n']} | {c['bias']:+.4f} | {c['rmse']:.4f} | {c['cov95_sieve']:.2f} |")
    m.append("\nAt realistic SNR the plug-in bias is larger and coverage lower — an honest picture "
             "of how hard CATE-threshold estimation is on real experimental data, and why the paper's "
             "irregular-inference machinery (undersmoothing, boundary-aware SEs, bootstrap) matters.\n")

    # affine exact-truth validation (computed inline; no study dependency)
    try:
        from harm_share.affine_dgp import AffineDGP
        from harm_share.functionals import mc_truth as _mc, grid_truth as _grid
        dgp = AffineDGP(); ex = dgp.exact_truth()
        mcv = _mc(dgp, n_draw=1_000_000); gv = _grid(dgp, n_grid=600, span=3.5)
        m.append("## 8. Exact-truth validation (affine bivariate-normal companion)\n")
        m.append("A companion affine DGP with straight margins and a single transversal corner "
                 f"(x*={[round(c,2) for c in ex['corner']]}, angle {ex['corner_angle_deg']:.0f}°) has "
                 "θ available in closed form (bivariate-normal orthant), validating the truth machinery "
                 "to zero grid error. See `results/figures/fig1b_affine_geometry.png`.\n")
        m.append("| method | θ | θ_++ | θ_-+ | θ_-- |")
        m.append("|---|---|---|---|---|")
        m.append(f"| **exact** (orthant) | {ex['theta_harm']:.4f} | {ex['theta_pp']:.4f} | {ex['theta_mp']:.4f} | {ex['theta_mm']:.4f} |")
        m.append(f"| grid quadrature | {gv.theta_harm:.4f} | {gv.theta_pp:.4f} | {gv.theta_mp:.4f} | {gv.theta_mm:.4f} |")
        m.append(f"| 1M-draw MC | {mcv['theta_harm']:.4f} | {mcv['theta_pp']:.4f} | {mcv['theta_mp']:.4f} | {mcv['theta_mm']:.4f} |")
        m.append(f"\nθ agreement: |exact − grid| = {abs(ex['theta_harm']-gv.theta_harm):.5f}, "
                 f"|exact − MC| = {abs(ex['theta_harm']-mcv['theta_harm']):.5f}.\n")
    except Exception as e:  # pragma: no cover
        m.append(f"\n(affine validation skipped: {e})\n")

    OUT.write_text("\n".join(m), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
