import os
import sys

# Ensure project root is on sys.path when running the script directly.
# When running "python experiments/monte_carlo_convergence.py", sys.path[0] is the
# experiments directory, so sibling packages (e.g., `dgp`) aren't found. Add
# the project root so top-level package imports work.
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import numpy as np
from dgp.dgp import generate_data
from estimation.t_learner import fit_t_learner
from welfare.welfare import true_welfare, plugin_welfare


def monte_carlo(n, R=200, method="t_learner", t_estimator="linear", out_dir=None, show_progress=True, progress_percent=10, seed=0, delta=0.9, true_welfare_val=None):
    """Add `seed` (int) to set a reproducible base RNG. Per-run seeds are derived as `seed + r`."""
    """Run Monte Carlo and collect CATE estimates and welfare gaps.

    Parameters
    - show_progress: whether to print periodic progress updates
    - progress_percent: frequency of updates in percent (e.g., 10 => every 10%)

    Returns a dict with keys:
      - tau_S_hat: array shape (R, n)
      - tau_Y_hat: array shape (R, n)
      - tau_S_true: array shape (R, n)
      - tau_Y_true: array shape (R, n)
      - W_hat: array shape (R,)
      - W_true: array shape (R,)
      - gaps: array shape (R,)
    """
    if show_progress:
        if not (isinstance(progress_percent, int) and 1 <= progress_percent <= 100):
            raise ValueError("progress_percent must be an int between 1 and 100")
        step = max(1, int(np.ceil(R * (progress_percent / 100.0))))
        next_report_at = step

    gaps = np.empty(R, dtype=float)
    W_hat_arr = np.empty(R, dtype=float)

    tau_S_hat = np.empty((R, n), dtype=float)
    tau_Y_hat = np.empty((R, n), dtype=float)
    tau_S_true = np.empty((R, n), dtype=float)
    tau_Y_true = np.empty((R, n), dtype=float)

    for r in range(R):
        run_seed = None if seed is None else int(seed) + r
        df, delta = generate_data(n=n, seed=run_seed)

        tau_S_true[r, :] = df.tau_S.values
        tau_Y_true[r, :] = df.tau_Y.values

        # Choose estimator
        if method == "causal_forest":
            try:
                from estimation.causal_forest import fit_causal_forest
            except Exception as e:
                raise RuntimeError("Failed to import causal forest (is econml installed?).") from e

            cf_S = fit_causal_forest(df, "S", random_state=run_seed)
            cf_Y = fit_causal_forest(df, "Y", random_state=run_seed + 1000 if run_seed is not None else None)
        elif method == "t_learner":
            cf_S = fit_t_learner(df, "S", estimator=t_estimator, random_state=run_seed)
            cf_Y = fit_t_learner(df, "Y", estimator=t_estimator, random_state=run_seed + 1000 if run_seed is not None else None)
        else:
            raise ValueError("Unknown method: choose 'causal_forest' or 't_learner'.")

        X = df[["X1", "X2"]].values
        tau_S_hat[r, :] = cf_S.effect(X)
        tau_Y_hat[r, :] = cf_Y.effect(X)

        # Only compute plugin (estimated) welfare; true welfare is fixed and computed once outside
        W_hat = plugin_welfare(tau_S_hat[r, :], tau_Y_hat[r, :], delta)
        W_hat_arr[r] = W_hat
        # Use the fixed true_welfare_val for all runs
        true_val = true_welfare_val if true_welfare_val is not None else 0.0
        gaps[r] = W_hat - true_val

        # Progress reporting
        if show_progress:
            if (r + 1) < R and (r + 1) >= next_report_at:
                pct = int(round(100.0 * (r + 1) / R))
                print(f"Progress: {pct}% ({r+1}/{R})", flush=True)
                next_report_at += step
            elif (r + 1) == R:
                print(f"Progress: 100% ({R}/{R})", flush=True)

    results = {
        "tau_S_hat": tau_S_hat,
        "tau_Y_hat": tau_Y_hat,
        "tau_S_true": tau_S_true,
        "tau_Y_true": tau_Y_true,
        "W_hat": W_hat_arr,
        # "W_true" removed, now only a fixed value is used
        "gaps": gaps,
        "method": method,
        "t_estimator": t_estimator,
        "n": n,
        "R": R,
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"results_{method}_{t_estimator}_n{n}_R{R}.npz")
        np.savez_compressed(fname, **results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo convergence experiment")
    parser.add_argument("--method", choices=["causal_forest", "t_learner"], default="t_learner", help="CATE estimation method to use (default: t_learner)")
    parser.add_argument("--t-estimator", choices=["linear", "spline"], default="linear", help="Base learner for T-learner (if selected)")
    parser.add_argument("--R", type=int, default=200, help="Number of Monte Carlo repetitions")
    parser.add_argument("--ns", nargs="*", type=int, default=[1000, 2000, 5000, 10000], help="List of sample sizes to run")
    parser.add_argument("--out-dir", type=str, default="experiments/results", help="Directory to save full results (npz)")
    parser.add_argument("--delta", type=float, default=0.9, help="Discount value delta for welfare calculation")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for simulation runs (per-run seed = seed + r). Use blank or omit for non-fixed seeds by setting to None manually)")
    parser.add_argument("--progress", action="store_true", help="Show progress updates during runs")
    parser.add_argument("--progress-percent", type=int, default=10, help="Progress update frequency as a percent (1-100)")

    args = parser.parse_args()


    # Collect results for all configurations
    import matplotlib.pyplot as plt
    all_results = []
    vals_list = []
    ns_list = []
    # Compute true welfare parameter once using a large sample
    from welfare.welfare import true_welfare
    df_true, _ = generate_data(n=100000, seed=12345)
    true_param = true_welfare(df_true.tau_S.values, df_true.tau_Y.values, args.delta)

    for n in args.ns:
        results = monte_carlo(n, R=args.R, method=args.method, t_estimator=args.t_estimator, out_dir=args.out_dir, show_progress=args.progress, progress_percent=args.progress_percent, seed=args.seed, delta=args.delta, true_welfare_val=true_param)
        gaps = results["gaps"]
        all_results.append({
            "method": args.method,
            "t_estimator": args.t_estimator,
            "n": n,
            "R": args.R,
            "W_Bias": gaps.mean(),
            "W_SD_sample": np.std(gaps, ddof=1),
            "True_Welfare": true_param
        })
        vals = (gaps - gaps.mean()) / (np.std(gaps, ddof=1) if np.std(gaps, ddof=1) > 0 else 1)
        vals_list.append(vals)
        ns_list.append(n)

    # Print a summary table (without result file)
    print("\nSummary of Monte Carlo Results:")
    print(f"{'Method':<12} {'T_Est':<10} {'n':>6} {'R':>5} {'W_Bias':>10} {'W_SD':>10} {'True_W':>12}")
    print("-"*75)
    for res in all_results:
        print(f"{res['method']:<12} {res['t_estimator']:<10} {res['n']:>6} {res['R']:>5} {res['W_Bias']:>10.4f} {res['W_SD_sample']:>10.4f} {res['True_Welfare']:>12.4f}")

    # Plot all standardized distributions in one figure with multiple panels
    if vals_list:
        n_panels = len(vals_list)
        ncols = min(2, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 4*nrows))
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        x = np.linspace(-4, 4, 400)
        for i, (vals, n) in enumerate(zip(vals_list, ns_list)):
            ax = axes[i]
            ax.hist(vals, bins=50, density=True, alpha=0.6, color="C0", label="Empirical")
            ax.plot(x, (np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)), "k--", lw=1.5, label="Standard Normal PDF")
            ax.set_title(f"Normalized distribution of gaps (n={n})")
            ax.set_xlabel("Normalized value (sample mean subtracted; sample SD)")
            ax.set_ylabel("Density")
            ax.legend()
            # Report mean and sample SD of the standardized empirical distribution (should be close to 0 and 1, but report actual values)
            emp_mean = np.mean(vals)
            emp_sd = np.std(vals, ddof=1)
            txt = f"Empirical (standardized):\nmean={emp_mean:+.4f}\nsample_sd={emp_sd:.4f}\nn={len(vals):,}"
            ax.text(0.95, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        # Hide unused axes if any
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        fig.tight_layout()
        plt.show()
