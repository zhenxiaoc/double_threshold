"""Plot normalized empirical distributions of Monte Carlo estimates against N(0,1).

Usage examples:

python experiments/plot_estimates.py \
    --npz experiments/tmp_results/results_t_learner_linear_n200_R10.npz \
    --variable tau_S_hat --normalize --bins 50 --output experiments/tmp_results/tauS_norm.png

python experiments/plot_estimates.py --npz experiments/results/results_t_learner_linear_n200_R6.npz --variable W_hat --normalize
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _std_normal_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def plot_normalized_distribution(npz_path, variable="tau_S_hat", normalize=True, bins=50, output=None):
    data = np.load(npz_path)

    if variable not in data:
        raise ValueError(f"Variable '{variable}' not found in {npz_path}. Available keys: {list(data.keys())}")

    arr = data[variable]
    # For tau estimates, if true values are available, compute estimation errors
    if variable == "tau_S_hat" and "tau_S_true" in data:
        vals = (arr - data["tau_S_true"]).ravel()
        title_extra = "(est - true) for Tau S"
    elif variable == "tau_Y_hat" and "tau_Y_true" in data:
        vals = (arr - data["tau_Y_true"]).ravel()
        title_extra = "(est - true) for Tau Y"
    elif variable == "W_hat" and "W_true" in data:
        vals = (arr - data["W_true"]).ravel()
        title_extra = "(est - true) for Welfare"
    else:
        vals = arr.ravel()
        title_extra = variable

    # Normalize if requested
    if normalize:
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1)
        if sd == 0:
            raise RuntimeError("Standard deviation is zero; cannot normalize")
        vals_norm = (vals - mu) / sd
        xlabel = "Normalized value (sample mean subtracted; sample SD)"
    else:
        vals_norm = vals
        xlabel = "Value"

    # Prepare plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals_norm, bins=bins, density=True, alpha=0.6, color="C0", label="Empirical")

    # Overlay standard normal pdf
    x = np.linspace(-4, 4, 400)
    ax.plot(x, _std_normal_pdf(x), "k--", lw=1.5, label="Standard Normal PDF")

    ax.set_title(f"Normalized distribution of {title_extra}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()

    # Add text with sample moments
    txt = f"n={len(vals):,}\nmean={np.mean(vals):+.4f}\nsample_sd={np.std(vals, ddof=1):.4f}"
    ax.text(0.95, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if output is None:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        output = os.path.join(os.path.dirname(npz_path), f"{base}_{variable}_norm.png")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
    plt.show()  # Show the plot as a pop-up window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="NPZ file produced by monte carlo script")
    parser.add_argument("--variable", choices=["tau_S_hat", "tau_Y_hat", "W_hat", "gaps"], default="tau_S_hat")
    parser.add_argument("--normalize", action="store_true", help="Subtract mean and divide by sample SD")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()
    plot_normalized_distribution(args.npz, variable=args.variable, normalize=args.normalize, bins=args.bins, output=args.output)


if __name__ == "__main__":
    main()
