"""Shared Monte Carlo engine for OptTreat simulation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import re

import numpy as np
import pandas as pd

from opttreat.config import EstimatorConfig, ParameterConfig, ParameterType, VarianceConfig
from opttreat.data import common_support_mask, ensure_2d_features, normalize_treatment, split_treated_control
from opttreat.estimation import get_estimator
from opttreat.parameters import get_parameter
from opttreat.variance import get_variance_estimator


@dataclass(frozen=True)
class SimulationSpec:
    """One model/estimator/parameter/variance simulation design."""

    label: str
    model_factory: Callable[[], Any]
    parameter_config: ParameterConfig
    estimator_config: EstimatorConfig
    variance_config: VarianceConfig | None


@dataclass(frozen=True)
class SimulationRunConfig:
    """Run-size and reproducibility controls for a group of specs."""

    replications: int
    n_values: tuple[int, ...]
    seed: int
    jobs: int = 1
    progress_every: int = 100


def _evaluate_parameter(parameter: Any, param_type: ParameterType, h_hat: Any, X_eval: np.ndarray) -> float:
    """Evaluate known-distribution parameters without X and unknown ones with X."""
    if param_type in (ParameterType.WELFARE_UNKNOWN_DIST, ParameterType.VALUE_UNKNOWN_DIST):
        return float(parameter.evaluate(h_hat, X_eval))
    return float(parameter.evaluate(h_hat))


def _pooled_features_and_treatment(data: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract pooled features and treatment from formats that carry both."""
    if isinstance(data, pd.DataFrame):
        X = ensure_2d_features(data.filter(like="X").to_numpy(), name="X")
        d = normalize_treatment(data["d"].to_numpy(), X.shape[0])
        return X, d

    if isinstance(data, dict) and "X" in data and "d" in data:
        X = ensure_2d_features(data["X"], name="X")
        d = normalize_treatment(data["d"], X.shape[0])
        return X, d

    return None


def _evaluation_sample(
    data: Any,
    estimator_output: dict[str, Any],
    parameter_config: ParameterConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the X sample for unknown-distribution parameters."""
    X_eval = ensure_2d_features(estimator_output["X_all"], name="X_all")
    if not parameter_config.options.get("common_support", False):
        return X_eval, estimator_output

    pooled = _pooled_features_and_treatment(data)
    if pooled is None:
        raise ValueError("common_support=True requires pooled data with X and d.")

    X_pooled, d_pooled = pooled
    mask = common_support_mask(X_pooled, d_pooled, strict=True)
    X_eval = X_pooled[mask]
    if X_eval.shape[0] == 0:
        raise ValueError("common_support=True produced an empty evaluation sample.")

    output = dict(estimator_output)
    output["X_eval"] = X_eval
    return X_eval, output


def _summarize_draws(draws: pd.DataFrame, *, has_variance: bool, replications: int) -> dict[str, Any]:
    W_hat = draws["W_hat"].to_numpy(dtype=float)
    W_true = float(draws["W_true"].iloc[0])
    row: dict[str, Any] = {
        "spec": draws["spec"].iloc[0],
        "n": int(draws["n"].iloc[0]),
        "W_true": W_true,
        "W_mean": float(W_hat.mean()),
        "bias": float(W_hat.mean() - W_true),
        "sd": float(W_hat.std(ddof=1)) if W_hat.shape[0] > 1 else 0.0,
        "replications": replications,
    }

    if has_variance:
        se = draws["se"].to_numpy(dtype=float)
        row["se"] = float(se.mean())
        row["sd_se"] = float(se.std(ddof=1)) if se.shape[0] > 1 else 0.0
        row["coverage"] = float(np.mean((W_hat - 1.96 * se <= W_true) & (W_true <= W_hat + 1.96 * se)))

    return row


def run_simulation_specs(
    specs: list[SimulationSpec],
    run_config: SimulationRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run simulation specs and return summary and draw-level tables."""
    if run_config.replications <= 0:
        raise ValueError("SimulationRunConfig.replications must be positive.")
    if not run_config.n_values:
        raise ValueError("SimulationRunConfig.n_values must be nonempty.")

    rng_state = np.random.get_state()
    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []

    try:
        for spec_index, spec in enumerate(specs):
            parameter = get_parameter(spec.parameter_config)
            truth_model = spec.model_factory()
            W_true = float(parameter.get_true_value(truth_model))
            has_variance = spec.variance_config is not None

            for n_index, n in enumerate(run_config.n_values):
                if run_config.progress_every:
                    print(f"Running {spec.label} | n={n} | rep={run_config.replications}", flush=True)

                cell_rows: list[dict[str, Any]] = []
                seed_base = int(run_config.seed) + 100_000 * spec_index + 10_000 * n_index
                for rep in range(run_config.replications):
                    np.random.seed(seed_base + rep)
                    model = spec.model_factory()
                    data = model.generate_data(n)
                    parsed = split_treated_control(data)

                    estimator = get_estimator(spec.estimator_config)
                    estimator_output = estimator.fit(parsed)
                    X_eval, output_for_variance = _evaluation_sample(data, estimator_output, spec.parameter_config)

                    W_hat = _evaluate_parameter(
                        parameter,
                        spec.parameter_config.param_type,
                        estimator_output["h_hat"],
                        X_eval,
                    )
                    row: dict[str, Any] = {
                        "spec": spec.label,
                        "n": n,
                        "rep": rep,
                        "W_hat": W_hat,
                        "W_true": W_true,
                    }

                    if has_variance:
                        variance = get_variance_estimator(spec.variance_config)
                        if variance is None:
                            raise RuntimeError(f"{spec.label} expected a variance estimator.")
                        var_hat = float(variance.fit(output_for_variance))
                        row["se"] = float(np.sqrt(max(var_hat, 0.0)))

                    cell_rows.append(row)
                    if run_config.progress_every and (rep + 1) % run_config.progress_every == 0:
                        print(f"  {spec.label} n={n}: completed {rep + 1}/{run_config.replications}", flush=True)

                cell_df = pd.DataFrame(cell_rows)
                draw_rows.extend(cell_rows)
                summary_rows.append(
                    _summarize_draws(cell_df, has_variance=has_variance, replications=run_config.replications)
                )

        return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)
    finally:
        np.random.set_state(rng_state)


def _slug(value: Any) -> str:
    """Return a filename-safe token."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def _sequence_token(prefix: str, values: tuple[int, ...]) -> str:
    """Format integer tuples as tokens such as n1500_3000_6000."""
    return f"{prefix}{'_'.join(str(int(value)) for value in values)}"


def simulation_output_suffix(run_config: SimulationRunConfig, *extra_parts: Any) -> str:
    """Build the canonical simulation output suffix.

    The suffix always starts with the sample sizes and replication count:
    `n<values>_rep<count>`. Simulation-specific tags, such as feature count or
    model family, are appended afterward.
    """
    parts = [
        _sequence_token("n", run_config.n_values),
        f"rep{int(run_config.replications)}",
    ]
    parts.extend(_slug(part) for part in extra_parts if part not in (None, ""))
    return "_".join(parts)


def _settings_table(settings: dict[str, Any]) -> str:
    rows = [{"setting": key, "value": value} for key, value in settings.items()]
    return pd.DataFrame(rows).to_markdown(index=False)


def _summary_table(summary: pd.DataFrame) -> str:
    table = summary.copy()
    float_cols = table.select_dtypes(include=["floating"]).columns
    table[float_cols] = table[float_cols].round(6)
    return table.to_markdown(index=False)


def _bias_table(summary: pd.DataFrame) -> str | None:
    if "bias" not in summary.columns:
        return None

    table = summary.copy()
    table["abs_bias"] = table["bias"].abs()
    group_cols = [col for col in ["K", "expansion", "model", "spec"] if col in table.columns]
    if group_cols:
        out = table.groupby(group_cols, as_index=False)["abs_bias"].max()
    else:
        out = pd.DataFrame({"abs_bias": [table["abs_bias"].max()]})
    float_cols = out.select_dtypes(include=["floating"]).columns
    out[float_cols] = out[float_cols].round(6)
    return out.to_markdown(index=False)


def write_simulation_outputs(
    *,
    output_dir: Path,
    stem: str,
    summary: pd.DataFrame,
    draws: pd.DataFrame,
    run_config: SimulationRunConfig,
    suffix_parts: tuple[Any, ...] = (),
    settings: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Path]:
    """Write summary, draw-level, and Markdown report outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = simulation_output_suffix(run_config, *suffix_parts)

    summary_path = output_dir / f"{stem}_summary_{suffix}.csv"
    draws_path = output_dir / f"{stem}_draws_{suffix}.csv"
    report_path = output_dir / f"{stem}_results_{suffix}.md"

    summary.to_csv(summary_path, index=False)
    draws.to_csv(draws_path, index=False)

    settings = settings or {}
    notes = notes or []
    bias_table = _bias_table(summary)

    lines = [
        f"# {stem} Results",
        "",
        "## Run Settings",
        "",
        _settings_table(settings) if settings else "_No settings were provided._",
        "",
        "## Output Files",
        "",
        f"- Summary: `{summary_path.name}`",
        f"- Draws: `{draws_path.name}`",
        "",
        "## Summary Results",
        "",
        _summary_table(summary),
    ]
    if bias_table is not None:
        lines.extend(["", "## Maximum Absolute Bias", "", bias_table])
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": summary_path, "draws": draws_path, "report": report_path}


__all__ = [
    "SimulationSpec",
    "SimulationRunConfig",
    "run_simulation_specs",
    "simulation_output_suffix",
    "write_simulation_outputs",
]
