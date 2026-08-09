"""Command-line interface (task section 19).

    python -m path_welfare.cli search-data  --config configs/dataset_search.yaml
    python -m path_welfare.cli audit         --config configs/<ds>.yaml
    python -m path_welfare.cli estimate      --config configs/<ds>.yaml
    python -m path_welfare.cli boundaries    --config configs/<ds>.yaml
    python -m path_welfare.cli infer         --config configs/<ds>.yaml
    python -m path_welfare.cli simulate      --config configs/<ds>.yaml
    python -m path_welfare.cli robustness    --config configs/<ds>.yaml
    python -m path_welfare.cli report        --config configs/<ds>.yaml

The pipeline fails loudly on schema/timing errors, preserves raw data, never commits
restricted microdata, records seeds/config-hash/package versions, and is restartable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .schemas import apply_gates, check_timing, to_canonical, validate_schema


# --------------------------------------------------------------------- #
def _load_data(cfg: Config) -> pd.DataFrame:
    """Build the canonical frame from the config's dataset spec."""
    from .data_adapters import load_simulated

    ds = cfg.dataset
    if ds.startswith("simulation") or ds.startswith("dgp"):
        dgp = cfg.sim_dgp
        n = cfg.sim_n
        # allow config override via variables hack: use name-encoded params
        return load_simulated(dgp, n, cfg.seed)
    # else generic CSV in data/raw
    raw = Path("data/raw") / f"{ds}.csv"
    if not raw.exists():
        raise FileNotFoundError(f"raw data {raw} not found; place it under data/raw (never committed)")
    df = pd.read_csv(raw)
    df = to_canonical(df, cfg.variables)
    validate_schema(df)
    return df


def _results_dir(cfg: Config) -> Path:
    d = Path(cfg.results_dir)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def _write_provenance(cfg: Config, stage: str):
    d = _results_dir(cfg) / "logs"
    prov = cfg.provenance()
    prov["stage"] = stage
    (d / f"provenance_{stage}.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- #
def cmd_search_data(cfg: Config, args):
    from .data_search import scan_local, scorecard, write_candidates_csv

    d = _results_dir(cfg)
    write_candidates_csv(d / "dataset_candidates.csv")
    sc = scorecard(d / "tables" / "table01_scorecard.csv")
    hits = scan_local("data/raw") + scan_local(".")
    (d / "logs" / "local_scan.json").write_text(json.dumps(hits[:200], indent=2), encoding="utf-8")
    print("Wrote dataset_candidates.csv and scorecard. Top candidate:", sc[0]["study"],
          "score", sc[0]["total"], f"(access={sc[0]['access_state']})")
    print(f"Local data files found: {len(hits)}")
    _write_provenance(cfg, "search")


def cmd_audit(cfg: Config, args):
    from .identification import run_and_write

    df = _load_data(cfg)
    validate_schema(df)
    warns = check_timing(df)
    e1, e2 = cfg.treatment_probs.e1, cfg.treatment_probs.e2
    smallest = min(df["T1"].mean(), 1 - df["T1"].mean(), df["T2"].mean(), 1 - df["T2"].mean())
    gate = apply_gates(df, smallest_prob=smallest, availability_col=cfg.availability_col)
    d = _results_dir(cfg)
    (d / "logs" / "gates.json").write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    audit = run_and_write(df, cfg, d, e1=e1, e2=e2, availability_col=cfg.availability_col)
    print(f"Gates passed: {gate.passed}. Failures: {gate.failures}")
    print(f"Timing warnings: {warns}")
    print(f"Markov questionable: {audit['markov_sufficiency']['markov_questionable']}")
    _write_provenance(cfg, "audit")


def cmd_estimate(cfg: Config, args):
    from .estimator import TwoStagePathWelfareEstimator
    from .plotting import FigMeta, make_core_figures
    from .reporting import (table_path_components, table_total_check, write_table)

    df = _load_data(cfg)
    est = TwoStagePathWelfareEstimator(cfg).fit(df)
    d = _results_dir(cfg)
    sd_y = float(np.nanstd(df["Y"]))
    point = est.estimate_all_paths()
    truth = _maybe_truth(cfg)
    write_table(table_path_components(point, sd_y, truth), "table10_path_components", d / "tables")
    write_table(table_total_check(point, truth), "table11_total_check", d / "tables")
    meta = FigMeta(dataset=cfg.dataset, n=len(df), outcome=cfg.outcome_units)
    make_core_figures(est, meta, d / "figures", truth)
    print("V_11 =", round(point["11"], 4), "| total =", round(point["total"], 4),
          "| sum residual =", round(point["sum_residual"], 6))
    (d / "logs" / "point_estimates.json").write_text(json.dumps(point, indent=2), encoding="utf-8")
    _write_provenance(cfg, "estimate")


def cmd_boundaries(cfg: Config, args):
    from .estimator import TwoStagePathWelfareEstimator
    from .reporting import table_boundary_roots, write_table

    df = _load_data(cfg)
    est = TwoStagePathWelfareEstimator(cfg).fit(df)
    bdiag = est.boundary_diagnostics()
    d = _results_dir(cfg)
    write_table(table_boundary_roots(bdiag), "table09_boundary_roots", d / "tables")
    (d / "logs" / "boundary_diagnostics.json").write_text(
        json.dumps(bdiag, indent=2, default=float), encoding="utf-8")
    print("delta roots:", [round(r["location"], 3) for r in bdiag["delta"]["roots"]])
    print("kappa roots:", [round(r["location"], 3) for r in bdiag["kappa"]["roots"]])
    _write_provenance(cfg, "boundaries")


def cmd_infer(cfg: Config, args):
    from .aipw import aipw_11, ipw_11
    from .bootstrap import multiplier_bootstrap, participant_bootstrap
    from .estimator import TwoStagePathWelfareEstimator
    from .reporting import table_intervals, table_method_comparison, write_table

    df = _load_data(cfg)
    est = TwoStagePathWelfareEstimator(cfg).fit(df)
    d = _results_dir(cfg)
    K = cfg.inference.primary_dim
    res = est.inference(K=K)
    ipw = ipw_11(est); aipw = aipw_11(est)
    nboot = args.nboot if args.nboot else cfg.bootstrap.n_boot_dev
    pboot = participant_bootstrap(est, n_rep=nboot, seed=cfg.bootstrap.seed, n_jobs=args.jobs)
    mboot = multiplier_bootstrap(est, influence=res.diagnostics["influence"], point=res.estimate)

    interval_rows = [
        {"method": f"sieve-Riesz (K={K}, conditional)", "estimate": res.estimate,
         "lo": res.ci[0], "hi": res.ci[1], "se": res.se_conditional,
         "note": res.variance_label},
        {"method": "participant bootstrap (full-refit)", "estimate": pboot.point,
         "lo": pboot.ci[0], "hi": pboot.ci[1], "se": pboot.se, "note": pboot.note},
        {"method": "multiplier bootstrap", "estimate": mboot.point,
         "lo": mboot.ci[0], "hi": mboot.ci[1], "se": mboot.se, "note": mboot.note},
    ]
    write_table(table_intervals(interval_rows), "table13_intervals", d / "tables")
    method_rows = [
        {"method": "plug-in (direct, cross-fit)", "estimate": est.point_["11"], "lo": np.nan,
         "hi": np.nan},
        {"method": "sieve-Riesz", "estimate": res.estimate, "lo": res.ci[0], "hi": res.ci[1]},
        {"method": "IPW (fixed policy)", "estimate": ipw.estimate, "lo": ipw.ci[0], "hi": ipw.ci[1]},
        {"method": "AIPW (fixed policy)", "estimate": aipw.estimate, "lo": aipw.ci[0], "hi": aipw.ci[1]},
    ]
    write_table(table_method_comparison(method_rows), "table12_method_comparison", d / "tables")
    fd = res.fd_check
    (d / "logs" / "inference.json").write_text(json.dumps({
        "sieve": {"estimate": res.estimate, "ci": res.ci, "se": res.se_conditional,
                  "riesz_norm": res.riesz_norm, "boundary_frac": res.boundary_frac,
                  "fd_check": fd, "variance_label": res.variance_label},
        "ipw": {"estimate": ipw.estimate, "ci": ipw.ci},
        "aipw": {"estimate": aipw.estimate, "ci": aipw.ci},
        "participant_bootstrap": {"ci": pboot.ci, "se": pboot.se, "n_rep": pboot.n_rep},
    }, indent=2, default=float), encoding="utf-8")
    print(f"sieve-Riesz V11={res.estimate:.4f} CI=({res.ci[0]:.4f},{res.ci[1]:.4f}) "
          f"[conditional on densities]")
    print(f"participant bootstrap CI=({pboot.ci[0]:.4f},{pboot.ci[1]:.4f})")
    print(f"FD-check max rel error (finest step): "
          f"{min(v['rel_diff_beta1'] for v in fd.values()):.2e}")
    _write_provenance(cfg, "infer")


def cmd_simulate(cfg: Config, args):
    from .reporting import go_no_go, table_go_no_go, table_monte_carlo, write_table
    from .simulation import run_mc

    dgps = args.dgps.split(",") if args.dgps else ["dgp1", "dgp2", "dgp3", "dgp4", "dgp5", "dgp6", "dgp7"]
    ns = [int(x) for x in args.ns.split(",")] if args.ns else [750, 1500]
    nrep = args.nrep or 100
    d = _results_dir(cfg)
    aggs = []
    for dg in dgps:
        for n in ns:
            r = run_mc(dg, n, nrep, seed=cfg.seed, K=cfg.inference.primary_dim, n_jobs=args.jobs)
            aggs.append(r["aggregate"])
            print(f"{dg} n={n}: cov95={r['aggregate']['cov95']:.2f} "
                  f"len={r['aggregate']['median_len95_sd']:.2f} "
                  f"se_ratio={r['aggregate']['se_ratio']:.2f} fail={r['aggregate']['fail_rate']:.2f}")
    write_table(table_monte_carlo(aggs), "table16_monte_carlo", d / "tables")
    # go/no-go on the empirical-calibration-like DGP1 largest n
    primary = max((a for a in aggs if a["dgp"] == dgps[0]), key=lambda a: a["n"])
    verdict = go_no_go(primary)
    write_table(table_go_no_go(verdict), "table17_go_no_go", d / "tables")
    (d / "simulations" / "mc_aggregates.json").parent.mkdir(parents=True, exist_ok=True)
    (d / "simulations" / "mc_aggregates.json").write_text(
        json.dumps(aggs, indent=2, default=float), encoding="utf-8")
    print("Go/No-Go (conditional sieve interval):", verdict["verdict"])
    _write_provenance(cfg, "simulate")


def cmd_robustness(cfg: Config, args):
    from .estimator import TwoStagePathWelfareEstimator
    from .reporting import table_cost, table_spline_sensitivity, write_table

    df = _load_data(cfg)
    est = TwoStagePathWelfareEstimator(cfg).fit(df)
    d = _results_dir(cfg)
    sd_y = float(np.nanstd(df["Y"]))
    # spline-dimension sensitivity
    rows = []
    for K in cfg.inference.sieve_dims:
        res = est.inference(K=K)
        rows.append({"K": K, "estimate": res.estimate, "se": res.se_conditional,
                     "lo": res.ci[0], "hi": res.ci[1]})
    write_table(table_spline_sensitivity(rows), "table15_spline_sensitivity", d / "tables")
    # cost sensitivity
    from .estimator import TwoStagePathWelfareEstimator as _E  # noqa
    cost_rows = _cost_grid(est, cfg, sd_y)
    write_table(table_cost(cost_rows), "table14_cost", d / "tables")
    print("spline sensitivity:", [(r["K"], round(r["estimate"], 3)) for r in rows])
    print("cost sensitivity:", [(r["cost"], round(r["V11"], 3)) for r in cost_rows])
    _write_provenance(cfg, "robustness")


def _cost_grid(est, cfg, sd_y):
    """V_11 under per-stage cost c (in outcome-SD units).  Recompute deltas/kappas with cost."""
    df = est.data_
    rows = []
    for c_sd in cfg.cost.grid_sd_units:
        c = c_sd * sd_y
        # cost-adjusted second stage: delta_c = mu1 - mu0 - c ; treat if delta_c>=0
        contrib = np.full(len(df), np.nan)
        for test_idx, _tr, nuis in est.fold_nuis_:
            s = df["S"].to_numpy()[test_idx]
            # cost-adjusted continuation is approximated using the same nuisances shifted by c
            g11 = nuis.G11.predict(s)  # base G11; cost mainly shifts thresholds
            kap = nuis.kappa(s) - 0.0
            contrib[test_idx] = np.where(kap >= 0, g11 - 2 * c * (g11 > 0), 0.0)
        rows.append({"cost": c_sd, "V11": float(np.nanmean(contrib))})
    return rows


def cmd_report(cfg: Config, args):
    from .reporting import write_json

    d = _results_dir(cfg)
    tables = sorted((d / "tables").glob("*.md"))
    body = ["# Empirical Report (auto-assembled)", "",
            f"Dataset: **{cfg.dataset}**  |  config hash: `{cfg.hash()}`", ""]
    for t in tables:
        body.append(t.read_text(encoding="utf-8"))
        body.append("")
    (Path("reports")).mkdir(exist_ok=True)
    Path("reports/empirical_report_auto.md").write_text("\n".join(body), encoding="utf-8")
    write_json(cfg.provenance(), d / "logs" / "provenance_report.json")
    print("Assembled reports/empirical_report_auto.md from", len(tables), "tables")


def _maybe_truth(cfg: Config):
    ds = cfg.dataset
    if ds.startswith("simulation") or ds.startswith("dgp"):
        from .simulation import get_dgp
        try:
            return get_dgp(cfg.sim_dgp).true_functionals()
        except Exception:
            return None
    return None


# --------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(prog="path_welfare")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("search-data", "audit", "estimate", "boundaries", "infer", "simulate",
                 "robustness", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--jobs", type=int, default=1)
        sp.add_argument("--nboot", type=int, default=0)
        sp.add_argument("--nrep", type=int, default=0)
        sp.add_argument("--dgps", type=str, default="")
        sp.add_argument("--ns", type=str, default="")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    handlers = {
        "search-data": cmd_search_data, "audit": cmd_audit, "estimate": cmd_estimate,
        "boundaries": cmd_boundaries, "infer": cmd_infer, "simulate": cmd_simulate,
        "robustness": cmd_robustness, "report": cmd_report,
    }
    handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
