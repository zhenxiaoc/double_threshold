from __future__ import annotations

import numpy as np

from opttreat.simulations.TaylorModel import run_taylor_rf


def test_taylor_model_specs_and_smoke_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(run_taylor_rf, "EXPANSIONS", ("tan2", "sinh2", "rational", "hyperbolic", "exp_pm"))
    monkeypatch.setattr(run_taylor_rf, "K_VALUES", (10,))
    monkeypatch.setattr(run_taylor_rf, "REPLICATIONS", 2)
    monkeypatch.setattr(run_taylor_rf, "N_VALUES", (120,))
    monkeypatch.setattr(run_taylor_rf, "THETA_SOBOL", 128)
    monkeypatch.setattr(run_taylor_rf, "N_FEATURES", 20)
    monkeypatch.setattr(run_taylor_rf, "ALPHA", 1e-3)
    monkeypatch.setattr(run_taylor_rf, "PROGRESS_EVERY", 0)
    monkeypatch.setattr(run_taylor_rf, "OUTPUT_DIR", tmp_path)

    specs = run_taylor_rf.build_specs()
    assert [spec.label for spec in specs] == [
        "tan2_K10_p10",
        "sinh2_K10_p10",
        "rational_K10_p10",
        "hyperbolic_K10_p10",
        "exp_pm_K10_p10",
    ]

    summary, draws = run_taylor_rf.main()

    assert summary.shape[0] == 5
    assert draws.shape[0] == 10
    assert np.isfinite(summary[["W_true", "W_mean", "bias", "sd"]].to_numpy()).all()
    assert np.isfinite(draws["W_hat"].to_numpy()).all()
    assert {"se", "coverage"}.isdisjoint(summary.columns)
    assert (tmp_path / "TaylorModel_rf_summary_n120_rep2_nf20_K10.csv").exists()
    assert (tmp_path / "TaylorModel_rf_draws_n120_rep2_nf20_K10.csv").exists()
    assert (tmp_path / "TaylorModel_rf_results_n120_rep2_nf20_K10.md").exists()
