from __future__ import annotations

import numpy as np

from opttreat.models import TaylorExpansionModel
from opttreat.simulations.high_D_tan2 import run_high_d_tan2_rf as high_d


def test_high_d_tan2_uses_explicit_taylor_specs() -> None:
    specs = high_d.build_specs()
    assert [spec.label for spec in specs] == ["tan2_K3_p3", "tan2_K7_p7", "tan2_K10_p10"]

    models = [spec.model_factory() for spec in specs]
    assert all(isinstance(model, TaylorExpansionModel) for model in models)
    assert [(model.K, model.p, model.expansion) for model in models] == [(3, 3, "tan2"), (7, 7, "tan2"), (10, 10, "tan2")]


def test_high_d_tan2_rf_smoke_has_estimation_only_outputs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(high_d, "OUTPUT_DIR", tmp_path)
    summary, draws = high_d.main()

    assert summary.shape[0] == 3
    assert draws.shape[0] == 6
    assert set(summary["K"]) == {3, 7, 10}
    assert set(summary["p"]) == {3, 7, 10}
    assert np.allclose(summary["W_true"], 1.0)
    assert np.allclose(draws["W_true"], 1.0)
    assert np.isfinite(summary[["W_mean", "bias", "sd"]].to_numpy()).all()
    assert np.isfinite(draws["W_hat"].to_numpy()).all()

    inference_columns = {"se", "coverage", "ci_l", "ci_u", "lower", "upper"}
    assert inference_columns.isdisjoint(summary.columns)
    assert inference_columns.isdisjoint(draws.columns)
    assert (tmp_path / "high_D_tan2_rf_summary_n120_rep2_nf40_K3_7_10.csv").exists()
    assert (tmp_path / "high_D_tan2_rf_draws_n120_rep2_nf40_K3_7_10.csv").exists()
    assert (tmp_path / "high_D_tan2_rf_results_n120_rep2_nf40_K3_7_10.md").exists()
