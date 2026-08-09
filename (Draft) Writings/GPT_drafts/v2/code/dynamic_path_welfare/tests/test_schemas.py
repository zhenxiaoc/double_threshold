import numpy as np
import pytest

from path_welfare.schemas import (SchemaError, apply_gates, continuity_report,
                                   path_counts, validate_schema)


def test_validate_schema_ok(sample_df):
    validate_schema(sample_df)


def test_validate_schema_nonbinary_treatment(sample_df):
    df = sample_df.copy()
    df.loc[0, "T1"] = 2
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_validate_schema_nonnumeric_state(sample_df):
    df = sample_df.copy()
    df["S"] = df["S"].astype(str)
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_continuity_report_flags_coarse():
    x = np.repeat(np.arange(3), 400)  # 3 unique values, high tie mass
    rep = continuity_report(x, "S")
    assert not rep.effectively_continuous
    assert rep.n_unique == 3


def test_continuity_report_continuous():
    rng = np.random.default_rng(1)
    rep = continuity_report(rng.normal(size=2000), "X")
    assert rep.effectively_continuous
    assert rep.max_point_mass <= 0.05


def test_gates_pass_on_good_sample(sample_df):
    g = apply_gates(sample_df, smallest_prob=0.5, min_units=1000)
    assert g.passed, g.failures


def test_gates_fail_small_n(sample_df):
    g = apply_gates(sample_df.iloc[:400], smallest_prob=0.5, min_units=1000)
    assert not g.passed
    assert any("units" in f for f in g.failures)


def test_path_counts_sum(sample_df):
    pc = path_counts(sample_df)
    assert sum(pc.values()) == len(sample_df)
