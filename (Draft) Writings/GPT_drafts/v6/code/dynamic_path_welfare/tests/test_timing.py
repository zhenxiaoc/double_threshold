
from path_welfare.schemas import check_timing


def test_timing_flags_S_equals_X(sample_df):
    df = sample_df.copy()
    df["X"] = df["S"]
    warns = check_timing(df)
    assert any("S == X" in w for w in warns)


def test_timing_clean(sample_df):
    warns = check_timing(sample_df)
    assert warns == []


def test_timing_flags_X_from_future_T2(sample_df):
    df = sample_df.copy()
    df["X"] = df["T2"].astype(float)  # X perfectly determined by future T2
    warns = check_timing(df)
    assert any("future" in w for w in warns)
