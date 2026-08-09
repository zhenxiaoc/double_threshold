import numpy as np
import pytest

from path_welfare.config import Config
from path_welfare.simulation import get_dgp


@pytest.fixture
def dgp1():
    return get_dgp("dgp1")


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    df = get_dgp("dgp1").sample(1200, rng)
    df["group"] = np.arange(len(df))
    return df


@pytest.fixture
def cfg():
    return Config(name="test", treatment_probs={"e1": 0.5, "e2": 0.5})
