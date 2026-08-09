import numpy as np

from path_welfare.boundaries import classify_roots, find_roots


def test_find_roots_polynomial():
    f = lambda x: (x - 0.3) * (x + 0.7)  # roots at 0.3, -0.7
    roots = find_roots(f, -2, 2, n_grid=2001)
    assert len(roots) == 2
    assert min(abs(np.array(roots) - 0.3)) < 1e-4
    assert min(abs(np.array(roots) + 0.7)) < 1e-4


def test_no_roots_when_one_sign():
    f = lambda x: x ** 2 + 1.0
    roots = find_roots(f, -2, 2)
    assert roots == []


def test_classify_regular_root():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 3000)
    f = lambda x: 0.8 * np.asarray(x, float)  # root at 0, steep
    fp = lambda x: np.full(np.shape(x), 0.8)
    res = classify_roots("delta", f, fp, data)
    assert res.has_crossing
    assert len(res.roots) == 1
    assert res.roots[0].regular  # interior, steep, well supported


def test_classify_no_root_no_artifact():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 3000)
    f = lambda x: 0.5 + 0.0 * np.asarray(x, float)  # constant positive
    fp = lambda x: np.zeros(np.shape(x))
    res = classify_roots("delta", f, fp, data)
    assert not res.has_crossing
    assert res.roots == []


def test_multiple_roots():
    f = lambda x: 0.5 * (np.asarray(x, float) ** 2 - 1.0)  # roots +-1
    roots = find_roots(f, -2, 2)
    assert len(roots) == 2
