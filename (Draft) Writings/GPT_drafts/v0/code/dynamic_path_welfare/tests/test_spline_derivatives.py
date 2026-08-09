import numpy as np

from path_welfare.splines import fit_spline, make_basis


def test_basis_partition_of_unity():
    x = np.linspace(-2, 2, 50)
    basis = make_basis(x, 8)
    B = basis.design(x)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-8)  # clamped B-splines sum to 1


def test_analytic_derivative_matches_fd():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 2000)
    f = lambda x: np.sin(1.2 * x) + 0.3 * x ** 2
    y = f(x)
    fit = fit_spline(x, y, 10)
    xg = np.linspace(-1.8, 1.8, 40)
    d_an = fit.predict_deriv(xg)
    h = 1e-5
    d_fd = (fit.predict(xg + h) - fit.predict(xg - h)) / (2 * h)
    assert np.max(np.abs(d_an - d_fd)) < 1e-4


def test_second_derivative_shape():
    x = np.linspace(-2, 2, 500)
    basis = make_basis(x, 8)
    D2 = basis.design_deriv(x, order=2)
    assert D2.shape == (500, 8)
