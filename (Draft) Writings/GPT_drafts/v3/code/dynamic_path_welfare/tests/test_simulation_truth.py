
from path_welfare.simulation import get_dgp


def test_no_second_stage_boundary_has_no_delta_root():
    t = get_dgp("dgp4").true_functionals()
    assert len(t["roots_delta"]) == 0  # delta one-signed by construction


def test_no_first_stage_boundary_has_no_kappa_root():
    t = get_dgp("dgp5").true_functionals()
    assert len(t["roots_kappa"]) == 0  # kappa one-signed by construction


def test_regular_dgp_has_one_root_each():
    t = get_dgp("dgp1").true_functionals()
    assert len(t["roots_delta"]) == 1
    assert len(t["roots_kappa"]) == 1


def test_multi_root_dgp_has_two_delta_roots():
    t = get_dgp("dgp6").true_functionals()
    assert len(t["roots_delta"]) == 2


def test_weak_boundary_has_small_derivative():
    reg = get_dgp("dgp1").true_functionals()
    weak = get_dgp("dgp2").true_functionals()
    # the weak-delta DGP must have a shallower derivative at its root than the regular one
    assert abs(weak["delta_deriv_at_roots"][0]) < abs(reg["delta_deriv_at_roots"][0])


def test_truth_deterministic():
    a = get_dgp("dgp1").true_functionals()["V11"]
    b = get_dgp("dgp1").true_functionals()["V11"]
    assert a == b
