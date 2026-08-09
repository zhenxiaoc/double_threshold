"""path_welfare
================

Plug-in estimation and *irregular* inference for path-specific welfare components
of an estimated optimal two-stage dynamic treatment regime.

Target parameter (see ``docs/theory_summary.md``):

    V_11^* = E[ Y^(1,1) * 1{delta(X^(1)) >= 0} * 1{kappa(S) >= 0} ]
           = \\int_{D1+} \\int_{D2+} mu_1(x) p_1(x|s) m(s) dx ds

which is the welfare contributed by the (1,1) treatment path under the optimal
regime -- NOT the value of a fixed policy and NOT the total optimal value V^*.

The public entry points are the estimator class and the simulation registry.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Path labels used everywhere. A path is (t1, t2).
PATHS = ((0, 0), (0, 1), (1, 0), (1, 1))
PATH_LABELS = {(0, 0): "00", (0, 1): "01", (1, 0): "10", (1, 1): "11"}

__all__ = ["PATHS", "PATH_LABELS", "__version__"]
