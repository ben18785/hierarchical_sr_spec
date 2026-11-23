"""Known-property constraints: exact limits by elimination, inequalities by truncation."""
import numpy as np

from hsr.candidates import skeleton_from_key
from hsr.constraints import Spec, impose_exact, prepare, satisfied, truncation_logratio
from hsr.likelihood import CompiledSkeleton

CDF = "range=0,1;limit_inf=1;monotone=increasing"


def test_parse():
    s = Spec.parse(CDF)
    assert (s.lower, s.upper, s.limit_inf, s.monotone) == (0.0, 1.0, 1.0, "increasing")
    assert s.fixes_y_scale and s.exact
    assert Spec.parse("nonneg").lower == 0.0 and not Spec.parse("nonneg").fixes_y_scale
    assert Spec.parse(None) is None


def test_impose_exact_limit():
    s = Spec.parse(CDF)
    assert impose_exact(skeleton_from_key("c0*x1/(c1 + x1)"), s) == {"x1/(c0 + x1)"}
    assert impose_exact(skeleton_from_key("c0*exp(c1*x1) + c2"), s) == {"c0*exp(c1*x1) + 1"}
    assert impose_exact(skeleton_from_key("c0 - exp(c1*x1)"), s) == {"1 - exp(c0*x1)"}
    assert impose_exact(skeleton_from_key("c0*x1"), s) == set()
    assert impose_exact(skeleton_from_key("c0*log(x1) + c1"), s) == set()


def test_satisfied_checks_signs_range_and_monotonicity():
    cs = CompiledSkeleton(skeleton_from_key("c0*exp(c1*x1) + 1"))
    grid = np.geomspace(0.01, 10, 50)
    phis = np.array([[-1.0, -0.5],    # 1 - exp(-x/2): a CDF
                     [-1.0, 0.5],     # diverges: wrong sign of the rate
                     [-2.0, -0.5],    # starts at -1: below the range
                     [1.0, -0.5]])    # decreasing towards 1
    assert satisfied(cs, phis, grid, Spec.parse(CDF)).tolist() == [True, False, False, False]


def test_truncation_penalises_forms_whose_fits_break_the_property():
    s = Spec.parse(CDF)
    grid = np.geomspace(0.01, 10, 50)
    cs = CompiledSkeleton(skeleton_from_key("c0*exp(c1*x1) + 1"))
    good = truncation_logratio(cs, np.array([-1.0, -0.5]), np.eye(3) * 1e-4, grid, s, tau=3.0)
    bad = truncation_logratio(cs, np.array([-1.0, 0.5]), np.eye(3) * 1e-4, grid, s, tau=3.0)
    assert good > 0          # posterior inside A, prior mostly outside: evidence rises
    assert bad == -np.inf    # posterior entirely outside A


def test_prepare_screens_added_skeletons():
    _, admit = prepare([], Spec.parse(CDF), scale_invariant=True)
    assert admit(skeleton_from_key("1 - exp(c0*x1**c1)"))
    assert admit(skeleton_from_key("x1/(c0 + x1)"))          # no amplitude needed when y is fixed
    assert not admit(skeleton_from_key("c0*x1/(c1 + x1)"))   # limit 1 only on a null set
    assert not admit(skeleton_from_key("exp(c0*x1)"))
    assert not admit(skeleton_from_key("c0*tanh(c1*x1)**c2"))    # limit c0: 1 only on a null set
    assert not admit(skeleton_from_key("tanh(c0*x1/(c1 + x1))"))  # limit tanh(c0): near 1, never equal
    assert admit(skeleton_from_key("(c0 + x1)/sqrt(c1 + x1**2)"))  # algebraic convergence to 1
