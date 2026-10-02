"""Tests for phonlab.utils.smooth1d_.smooth1d.

The golden data in test/golden/smooth1d_golden.npz were made with phonlab.smoothn (see
make_smooth1d_golden.py). smoothn iterates until its smooth changes by less than TolZ = 1e-3,
whereas smooth1d solves for the smooth directly, so they agree to about that tolerance.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phonlab.utils.smooth1d_ import smooth1d, _solve, _solve_iterative

sys.path.insert(0, str(Path(__file__).parent))
from make_smooth1d_golden import CASES, make_inputs  # noqa: E402

GOLDEN = np.load(Path(__file__).parent / "golden" / "smooth1d_golden.npz")
INPUTS = make_inputs()


#### Agreement with smoothn ####

# Allowed differences from smoothn (relative error in s, and max error in z as a fraction of the range
# of the smooth). The defaults are about smoothn's own convergence tolerance. The exception is the short
# curve with robust smoothing: the first smooth of its 32 points is at the lower bound for s (nearly
# interpolating the data), and the final s depends on how long the iteration for s is allowed to run.
# smoothn stops at s = 0.526 and smooth1d at 0.585 (continuing to convergence gives 0.597), and z is
# very sensitive to s there.
DEFAULT_TOL = dict(s=0.05, z=3e-3)
TOLERANCES = {"curve_robust": dict(s=0.15, z=8e-2)}

def test_golden_inputs_unchanged():
    """The generator still produces the signals the golden data were made from."""
    for name, (y, extra) in INPUTS.items():
        np.testing.assert_allclose(y, GOLDEN[f"in_{name}"], rtol=0, atol=1e-12, equal_nan=True)
        for k, v in extra.items():
            np.testing.assert_allclose(v, GOLDEN[f"in_{name}_{k}"], rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name", list(CASES))
def test_matches_smoothn(name):
    """z and s are close to those from smoothn."""
    inp, kwargs = CASES[name]
    y, extra = INPUTS[inp]
    tol = TOLERANCES.get(name, DEFAULT_TOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, s, exitflag = smooth1d(y, **extra, **kwargs)
    z_old, s_old = GOLDEN[f"z_{name}"], float(GOLDEN[f"s_{name}"])
    assert exitflag
    assert z.shape == z_old.shape
    if "s" in kwargs:
        assert s == kwargs["s"]
    else:
        assert s == pytest.approx(s_old, rel=tol["s"])
    assert np.max(np.abs(z - z_old)) / np.ptp(z_old) < tol["z"]


@pytest.mark.parametrize("name", ["curve_s5", "curve_auto", "long_auto"])
def test_matches_smoothn_closely_without_iteration(name):
    """With equal weights smoothn does not iterate, so the results are almost identical."""
    inp, kwargs = CASES[name]
    y, extra = INPUTS[inp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, s, _ = smooth1d(y, **kwargs)
    np.testing.assert_allclose(z, GOLDEN[f"z_{name}"], atol=1e-4 * np.ptp(z))


#### Behavior ####

def test_input_not_modified():
    """Missing values (and everything else) in the caller's array are left alone."""
    y = INPUTS["gappy"][0].copy()
    orig = y.copy()
    smooth1d(y, isrobust=True)
    np.testing.assert_array_equal(y, orig)   # equal_nan is the default for assert_array_equal


def test_no_nan_in_output_with_missing_data():
    """Missing values are filled in by the smooth."""
    y = INPUTS["gappy"][0]
    z, _, _ = smooth1d(y, isrobust=True)
    assert np.all(np.isfinite(z))


def test_robust_ignores_outliers():
    """The robust smooth is closer to the underlying curve than the ordinary smooth."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 300)
    truth = np.sin(t)
    y = truth + rng.normal(0, .1, len(t))
    y[[50, 120, 200]] += [4, -5, 6]
    z, s, _ = smooth1d(y, s=50)
    zr, sr, _ = smooth1d(y, s=50, isrobust=True)
    assert np.sqrt(np.mean((zr - truth) ** 2)) < 0.5 * np.sqrt(np.mean((z - truth) ** 2))


def test_larger_s_is_smoother():
    """Increasing s decreases the roughness of the smooth."""
    y = INPUTS["long"][0]
    rough = [np.sum(np.diff(smooth1d(y, s=s)[0], 2) ** 2) for s in (1, 10, 100, 1000)]
    assert rough == sorted(rough, reverse=True)


def test_zero_weight_is_like_missing():
    """A point with zero weight has no influence on the smooth, just like a NaN."""
    y = INPUTS["long"][0].copy()
    w = np.ones(len(y))
    w[100:120] = 0
    y_nan = y.copy()
    y_nan[100:120] = np.nan
    z_w, _, _ = smooth1d(y, s=100, W=w)
    z_nan, _, _ = smooth1d(y_nan, s=100)
    np.testing.assert_allclose(z_w, z_nan, atol=1e-8)


def test_weights_are_scale_invariant():
    """Only the relative sizes of the weights matter."""
    y, extra = INPUTS["long_W"]
    z1, _, _ = smooth1d(y, s=100, W=extra["W"])
    z2, _, _ = smooth1d(y, s=100, W=extra["W"] * 17)
    np.testing.assert_allclose(z1, z2, atol=1e-10)


def test_constant_signal():
    """A constant signal is returned as is, with no division by zero in the robust weights."""
    y = np.full(100, 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, s, _ = smooth1d(y, isrobust=True)
    np.testing.assert_allclose(z, 3.0, atol=1e-6)


def test_linear_signal_is_not_distorted_much_with_large_s():
    """Very large s tends toward the mean of the data (reflecting ends), and nothing breaks."""
    y = INPUTS["long"][0]
    ym = y.copy()
    ym[100:120] = np.nan
    for s in (1e5, 1e7, 1e11):
        z, _, _ = smooth1d(ym, s=s)
        assert np.all(np.isfinite(z))
        assert np.ptp(z) < np.ptp(y)


@pytest.mark.parametrize("s", [1e2, 1e4])
def test_iterative_and_direct_solutions_agree(s):
    """The fallback used for very large s finds the same smooth as the direct solution."""
    y = INPUTS["gappy"][0]
    finite = np.isfinite(y)
    w = finite.astype(float)
    y0 = np.where(finite, y, 0.0)
    z_direct = _solve(y0, w, s, False, y0)
    z_iter = _solve_iterative(y0, w, s, np.zeros(len(y)))
    np.testing.assert_allclose(z_iter, z_direct, atol=1e-5)


def test_bound_warning():
    """A warning says when the search for s ends up at a bound."""
    y = np.linspace(0, 1, 200)   # a straight line wants the smoothest possible smooth
    with pytest.warns(UserWarning, match="bound"):
        smooth1d(y)


def test_no_warning_for_ordinary_data():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smooth1d(INPUTS["long"][0], isrobust=True)


#### Input types and errors ####

def test_pandas_series():
    y = INPUTS["gappy"][0]
    z1, s1, _ = smooth1d(y, isrobust=True)
    z2, s2, _ = smooth1d(pd.Series(y, index=np.arange(len(y)) + 100), isrobust=True)
    assert isinstance(z2, np.ndarray)
    np.testing.assert_array_equal(z1, z2)


def test_masked_array():
    """Masked values are treated as missing."""
    y = INPUTS["long"][0]
    mask = np.zeros(len(y), dtype=bool)
    mask[100:120] = True
    z1, _, _ = smooth1d(np.ma.masked_array(y, mask), s=100)
    y_nan = y.copy()
    y_nan[mask] = np.nan
    z2, _, _ = smooth1d(y_nan, s=100)
    np.testing.assert_array_equal(z1, z2)


def test_list_input():
    z, _, _ = smooth1d([1., 2., 5., 3., 4., 6., 5.], s=1)
    assert z.shape == (7,)


def test_integer_input():
    z, _, _ = smooth1d(np.arange(20) % 3, s=1)
    assert z.dtype == np.float64


def test_short_input():
    """A single value has nothing to smooth."""
    z, s, e = smooth1d(np.array([2.0]))
    np.testing.assert_array_equal(z, [2.0])
    assert s is None
    z, s, e = smooth1d(np.array([2.0, 4.0, 3.0]), s=1)   # shortest input that is actually smoothed
    assert z.shape == (3,)
    z, s, e = smooth1d(np.array([2.0, 4.0]), s=1)
    assert z.shape == (2,)


def test_errors():
    y = INPUTS["long"][0]
    with pytest.raises(ValueError, match="one-dimensional"):
        smooth1d(np.ones((10, 10)))
    with pytest.raises(ValueError, match="positive"):
        smooth1d(y, s=0)
    with pytest.raises(ValueError, match="positive"):
        smooth1d(y, s=-1)
    with pytest.raises(ValueError, match="no finite"):
        smooth1d(np.full(10, np.nan))
    with pytest.raises(ValueError, match="weights"):
        smooth1d(y, W=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="weights"):
        smooth1d(y, W=-np.ones(len(y)))
