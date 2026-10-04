"""Tests for phonlab.prep_audio, including golden output comparisons."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from phonlab.utils.prep_audio_ import prep_audio

sys.path.insert(0, str(Path(__file__).parent))
from make_prep_audio_golden import CASES, FS, make_input  # noqa: E402

GOLDEN = np.load(Path(__file__).parent / "golden" / "prep_audio_golden.npz")


def dbfs(y):
    return 20 * np.log10(np.max(np.abs(y)))


@pytest.fixture
def x():
    return make_input()


#### Golden data ####

def test_golden_input_unchanged(x):
    """The generator still produces the signal the golden data was made from."""
    np.testing.assert_allclose(x, GOLDEN["input"], rtol=0, atol=1e-12)  # tolerate float noise across numpy/scipy versions


@pytest.mark.parametrize("name", list(CASES))
def test_golden(x, name):
    """prep_audio output matches the stored golden output."""
    y, fs = prep_audio(x, FS, add_tiny_noise=False, **CASES[name])
    assert fs == int(GOLDEN[name + "_fs"])
    assert y.shape == GOLDEN[name].shape
    assert y.dtype == GOLDEN[name].dtype
    np.testing.assert_allclose(y, GOLDEN[name], atol=1e-6)


#### scale ####

def test_scale_true_peaks_at_minus_1_dbfs(x):
    """scale=True puts the absolute peak at 0.89125."""
    y, _ = prep_audio(x, FS, target_fs=None, scale=True)
    assert np.max(np.abs(y)) == pytest.approx(0.89125)
    assert dbfs(y) == pytest.approx(-1.0, abs=1e-3)


def test_scale_uses_absolute_peak(x):
    """A negative peak larger than the positive peak sets the scaling."""
    assert -np.min(x) > np.max(x)
    y, _ = prep_audio(x, FS, target_fs=None, scale=True)
    assert np.min(y) == pytest.approx(-0.89125)
    assert np.max(y) < 0.89125


@pytest.mark.parametrize("db", [0, -3, -6, -20, -0.5])
def test_scale_dbfs(x, db):
    """A numeric scale value sets the absolute peak to that many dBFS."""
    y, _ = prep_audio(x, FS, target_fs=None, scale=db)
    assert dbfs(y) == pytest.approx(db, abs=1e-6)
    assert np.max(np.abs(y)) == pytest.approx(10 ** (db / 20))


def test_scale_numpy_bool(x):
    """np.True_ is treated like True, not as +1 dBFS."""
    y1, _ = prep_audio(x, FS, target_fs=None, scale=np.True_)
    y2, _ = prep_audio(x, FS, target_fs=None, scale=True)
    np.testing.assert_array_equal(y1, y2)


def test_scale_numpy_false(x):
    """np.False_ is treated like False, not as True."""
    y, _ = prep_audio(x, FS, target_fs=None, scale=np.False_)
    np.testing.assert_array_equal(y, x)


@pytest.mark.parametrize("off", [False, None])
def test_scale_off(x, off):
    """scale=False or None leaves the amplitude alone."""
    y, _ = prep_audio(x, FS, target_fs=None, scale=off)
    np.testing.assert_array_equal(y, x)


def test_scale_silence_is_not_nan():
    """An all-zero signal is not divided by its (zero) peak."""
    y, _ = prep_audio(np.zeros(1000), 16000, target_fs=None, scale=-3, add_tiny_noise=False)
    assert np.all(y == 0)
    y, _ = prep_audio(np.zeros(1000), 16000, target_fs=None, scale=True, add_tiny_noise=False)
    assert np.all(y == 0)


def test_input_not_modified(x):
    """prep_audio does not change the caller's array."""
    orig = x.copy()
    prep_audio(x, FS, target_fs=None, scale=-3)
    np.testing.assert_array_equal(x, orig)


#### integer conversion ####

def test_int_output_no_warning(x):
    """A signal within full scale converts to int16 without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, _ = prep_audio(x, FS, target_fs=None, scale=-1, outtype="int")
    assert y.dtype == np.int16
    assert np.max(np.abs(y)) == round(0.89125 * 32767)


def test_int_full_scale(x):
    """scale=0 reaches 32767 (full scale) without a warning or wraparound."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, _ = prep_audio(x, FS, target_fs=None, scale=0, outtype="int")
    assert np.max(np.abs(y)) == 32767


def test_int_clipping_warns_and_clips(x):
    """Positive gain beyond full scale warns, and clips rather than wrapping around."""
    with pytest.warns(UserWarning, match="clipped"):
        y, _ = prep_audio(x, FS, target_fs=None, scale=+3, outtype="int")
    assert y.dtype == np.int16
    assert np.min(y) == -32767          # the negative peak, clipped to full scale
    assert np.max(y) <= 32767
    assert np.all(np.sign(y[np.abs(x) > 0.15]) == np.sign(x[np.abs(x) > 0.15]))   # no sign flips from wraparound


def test_int_clipping_from_preemphasis_warns():
    """Pre-emphasis applied after scaling can push the peak over full scale."""
    nyquist = np.tile([1.0, -1.0], 500)   # pre-emphasis of 1.0 doubles the amplitude of this signal
    with pytest.warns(UserWarning, match="clipped"):
        y, _ = prep_audio(nyquist, FS, target_fs=None, scale=0, pre=1.0, outtype="int")
    assert np.max(np.abs(y)) == 32767


def test_float_output_not_clipped_or_warned(x):
    """Float output may exceed full scale: no clipping, no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, _ = prep_audio(x, FS, target_fs=None, scale=+6)
    assert dbfs(y) == pytest.approx(6.0, abs=1e-6)
