"""Tests for glottal closure instant (GCI) detection at low F0, and for the
pre-emphasis parameter of TVLP_formants()."""
import inspect
import warnings

import numpy as np
import pytest
from scipy.signal import buttord, butter, sosfiltfilt, filtfilt, windows, lfilter

import phonlab as phon
from phonlab.acoustic.gci import get_mbs

warnings.filterwarnings("ignore")


def synth_voice(f0, formants, bandwidths, fs, dur=2.0, jitter=0.01, seed=0):
    """Pulse train at f0 (with a little jitter) through a cascade of resonators."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    src = np.zeros(n)
    t = 0.0
    while t < dur:
        src[int(t * fs)] = 1.0
        t += (1 / f0) * (1 + jitter * rng.standard_normal())
    y = lfilter([1], [1, -0.97], src)  # glottal-like spectral tilt
    for F, B in zip(formants, bandwidths):
        r = np.exp(-np.pi * B / fs)
        th = 2 * np.pi * F / fs
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return y + 1e-4 * rng.standard_normal(n)


def old_get_mbs(x, fs, f0median, width=1.4):
    """get_mbs() as it was before the highpass cutoff depended on f0median."""
    t0mean = round(fs / f0median)
    halfL = round((0.5 * width) * t0mean)
    w = windows.blackman(2 * halfL + 1)
    mbs = filtfilt(w, 1, x)
    n, Wp = buttord(50, 30, 3, 60, fs=fs)
    sos = butter(n, Wp, 'hp', output='sos', fs=fs)
    mbs = sosfiltfilt(sos, mbs)
    return mbs / np.max(np.abs(mbs))


@pytest.mark.parametrize("f0median", [100, 150, 200, 300])
def test_get_mbs_unchanged_for_typical_speech_f0(f0median):
    fs = 16000
    x = synth_voice(f0median, [500, 1500, 2500], [80, 100, 150], fs, dur=0.5)
    np.testing.assert_array_equal(get_mbs(x, fs, f0median), old_get_mbs(x, fs, f0median))


@pytest.mark.parametrize("fs", [2000, 4000])
@pytest.mark.parametrize("f0", [5, 7.5, 10, 12.5, 15, 20, 25])
def test_gci_sedreams_tracks_very_low_f0(f0, fs):
    dur = max(2.0, 20 / f0)  # at least 20 glottal cycles
    x = synth_voice(f0, [40, 170, 400, 650], [20, 40, 60, 80], fs, dur=dur)
    df, mbs, resid = phon.gci_sedreams(x, fs, f0median=f0, target_fs=fs)
    expected = dur * f0
    assert abs(len(df) - expected) <= 2
    assert np.median(df.f0) == pytest.approx(f0, rel=0.05)


def test_TVLP_formants_pre_default():
    assert inspect.signature(phon.TVLP_formants).parameters["pre"].default == 0.95


def test_TVLP_formants_pre_is_used():
    fs = 12000
    x = synth_voice(120, [500, 1500, 2500, 3500], [80, 100, 150, 200], fs, dur=0.5)
    kw = dict(order=12, f0median=120, verbose=False)
    default = phon.TVLP_formants(x, fs, **kw)
    explicit = phon.TVLP_formants(x, fs, pre=0.95, **kw)
    none = phon.TVLP_formants(x, fs, pre=0, **kw)
    cols = [c for c in default.columns if c.startswith("F")]
    np.testing.assert_array_equal(default[cols].values, explicit[cols].values)
    assert not np.allclose(default[cols].values, none[cols].values, equal_nan=True)
