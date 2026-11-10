"""Tests for choose_order() and TVLP_formants() with low-frequency vocalizations
(very low F0 and formant frequencies, like elephant rumbles)."""
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

import phonlab as phon
from phonlab.acoustic.choose_order_ import choose_order

warnings.filterwarnings("ignore")

FORMANTS = [40, 170, 400, 650]
BANDWIDTHS = [20, 40, 60, 80]
LOW_FREQ = dict(tf=1000, order=10, preemph=0, length=0.3)  # settings for low-frequency input


def synth_rumble(f0=15, fs=16000, dur=3.0, jitter=0.01, snr_db=60, seed=0):
    """Jittered pulse train through a glottal-like tilt and a cascade of resonators at
    FORMANTS, scaled to a peak of 1, plus white noise snr_db below the peak."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    src = np.zeros(n)
    t = 0.0
    while t < dur:
        src[int(t * fs)] = 1.0
        t += (1 / f0) * (1 + jitter * rng.standard_normal())
    y = lfilter([1], [1, -0.97], src)
    for F, B in zip(FORMANTS, BANDWIDTHS):
        r = np.exp(-np.pi * B / fs)
        th = 2 * np.pi * F / fs
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    y = y / np.max(np.abs(y))
    return y + 10 ** (-snr_db / 20) * rng.standard_normal(n), fs


def formant_medians(df, dur, frame):
    d = df[df.sec < dur - frame]  # drop the zero-padded last frame
    return [np.nanmedian(d[f"F{i + 1}"]) for i in range(len(FORMANTS))]


# --- choose_order() ------------------------------------------------------

@pytest.mark.parametrize("fs", [2000, 3000, 4000, 6000])
def test_choose_order_low_sample_rates(fs):
    # the 120-3000 Hz bandpass used to fail for fs <= 6000 (upper edge >= Nyquist)
    x, _ = synth_rumble(fs=fs, dur=1.0)
    order, t = choose_order(x, fs)
    assert order >= 8 and order % 2 == 0
    assert 0 <= t <= 1.0


@pytest.mark.parametrize("where", ["start", "end"])
def test_choose_order_peak_at_edge(where):
    # a frame centered on a peak near the start of the signal used to be empty
    fs = 16000
    rng = np.random.default_rng(0)
    x = 1e-3 * rng.standard_normal(fs // 2)
    burst = np.sin(2 * np.pi * 500 * np.arange(80) / fs)
    if where == "start":
        x[:80] += burst
    else:
        x[-80:] += burst
    order, t = choose_order(x, fs)
    assert order >= 8 and order % 2 == 0


# --- TVLP_formants() -----------------------------------------------------

def test_TVLP_formants_low_tf_automatic_order():
    x, fs = synth_rumble(dur=1.0)
    df = phon.TVLP_formants(x, fs, tf=1000, order=-1, f0median=15, preemph=0,
                            length=0.3)
    assert len(df) > 0


def test_TVLP_formants_low_frequency_settings():
    dur = 3.0
    x, fs = synth_rumble(dur=dur)
    df = phon.TVLP_formants(x, fs, f0median=15, nformants=4, **LOW_FREQ)
    np.testing.assert_allclose(formant_medians(df, dur, 0.3), FORMANTS, rtol=0.1)


def test_TVLP_formants_min_freq_default_is_f0median():
    x, fs = synth_rumble(dur=1.0)
    np.random.seed(0)  # prep_audio() fills the zero padding with random noise
    a = phon.TVLP_formants(x, fs, f0median=15, **LOW_FREQ)
    np.random.seed(0)
    b = phon.TVLP_formants(x, fs, f0median=15, min_freq=15, **LOW_FREQ)
    cols = [c for c in a.columns if c != "sec"]
    np.testing.assert_array_equal(a[cols].to_numpy(), b[cols].to_numpy())


def test_TVLP_formants_min_freq_below_f0median():
    # with f0median above F1, F1 is dropped and the formants shift down a column,
    # unless min_freq is set below F1
    dur = 3.0
    x, fs = synth_rumble(dur=dur)
    shifted = formant_medians(phon.TVLP_formants(x, fs, f0median=45, **LOW_FREQ), dur, 0.3)
    fixed = formant_medians(phon.TVLP_formants(x, fs, f0median=45, min_freq=20, **LOW_FREQ),
                            dur, 0.3)
    assert shifted[0] == pytest.approx(FORMANTS[1], rel=0.1)
    assert fixed[0] == pytest.approx(FORMANTS[0], rel=0.1)
    assert fixed[1] == pytest.approx(FORMANTS[1], rel=0.1)
