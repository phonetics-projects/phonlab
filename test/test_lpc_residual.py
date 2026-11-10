"""Tests for lpcresidual() when Burg's method returns invalid LPC coefficients
for some frames (low-F0 voices at fs=12000, telephone-band speech)."""
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter, resample_poly

import phonlab as phon
import phonlab.acoustic.lpc_residual as lr
from phonlab.acoustic.lpc_residual import _invalid_lpc

warnings.filterwarnings("ignore")
AUDIO = Path(phon.__file__).parent / "data" / "example_audio"


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


def _lpc_with_failures(real_lpc, bad_rows, bad_value, fail_float64):
    """Wrap librosa.lpc so the float32 call returns garbage coefficients (bad_value: NaN,
    or finite but huge) for bad_rows, as Burg's method can for badly conditioned float32
    frames, and optionally so the float64 retry fails too."""
    def wrapped(frames, order, **kw):
        A = real_lpc(frames, order=order, **kw)
        if frames.dtype == np.float32:
            A[bad_rows, 1:] = bad_value
        elif fail_float64:
            A[:, 1:] = bad_value
        return A
    return wrapped


@pytest.mark.parametrize("bad_value", [np.nan, 1e38])
@pytest.mark.parametrize("fail_float64", [False, True])
def test_lpcresidual_survives_bad_lpc_frames(monkeypatch, bad_value, fail_float64):
    fs = 12000
    x = synth_voice(120, [500, 1500, 2500, 3500], [80, 100, 150, 200], fs, dur=1.0)
    ref, _ = lr.lpcresidual(x, fs, fs, order=14)
    monkeypatch.setattr(lr, "lpc", _lpc_with_failures(lr.lpc, [5, 50, 100], bad_value, fail_float64))
    resid, _ = lr.lpcresidual(x, fs, fs, order=14)
    assert np.isfinite(resid).all()
    if not fail_float64:  # frames redone in float64 should match the float32 result closely
        np.testing.assert_allclose(resid, ref, atol=1e-3)


def test_invalid_lpc_bound():
    order = 4
    stable = np.poly([0.9, -0.9, 0.5 + 0.5j, 0.5 - 0.5j]).real  # roots inside unit circle
    edge = np.poly([0.999] * order).real                          # |a_k| just below C(4,k)
    unstable = np.poly([5.0, 0.2, 0.1, -0.3]).real                # root far outside -> |a_1| = 5 > C(4,1)
    A = np.array([stable, edge, unstable, [1, np.nan, 0, 0, 0]], dtype=np.float32)
    assert _invalid_lpc(A, order).tolist() == [False, False, True, True]


@pytest.mark.parametrize("f0", [5, 15, 25])
def test_lpcresidual_finite_for_low_f0_voice_at_12k(f0):
    # A low-F0 voice with no energy above ~700 Hz, analyzed at fs=12000 with order 14
    # (as gci() does inside TVLP_formants() with the default tf=6000). With
    # numba-compiled librosa.lpc in float32, some frames get garbage coefficients, which
    # used to make the entire residual NaN.
    fs = 12000
    x = synth_voice(f0, [40, 170, 400, 650], [20, 40, 60, 80], fs, dur=max(2.0, 20 / f0))
    resid, _ = lr.lpcresidual(x, fs, fs, order=14)
    assert np.isfinite(resid).all()
    assert np.max(np.abs(resid)) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["three_plus_five.wav", "two_plus_two.wav"])
def test_lpcresidual_telephone_band_speech(name):
    # Speech with no energy above 4 kHz (8 kHz audio upsampled to 16 kHz, which is what
    # lpcresidual() does by default with telephone recordings) gives invalid float32 LPC
    # fits in some frames. Those frames used to dominate the residual and crush the rest
    # of it to ~0, so gci(cthresh=0.01) rejected nearly all GCIs.
    x, fs = phon.loadsig(str(AUDIO / name), chansel=[0], fs=16000)
    tel = resample_poly(resample_poly(x.astype(float), 1, 2), 2, 1)
    resid, _ = lr.lpcresidual(tel, fs, fs, order=18)
    assert np.isfinite(resid).all()
    assert np.mean(np.abs(resid) < 1e-6) < 0.01
    wide, _, _ = phon.gci(x, fs, f0median=150, target_fs=fs, cthresh=0.01)
    narrow, _, _ = phon.gci(tel, fs, f0median=150, target_fs=fs, cthresh=0.01)
    assert len(narrow) >= 0.5 * len(wide)


# ---- the `gain` and `normalize` parameters ----

def _sf3():
    import importlib.resources as res
    x, fs = phon.loadsig(res.files("phonlab") / "data" / "example_audio" / "sf3_cln.wav", chansel=[0])
    return x, fs


def _level_db(s, n=640):
    return np.array([20 * np.log10(np.sqrt(np.mean(s[i:i + n] ** 2)) + 1e-9) for i in range(0, len(s) - n, n)])


def test_gain_speech_makes_the_residual_level_follow_the_speech_level():
    x, fs = _sf3()
    xs, _ = phon.prep_audio(x, fs, target_fs=16000, preemph=0)
    speech = _level_db(xs)
    follows = np.corrcoef(_level_db(phon.lpcresidual(x, fs, gain="speech")[0]), speech)[0, 1]
    raw = np.corrcoef(_level_db(phon.lpcresidual(x, fs, gain=None)[0]), speech)[0, 1]
    assert follows > 0.95 and raw < 0.9      # the raw prediction error is much smaller than the speech in voiced stretches


def test_gain_speech_is_the_default():
    x, fs = _sf3()
    np.random.seed(0)   # prep_audio() puts random noise in zero samples
    a = phon.lpcresidual(x, fs)[0]
    np.random.seed(0)
    np.testing.assert_array_equal(a, phon.lpcresidual(x, fs, gain="speech")[0])


def test_normalize_false_is_the_residual_without_the_peak_scaling():
    x, fs = _sf3()
    for gain in ("speech", None):
        np.random.seed(0)
        unnorm, _ = phon.lpcresidual(x, fs, gain=gain, normalize=False)
        np.random.seed(0)
        norm, _ = phon.lpcresidual(x, fs, gain=gain)
        assert np.max(np.abs(norm)) == pytest.approx(1.0)
        np.testing.assert_allclose(norm, unnorm / np.max(np.abs(unnorm)), rtol=1e-4, atol=1e-6)


def test_raw_residual_resynthesizes_the_audio():
    """With gain=None, normalize=False the residual is the prediction error: the LPC filters turn it back into speech."""
    from librosa import util, lpc
    from scipy.signal import lfilter, windows
    from phonlab.acoustic.lpc_residual import overlap_add
    x, fs = _sf3()
    resid, rfs = phon.lpcresidual(x, fs, gain=None, normalize=False)
    xs, _ = phon.prep_audio(x, fs, target_fs=16000, preemph=0, fix_polarity=True)
    N, hop = 640, 80
    frames = util.frame(xs.astype(np.float64), frame_length=N, hop_length=hop, axis=0)
    A = lpc(frames, order=18)
    w = windows.get_window("cosine", N, fftbins=False)
    wsum = np.pad(w, (0, 0)).reshape(-1, hop).sum(axis=0).max() if N % hop == 0 else None
    out = np.stack([lfilter([1.0], A[i], resid[i * hop:i * hop + N]) for i in range(len(frames))])
    y = overlap_add(out, hop, w) / wsum
    a, b = N, (len(frames) - 1) * hop
    snr = 10 * np.log10(np.sum(xs[a:b] ** 2) / np.sum((xs[a:b] - y[a:b]) ** 2))
    assert snr > 15


def test_bad_gain_is_an_error():
    x, fs = _sf3()
    with pytest.raises(ValueError, match="gain must be"):
        phon.lpcresidual(x, fs, gain="unit")


def test_silence_gives_a_finite_residual():
    for gain in ("speech", None):
        r, _ = phon.lpcresidual(np.zeros(16000), 16000, gain=gain)
        assert np.isfinite(r).all()
