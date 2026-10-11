"""Tests against known analytic answers and invariants (not snapshots)."""
import warnings

import numpy as np
import pandas as pd
import pytest

import phonlab as phon
from pathlib import Path

warnings.filterwarnings("ignore")
AUDIO = Path(phon.__file__).parent / "data" / "example_audio"


# --- signal / scale utilities -------------------------------------------

def test_freq_scale_roundtrips():
    f = np.array([50.0, 300.0, 1000.0, 4000.0, 8000.0])
    np.testing.assert_allclose(phon.mel_to_Hz(phon.Hz_to_mel(f)), f, rtol=1e-6)
    np.testing.assert_allclose(phon.bark2hz(phon.hz2bark(f)), f, rtol=1e-3)


def test_freq_scales_monotonic():
    f = np.linspace(20, 8000, 50)
    assert (np.diff(phon.Hz_to_mel(f)) > 0).all()
    assert (np.diff(phon.hz2bark(f)) > 0).all()


def test_test_signal_sine_frequency():
    x = phon.test_signal("sine", fs=10000, dur=0.5, freq=500)
    spec = np.abs(np.fft.rfft(x))
    assert np.fft.rfftfreq(len(x), 1 / 10000)[spec.argmax()] == pytest.approx(500, abs=3)


def test_peak_rms_scales_with_amplitude():
    x = phon.test_signal("sine", fs=10000, dur=1.0, freq=500, amp=0.5)
    assert phon.peak_rms(2 * x) == pytest.approx(2 * phon.peak_rms(x), rel=1e-4)


def test_loadsig_shapes_and_rate():
    x, fs = phon.loadsig(str(AUDIO / "sf3_cln.wav"), chansel=[0])
    assert fs == 16000 and x.ndim == 1
    x2, fs2 = phon.loadsig(str(AUDIO / "sf3_cln.wav"), chansel=[0], fs=8000)
    assert fs2 == 8000 and len(x2) == pytest.approx(len(x) / 2, abs=2)
    xd, _ = phon.loadsig(str(AUDIO / "sf3_cln.wav"), chansel=[0], duration=0.5)
    assert len(xd) == 8000


def test_loadsig_stereo_channels():
    left, right, fs = phon.loadsig(str(AUDIO / "stereo.wav"))
    assert left.shape == right.shape and fs == 48000
    one, fs1 = phon.loadsig(str(AUDIO / "stereo.wav"), chansel=[1])
    np.testing.assert_allclose(one, right)


def test_channels_are_duplicates():
    x = np.random.RandomState(0).randn(1000)
    assert phon.channels_are_duplicates(x, x.copy())
    assert not phon.channels_are_duplicates(x, np.random.RandomState(1).randn(1000))


def test_prep_audio_resamples_and_is_finite():
    x, fs = phon.loadsig(str(AUDIO / "sf3_cln.wav"), chansel=[0])
    y, fs2 = phon.prep_audio(x, fs, target_fs=32000)
    assert fs2 == 32000
    assert len(y) == pytest.approx(len(x) * 2, abs=4)
    assert np.isfinite(y).all()


def test_prep_audio_silence_has_no_nan():
    y, _ = phon.prep_audio(np.zeros(16000, dtype=np.float32), 16000)
    assert np.isfinite(y).all()


def test_add_noise_snr():
    x = phon.test_signal("sine", fs=16000, dur=1.0, freq=300, amp=0.5)
    y, fs = phon.add_noise(x, 16000, noise_type="white", snr=10)
    assert fs == 16000
    assert phon.peak_rms(y) != phon.peak_rms(x)  # noise was actually added
    assert np.isfinite(y).all()


def test_smoothn_reduces_noise():
    t = np.linspace(0, 6, 200)
    clean = np.sin(t)
    noisy = clean + np.random.RandomState(0).randn(200) * 0.2
    z = phon.smoothn(noisy)[0]
    assert np.abs(z - clean).mean() < np.abs(noisy - clean).mean()


# --- acoustic measures on synthetic input -------------------------------

def _vowelish(f0=120, fs=16000, dur=1.0):
    """A pulse train through two resonators: known f0, crude formants."""
    import scipy.signal as sig
    n = int(fs * dur)
    src = np.zeros(n)
    src[:: int(fs / f0)] = 1.0
    y = src
    for f, bw in [(700, 80), (1200, 90), (2600, 120), (3400, 150)]:
        r = np.exp(-np.pi * bw / fs)
        th = 2 * np.pi * f / fs
        y = sig.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return (y / np.abs(y).max() * 0.5).astype(np.float32), fs


def _harmonic(f0=120, fs=16000, dur=1.0):
    """Harmonics with 1/k amplitudes: a simple periodic voice-like source."""
    t = np.arange(int(fs * dur)) / fs
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, int(4000 // f0)))
    return (y / np.abs(y).max() * 0.5).astype(np.float32), fs


@pytest.mark.parametrize("fn", ["get_f0", "get_f0_acd", "get_f0_shs"])
def test_f0_trackers_find_known_f0(fn):
    x, fs = _harmonic(f0=120)
    df = getattr(phon, fn)(x, fs)
    f0 = df["f0"].dropna()
    f0 = f0[f0 > 0]
    assert len(f0) > 20
    assert np.median(f0) == pytest.approx(120, rel=0.05)


def test_formant_trackers_find_known_formants():
    x, fs = _vowelish(f0=120)
    df = phon.track_formants(x, fs, quiet=True)
    mid = df.iloc[len(df) // 4: -len(df) // 4]
    assert mid["F1"].median() == pytest.approx(700, rel=0.1)
    assert mid["F2"].median() == pytest.approx(1200, rel=0.1)


def test_track_formants_rejects_unknown_method():
    x, fs = _vowelish()
    with pytest.raises(ValueError, match="unknown method 'rlpc'"):
        phon.track_formants(x, fs, method="rlpc", quiet=True)


def test_get_rms_columns_and_length():
    x, fs = _vowelish()
    df = phon.get_rms(x, fs, s=0.005)
    assert "rms" in df.columns and 150 < len(df) <= 200
    louder = phon.get_rms(x * 2, fs, s=0.005)["rms"]
    assert (louder - df["rms"]).mean() == pytest.approx(6.02, abs=0.1)  # dB


def test_cpp_higher_for_periodic_than_noise():
    x, fs = _vowelish(f0=120)
    noise = np.random.RandomState(0).randn(len(x)).astype(np.float32) * 0.1
    assert phon.CPP(x, fs)["cpp"].median() > phon.CPP(noise, fs)["cpp"].median()


def test_hnr_higher_for_periodic_than_noise():
    x, fs = _vowelish(f0=120)
    noise = np.random.RandomState(0).randn(len(x)).astype(np.float32) * 0.1
    col = "hnr_500"
    assert phon.HNR(x, fs)[col].median() > phon.HNR(noise, fs)[col].median()


def test_amplitude_envelope_follows_gating():
    fs = 22050
    x = np.random.RandomState(0).randn(fs)
    x[: fs // 2] *= 0.01
    env, efs = phon.amplitude_envelope(x, fs)
    assert env[int(efs * 0.75)] > 10 * env[int(efs * 0.25)]


# --- textgrid / dataframe utilities -------------------------------------

def test_tg_tiernames_matches_tg_to_df():
    tg = str(AUDIO / "im_twelve.TextGrid")
    names = phon.tg_tiernames(tg)
    dfs = phon.tg_to_df(tg)
    assert len(names) == len(dfs) == 3


def test_tg_roundtrip(tmp_path):
    words, phones, _ = phon.tg_to_df(str(AUDIO / "im_twelve.TextGrid"))
    out = tmp_path / "out.TextGrid"
    phon.df_to_tg([words, phones], tiercols=["word", "phone"],
                  ts=["t1", "t2"], outfile=str(out))
    w2, p2 = phon.tg_to_df(str(out))
    for a, b, col in [(words, w2, "word"), (phones, p2, "phone")]:
        assert list(a[col]) == list(b[col])
        np.testing.assert_allclose(a["t1"], b["t1"], atol=1e-6)
        np.testing.assert_allclose(a["t2"], b["t2"], atol=1e-6)


def test_add_context():
    df = pd.DataFrame({"ph": list("abcd")})
    out = phon.add_context(df, "ph", 1, 1)
    assert list(out["prev_ph1"]) == ["", "a", "b", "c"]
    assert list(out["next_ph1"]) == ["b", "c", "d", ""]


def test_explode_intervals_covers_span():
    divs = pd.DataFrame({"t1": [0.0, 1.0], "t2": [1.0, 2.0]})
    out = phon.explode_intervals([0.25, 0.5], ts=["t1", "t2"], df=divs)
    np.testing.assert_allclose(out["obs_t"].astype(float), [0.25, 0.5, 1.25, 1.5])


def test_merge_tiers_assigns_inner_to_outer():
    outer = pd.DataFrame({"t1": [0.0, 1.0], "t2": [1.0, 2.0], "w": ["x", "y"]})
    inner = pd.DataFrame({"t1": [0.1, 1.1, 1.5], "t2": [0.9, 1.4, 1.9],
                          "p": ["a", "b", "c"]})
    m = phon.merge_tiers(inner, outer, suffixes=["_p", "_w"])
    assert len(m) == 3
    assert list(m["w"]) == ["x", "y", "y"]


def test_interpolate_measures_linear():
    meas = pd.DataFrame({"sec": np.linspace(0, 1, 101)})
    meas["v"] = meas["sec"] * 3
    r = phon.interpolate_measures(meas, "sec", interp_ts=np.array([0.25, 0.5]))
    np.testing.assert_allclose(r["v"], [0.75, 1.5], atol=1e-9)
