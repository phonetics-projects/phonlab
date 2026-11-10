"""Tests for mfcc()."""
import importlib.resources as res
import warnings

import librosa
import numpy as np
import pytest
from scipy.fft import dct
from scipy.signal import get_window

import phonlab as phon
import phonlab.auditory.mfcc_ as mf

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def audio():
    x, fs = phon.loadsig(res.files("phonlab") / "data" / "example_audio" / "sf3_cln.wav", chansel=[0])
    return x[:int(1.0 * fs)], fs


def reference_log_mel(x, fs, length=0.025, step=0.01, preemph=0.97, n_mels=40, fmin=0.0, fmax=None, window="hamming"):
    """The log filter outputs, built frame by frame with plain numpy."""
    y, _ = phon.prep_audio(x, fs, target_fs=None, preemph=preemph, scale=True, add_tiny_noise=False)
    N, hop = int(round(length * fs)), int(round(step * fs))
    nfft = int(2 ** np.ceil(np.log2(N)))
    w = get_window(window, N, fftbins=False)
    W = phon.linear_to_mel_weight_matrix(n_mels, nfft // 2 + 1, fs, fmin, fs / 2 if fmax is None else fmax).T
    n = 1 + (len(y) - N) // hop
    return np.array([np.log(np.maximum(W @ (np.abs(np.fft.rfft(y[i * hop:i * hop + N] * w, nfft)) ** 2 / np.sum(w ** 2)), 1e-10))
                     for i in range(n)])


def test_mfcc_matches_a_frame_by_frame_reference(audio):
    x, fs = audio
    df = phon.mfcc(x, fs)
    ref = dct(reference_log_mel(x, fs), type=2, norm="ortho", axis=1)[:, :13]
    assert df.shape == (len(ref), 14)
    np.testing.assert_allclose(df.drop(columns="sec").to_numpy(), ref, rtol=0, atol=1e-9)


def test_mfcc_dct_scaling_is_the_same_as_librosa(audio):
    x, fs = audio
    logmel = reference_log_mel(x, fs)
    lib = librosa.feature.mfcc(S=(10 / np.log(10)) * logmel.T, n_mfcc=13, dct_type=2, norm="ortho").T * (np.log(10) / 10)
    np.testing.assert_allclose(phon.mfcc(x, fs).drop(columns="sec").to_numpy(), lib, atol=1e-9)


def test_columns_and_times(audio):
    x, fs = audio
    df = phon.mfcc(x, fs, n_mfcc=8, length=0.03, step=0.005)
    assert list(df.columns) == ["sec"] + [f"c{i}" for i in range(9)]
    N, hop = int(round(0.03 * fs)), int(round(0.005 * fs))
    assert len(df) == 1 + (len(x) - N) // hop
    np.testing.assert_allclose(np.diff(df.sec), 0.005)
    assert df.sec.iloc[0] == pytest.approx(0.015)          # the middle of the first frame


def test_c0_false_drops_c0_and_its_deltas(audio):
    x, fs = audio
    df = phon.mfcc(x, fs, c0=False, delta=2)
    assert "c0" not in df and "d_c0" not in df and "dd_c0" not in df
    assert {"c1", "d_c1", "dd_c12"} <= set(df.columns) and df.shape[1] == 1 + 36


def test_lifter_cmn_and_the_other_options_do_what_they_say(audio):
    x, fs = audio
    plain = phon.mfcc(x, fs)
    cols = [f"c{i}" for i in range(1, 13)]
    lif = phon.mfcc(x, fs, lifter=22)
    factor = 1 + 11 * np.sin(np.pi * np.arange(1, 13) / 22)
    np.testing.assert_allclose(lif[cols].to_numpy(), plain[cols].to_numpy() * factor)
    np.testing.assert_array_equal(lif.c0, plain.c0)
    cm = phon.mfcc(x, fs, cmn=True)
    np.testing.assert_allclose(cm[cols].mean().to_numpy(), 0, atol=1e-10)
    np.testing.assert_allclose(cm[cols].to_numpy(), plain[cols].to_numpy() - plain[cols].mean().to_numpy())
    np.testing.assert_array_equal(cm.c0, plain.c0)


def test_deltas_are_slopes():
    ramp = np.arange(12.0)[:, None] * np.array([[1.0, -2.0]])
    np.testing.assert_allclose(mf._deltas(ramp)[2:-2], np.tile([1.0, -2.0], (8, 1)))
    d = phon.mfcc(np.random.default_rng(0).standard_normal(8000), 8000, delta=2)
    assert sum(c.startswith("d_") for c in d) == 13 and sum(c.startswith("dd_") for c in d) == 13


def test_other_sampling_rates_and_filterbank_limits(audio):
    x, fs = audio
    y = np.interp(np.arange(0, len(x) - 1, fs / 22050), np.arange(len(x)), x)
    df = phon.mfcc(y, 22050, n_mels=26, fmin=100, fmax=8000, window="hann", preemph=0)
    ref = dct(reference_log_mel(y, 22050, n_mels=26, fmin=100, fmax=8000, window="hann", preemph=0), type=2, norm="ortho", axis=1)[:, :13]
    np.testing.assert_allclose(df.drop(columns="sec").to_numpy(), ref, atol=1e-9)


def test_c0_does_not_depend_on_the_level_of_the_recording(audio):
    x, fs = audio
    np.testing.assert_allclose(phon.mfcc(x, fs).to_numpy(), phon.mfcc(0.2 * x, fs).to_numpy(), atol=1e-4)


def test_processing_in_chunks_gives_the_same_values(audio, monkeypatch):
    x, fs = audio
    whole = phon.mfcc(x, fs)
    monkeypatch.setattr(mf, "CHUNK_ELEMENTS", 3 * 512)    # three frames at a time
    np.testing.assert_array_equal(phon.mfcc(x, fs).to_numpy(), whole.to_numpy())


def test_silence_is_finite_and_the_same_every_time():
    a = phon.mfcc(np.zeros(8000), 8000)
    b = phon.mfcc(np.zeros(8000), 8000)
    assert np.isfinite(a.to_numpy()).all()
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_similar_to_praat(audio):
    import parselmouth
    path = res.files("phonlab") / "data" / "example_audio" / "sf3_cln.wav"
    snd = parselmouth.Sound(str(path)).extract_left_channel()
    pm = phon.mfcc_to_df(snd.to_mfcc(number_of_coefficients=12, window_length=0.025, time_step=0.01), num=12, include_energy=True)
    mine = phon.mfcc(*phon.loadsig(path, chansel=[0]))
    for k in (1, 2, 3):      # (different filterbanks and scaling: the same shape, not the same values)
        assert np.corrcoef(np.interp(mine.sec, pm.sec, pm[f"c{k}"]), mine[f"c{k}"])[0, 1] > 0.9


@pytest.mark.parametrize("kw,msg", [
    (dict(n_mfcc=40), "smaller than n_mels"),
    (dict(n_mfcc=0), "at least 1"),
    (dict(delta=3), "delta must be"),
    (dict(length=0), "length must be positive"),
    (dict(lifter=-1), "lifter"),
    (dict(fmax=20000), "Nyquist"),
])
def test_bad_arguments(audio, kw, msg):
    x, fs = audio
    with pytest.raises(ValueError, match=msg):
        phon.mfcc(x, fs, **kw)


def test_audio_shorter_than_a_frame_is_an_error():
    with pytest.raises(ValueError, match="shorter than one"):
        phon.mfcc(np.zeros(100), 16000)
    with pytest.raises(ValueError, match="one-dimensional"):
        phon.mfcc(np.zeros((2, 16000)), 16000)
