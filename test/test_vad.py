"""VAD() predicts voicing from a low band amplitude envelope and cepstral peak prominence."""
import importlib.resources
import warnings

import numpy as np
import pytest

import phonlab as phon
from phonlab.acoustic.vad import vad_features

warnings.filterwarnings("ignore")


def _load(name):
    f = importlib.resources.files('phonlab') / 'data' / 'example_audio' / name
    return phon.loadsig(f, chansel=[0])


@pytest.fixture(scope="module")
def im12():
    return _load('im_twelve.wav')


def test_columns_and_grid(im12):
    df = phon.VAD(*im12, s=0.005)
    assert list(df.columns) == ['sec', 'f0', 'cpp', 'amp', 'probv', 'voiced']
    assert np.allclose(np.diff(df.sec), 0.005, atol=1e-4)      # the frames of CPP()
    assert np.isfinite(df[['cpp', 'amp', 'probv']]).all().all()
    assert ((df.probv >= 0) & (df.probv <= 1)).all()
    assert (df.voiced == (df.probv > 0.5)).all()


def test_roughly_agrees_with_get_f0(im12):
    ref = phon.get_f0(*im12)
    df = phon.VAD(*im12)
    assert (np.interp(ref.sec, df.sec, df.probv) > 0.5).astype(bool).__eq__(ref.voiced).mean() > 0.8


def test_features_are_on_the_cpp_frames(im12):
    df = vad_features(*im12)
    cpp = phon.CPP(*im12, norm=False, smooth=0)
    assert np.array_equal(df.sec, cpp.sec) and np.array_equal(df.cpp, cpp.cpp)
    assert list(df.columns) == ['sec', 'f0', 'cpp', 'amp'] and np.isfinite(df).all().all()


def test_silence_is_finite_and_unvoiced():
    # all-zero frames give cpp == 0 (CPP replaces NaN by 0) and an all-zero envelope
    df = phon.VAD(np.zeros(16000), 16000)
    assert np.isfinite(df[['cpp', 'amp', 'probv']]).all().all()
    assert not df.voiced.any()


def test_noise_only_input_is_finite():
    # amp and cpp are relative to the file's own peak, so noise-only audio is not reliably unvoiced
    y = np.random.default_rng(0).normal(0, 1e-9, 16000)
    assert np.isfinite(phon.VAD(y, 16000)[['cpp', 'amp', 'probv']]).all().all()
