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


def test_voiced_runs_follow_the_textgrid(im12):
    """nonzero_groups() turns VAD()'s frame decisions into the voiced stretches of im_twelve.wav."""
    df = phon.VAD(*im12)
    phones = phon.tg_to_df(importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid')[1]
    runs = phon.nonzero_groups(df.voiced, minlen=4)
    assert 3 <= len(runs) <= 8 and all(len(r) >= 4 for r in runs)
    # Each vowel midpoint lies in exactly one voiced run; the middle of a voiceless stop lies in none.
    for _, p in phones.iterrows():
        i = np.argmin(np.abs(df.sec - (p.t1 + p.t2) / 2))     # the frame nearest the midpoint
        n = len(phon.nonzero_groups(df.voiced, containing=i))
        if p.phone in ('ay', 'eh', 'iy', 'aa', 'aw'):
            assert n == 1, p.phone
        elif p.phone in ('t', 'k'):
            assert n == 0, p.phone
    # Dropping short runs and bridging short gaps (the docstring example) leaves neither behind.
    clean = np.zeros(len(df), dtype=bool)
    for run in runs:
        clean[run] = True
    for gap in phon.nonzero_groups(~clean):
        if len(gap) < 4 and gap[0] > 0 and gap[-1] < len(clean) - 1:
            clean[gap] = True
    assert all(len(r) >= 4 for r in phon.nonzero_groups(clean))
    assert all(len(g) >= 4 for g in phon.nonzero_groups(~clean) if g[0] > 0 and g[-1] < len(clean) - 1)
    assert (clean != df.voiced.to_numpy()).mean() < 0.05   # cleaning touches only a few frames


def test_noise_only_input_is_finite():
    # amp and cpp are relative to the file's own peak, so noise-only audio is not reliably unvoiced
    y = np.random.default_rng(0).normal(0, 1e-9, 16000)
    assert np.isfinite(phon.VAD(y, 16000)[['cpp', 'amp', 'probv']]).all().all()
