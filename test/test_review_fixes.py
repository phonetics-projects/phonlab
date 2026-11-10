"""Regression tests for bugs found in the 0.2.0 code review."""
import importlib.resources as res
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.signal as ss

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import phonlab as phon  # noqa: E402


@pytest.fixture(scope="module")
def audio():
    path = res.files("phonlab") / "data" / "example_audio" / "sf3_cln.wav"
    x, fs = phon.loadsig(path, chansel=[0])
    return x[:fs], fs


@pytest.mark.parametrize("fs", [16000, 22050, 32000, 44100, 48000])
def test_mel_sgram_step_is_the_requested_step(fs):
    x = np.random.default_rng(0).standard_normal(fs)
    t, f, S = phon.compute_mel_sgram(x, fs, step=0.01)
    assert np.diff(t).mean() == pytest.approx(0.01, abs=1e-4)


def test_burst_finds_the_same_burst_at_other_sampling_rates(audio):
    x, fs = audio
    t16, _ = phon.burst(x, fs, 0.5, 0.6)
    y = ss.resample_poly(x, 44100, fs)
    t44, score = phon.burst(y, 44100, 0.5, 0.6)
    assert t16 > 0 and score > 0
    assert t44 == pytest.approx(t16, abs=0.005)


def test_sgram_slice_time_with_ax_warns_instead_of_failing(audio):
    x, fs = audio
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match="slice_time is ignored"):
        phon.sgram(x, fs, ax=ax, slice_time=0.5)
    plt.close(fig)


def test_get_f0_acd_test_time_diagnostics(audio, capsys):
    x, fs = audio
    df = phon.get_f0_acd(x, fs, test_time=0.5)
    assert len(df) > 0
    plt.close("all")


def test_deltaF_norm_with_column_and_deltaF():
    df = pd.DataFrame({"F1": [500., 600, 700, 800], "F2": [1500., 1600, 1700, 1800],
                       "F3": [2500., 2600, 2700, 2800], "F4": [3500., 3600, 3700, 3800],
                       "spk": ["a", "a", "b", "b"]})
    out = phon.deltaF_norm(df, column="spk", deltaF=1000)
    assert set(out["deltaF"]) == {1000}
    np.testing.assert_allclose(out["F1/∆F"], out["F1"] / 1000)
    assert "deltaF" not in df.columns       # a new dataframe: the input is not changed
    # no column: the input is changed in place
    df2 = df.drop(columns="spk").copy()
    assert phon.deltaF_norm(df2) is df2 and "deltaF" in df2.columns


def test_shr_pitch_runs(audio):
    from phonlab.acoustic.shs import _shr_pitch
    x, fs = audio
    assert len(_shr_pitch(x, fs)) > 0


def test_split_speaker_df_warns_about_overlaps(capsys):
    df2 = pd.DataFrame({"t1": [0.0, 1.0], "t2": [1.5, 2.0], "text": ["A: one", "A: two"]})
    phon.split_speaker_df(df2, "text", ts=["t1", "t2"], include=["A"], exclude=[], sep=": ")
    assert "overlaps" in capsys.readouterr().err


@pytest.mark.parametrize("rate", [22050, 44100])
def test_h2h1_does_not_depend_on_the_sampling_rate_of_the_residual(audio, rate):
    """h2h1() filtered the residual at its own rate with a filter designed for 16000 Hz."""
    x, fs = audio
    resid, fs16 = phon.lpcresidual(x, fs)
    f0df = phon.get_f0_acd(x, fs)
    ref = phon.h2h1(resid, fs16, f0df).h2h1.to_numpy()
    other = phon.h2h1(ss.resample_poly(resid, rate, fs16), rate, f0df).h2h1.to_numpy()
    assert np.nanmean(np.abs(other - ref)) < 1.0   # was about 17-19 dB


def _h2h1_inputs(audio):
    x, fs = audio
    resid, fs16 = phon.lpcresidual(x, fs)
    return resid, fs16, phon.get_f0_acd(x, fs)


def test_h2h1_of_a_filtered_dataframe_matches_the_full_dataframe(audio):
    resid, fs, f0df = _h2h1_inputs(audio)
    full = phon.h2h1(resid, fs, f0df, f0med=150.0)
    sub = f0df[f0df.f0.notna()].iloc[5:60:3]          # voiced rows only: an index that does not start at 0, with gaps
    assert sub.index[0] != 0
    out = phon.h2h1(resid, fs, sub, f0med=150.0, length=0.05)
    assert out.index.equals(sub.index)
    np.testing.assert_allclose(out.h2h1.to_numpy(), full.loc[sub.index, "h2h1"].to_numpy())


def test_h2h1_needs_length_when_the_first_row_is_not_the_first_frame(audio):
    resid, fs, f0df = _h2h1_inputs(audio)
    with pytest.raises(ValueError, match="Give the `length`"):
        phon.h2h1(resid, fs, f0df.iloc[40:])


def test_h2h1_with_an_uneven_step(audio):
    resid, fs, f0df = _h2h1_inputs(audio)
    uneven = f0df.iloc[[0, 1, 2, 5, 6, 11, 12, 30]]      # no constant step between the rows
    out = phon.h2h1(resid, fs, uneven, f0med=150.0)
    assert len(out) == len(uneven)


# ---- phonlab no longer changes numpy's floating point error handling ----

def test_importing_and_using_phonlab_does_not_change_numpy_error_handling():
    import subprocess, sys
    code = ("import numpy as np; before = np.geterr(); import phonlab as phon; "
            "[getattr(phon, n) for n in phon.__all__]; assert np.geterr() == before, (before, np.geterr())")
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("name,call", [
    ("choose_order", lambda: phon.choose_order(np.zeros(16000), 16000)),
    ("egg_to_oq", lambda: phon.egg_to_oq(np.zeros(16000), 16000)),
    ("rhythmogram", lambda: phon.rhythmogram(np.zeros(16000 * 6), 16000)),
])
def test_silence_gives_no_floating_point_warnings(name, call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        call()


def test_get_f0_srh_of_a_pure_tone():
    df = phon.get_f0_srh(phon.test_signal("sine", fs=16000, dur=1.0, freq=150), 16000)
    assert len(df) > 0
