"""Tests for the speaker-parameter dict (including ref_f1) and the frame_length / step arguments of IFC_formants()."""
import numpy as np
import pytest
from scipy import signal

import phonlab as phon
from phonlab.acoustic.ifc_formant import resolve_speaker_params, params, analysis_rate, _REQUIRED_KEYS


@pytest.fixture(scope="module")
def speechlike():
    """One second of a pulse train through four resonators, 16 kHz, float32."""
    fs, n = 16000, 16000
    rng = np.random.default_rng(0)
    p = np.zeros(n); p[::133] = 1.0
    x = p + 0.05 * rng.normal(size=n)
    for f, bw in ((600, 80), (1400, 100), (2600, 140), (3600, 200)):
        r = np.exp(-np.pi * bw / fs)
        x = signal.lfilter([1], [1, -2 * r * np.cos(2 * np.pi * f / fs), r * r], x)
    return x.astype(np.float32), fs


def _as_dict(spk):
    """A preset as a plain dict of lists/floats, the way a caller would write one."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params[spk].items()}


@pytest.mark.parametrize("spk", [0, 1, 2])
def test_preset_as_dict_equals_preset(speechlike, spk):
    x, fs = speechlike
    a = phon.IFC_formants(x, fs, speaker=spk, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=_as_dict(spk), parallel=False)
    np.testing.assert_array_equal(a.values, b.values)


def test_optional_keys_default_to_preset_behaviour(speechlike):
    """Presets 1 and 2 use no extra upper filter and the default 200-6000 Hz clamp, so a dict with
    only the required keys plus ref_f1 matches them.  ref_f1 has to be kept: without it the dict
    would default to fr[1] = 620 Hz and the analysis would run at 620/500 the speech scale."""
    x, fs = speechlike
    d = {k: v for k, v in _as_dict(1).items() if k in _REQUIRED_KEYS + ("ref_f1",)}
    a = phon.IFC_formants(x, fs, speaker=1, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    np.testing.assert_array_equal(a.values, b.values)
    del d["ref_f1"]
    c = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    assert len(c) != len(a)                                   # 16 ms frames instead of 20 ms


def test_custom_dict_changes_result(speechlike):
    """Dropping the upper-formant pre-filters and widening the inverse filters changes the tracks."""
    x, fs = speechlike
    a = phon.IFC_formants(x, fs, speaker=0, parallel=False)
    d = dict(_as_dict(0), upper_fs=[], upper_bws=[], bws=[200, 200, 200, 200, 200])
    b = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    assert not np.allclose(a[["F1", "F2", "F3", "F4"]].values, b[["F1", "F2", "F3", "F4"]].values)


def test_custom_dict_tracks_shifted_formants(speechlike):
    """Expectations shifted to the synthetic vowel's actual formants still find them."""
    x, fs = speechlike
    d = dict(_as_dict(0), fr=[60, 600, 1400, 2600, 3600])
    df = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    assert abs(df["F1"].median() - 600) < 60
    assert abs(df["F2"].median() - 1400) < 140


@pytest.fixture(scope="module")
def lowlike():
    """Three seconds of an 18 Hz pulse train through resonators at 35 and 105 Hz (a very long tube), 1 kHz."""
    fs, n = 1000, 3000
    rng = np.random.default_rng(1)
    p = np.zeros(n); p[::55] = 1.0
    x = p + 0.02 * rng.normal(size=n)
    for f, bw in ((35, 9), (105, 20)):
        r = np.exp(-np.pi * bw / fs)
        x = signal.lfilter([1], [1, -2 * r * np.cos(2 * np.pi * f / fs), r * r], x)
    return x.astype(np.float32), fs


S = 500 / 35   # the male preset's F1 expectation over a 35 Hz one: the frequency ratio of the long tube


def _male_at(ref_f1):
    """The male preset with every frequency divided by 500/ref_f1: the same speaker, ref_f1/500 times the size."""
    s = 500 / ref_f1
    d = {k: (np.asarray(v, float) / s).tolist() for k, v in params[0].items()
         if k in ("fr", "bws", "upper_fs", "upper_bws", "bands", "extra_upper_fs", "extra_upper_bws", "spec_bounds")}
    d.update(du1=params[0]["du1"] / s, du2=params[0]["du2"] / s, dh1=0, dh2=0, fmin=200 / s, fmax=6000 / s, ref_f1=ref_f1)
    return d


def test_ref_f1_matches_frequency_scaling(lowlike):
    """A dict in real Hz with ref_f1 = 35 reproduces the male preset applied to the audio with its
    sampling rate pretended to be 500/35 times higher (the manual frequency-scaling trick)."""
    x, fs = lowlike
    manual = phon.IFC_formants(x, int(round(fs * S)), speaker=0, parallel=False)
    manual["sec"] *= S
    manual[["F1", "F2", "F3", "F4"]] /= S
    low = phon.IFC_formants(x, fs, speaker=_male_at(35), parallel=False)
    # the analysis rates differ by rounding (12000 vs 840 Hz on different signals), so not bit-identical
    n = min(len(manual), len(low))
    np.testing.assert_allclose(low["sec"].values[:n], manual["sec"].values[:n], atol=2e-3)
    np.testing.assert_allclose(low[["F1", "F2"]].median().values, manual[["F1", "F2"]].median().values, rtol=0.03)
    close = np.isclose(low[["F1", "F2"]].values[:n], manual[["F1", "F2"]].values[:n], rtol=0.05, atol=1.0)
    assert close.mean() > 0.9
    assert abs(low["F1"].median() - 35) < 5 and abs(low["F2"].median() - 105) < 10


def test_ref_f1_defaults_to_fr1(lowlike):
    x, fs = lowlike
    d = _male_at(35); del d["ref_f1"]
    assert resolve_speaker_params(d)["ref_f1"] == d["fr"][1] == 35.0
    assert resolve_speaker_params(_as_dict(1))["ref_f1"] == 500.0          # presets carry an explicit 500
    a = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=_male_at(35), parallel=False)
    np.testing.assert_array_equal(a.values, b.values)


def test_fmin_fmax_default_scaled(lowlike):
    """Omitting fmin/fmax in a low-frequency dict gives the speech clamp scaled by ref_f1/500, not 200-6000 Hz."""
    x, fs = lowlike
    d = _male_at(35); del d["fmin"], d["fmax"]
    r = resolve_speaker_params(d)
    assert np.isclose(r["fmin"], 200 * 35 / 500) and np.isclose(r["fmax"], 6000 * 35 / 500)
    a = phon.IFC_formants(x, fs, speaker=d, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=_male_at(35), parallel=False)
    np.testing.assert_allclose(a.values, b.values, atol=1e-6)


def test_ref_f1_scales_analysis_rate_and_frame(lowlike):
    x, fs = lowlike
    assert analysis_rate() == 12000 and analysis_rate(35) == 840
    df = phon.IFC_formants(x, fs, speaker=_male_at(35), parallel=False)
    assert np.allclose(np.diff(df["sec"].values), 0.01 * S, atol=1e-3)       # step = half of the 0.286 s frame
    assert abs(df["sec"].iloc[0] - 0.01 * S) < 2e-3                           # first frame centre


def test_low_ref_f1_without_scaling_fails_where_scaled_succeeds(lowlike):
    """The same real-Hz parameters at the speech analysis rate (ref_f1 left at 500) do not track the
    35 Hz tube: that is what ref_f1 is for."""
    x, fs = lowlike
    d = dict(_male_at(35), ref_f1=500)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = phon.IFC_formants(x, fs, speaker=d, parallel=False, frame_length=0.02 * S)
    assert not (abs(df["F1"].median() - 35) < 5 and abs(df["F2"].median() - 105) < 10)


@pytest.mark.parametrize("spk", [0, 1, 2])
def test_presets_resolve(spk):
    """Every preset passes validation (the child preset has an inverse filter at exactly 6 kHz)."""
    p = resolve_speaker_params(spk)
    assert p["fmin"] == 200.0 and p["fmax"] == 6000.0 and p["ref_f1"] == 500.0
    assert p["extra_upper_fs"].shape == ((1,) if spk == 0 else (0,))


def test_resolve_returns_copy():
    p = resolve_speaker_params(0)
    p["fr"][1] = 1.0
    assert params[0]["fr"][1] == 500


@pytest.mark.parametrize("bad", [
    {**_as_dict(0), "fr": [1, 2, 3]},                                                   # wrong shape
    {**_as_dict(0), "nope": 1},                                                         # unknown key
    {k: v for k, v in _as_dict(0).items() if k != "bands"},                             # missing key
    {**_as_dict(0), "upper_fs": [1, 2], "upper_bws": [1]},                              # unpaired
    {**_as_dict(0), "fmin": 500, "fmax": 400},
    {**_as_dict(0), "bands": [[200, 1100], [700, 2200], [2000, 3500], [3000, 6000]]},  # edge at Nyquist
    {**_as_dict(0), "upper_fs": [7000, 5500, 4500]},                                   # above Nyquist
    {**_as_dict(0), "ref_f1": 0},
    {**_as_dict(0), "ref_f1": 35},                                                      # speech bands above the 420 Hz Nyquist
    3,
    "male",
])
def test_invalid_params_raise(bad):
    with pytest.raises(ValueError):
        resolve_speaker_params(bad)


def test_default_frame_and_step_explicit(speechlike):
    x, fs = speechlike
    a = phon.IFC_formants(x, fs, speaker=0, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=0, parallel=False, frame_length=0.02, step=0.01)
    np.testing.assert_array_equal(a.values, b.values)
    assert np.allclose(np.diff(a["sec"].values), 0.01)


def test_frame_length_and_step_change_timing(speechlike):
    x, fs = speechlike
    df = phon.IFC_formants(x, fs, speaker=0, parallel=False, frame_length=0.04)   # step defaults to half
    assert np.allclose(np.diff(df["sec"].values), 0.02)
    assert abs(df["sec"].iloc[0] - 0.02) < 1e-6                                   # first frame centre
    df2 = phon.IFC_formants(x, fs, speaker=0, parallel=False, frame_length=0.04, step=0.005)
    assert np.allclose(np.diff(df2["sec"].values), 0.005)
    # a longer frame still finds the same vowel
    assert abs(df["F1"].median() - 600) < 60 and abs(df["F2"].median() - 1400) < 140


def test_frame_length_mismatch_warns(speechlike):
    x, fs = speechlike
    low = dict(_as_dict(0), fr=[4.0, 42, 125, 208, 292])          # f0 notch at 4 Hz
    with pytest.warns(UserWarning, match="frame_length"):
        phon.IFC_formants(x, fs, speaker=low, parallel=False)      # 20 ms frame: far too short
    with pytest.warns(UserWarning, match="frame_length"):
        phon.IFC_formants(x, fs, speaker=0, parallel=False, frame_length=0.2)   # 12 notch periods
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        phon.IFC_formants(x, fs, speaker=0, parallel=False)
        phon.IFC_formants(x, fs, speaker=_male_at(35), parallel=False, frame_length=0.3)   # matched pair


@pytest.mark.parametrize("kw", [dict(frame_length=0.005), dict(frame_length=0), dict(step=0), dict(step=-0.01),
                                dict(step=float("nan")), dict(step="0.01"), dict(step=None)])
def test_bad_frame_or_step_raises(speechlike, kw):
    x, fs = speechlike
    with pytest.raises(ValueError):
        phon.IFC_formants(x, fs, parallel=False, **kw)


@pytest.mark.parametrize("spk", [0, 1, 2])
def test_ifc_speaker_params_is_plain_and_equivalent(speechlike, spk):
    """ifc_speaker_params() gives a plain dict (lists and floats, optional keys filled, reserved keys
    dropped) that reproduces the preset exactly and survives a JSON round trip."""
    import json
    x, fs = speechlike
    p = phon.ifc_speaker_params(spk)
    assert set(p) == set(_REQUIRED_KEYS) | {"extra_upper_fs", "extra_upper_bws", "fmin", "fmax", "ref_f1"}
    assert all(isinstance(v, (list, float)) for v in p.values())
    assert p["ref_f1"] == 500.0 and p["fmin"] == 200.0 and p["fmax"] == 6000.0
    assert p["extra_upper_fs"] == ([5500.0] if spk == 0 else [])
    a = phon.IFC_formants(x, fs, speaker=spk, parallel=False)
    b = phon.IFC_formants(x, fs, speaker=json.loads(json.dumps(p)), parallel=False)
    np.testing.assert_array_equal(a.values, b.values)


def test_ifc_speaker_params_completes_a_dict():
    d = _male_at(35); del d["fmin"], d["fmax"], d["ref_f1"]
    p = phon.ifc_speaker_params(d)
    assert p["ref_f1"] == 35.0 and np.isclose(p["fmin"], 14.0) and np.isclose(p["fmax"], 420.0)
    assert phon.ifc_speaker_params(p) == p                      # idempotent
    p["fr"][1] = 1.0
    assert phon.ifc_speaker_params(d)["fr"][1] == 35.0           # a fresh copy each call
