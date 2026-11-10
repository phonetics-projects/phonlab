"""The step (``step``) and window length (``length``) parameters have the same names in every function.

The pre-emphasis factor is always ``preemph``.

The older names (``s``, ``l``, ``w``, ``hop_dur``, ``step_sec``, ``step_size``, ``frame_duration_sec``, ``pre`` and
``preemphasis``) still work by keyword, with a DeprecationWarning.
"""
import importlib.resources as res
import inspect
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import phonlab as phon  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

OLD_NAMES = {"s", "l", "w", "hop_dur", "step_sec", "step_size", "frame_duration_sec", "pre", "preemphasis"}
# s is not a step in these functions: it is the smoothing parameter
SMOOTHERS = {"smooth1d", "smoothn"}


@pytest.fixture(scope="module")
def audio():
    path = res.files("phonlab") / "data" / "example_audio" / "sf3_cln.wav"
    x, fs = phon.loadsig(path, chansel=[0])
    return x[: int(1.2 * fs)], fs


def _run(func, *args, **kwargs):
    """Call `func` with the random number generator in the same state every time.

    prep_audio() puts a tiny amount of random noise in any silent (exactly zero) samples, so two
    calls would otherwise differ slightly.
    """
    np.random.seed(0)
    return func(*args, **kwargs)


def _arrays(result):
    """Flatten a function result into a list of numpy arrays, for comparison."""
    if hasattr(result, "to_numpy"):
        return [result.to_numpy()]
    if isinstance(result, tuple):
        return [a for r in result for a in _arrays(r)]
    return [np.asarray(result)]


def _same(a, b):
    a, b = _arrays(a), _arrays(b)
    return len(a) == len(b) and all(
        x.shape == y.shape and np.allclose(x.astype(float), y.astype(float), equal_nan=True)
        for x, y in zip(a, b))


# ---- no function uses the old names (the deprecated names are accepted only by the decorator) ----

def test_signatures_use_the_new_names():
    for name in phon.__all__:
        obj = getattr(phon, name)
        if name in SMOOTHERS or not callable(obj):
            continue
        params = set(inspect.signature(obj).parameters)
        assert not (params & OLD_NAMES), f"{name}() still has parameter(s) {sorted(params & OLD_NAMES)}"


STEP_FUNCTIONS = ["CPP", "DPPT_formants", "HNR", "LPC_formants", "RLPC_formants", "TVLP_formants", "VAD",
                  "cepstral_smooth", "compute_cepstrogram", "compute_mel_sgram", "compute_mt_sgram",
                  "compute_sgram", "egg_to_oq", "get_f0", "get_f0_acd", "get_f0_shs", "get_f0_srh",
                  "get_rms", "lpcresidual", "nasality", "nasalance", "tandem_fric", "Audspec"]
LENGTH_FUNCTIONS = ["CPP", "DPPT_formants", "HNR", "LPC_formants", "RLPC_formants", "TVLP_formants",
                    "cepstral_smooth", "choose_order", "compute_cepstrogram", "compute_mt_sgram",
                    "compute_sgram", "fricative", "get_f0_acd", "get_f0_shs", "get_f0_srh", "get_rms",
                    "lpcresidual", "nasality", "tandem_fric"]


@pytest.mark.parametrize("name", STEP_FUNCTIONS)
def test_step_parameter_name(name):
    obj = getattr(phon, name)
    func = obj.__init__ if inspect.isclass(obj) else obj
    assert "step" in inspect.signature(func).parameters


@pytest.mark.parametrize("name", LENGTH_FUNCTIONS)
def test_length_parameter_name(name):
    assert "length" in inspect.signature(getattr(phon, name)).parameters


# ---- the old names still work, with a warning ----

# (function, keyword arguments with the new names, the same with the old names)
CASES = [
    ("get_f0", dict(step=0.007), dict(s=0.007)),
    ("get_f0_acd", dict(length=0.06, step=0.008), dict(l=0.06, s=0.008)),
    ("get_f0_shs", dict(length=0.07, step=0.008), dict(l=0.07, s=0.008)),
    ("get_f0_srh", dict(length=0.07, step=0.008), dict(l=0.07, s=0.008)),
    ("get_rms", dict(length=0.03, step=0.007), dict(l=0.03, s=0.007)),
    ("HNR", dict(length=0.07, step=0.008), dict(l=0.07, s=0.008)),
    ("CPP", dict(length=0.05, step=0.008), dict(l=0.05, s=0.008)),
    ("VAD", dict(step=0.007), dict(s=0.007)),
    ("LPC_formants", dict(length=0.04, step=0.007), dict(l=0.04, s=0.007)),
    ("RLPC_formants", dict(length=0.04, step=0.007), dict(l=0.04, s=0.007)),
    ("DPPT_formants", dict(length=0.04, step=0.012), dict(l=0.04, s=0.012)),
    ("cepstral_smooth", dict(length=0.05, step=0.007), dict(l=0.05, s=0.007)),
    ("compute_cepstrogram", dict(length=0.05, step=0.007), dict(l=0.05, s=0.007)),
    ("lpcresidual", dict(length=0.05, step=0.006), dict(l=0.05, s=0.006)),
    ("choose_order", dict(length=0.04), dict(l=0.04)),
    ("compute_mt_sgram", dict(length=0.03, step=0.004), dict(w=0.03, s=0.004)),
    ("compute_mel_sgram", dict(step=0.007), dict(step_sec=0.007)),
]


@pytest.mark.parametrize("name,new,old", CASES, ids=[c[0] for c in CASES])
def test_old_names_still_work(audio, name, new, old):
    x, fs = audio
    func = getattr(phon, name)
    expected = _run(func, x, fs, **new)
    with pytest.warns(DeprecationWarning, match=r"has been renamed"):
        result = _run(func, x, fs, **old)
    assert _same(result, expected)


def test_old_names_still_work_compute_sgram(audio):
    x, fs = audio
    expected = phon.compute_sgram(x, fs, length=0.008, step=0.004)
    with pytest.warns(DeprecationWarning, match=r"'w' parameter has been renamed 'length'"):
        result = phon.compute_sgram(x, fs, w=0.008, s=0.004)
    assert _same(result, expected)


def test_old_names_still_work_fricative_functions(audio):
    x, fs = audio
    for func in (phon.fricative, phon.tandem_fric):
        expected = _run(func, x, fs, 0.5, length=0.025)
        with pytest.warns(DeprecationWarning, match=r"'w' parameter has been renamed 'length'"):
            result = _run(func, x, fs, 0.5, w=0.025)
        assert _same(tuple(v for v in result[:5] if v is not None),
                     tuple(v for v in expected[:5] if v is not None))


def test_old_names_still_work_egg_to_oq():
    t = np.arange(0, 1.0, 1 / 16000)
    egg = np.sin(2 * np.pi * 110 * t)
    expected = _run(phon.egg_to_oq, egg, 16000, step=0.01)
    with pytest.warns(DeprecationWarning, match=r"'hop_dur' parameter has been renamed 'step'"):
        result = _run(phon.egg_to_oq, egg, 16000, hop_dur=0.01)
    assert _same(result, expected)


def test_old_name_frame_duration_sec_still_works():
    x = phon.test_signal("sawtooth", fs=8000, dur=0.6, freq=120)
    kw = dict(tf=3000, order=8, use_parallel=False)
    expected = _run(phon.TVLP_formants, x, 8000, length=0.1, step=0.02, **kw)
    with pytest.warns(DeprecationWarning, match=r"'frame_duration_sec' parameter has been renamed 'length'"):
        result = _run(phon.TVLP_formants, x, 8000, frame_duration_sec=0.1, step=0.02, **kw)
    assert _same(result, expected)


# ---- the pre-emphasis factor is always `preemph` ----

PREEMPH_FUNCTIONS = ["DPPT_formants", "IFC_formants", "LPC_formants", "RLPC_formants", "TVLP_formants",
                     "cepstral_smooth", "nasality", "prep_audio", "sgram", "track_formants"]


@pytest.mark.parametrize("name", PREEMPH_FUNCTIONS)
def test_preemph_parameter_name(name):
    assert "preemph" in inspect.signature(getattr(phon, name)).parameters


PREEMPH_CASES = [
    ("prep_audio", dict(preemph=0.9), dict(pre=0.9)),
    ("LPC_formants", dict(preemph=0.9), dict(preemphasis=0.9)),
    ("RLPC_formants", dict(preemph=0.9), dict(preemphasis=0.9)),
    ("IFC_formants", dict(preemph=0.9), dict(preemphasis=0.9)),
    ("DPPT_formants", dict(preemph=0.5), dict(pre=0.5)),
    ("DPPT_formants", dict(preemph=0.5), dict(preemphasis=0.5)),
    ("cepstral_smooth", dict(preemph=0.9), dict(preemphasis=0.9)),
    ("track_formants", dict(preemph=0.9), dict(preemphasis=0.9)),
]


@pytest.mark.parametrize("name,new,old", PREEMPH_CASES, ids=[f"{c[0]}-{list(c[2])[0]}" for c in PREEMPH_CASES])
def test_old_preemph_names_still_work(audio, name, new, old):
    x, fs = audio
    func = getattr(phon, name)
    expected = _run(func, x, fs, **new)
    with pytest.warns(DeprecationWarning, match=r"has been renamed 'preemph'"):
        result = _run(func, x, fs, **old)
    assert _same(result, expected)


def test_old_preemph_names_still_work_nasality_and_tvlp():
    x = phon.test_signal("sawtooth", fs=8000, dur=0.6, freq=120)
    kw = dict(tf=3000, order=8, use_parallel=False, length=0.1, step=0.02)
    expected = _run(phon.TVLP_formants, x, 8000, preemph=0.9, **kw)
    with pytest.warns(DeprecationWarning, match=r"'pre' parameter has been renamed 'preemph'"):
        result = _run(phon.TVLP_formants, x, 8000, pre=0.9, **kw)
    assert _same(result, expected)
    x12 = np.random.default_rng(0).standard_normal(24000)
    with pytest.warns(DeprecationWarning, match=r"'preemphasis' parameter has been renamed 'preemph'"):
        old = phon.nasality(x12, 12000, preemphasis=0.9)
    assert _same(old, phon.nasality(x12, 12000, preemph=0.9))


def test_audspec_old_names():
    with pytest.warns(DeprecationWarning, match=r"Audspec\(\): the 'step_size' parameter"):
        aud = phon.Audspec(step_size=0.02)
    assert aud.step == pytest.approx(0.02)
    with pytest.warns(DeprecationWarning, match=r"Audspec.step_size has been renamed"):
        assert aud.step_size == aud.step


def test_old_and_new_name_together_is_an_error(audio):
    x, fs = audio
    with pytest.raises(TypeError, match=r"both 's' .* and 'step'"):
        phon.get_f0(x, fs, s=0.01, step=0.01)


def test_the_new_names_do_not_warn(audio):
    x, fs = audio
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        phon.get_f0(x, fs, step=0.01)
        phon.LPC_formants(x, fs, step=0.01, length=0.03)


def test_smooth1d_s_is_not_renamed(audio):
    """`s` is the smoothing parameter in the smoothers, not a step."""
    y = np.cumsum(np.random.default_rng(1).normal(size=200))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        phon.smooth1d(y, s=5)
        phon.smoothn(y, s=5)


# ---- sgram() takes the same step and length for every type ----

TYPES = ["fft", "multitaper", "cepstral", "lpc", "mel", "bark"]


def _frames(x, fs, **kwargs):
    plt.figure()
    np.random.seed(0)
    ax, f, ts, Sxx = phon.sgram(x, fs, **kwargs)
    plt.close("all")
    return ts


@pytest.mark.parametrize("type", TYPES)
def test_sgram_step(audio, type):
    x, fs = audio
    ts = _frames(x, fs, type=type, step=0.02)
    assert np.diff(ts).mean() == pytest.approx(0.02, rel=0.02)


@pytest.mark.parametrize("type", ["fft", "multitaper", "cepstral", "lpc"])
def test_sgram_length(audio, type):
    x, fs = audio
    _frames(x, fs, type=type, length=0.03, step=0.01)


@pytest.mark.parametrize("type", ["mel", "bark"])
def test_sgram_length_is_an_error_for_mel_and_bark(audio, type):
    x, fs = audio
    with pytest.raises(ValueError, match="length"):
        _frames(x, fs, type=type, length=0.03)


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("step", [0, -0.01])
def test_sgram_step_must_be_positive(audio, type, step):
    x, fs = audio
    with pytest.raises(ValueError, match="step"):
        _frames(x, fs, type=type, step=step)


@pytest.mark.parametrize("old,new", [("s", "step"), ("step_sec", "step"), ("step_size", "step"),
                                     ("w", "length"), ("l", "length")])
def test_sgram_old_names(audio, old, new):
    x, fs = audio
    type = "fft" if old == "w" else "cepstral" if old == "l" else "mel" if old == "step_sec" \
        else "bark" if old == "step_size" else "fft"
    value = 0.03 if new == "length" else 0.02
    expected = _frames(x, fs, type=type, **{new: value})
    with pytest.warns(DeprecationWarning, match=f"'{old}' parameter has been renamed '{new}'"):
        result = _frames(x, fs, type=type, **{old: value})
    assert np.allclose(result, expected)


def test_get_f0_srh_does_not_change_its_f0_range(audio):
    x, fs = audio
    f0_range = [60, 400]
    phon.get_f0_srh(x, fs, f0_range=f0_range)
    assert f0_range == [60, 400]
    assert inspect.signature(phon.get_f0_srh).parameters["f0_range"].default == [60, 400]
