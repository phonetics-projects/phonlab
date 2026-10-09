"""Golden-output regression tests.

Each case runs a phonlab function on a short, fixed clip of bundled example
audio and compares the result to a stored snapshot in ``test/golden/``.  A
failure means the function's output changed.  If the change is intended,
regenerate the snapshots and review the diff:

    python -m pytest test/test_regression.py --update-golden
"""
import random
import warnings

import numpy as np
import pandas as pd
import pytest

import phonlab as phon
from pathlib import Path

warnings.filterwarnings("ignore")

AUDIO = Path(phon.__file__).parent / "data" / "example_audio"
GOLDEN = Path(__file__).parent / "golden"
RTOL, ATOL = 1e-4, 1e-6
MAX_STORED = 2000


def _clip(name="sf3_cln.wav", dur=1.0, chansel=(0,)):
    x, fs = phon.loadsig(str(AUDIO / name), chansel=list(chansel), duration=dur)
    return x, fs


def flatten(obj, prefix="r"):
    """Flatten a function result into {key: ndarray} for saving/comparing."""
    out = {}
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            out[f"{prefix}.{c}"] = obj[c].to_numpy()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}"))
    elif isinstance(obj, (tuple, list)) and not (
        obj and all(isinstance(i, (int, float, np.floating)) for i in obj)
    ):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}.{i}"))
    else:
        a = np.asarray(obj)
        if a.size > MAX_STORED:  # keep golden files small: store a strided sample
            out[f"{prefix}#shape"] = np.array(a.shape)
            a = a.ravel()[:: -(-a.size // MAX_STORED)]
        out[prefix] = a
    return out


def _fricative(x, fs):
    r = phon.fricative(x, fs, 0.3)
    return list(r)


def _cases():
    x, fs = _clip()
    xs, _ = _clip(dur=0.4)  # shorter clip for the slow trackers
    xl, fsl = phon.loadsig(str(Path(__file__).parent / "s1202a.wav"),
                           chansel=[0], duration=8.0)  # rhythm needs >4 s
    bands = phon.shannon_bands(4)
    f0df = phon.get_f0(x, fs)
    resid, rfs = phon.lpcresidual(x, fs)
    return {
        "CPP": lambda: phon.CPP(x, fs),
        "compute_cepstrogram": lambda: phon.compute_cepstrogram(x, fs),
        "cepstral_smooth": lambda: phon.cepstral_smooth(x, fs),
        "DPPT_formants": lambda: phon.DPPT_formants(x, fs),
        "HNR": lambda: phon.HNR(x, fs),
        "RLPC_formants": lambda: phon.RLPC_formants(x, fs),
        "TVLP_formants": lambda: phon.TVLP_formants(xs, fs),
        "choose_order": lambda: phon.choose_order(x, fs),
        "amplitude_envelope": lambda: phon.amplitude_envelope(x, fs),
        "burst": lambda: phon.burst(x, fs, 0.1, 0.2),
        "fricative": lambda: _fricative(x, fs),
        "gci_sedreams": lambda: phon.gci_sedreams(x, fs),
        "get_f0": lambda: phon.get_f0(x, fs),
        "get_f0_acd": lambda: phon.get_f0_acd(x, fs),
        "get_f0_shs": lambda: phon.get_f0_shs(x, fs),
        "get_f0_srh": lambda: phon.get_f0_srh(x, fs),
        "get_rms": lambda: phon.get_rms(x, fs),
        "h2h1": lambda: phon.h2h1(resid, rfs, f0df),
        "lpcresidual": lambda: phon.lpcresidual(x, fs),
        "get_rhythm_spectrum": lambda: phon.get_rhythm_spectrum(xl, fsl),
        "rhythmogram": lambda: phon.rhythmogram(xl, fsl),
        "compute_sgram": lambda: phon.compute_sgram(x, fs, 0.005),
        "compute_mel_sgram": lambda: phon.compute_mel_sgram(x, fs),
        "nasality": lambda: phon.nasality(x, fs),
        "track_formants_lpc": lambda: phon.track_formants(x, fs, quiet=True),
        "track_formants_ifc_fast": lambda: phon.track_formants(
            xs, fs, method="ifc_fast", quiet=True),
        "track_formants_ifc": lambda: phon.track_formants(
            x, fs, method="ifc", quiet=True),
        "prep_audio": lambda: phon.prep_audio(x, fs),
        "peak_rms": lambda: phon.peak_rms(x),
        "sigcor_noise": lambda: phon.sigcor_noise(x, fs),
        "vocode": lambda: phon.vocode(x, fs, bands, target_fs=16000),
        "third_octave_bands": lambda: phon.third_octave_bands(),
        "shannon_bands": lambda: phon.shannon_bands(),
        "sine_synth": lambda: phon.sine_synth(
            phon.track_formants(x, fs, quiet=True)),
        "test_signal_sine": lambda: phon.test_signal("sine"),
        "smoothn": lambda: phon.smoothn(
            np.sin(np.linspace(0, 6, 60))
            + np.random.RandomState(0).randn(60) * 0.1),
        "freq_scales": lambda: (
            phon.Hz_to_mel(np.array([100., 1000, 5000])),
            phon.mel_to_Hz(np.array([100., 1000, 2000])),
            phon.hz2bark(np.array([100., 1000, 5000])),
            phon.bark2hz(np.array([1., 8, 20])),
        ),
        "tg_to_df": lambda: phon.tg_to_df(str(AUDIO / "im_twelve.TextGrid")),
    }


CASES = None


def _get_cases():
    global CASES
    if CASES is None:
        CASES = _cases()
    return CASES


@pytest.mark.parametrize("name", [
    "CPP", "compute_cepstrogram", "cepstral_smooth", "DPPT_formants", "HNR",
    "RLPC_formants", "TVLP_formants", "choose_order", "amplitude_envelope",
    "burst", "fricative", "gci_sedreams", "get_f0", "get_f0_acd",
    "get_f0_shs", "get_f0_srh", "get_rms", "h2h1", "lpcresidual",
    "get_rhythm_spectrum", "rhythmogram", "compute_sgram",
    "compute_mel_sgram", "nasality", "track_formants_lpc",
    "track_formants_ifc_fast", "track_formants_ifc", "prep_audio", "peak_rms",
    "sigcor_noise", "vocode", "third_octave_bands", "shannon_bands",
    "sine_synth", "test_signal_sine", "smoothn", "freq_scales", "tg_to_df",
])
def test_golden(name, request):
    np.random.seed(1234)
    random.seed(1234)
    result = flatten(_get_cases()[name]())
    path = GOLDEN / f"{name}.npz"

    if request.config.getoption("--update-golden"):
        GOLDEN.mkdir(exist_ok=True)
        np.savez_compressed(path, **result)
        pytest.skip(f"golden file for {name} updated")

    assert path.exists(), (
        f"no golden file for {name}; run with --update-golden to create it")
    with np.load(path, allow_pickle=True) as g:
        expected = {k: g[k] for k in g.files}

    assert set(result) == set(expected), "output structure changed"
    for k, exp in expected.items():
        got = result[k]
        assert got.shape == exp.shape, f"{k}: shape {got.shape} != {exp.shape}"
        if exp.dtype.kind in "fiu" and got.dtype.kind in "fiu":
            np.testing.assert_allclose(
                got.astype(float), exp.astype(float), rtol=RTOL, atol=ATOL,
                equal_nan=True, err_msg=k)
        else:
            assert (got == exp).all(), f"{k} differs"
