"""Every function with a `step` parameter rejects a step that is not a positive number of seconds."""
import inspect
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import phonlab as phon  # noqa: E402
from phonlab.utils.validate import check_step  # noqa: E402

warnings.filterwarnings("ignore")

FS = 16000
X = np.random.default_rng(0).standard_normal(FS)

# call each function with its required arguments, and the bad step
CALLS = {
    "compute_sgram": lambda s: phon.compute_sgram(X, FS, 0.01, step=s),
    "compute_mt_sgram": lambda s: phon.compute_mt_sgram(X, FS, step=s),
    "compute_cepstrogram": lambda s: phon.compute_cepstrogram(X, FS, step=s),
    "CPP": lambda s: phon.CPP(X, FS, step=s),
    "cepstral_smooth": lambda s: phon.cepstral_smooth(X, FS, step=s),
    "get_rms": lambda s: phon.get_rms(X, FS, step=s),
    "get_f0": lambda s: phon.get_f0(X, FS, step=s),
    "get_f0_srh": lambda s: phon.get_f0_srh(X, FS, step=s),
    "get_f0_acd": lambda s: phon.get_f0_acd(X, FS, step=s),
    "get_f0_shs": lambda s: phon.get_f0_shs(X, FS, step=s),
    "HNR": lambda s: phon.HNR(X, FS, step=s),
    "VAD": lambda s: phon.VAD(X, FS, step=s),
    "LPC_formants": lambda s: phon.LPC_formants(X, FS, step=s),
    "RLPC_formants": lambda s: phon.RLPC_formants(X, FS, step=s),
    "DPPT_formants": lambda s: phon.DPPT_formants(X, FS, step=s),
    "TVLP_formants": lambda s: phon.TVLP_formants(X, FS, step=s),
    "egg_to_oq": lambda s: phon.egg_to_oq(X, FS, step=s),
    "compute_mel_sgram": lambda s: phon.compute_mel_sgram(X, FS, step=s),
    "Audspec": lambda s: phon.Audspec(fs=FS, step=s),
    "lpcresidual": lambda s: phon.lpcresidual(X, FS, step=s),
    "nasality": lambda s: phon.nasality(X, FS, step=s),
    "tandem_fric": lambda s: phon.tandem_fric(X, FS, 0.5, step=s),
    "sgram": lambda s: phon.sgram(X, FS, step=s),
    "sgram_mel": lambda s: phon.sgram(X, FS, type="mel", step=s),
    "sgram_old_name": lambda s: phon.sgram(X, FS, s=s),
}


@pytest.mark.parametrize("bad", [0, -0.005, float("nan"), "0.005", None])
@pytest.mark.parametrize("name", [n for n in CALLS if n not in ("sgram", "sgram_mel", "sgram_old_name")])
def test_bad_step_is_an_error(name, bad):
    with pytest.raises(ValueError, match="positive number of seconds"):
        CALLS[name](bad)


@pytest.mark.parametrize("bad", [0, -0.005, float("nan")])
@pytest.mark.parametrize("name", ["sgram", "sgram_mel", "sgram_old_name"])
def test_bad_step_is_an_error_in_sgram(name, bad):
    with pytest.raises(ValueError, match="positive number of seconds"):
        CALLS[name](bad)


def test_every_public_function_with_a_step_is_checked():
    """A function that has a `step` parameter must be in CALLS (so a new one is not forgotten)."""
    unchecked = []
    for name in phon.__all__:
        obj = getattr(phon, name)
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if "step" in params and name not in CALLS and name != "nasalance":
            unchecked.append(name)
    assert not unchecked, f"add these to CALLS in test_step_check.py: {unchecked}"


def test_good_step_still_works():
    assert len(phon.get_rms(X, FS, 0.01)) > 0
    assert len(phon.compute_sgram(X, FS, 0.01, 0.005)[0]) > 0
    assert check_step(0.005) == 0.005


# ---- the step is converted to samples in one way, in every function ----

@pytest.mark.parametrize("fs", [8000, 10000, 12000, 16000, 22050])
@pytest.mark.parametrize("length,step", [(0.03, 0.01), (0.03, 0.005), (0.03, 0.0125), (0.06, 0.01), (0.04, 0.002)])
def test_compute_sgram_hop_is_the_step(fs, length, step):
    from phonlab.utils.validate import step_samples
    x = np.random.default_rng(0).standard_normal(fs)
    t, f, S = phon.compute_sgram(x, fs, length, step)
    assert np.diff(t)[0] * fs == pytest.approx(step_samples(step, fs), abs=1e-6)


@pytest.mark.parametrize("step", [0.005, 0.01, 0.0125])
def test_nasality_frames_line_up_at_any_step(step):
    """compute_sgram() and RLPC_formants() must have the same frames (this failed at step=0.01, fs=12000)."""
    x = np.random.default_rng(0).standard_normal(24000)
    assert len(phon.nasality(x, 12000, step=step)) > 0


def test_step_shorter_than_a_sample_is_an_error():
    with pytest.raises(ValueError, match="shorter than one sample"):
        phon.compute_mt_sgram(X, FS, step=0.00001)
