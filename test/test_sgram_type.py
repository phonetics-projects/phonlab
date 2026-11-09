import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import phonlab as phon


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(0)
    fs = 16000
    t = np.arange(fs) / fs
    return np.sin(2 * np.pi * 500 * t) + 0.1 * rng.standard_normal(fs), fs


@pytest.mark.parametrize("type", ["fft", "multitaper", "cepstral", "lpc", "mel", "bark"])
def test_sgram_types_return_common_layout(sig, type):
    x, fs = sig
    ax, f, t, Sxx = phon.sgram(x, fs, type=type)
    plt.close("all")
    assert Sxx.shape == (len(f), len(t))
    assert np.isfinite(Sxx).all()


def test_sgram_default_is_fft(sig):
    x, fs = sig
    _, f, t, Sxx = phon.sgram(x, fs)
    _, f2, t2, Sxx2 = phon.sgram(x, fs, type="fft")
    plt.close("all")
    np.testing.assert_array_equal(Sxx, Sxx2)


def test_sgram_kwargs_forwarded(sig):
    x, fs = sig
    _, _, t, _ = phon.sgram(x, fs, type="mel", step_sec=0.02)
    _, _, t2, _ = phon.sgram(x, fs, type="mel", step_sec=0.01)
    plt.close("all")
    assert len(t2) > len(t)


def test_sgram_unknown_type_raises(sig):
    x, fs = sig
    with pytest.raises(ValueError):
        phon.sgram(x, fs, type="bogus")


def test_sgram_lpc_order_kwarg(sig):
    x, fs = sig
    _, f, t, S10 = phon.sgram(x, fs, type="lpc", order=10)
    _, _, _, S16 = phon.sgram(x, fs, type="lpc", order=16)
    plt.close("all")
    assert S10.shape == S16.shape and not np.allclose(S10, S16)


def test_sgram_min_prop_defaults_and_percentiles(sig):
    x, fs = sig
    for type, prop in [("fft", -0.1), ("multitaper", 0), ("cepstral", 0),
                       ("lpc", 0), ("mel", 0), ("bark", -0.1)]:
        ax, f, t, Sxx = phon.sgram(x, fs, type=type)
        lo, hi = np.percentile(Sxx, [20, 80])
        vmin = ax.images[0].get_clim()[0]
        plt.close("all")
        assert vmin == pytest.approx(lo + (hi - lo) * prop)


def test_sgram_min_prop_explicit_overrides_default(sig):
    x, fs = sig
    ax, f, t, Sxx = phon.sgram(x, fs, type="mel", min_prop=0.5)
    lo, hi = np.percentile(Sxx, [20, 80])
    vmin = ax.images[0].get_clim()[0]
    plt.close("all")
    assert vmin == pytest.approx(lo + (hi - lo) * 0.5)


def test_sgram_verbose_passed_to_prep_audio(sig, capsys):
    x, fs = sig
    phon.sgram(x, fs, verbose=True)
    plt.close("all")
    assert capsys.readouterr().out != ""


@pytest.mark.parametrize("type", ["fft", "multitaper", "cepstral", "lpc"])
def test_sgram_w_overrides_band_window(sig, type):
    x, fs = sig
    _, _, t1, _ = phon.sgram(x, fs, type=type, w=0.02)
    _, _, t2, _ = phon.sgram(x, fs, type=type, w=0.06)
    plt.close("all")
    assert t1[0] == pytest.approx(0.01, abs=0.005) and t2[0] == pytest.approx(0.03, abs=0.005)  # first frame is centered


@pytest.mark.parametrize("type", ["mel", "bark"])
def test_sgram_w_rejected_for_fixed_window_types(sig, type):
    x, fs = sig
    with pytest.raises(ValueError):
        phon.sgram(x, fs, type=type, w=0.02)


def test_sgram_verbose_passed_to_rlpc(sig, capsys):
    x, fs = sig
    phon.sgram(x, fs, type="lpc", verbose=False)
    quiet = capsys.readouterr().out
    phon.sgram(x, fs, type="lpc", verbose=True)
    loud = capsys.readouterr().out
    plt.close("all")
    assert "order" in loud and len(loud) > len(quiet)


def test_sgram_multitaper_default_bandwidth(sig):
    x, fs = sig
    _, f, t, S = phon.sgram(x, fs, type="multitaper")
    x2, fs2 = phon.prep_audio(x, fs, target_fs=16000, pre=0.94, verbose=False)  # as in sgram()
    t2, f2, S2 = phon.compute_mt_sgram(x2, fs2, w=0.02, s=0.001, NW=3, order=13)
    plt.close("all")
    assert S.shape == S2.shape
    np.testing.assert_allclose(S, S2)
