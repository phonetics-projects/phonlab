import numpy as np
import pytest
import phonlab as phon


def test_mt_sgram_psd_integrates_to_variance():
    rng = np.random.default_rng(1)
    fs = 16000
    x = rng.standard_normal(2 * fs) * 0.5
    t, f, S = phon.compute_mt_sgram(x, fs, w=0.02, s=0.02, order=12)
    p = np.mean(10 ** (S / 10), axis=1)
    assert np.trapezoid(p, f) == pytest.approx(x.var(), rel=0.05)


def test_mt_sgram_shape_matches_compute_sgram():
    x = np.random.default_rng(2).standard_normal(16000)
    t, f, S = phon.compute_mt_sgram(x, 16000)
    t2, f2, S2 = phon.compute_sgram(x, 16000, 0.008)
    assert S.shape == S2.shape
    assert np.allclose(f, f2)


def test_mt_sgram_scale_nitime_offset():
    x = np.random.default_rng(3).standard_normal(4000)
    _, _, a = phon.compute_mt_sgram(x, 16000, scale='psd')
    _, _, b = phon.compute_mt_sgram(x, 16000, scale='nitime')
    assert np.allclose(b - a, 10 * np.log10(16000 / (2 * np.pi)))


def test_tandem_spectrum_averages_power_not_db():
    S = np.array([[0.0, 10.0]])   # one frequency, two frames
    assert phon.tandem_spectrum(S)[0] == pytest.approx(10 * np.log10(5.5))


def test_mt_sgram_order_none_has_no_padding():
    x = np.random.default_rng(5).standard_normal(3200)
    _, f, S = phon.compute_mt_sgram(x, 16000, w=0.02, s=0.02, order=None)
    assert len(f) == 320 // 2 + 1


def test_tandem_fric_total_and_edges():
    x = np.random.default_rng(4).standard_normal(32000)
    with pytest.raises(ValueError):
        phon.tandem_fric(x, 32000, 0.5, w=0.02, total=0.01)
    with pytest.raises(ValueError):
        phon.tandem_fric(x, 32000, 0.005)
    r = phon.tandem_fric(x, 32000, 0.5)
    r1 = phon.tandem_fric(x, 32000, 0.5, total=0.02)
    assert r.freq.shape == r1.freq.shape
