"""egg_to_oq() with and without its diagnostics."""
import importlib.resources
import inspect
import warnings

import numpy as np
import pytest

import phonlab as phon

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def egg():
    f = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'F1_bha24_1.wav'
    _, e, fs = phon.loadsig(f, chansel=[0, 1])
    return e, fs


def test_default_columns(egg):
    assert list(phon.egg_to_oq(*egg).columns) == ['sec', 'OQ', 'f0', 'voiced']


def test_diagnostics_do_not_change_results(egg):
    base, diag = phon.egg_to_oq(*egg), phon.egg_to_oq(*egg, diagnostics=True)
    assert base.equals(diag[['sec', 'OQ', 'f0', 'voiced']])


def test_diagnostics_explain_missing_oq(egg):
    d = phon.egg_to_oq(*egg, diagnostics=True)
    assert set(d.columns) >= {'n_gci', 'n_goi', 'degg_max', 'egg_min', 'egg_max', 'egg_range', 'fail'}
    assert ((d.fail == '') == d.OQ.notna()).all()
    assert set(d.fail) <= {'', 'few_gci', 'few_goi', 'no_opening'}
    ok = d[d.fail == '']
    assert (ok.n_gci >= 2).all() and (ok.n_goi >= 2).all()
    assert (d[d.fail == 'few_gci'].n_gci < 2).all()


def test_voiced_column(egg):
    d = phon.egg_to_oq(*egg, diagnostics=True)
    assert d.voiced.dtype == bool
    assert (d[d.voiced].n_gci >= 2).all()
    assert d.voiced[d.OQ.notna()].mean() > 0.9      # nearly every frame with an OQ has two closures a period apart


def test_normalization_options(egg):
    x, fs = egg
    long = np.tile(x, 12)                           # longer than any window
    base = phon.egg_to_oq(long, fs)
    for kw in (dict(norm_window=0.3, center=True), dict(norm_window=0.3, center=True, floor=True),
               dict(norm_window=1.5, egg_norm_window=0.5), dict(peak_height=0.3)):
        d = phon.egg_to_oq(long, fs, **kw)
        assert len(d) == len(base) and d.voiced.dtype == bool
    # a higher floor can only remove closing instants, so it cannot find more voiced frames than a lower one
    low = phon.egg_to_oq(long, fs, norm_window=0.3, center=True, floor=0.02).voiced.sum()
    high = phon.egg_to_oq(long, fs, norm_window=0.3, center=True, floor=0.5).voiced.sum()
    assert high <= low


def test_default_normalization_window():
    params = inspect.signature(phon.egg_to_oq).parameters
    assert params['norm_window'].default == 0.7 and params['center'].default is True
    assert params['floor'].default is True


def test_floor_silences_a_weak_signal():
    # a weak noise signal followed by a strong periodic one: the noise should not be found voiced
    fs = 16000
    rng = np.random.default_rng(0)
    t = np.arange(fs) / fs
    weak = rng.normal(0, 0.002, fs)
    strong = np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t) + rng.normal(0, 0.002, fs)
    x = np.r_[weak, strong]
    d = phon.egg_to_oq(x, fs)
    assert d.voiced[d.sec < 0.8].mean() < 0.05
    assert d.voiced[d.sec > 1.2].mean() > 0.5
    assert phon.egg_to_oq(x, fs, floor=None).voiced[d.sec < 0.8].mean() > 0.5    # without the floor the noise looks voiced
