"""Tests for phonlab.nonzero_groups."""
import numpy as np
import pandas as pd
import pytest

import phonlab as phon

A = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]


def _reference(a, minlen=1, containing=None):
    """Plain-Python version of nonzero_groups() to compare against."""
    runs, start = [], None
    for i, v in enumerate([bool(x) for x in a] + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append(list(range(start, i)))
            start = None
    if containing is not None:
        wanted = {i if i >= 0 else i + len(a) for i in np.atleast_1d(containing).tolist()}
        runs = [r for r in runs if wanted & set(r)]
    return tuple(np.array(r, dtype=np.intp) for r in runs if len(r) >= minlen)


def _assert_runs_equal(got, expected):
    assert isinstance(got, tuple)
    assert len(got) == len(expected), (got, expected)
    for g, e in zip(got, expected):
        assert isinstance(g, np.ndarray) and np.issubdtype(g.dtype, np.integer)
        np.testing.assert_array_equal(g, e)


def test_basic_runs():
    _assert_runs_equal(phon.nonzero_groups(A), ([1, 2], [4], [7, 8, 9]))


def test_runs_touching_the_ends():
    _assert_runs_equal(phon.nonzero_groups([1, 1, 0, 1]), ([0, 1], [3]))
    _assert_runs_equal(phon.nonzero_groups([1, 1, 1]), ([0, 1, 2],))
    _assert_runs_equal(phon.nonzero_groups([1]), ([0],))


def test_no_runs():
    assert phon.nonzero_groups([0, 0, 0]) == ()
    assert phon.nonzero_groups([]) == ()
    assert phon.nonzero_groups(np.array([], dtype=bool)) == ()


def test_input_is_coerced_to_bool():
    expected = ([1, 2], [4], [7, 8, 9])
    _assert_runs_equal(phon.nonzero_groups(np.array(A, dtype=bool)), expected)
    _assert_runs_equal(phon.nonzero_groups(np.array(A, dtype=float) * 0.5), expected)
    _assert_runs_equal(phon.nonzero_groups(pd.Series(A)), expected)
    _assert_runs_equal(phon.nonzero_groups([-2, 0, 3.5]), ([0], [2]))
    # NaN is truthy, so it counts as non-zero.
    _assert_runs_equal(phon.nonzero_groups([np.nan, 0.0, 1.0]), ([0], [2]))
    # Masked elements count as zero, and the input is not modified.
    m = np.ma.array([0, 1, 1, 0, 2], mask=[0, 0, 1, 0, 0])
    _assert_runs_equal(phon.nonzero_groups(m), ([1], [4]))
    assert m.data.tolist() == [0, 1, 1, 0, 2]


@pytest.mark.parametrize('a', [
    np.array([0, 3, -2, 0, 7]),
    np.array([0.0, -0.0, 1e-300, 5e-324, np.nan, np.inf, -np.inf]),
    np.array([0 + 0j, 0 + 1j, 1 + 0j, 0j]),
    np.array(['', 'a', ' ', '0', '']),
    np.array([None, 0, [], 'x', 0.0, [0], '', 1, {}, {'a': 1}], dtype=object),
    np.array(['NaT', '1970-01-01', '2000-01-01'], dtype='datetime64[s]'),
    np.array([0, 1, 'NaT'], dtype='timedelta64[s]'),
    pd.Series([0.0, np.nan, 1.0, 0.0, 2.0]),
    np.ma.array([1, 1, 0, 1, 1], mask=[0, 1, 0, 0, 1]),
    np.ma.array([1, 1], mask=[1, 1]),
], ids=lambda a: str(getattr(a, 'dtype', type(a).__name__)) + ('/masked' if np.ma.isMaskedArray(a) else ''))
def test_counts_the_same_elements_as_np_nonzero(a):
    runs = phon.nonzero_groups(a)
    got = np.concatenate(runs) if runs else np.array([], dtype=int)
    np.testing.assert_array_equal(got, np.nonzero(a)[0])


def test_runs_index_the_input():
    a = np.random.RandomState(0).rand(200) > 0.5
    runs = phon.nonzero_groups(a)
    assert sum(len(r) for r in runs) == a.sum()
    for r in runs:
        assert a[r].all()
        assert np.array_equal(r, np.arange(r[0], r[-1] + 1))
    assert np.array_equal(np.concatenate(runs), np.flatnonzero(a))


def test_minlen():
    _assert_runs_equal(phon.nonzero_groups(A, minlen=2), ([1, 2], [7, 8, 9]))
    _assert_runs_equal(phon.nonzero_groups(A, minlen=3), ([7, 8, 9],))
    assert phon.nonzero_groups(A, minlen=4) == ()
    _assert_runs_equal(phon.nonzero_groups(A, minlen=0), ([1, 2], [4], [7, 8, 9]))


def test_containing():
    _assert_runs_equal(phon.nonzero_groups(A, containing=4), ([4],))
    _assert_runs_equal(phon.nonzero_groups(A, containing=[4, 8]), ([4], [7, 8, 9]))
    _assert_runs_equal(phon.nonzero_groups(A, containing=np.array([1, 2])), ([1, 2],))
    _assert_runs_equal(phon.nonzero_groups(A, containing=[1, 9, 9]), ([1, 2], [7, 8, 9]))
    # Negative indexes count from the end, as in numpy.
    _assert_runs_equal(phon.nonzero_groups(A, containing=-1), ([7, 8, 9],))
    _assert_runs_equal(phon.nonzero_groups(A, containing=[-6, -len(A)]), ([4],))
    # Indexes on zero elements, or no indexes at all, match no run.
    assert phon.nonzero_groups(A, containing=3) == ()
    assert phon.nonzero_groups(A, containing=[0, 5, 6]) == ()
    assert phon.nonzero_groups(A, containing=-len(A)) == ()
    with pytest.warns(UserWarning, match='empty'):
        assert phon.nonzero_groups(A, containing=[]) == ()
    with pytest.warns(UserWarning, match='empty'):
        assert phon.nonzero_groups([], containing=[]) == ()


def test_minlen_and_containing_apply_together():
    _assert_runs_equal(phon.nonzero_groups(A, minlen=2, containing=[4, 8]), ([7, 8, 9],))
    assert phon.nonzero_groups(A, minlen=2, containing=4) == ()


def test_containing_must_be_integer():
    with pytest.raises(TypeError):
        phon.nonzero_groups(A, containing=1.5)
    with pytest.raises(TypeError):
        phon.nonzero_groups(A, containing=[True, False])


def test_containing_out_of_bounds_raises():
    for bad in (len(A), -len(A) - 1, [1, 99], [-99, 1], np.array([len(A)])):
        with pytest.raises(IndexError):
            phon.nonzero_groups(A, containing=bad)
    with pytest.raises(IndexError):
        phon.nonzero_groups([], containing=0)


def test_input_must_be_one_dimensional():
    with pytest.raises(ValueError):
        phon.nonzero_groups(True)
    with pytest.raises(ValueError):
        phon.nonzero_groups([[0, 1], [1, 0]])


def test_matches_reference_on_random_input():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(0, 40))
        a = rng.random(n) < rng.random()
        minlen = int(rng.integers(0, 5))
        r = rng.random()
        if r < 0.4 or n == 0:
            containing = None
        elif r < 0.6:
            containing = int(rng.integers(-n, n))                      # in bounds, negatives included
        else:
            containing = rng.integers(-n, n, size=rng.integers(1, 6)).tolist()
        _assert_runs_equal(phon.nonzero_groups(a, minlen=minlen, containing=containing),
                           _reference(a, minlen=minlen, containing=containing))
