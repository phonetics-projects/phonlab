"""Tests for phonlab.align_timeseries."""

import numpy as np
import pandas as pd
import pytest

from phonlab.utils.align_series import align_timeseries


def _base():
    return pd.DataFrame({"sec": [0.0, 0.10, 0.20, 0.30], "F1": [500, 510, 520, 530]})


def test_numeric_interpolated_and_base_unchanged():
    base = _base()
    orig = base.copy()
    other = pd.DataFrame({"sec": [0.0, 0.2, 0.4], "cpp": [0.0, 2.0, 4.0]})
    out = align_timeseries(base, other)
    np.testing.assert_allclose(out["cpp"], [0.0, 1.0, 2.0, 3.0])
    pd.testing.assert_frame_equal(out[["sec", "F1"]], orig)
    pd.testing.assert_frame_equal(base, orig)  # input not modified


def test_categorical_takes_nearest():
    base = pd.DataFrame({"sec": [0.0, 0.04, 0.06, 0.19]})
    other = pd.DataFrame({"sec": [0.0, 0.1, 0.2], "lab": ["a", "b", "c"]})
    out = align_timeseries(base, other)
    assert out["lab"].tolist() == ["a", "a", "b", "c"]


def test_numeric_with_nan_takes_nearest_not_interpolated():
    base = pd.DataFrame({"sec": [0.02, 0.06, 0.14, 0.18]})
    other = pd.DataFrame({"sec": [0.0, 0.1, 0.2], "f0": [100.0, np.nan, 200.0]})
    out = align_timeseries(base, other)
    # 0.02 -> t=0.0 (100); 0.06, 0.14 -> t=0.1 (NaN); 0.18 -> t=0.2 (200)
    assert out["f0"].iloc[0] == 100.0
    assert np.isnan(out["f0"].iloc[1]) and np.isnan(out["f0"].iloc[2])
    assert out["f0"].iloc[3] == 200.0


def test_numeric_with_nan_does_not_blend_values():
    base = pd.DataFrame({"sec": [0.05]})
    other = pd.DataFrame({"sec": [0.0, 0.1], "f0": [100.0, np.nan]})
    out = align_timeseries(base, other)
    # tie goes to the earlier time; value is 100, not an interpolation
    assert out["f0"].iloc[0] == 100.0


def test_bool_column_takes_nearest():
    base = pd.DataFrame({"sec": [0.01, 0.09]})
    other = pd.DataFrame({"sec": [0.0, 0.1], "voiced": [True, False]})
    out = align_timeseries(base, other)
    assert out["voiced"].tolist() == [True, False]


def test_outside_range_numeric_is_nan():
    base = pd.DataFrame({"sec": [-0.1, 0.05, 0.5]})
    other = pd.DataFrame({"sec": [0.0, 0.1], "x": [0.0, 1.0]})
    out = align_timeseries(base, other)
    assert np.isnan(out["x"].iloc[0]) and np.isnan(out["x"].iloc[2])
    assert out["x"].iloc[1] == pytest.approx(0.5)


def test_other_time_column_name_and_unsorted_input():
    base = _base()
    other = pd.DataFrame({"t": [0.3, 0.0, 0.2, 0.1], "y": [3.0, 0.0, 2.0, 1.0]})
    out = align_timeseries(base, other, ts="sec", other_ts="t")
    np.testing.assert_allclose(out["y"], [0.0, 1.0, 2.0, 3.0])
    assert "t" not in out.columns


def _clash_other():
    return pd.DataFrame({"sec": [0.0, 0.3], "F1": [1.0, 2.0], "new": [5.0, 6.0]})


def test_duplicate_names_ignored_by_default():
    out = align_timeseries(_base(), _clash_other())
    assert out["F1"].tolist() == [500, 510, 520, 530]  # base values kept
    assert list(out.columns) == ["sec", "F1", "new"]


def test_duplicate_names_renamed():
    out = align_timeseries(_base(), _clash_other(), ignore_duplicates=False)
    assert list(out.columns) == ["sec", "F1", "F1_2", "new"]
    assert out["F1"].tolist() == [500, 510, 520, 530]
    np.testing.assert_allclose(out["F1_2"], [1.0, 1 + 1 / 3, 1 + 2 / 3, 2.0])


def test_duplicate_names_chain_of_merges():
    base = _base()
    for _ in range(3):
        base = align_timeseries(base, _clash_other(), ignore_duplicates=False)
    assert list(base.columns) == ["sec", "F1", "F1_2", "new", "F1_3", "new_2", "F1_4", "new_3"]


def test_rename_avoids_other_columns_in_other():
    other = pd.DataFrame({"sec": [0.0, 0.3], "F1": [1.0, 2.0], "F1_2": [3.0, 4.0]})
    out = align_timeseries(_base(), other, ignore_duplicates=False)
    # other's F1 takes F1_2, so other's own F1_2 must move on to F1_2_2
    assert list(out.columns) == ["sec", "F1", "F1_2", "F1_2_2"]
    np.testing.assert_allclose(out["F1_2"].iloc[[0, -1]], [1.0, 2.0])
    np.testing.assert_allclose(out["F1_2_2"].iloc[[0, -1]], [3.0, 4.0])


def test_other_columns_selects_subset():
    other = pd.DataFrame({"sec": [0.0, 0.3], "a": [0.0, 3.0], "b": [1.0, 1.0], "c": ["x", "y"]})
    out = align_timeseries(_base(), other, other_columns=["a", "c"])
    assert list(out.columns) == ["sec", "F1", "a", "c"]


def test_other_columns_duplicate_respects_flag():
    out = align_timeseries(_base(), _clash_other(), other_columns=["F1"])
    assert list(out.columns) == ["sec", "F1"]
    out = align_timeseries(_base(), _clash_other(), other_columns=["F1"], ignore_duplicates=False)
    assert list(out.columns) == ["sec", "F1", "F1_2"]


def test_other_columns_missing_raises():
    with pytest.raises(KeyError):
        align_timeseries(_base(), _clash_other(), other_columns=["nope"])
