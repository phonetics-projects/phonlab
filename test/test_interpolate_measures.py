"""Tests for phonlab.interpolate_measures."""

import numpy as np
import pandas as pd
import pytest

from phonlab.utils.tidy import interpolate_measures


def _make_meas_df(start=0.0, stop=1.0, step=0.01, cols=("F1", "F2")):
    """Create a simple measurement dataframe with known linear columns."""
    ts = np.arange(start, stop + step / 2, step)
    data = {"sec": ts}
    for i, col in enumerate(cols, start=1):
        data[col] = ts * i  # perfectly linear: F1 = t, F2 = 2t, ...
    return pd.DataFrame(data)


def test_basic_interpolation():
    """Linearly interpolated values are correct."""
    meas = _make_meas_df()
    interp_times = np.array([0.02, 0.05, 0.08])
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times
    )
    np.testing.assert_allclose(result["F1"], interp_times, atol=1e-10)
    np.testing.assert_allclose(result["F2"], interp_times * 2, atol=1e-10)


def test_with_interp_df():
    """Results merge into the input dataframe."""
    meas = _make_meas_df()
    interp_df = pd.DataFrame({"obs_t": [0.1, 0.2, 0.3], "label": ["a", "b", "c"]})
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_df=interp_df, interp_ts="obs_t"
    )
    assert "label" in result.columns
    assert "F1" in result.columns
    assert "F2" in result.columns
    np.testing.assert_allclose(result["F1"], [0.1, 0.2, 0.3], atol=1e-10)


def test_without_interp_df_returns_standalone():
    """Returns standalone dataframe from array input."""
    meas = _make_meas_df()
    interp_times = np.array([0.05, 0.15])
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times
    )
    assert "tcol" in result.columns
    assert len(result) == 2


def test_default_tolerance():
    """Default tol is auto-computed as half the measurement timestep."""
    meas = _make_meas_df(step=0.01)  # step=0.01 → tol=0.005
    interp_times = np.array([0.013])  # 0.003 from nearest, within tol
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times
    )
    assert len(result) == 1


def test_out_of_range_raises():
    """Interp points beyond measurement range + tol raise ValueError."""
    meas = _make_meas_df(start=0.0, stop=1.0, step=0.01)
    interp_times = np.array([2.0])
    with pytest.raises(ValueError, match="tolerance"):
        interpolate_measures(
            meas_df=meas, meas_ts="sec", interp_ts=interp_times
        )


def test_gappy_data_raises():
    """Interp point in a gap wider than tol raises ValueError."""
    ts = np.concatenate([np.arange(0, 0.5, 0.01), np.arange(1.0, 1.5, 0.01)])
    data = {"sec": ts, "F1": ts}
    meas = pd.DataFrame(data)
    interp_times = np.array([0.75])
    with pytest.raises(ValueError, match="tolerance"):
        interpolate_measures(
            meas_df=meas, meas_ts="sec", interp_ts=interp_times, tol=0.01
        )


def test_gappy_data_large_tol_passes():
    """Passes when tolerance is wide enough to cover the gap."""
    ts = np.concatenate([np.arange(0, 0.5, 0.01), np.arange(1.0, 1.5, 0.01)])
    data = {"sec": ts, "F1": ts}
    meas = pd.DataFrame(data)
    interp_times = np.array([0.75])
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times, tol=1.0
    )
    assert len(result) == 1


def test_overwrite_true():
    """overwrite=True allows overlapping column names."""
    meas = _make_meas_df()
    interp_df = pd.DataFrame({"obs_t": [0.1, 0.2], "F1": [999.0, 999.0]})
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec",
        interp_df=interp_df, interp_ts="obs_t",
        overwrite=True,
    )
    np.testing.assert_allclose(result["F1"], [0.1, 0.2], atol=1e-10)


def test_overwrite_false_raises_on_overlap():
    """overwrite=False raises ValueError on column overlap."""
    meas = _make_meas_df()
    interp_df = pd.DataFrame({"obs_t": [0.1, 0.2], "F1": [999.0, 999.0]})
    with pytest.raises(ValueError, match="overlap"):
        interpolate_measures(
            meas_df=meas, meas_ts="sec",
            interp_df=interp_df, interp_ts="obs_t",
            overwrite=False,
        )


def test_non_numeric_column_raises():
    """Non-numeric column raises TypeError."""
    df = pd.DataFrame({"sec": [0.0, 0.1, 0.2], "label": ["a", "b", "c"]})
    interp_times = np.array([0.1])
    with pytest.raises(TypeError, match="interpolate"):
        interpolate_measures(
            meas_df=df, meas_ts="sec", interp_ts=interp_times
        )


def test_non_increasing_meas_ts_raises():
    """Non-increasing meas_ts raises ValueError."""
    df = pd.DataFrame({"sec": [0.0, 0.2, 0.1], "F1": [0.0, 0.2, 0.1]})
    interp_times = np.array([0.0])
    with pytest.raises(ValueError, match="increasing"):
        interpolate_measures(
            meas_df=df, meas_ts="sec", interp_ts=interp_times
        )


def test_exact_boundary_values():
    """Interp at first/last measurement time works."""
    meas = _make_meas_df(start=0.0, stop=1.0, step=0.01)
    first = meas["sec"].iloc[0]
    last = meas["sec"].iloc[-1]
    interp_times = np.array([first, last])
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times
    )
    np.testing.assert_allclose(result["F1"], [first, last], atol=1e-12)
    np.testing.assert_allclose(result["F2"], [first * 2, last * 2], atol=1e-12)


def test_interp_at_exact_measurement_times():
    """Returns exact values with no interpolation error."""
    meas = _make_meas_df()
    exact_times = meas["sec"].iloc[[0, 10, 50, 100]].values
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=exact_times
    )
    np.testing.assert_allclose(result["F1"], exact_times, atol=1e-12)
    np.testing.assert_allclose(result["F2"], exact_times * 2, atol=1e-12)


def test_no_interp_df_overwrite_false():
    """interp_df=None + overwrite=False should not crash."""
    meas = _make_meas_df()
    interp_times = np.array([0.05, 0.10])
    result = interpolate_measures(
        meas_df=meas, meas_ts="sec", interp_ts=interp_times, overwrite=False
    )
    assert len(result) == 2


def test_error_message_with_numpy_interp_ts():
    """ValueError message works when interp_ts is a numpy array."""
    meas = _make_meas_df(start=0.0, stop=1.0, step=0.01)
    interp_times = np.array([5.0])
    with pytest.raises(ValueError, match="tolerance"):
        interpolate_measures(
            meas_df=meas, meas_ts="sec", interp_ts=interp_times
        )
