import numpy as np
import pandas as pd

def _nearest_index(src_t, target_t):
    '''Index into the sorted array `src_t` of the value nearest each value of `target_t`.'''
    right = np.clip(np.searchsorted(src_t, target_t), 1, len(src_t) - 1)
    left = right - 1
    return np.where(np.abs(src_t[left] - target_t) <= np.abs(src_t[right] - target_t), left, right)

def align_timeseries(base_df, other_df, ts='sec', other_ts=None):
    '''
    Add the columns of one time-indexed dataframe to another, aligned in time. The rows
    and values of `base_df` are not changed, so the output has the same time axis as
    `base_df`.

    Numeric columns of `other_df` that have no missing values are linearly interpolated at
    the times of `base_df` (NaN outside the time range of `other_df`). All other columns
    (non-numeric, boolean, or containing NaN) are not interpolated: each takes the value
    (which may be NaN) at the time in `other_df` that is nearest to the target time.

Parameters
----------

base_df : dataframe
    The dataframe to add columns to.

other_df : dataframe
    The dataframe providing the new columns.

ts : str (default 'sec')
    Name of the time column in `base_df`.

other_ts : str (default None)
    Name of the time column in `other_df`. If None, the same name as `ts` is used.

Returns
-------

df : dataframe
    A copy of `base_df` with the columns of `other_df` (other than its time column) added.
    Column names that already occur in `base_df` raise a ValueError.

Example
-------

.. code-block:: Python

    x, fs = phon.loadsig(example_file, chansel=[0])
    IFCdf = phon.track_formants(x, fs, method='ifc', speaker=2)
    Voicingdf = phon.vad(x, fs)
    merged = phon.align_timeseries(IFCdf, Voicingdf, ts='sec')
    '''
    other_ts = ts if other_ts is None else other_ts
    clash = (set(other_df.columns) - {other_ts}) & set(base_df.columns)
    if clash:
        raise ValueError(f'Column(s) {sorted(clash)} of `other_df` already exist in `base_df`.')

    other = other_df.sort_values(other_ts)
    src_t = other[other_ts].to_numpy(dtype=float)
    target_t = base_df[ts].to_numpy(dtype=float)
    out = base_df.copy()
    if len(other) == 0:
        return out
    nearest = _nearest_index(src_t, target_t) if len(other) > 1 else np.zeros(len(target_t), dtype=int)

    new = {}
    for c in other.columns.drop(other_ts):
        s = other[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and not s.isna().any():
            new[c] = np.interp(target_t, src_t, s.to_numpy(dtype=float), left=np.nan, right=np.nan)
        else:
            new[c] = s.iloc[nearest].to_numpy()
    return out.assign(**new)
