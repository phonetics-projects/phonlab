import numpy as np
import pandas as pd

def _nearest_index(src_t, target_t):
    '''Index into the sorted array `src_t` of the value nearest each value of `target_t`.'''
    right = np.clip(np.searchsorted(src_t, target_t), 1, len(src_t) - 1)
    left = right - 1
    return np.where(np.abs(src_t[left] - target_t) <= np.abs(src_t[right] - target_t), left, right)

def _unique_name(name, taken):
    '''Return `name` if it is not in `taken`, else the first of name_2, name_3, ... that is not.'''
    new, n = name, 1
    while new in taken:
        n += 1
        new = f'{name}_{n}'
    return new

def align_timeseries(base_df, other_df, ts='sec', other_ts=None, other_columns=None,
                     ignore_duplicates=True):
    '''
    Add the columns of one time-indexed dataframe to another, aligned in time. The rows
    and values of `base_df` are not changed, so the output has the same time axis as
    `base_df`.

    Numeric columns of `other_df` that have no missing values are linearly interpolated at
    the times of `base_df` (NaN outside the time range of `other_df`). All other columns
    (non-numeric, boolean, or containing NaN) are not interpolated: each takes the value
    (which may be NaN) at the time in `other_df` that is nearest to the target time.

    If a column of `other_df` has the same name as a column of `base_df`, it is skipped
    (the default) or renamed with a numeric suffix (`ignore_duplicates=False`).

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

other_columns : list of str (default None)
    Names of the columns of `other_df` to add. If None, all columns except the time
    column are added. A KeyError is raised if a name is not a column of `other_df`.

ignore_duplicates : bool (default True)
    What to do with a column of `other_df` whose name is already a column of `base_df`.
    If True, the column is skipped and the `base_df` column is kept. If False, the column
    is added with a numeric suffix: `f0` becomes `f0_2`, or `f0_3` if `f0_2` is also
    taken, and so on. The check is against all the columns of `base_df`, so repeated
    calls that merge a series of dataframes that all have an `f0` column give `f0`, `f0_2`,
    `f0_3`, ....

Returns
-------

df : dataframe
    A copy of `base_df` with the selected columns of `other_df` added.

Example
-------

.. code-block:: Python

    x, fs = phon.loadsig(example_file, chansel=[0])
    IFCdf = phon.track_formants(x, fs, method='ifc', speaker=2)
    Voicingdf = phon.vad(x, fs)
    merged = phon.align_timeseries(IFCdf, Voicingdf, ts='sec')

    # only some columns, keeping (and renaming) any that duplicate column names
    merged = phon.align_timeseries(IFCdf, Voicingdf, other_columns=['f0', 'amp'],
                                   ignore_duplicates=False)
    '''
    other_ts = ts if other_ts is None else other_ts
    if other_columns is None:
        other_columns = [c for c in other_df.columns if c != other_ts]
    else:
        missing = [c for c in other_columns if c not in other_df.columns]
        if missing:
            raise KeyError(f'Column(s) {missing} not found in `other_df`.')
        other_columns = [c for c in other_columns if c != other_ts]

    # Map each column to add to its name in the output.
    taken = set(base_df.columns)
    names = {}
    for c in other_columns:
        if c in base_df.columns and ignore_duplicates:
            continue
        names[c] = _unique_name(c, taken)
        taken.add(names[c])

    other = other_df.sort_values(other_ts)
    src_t = other[other_ts].to_numpy(dtype=float)
    target_t = base_df[ts].to_numpy(dtype=float)
    out = base_df.copy()
    if len(other) == 0:
        return out
    nearest = _nearest_index(src_t, target_t) if len(other) > 1 else np.zeros(len(target_t), dtype=int)

    new = {}
    for c, newname in names.items():
        s = other[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and not s.isna().any():
            new[newname] = np.interp(target_t, src_t, s.to_numpy(dtype=float), left=np.nan, right=np.nan)
        else:
            new[newname] = s.iloc[nearest].to_numpy()
    return out.assign(**new)
