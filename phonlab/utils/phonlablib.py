__all__=['tg_to_df', 'add_context', 'merge_tiers', 'explode_intervals', 'interpolate_measures', 'loadsig']

import librosa
import numpy as np
import pandas as pd
from parselmouth.praat import call as pcall

def tg_to_df(tg, tiersel=[], names=None):
    '''
Read a Praat textgrid and return its tiers as a list of dataframes.

Parameters
----------

tg : path-like
    Filepath of input textgrid.

tiersel : list of str or int
    Selection of tiers to parse and include in the output list, identified by tier name for `str` or `0`-based integer index for `int`. If `[]` then all textgrid tiers are returned. The order of the tiers does not have to match the input textgrid.

names : None, str, or list of str (default None)
    Names of the label content columns in the output dataframes. If `None`, then the column name for each tier is the tier name. If `str` then the same column name will be used for all dataframes. If list, then the names match the corresponding tiers in `tiersel`.

Returns
-------

tiers : list of dataframes
    Textgrid tiers are parsed in the order selected by `tiersel` and returned as separate dataframes for each tier. The columns of each dataframe are named `t1` and `t2` for label start and end times of interval tiers, and `t1` for the timepoints of point tiers. The textgrid tier's name is used as the name of the column containing the label content unless column names are provided by `names`. If `tiers` is an empty list `[]` then all tiers are returned.
    '''
    tg = pcall('Read from file...', str(tg))[0]
    ntiers = pcall(tg, 'Get number of tiers')
    tiers = []
    tiermap = {pcall(tg, 'Get tier name...', n+1): n for n in range(ntiers)}
    if tiersel == []:
        tiersel = range(ntiers)
    else:
        for n in range(len(tiersel)):
            if not isinstance(tiersel[n], int):
                tiersel[n] = tiermap[tiersel[n]]
    if isinstance(names, str):
        names = [names] * ntiers
    for i, n in enumerate(tiersel):
        try:
            tiername = names[i] if names is not None else pcall(tg, 'Get tier name...', n+1)
        except IndexError:
            msg = f'Not enough names listed in `names`. There are {len(names)} names for {ntiers} selected tiers.'
            raise ValueError(msg) from None
        recs = []
        if pcall(tg, 'Is interval tier...', n+1) is True:
            nlabels = pcall(tg, 'Get number of intervals...', n+1)
            for i in range(nlabels):
                recs.append({
                    't1': pcall(tg, 'Get start time of interval...', n+1, i+1),
                    't2': pcall(tg, 'Get end time of interval...', n+1, i+1),
                    tiername: pcall(tg, 'Get label of interval...', n+1, i+1)
                })
        else:
            nlabels = pcall(tg, 'Get number of points...', n+1)
            for i in range(nlabels):
                recs.append({
                    't1': pcall(tg, 'Get time of point...', n+1, i+1),
                    tiername: pcall(tg, 'Get label of point...', n+1, i+1)
                })
        tiers.append(pd.DataFrame(recs))
    return tiers

def add_context(df, col, nprev, nnext, prefixes=['prev_', 'next_'], fillna='', ctxcol=None, sep=' '):
    '''
Add shifted versions of a dataframe column to provide context within rows.  For example, usee this function if you want to add a column in a dataframe full of vowel measurements for the preceeding or following segmental context.

Parameters
----------

df : dataframe
    The input dataframe.

col : str
    The name of the column for which context is desired.

nprev : int
    The number of preceding values of `col` to add as context.

nnext : int
    The number of following values of `col` to add as context.

prefixes : list of str (default [`'prev_'`, `'next_'`])
    Prefixes to use as column names. The first value of the list is the prefix to use for preceding context, and the second value is the prefix for following context, e.g. 'prev_word2', 'prev_word1', 'next_word1', 'next_word2'.

fillna : str (default '')
    Value to use to fill empty values created by `shift` at beginning and end of `col`.

ctxcol : str or None (default None)
    If not None, add a string column that `join`s `col` and its preceding/following context in order, separated by `sep`.

sep : str (default ' ')
        String separator used to `join` context in `ctxcol`.

Returns
-------

df : dataframe
    The original input dataframe with new context columns added. Note that the new columns are inserted in order around `col`.
    '''
    colidx = df.columns.get_loc(col)
    nextrng = range(nnext, 0, -1)
    nextcols = [f'{prefixes[1]}{col}{n}' for n in nextrng]
    for nshift, newcol in zip(nextrng, nextcols):
        df.insert(
            colidx+1, newcol, df[col].shift(nshift * -1).fillna(fillna), allow_duplicates=False
        )
    prevrng = range(1, nprev+1)
    prevcols = [f'{prefixes[0]}{col}{n}' for n in prevrng]
    for nshift, newcol in zip(prevrng, prevcols):
        df.insert(
            colidx, newcol, df[col].shift(nshift).fillna(fillna), allow_duplicates=False
        )
    if ctxcol is not None:
        newcol = df[prevcols + [col] + nextcols].apply(
            lambda x: sep.join(x),
            axis='columns'
        )
        df.insert(colidx + nprev + nnext + 1, ctxcol, newcol, allow_duplicates=False)
    return df

def merge_tiers(inner_df, outer_df, suffixes, inner_ts=['t1','t2'], outer_ts=['t1','t2'],
    drop_repeated_cols=None):
    '''
Merge hierarchical dataframe tiers based on their times.

Parameters
----------

inner_df : dataframe
     The dataframe whose intervals are properly contained inside `outer_df`.

outer_df : dataframe
      The dataframe whose intervals contain one or more intervals from
      `inner_df`.

suffixes : list of str
    List of suffixes to add to time columns in the output dataframe. The first
    suffix is added to the names in `inner_ts`, and the second suffix is added
    to the names in `outer_ts`. If the names in `inner_ts` and `outer_ts` do
    not overlap, then empty string suffixes may be appropriate.

inner_ts : list of str
    Names of the columns that define time intervals in `inner_df`. The first
    value is the start time of the interval, and the second value is the end
    time. For point tiers, only one column should be named.

outer_ts : list of str
    Names of the columns that define time intervals in `outer_df`. The first
    value is the start time of the interval, and the second value is the end
    time. For point tiers, only one column should be named.

drop_repeated_cols : str ('inner', 'inner_df', 'outer', 'outer_df', None)
    Drop each column from the specified dataframe if there is a column with an
    identical label in the other input dataframe. The `inner_ts` and `outer_ts`
    columns are excluded from being dropped. If None, no columns are dropped.

Returns
-------

mergedf : dataframe
    Merged dataframe of time-matched rows from `inner_df` and `outer_df`.
    '''
    common_cols = np.intersect1d(inner_df.columns, outer_df.columns)
    if drop_repeated_cols in ['inner', 'inner_df']:
        innerdropcols = np.setdiff1d(common_cols, inner_ts)
        outerdropcols = []
    elif drop_repeated_cols in ['outer', 'outer_df']:
        innerdropcols = []
        outerdropcols = np.setdiff1d(common_cols, outer_ts)
    else:
        innerdropcols = []
        outerdropcols = []
    innert1col = f'{inner_ts[0]}{suffixes[0]}'
    innerrenamecols = {inner_ts[0]: innert1col}
    if len(inner_ts) > 1:
        innerrenamecols[inner_ts[1]] =  f'{inner_ts[1]}{suffixes[0]}'
    outert1col = f'{outer_ts[0]}{suffixes[1]}'
    outerrenamecols = {outer_ts[0]: outert1col}
    if len(outer_ts) > 1:
        outerrenamecols[outer_ts[1]] =  f'{outer_ts[1]}{suffixes[1]}'
    try:
        mergedf = pd.merge_asof(
            inner_df.drop(columns=innerdropcols) \
                .rename(innerrenamecols, axis='columns'),
            outer_df.drop(columns=outerdropcols) \
                .rename(outerrenamecols, axis='columns'),
            left_on=innert1col,
            right_on=outert1col
        )
    except KeyError:
        msg = f'Time column(s) {inner_ts} not found in `inner_df` or {outer_ts} not found in `outer_df`. Select valid column names in the `inner_ts` and `outer_ts` parameters.'
        raise KeyError(msg) from None
    return mergedf

def explode_intervals(divs, ts=['t1', 't2'], df=None, prefix='obs_'):
    '''
    Divide a series of time intervals into subintervals and explode into long format,
    with one row per subinterval timepoint. An interval [2.0, 3.0] divided into two
    subintervals, for example, produces three output rows for the times corresponding
    to 0%, 50%, and 100% of the interval: 2.0, 2.5, 3.0.
    
    The subinterval divisions can be specified as an integer number of subdivisions,
    or as a list of interval proportions. For `int` the number of timepoints produced
    is the number of subintervals + 1, and for a list of proportions one timepoint
    is produced for each element of the list.

Parameters
----------

divs : int, list of float in range [0.0, 1.0]
    The subintervals to include. If `int`, the number of equal-duration subintervals
    each interval will be divided into. If a list, the values should be in the range
    [0.0, 1.0] and express proportions of the duration of each interval for which
    subinterval timepoints will be created. For example, `[0.25, 0.50, 0.75]` yield
    timepoints at 25%, 50%, and 75% of each input interval.

ts : list of str or list of Series/arrays
    If list of `str`, these are the names of time columns in the `df` dataframe. The
    first name defines the start time of the interval to be subdivided, and the second
    name defines the end time. If list of `pd.Series` or `np.array`, then the first
    Series/array provides the start times, and the second Series/array provides the
    end times.

df : dataframe
    A dataframe containing start and end times of the intervals to be subdivided, or
    None if `ts` provides the times directly as Series/arrays rather than names.
    An arbitrary number of additional columns may be included in the dataframe.

prefix : str (default 'obs_t')
    The prefix to use when naming the output columns of timepoints (f'{prefix}n') and
    timepoint identifiers (f'{prefix}id').

Returns
-------

divdf : dataframe
    A dataframe of subinterval timepoints with an index that matches the index of
    `ts`. The timepoints are in a column labelled `obs_t` by default. A second
    column that identifies the timepoint's location within the series of timepoints
    is named `obs_id` by default. If `divs` is an `int` these identifiers are
    integers in the range [0, divs]. If `divs` is a list of proportions, the
    proportions are used as the identifiers.


Note
----
    
    `divdf` is merged with the input dataframe `df` if it is provided. If this
    behavior is not desired, then `df` should be None. For example, use
    `ts=[df['t1'], df['t2']], df=None` instead of `ts=['t1', 't2'], df=df`.
    '''
    # TODO: test different kinds of indexes in input dataframe, e.g. MultiIndex
    t1col = ts[0] if df is None else df[ts[0]]
    t2col = ts[1] if df is None else df[ts[1]]
    id_vars = None if df is None else df.index.name
    tindex = t1col.index if isinstance(t1col, pd.Series) else np.arange(len(t1col))
    if isinstance(tindex, pd.Index):
        try:
            assert(~tindex.duplicated().any())
        except AssertionError:
            msg = 'The index of the input dataframe must not contain duplicate values.'
            raise ValueError(msg) from None
        try:
            assert((t1col.index == t2col.index).all())
        except AssertionError:
            msg = 'The indexes of the input `ts` must match each other.'
            raise ValueError(msg) from None

    if isinstance(divs, int):
        obs_t = np.linspace(t1col, t2col, num=divs+1, endpoint=True).transpose()
        obs_id = np.tile(np.arange(divs+1), len(t1col))
    else:
        obs_t = (
            np.expand_dims((t2col - t1col), axis=1) * np.expand_dims(divs, axis=0)
        ) + np.expand_dims(t1col, axis=1)
        obs_id = np.tile(np.array(divs), len(t1col))

    obsidcol, obstimecol = f'{prefix}id', f'{prefix}t'
    # .tolist() converts the 2d arrays into list of 1d arrays
    divdf = pd.DataFrame(
        {
            'obs_t': obs_t.tolist()
        }, index=tindex
    ) \
    .explode('obs_t')
    divdf['obs_id'] = obs_id.tolist()

    if df is not None:
        divdf = df.merge(divdf, left_index=True, right_index=True)
    return divdf

def interpolate_measures(meas_df, meas_ts, interp_df=None, interp_ts=None, tol=None):
    '''
    Interpolate measurements from an analysis dataframe consisting of a time-based
    column and one or more columns containing measurement values. Linear interpolation
    of measurement values is performed for times specified by the time column of
    another dataframe or from an array or list of times.

Parameters
----------

meas_df : dataframe
    Measurement dataframe containing a time column and one or more columns of
    measurements. All the measurement columns must be a numeric type and able to
    be interpolated. Non-numeric columns from an input dataframe must be removed
    before calling this function.

meas_ts : str
    Name of the time column in `meas_df`. Values in this column must be in increasing
    order.

interp_df : dataframe
    Dataframe containing a time column with times for which interpolated values are
    desired. If `None`, then `interp_ts` must provide the time values as an array or
    list.

interp_ts : str, array-like or list
    If a string, `interp_ts` is the name of a time column in `interp_df`. If an array
    or list of time values, then `interp_df` must be `None`.

tol : float (default None)
    Maximum allowed distance from each interpolation timepoint to its nearest
    measurement timepoint. If None, the tolerance will be automatically
    calculated as half the mean step between measurement timepoints.


Returns
-------

df : dataframe
    The output dataframe of measurements. If `interp_df` is a dataframe, then interpolated
    values from the measurement columns of `meas_df` are concatenated as new columns to
    `interp_df` and returned. Otherwise, a dataframe of interpolation times and corresponding
    measurement values is returned. If `interp_ts` has an index, that index is used as the
    returned dataframe's index, and a default index is assigned otherwise.
    '''
    
    interp_ts = interp_ts if interp_df is None else interp_df[interp_ts]
    meas_ts = meas_df[meas_ts]
    # Default tolerance is half the apparent measurement timestep.
    tol = np.mean(np.diff(meas_ts)) / 2 if tol is None else tol
    # An interpolation timepoint is out of tolerance if its minimum absolute
    # distance to a measurement timepoint is greater than `tol`
    outoftol = np.min(
        np.abs(
            np.expand_dims(meas_ts, axis=0) - np.expand_dims(interp_ts, axis=1)
        ),
        axis=1
    ) > tol
    try:
        assert(not outoftol.any())
    except AssertionError:
        with np.printoptions(threshold=3):
            msg = f'The maximum distance allowed from an interpolation timepoint to the nearest measurement timepoint is {tol}, and that tolerance is exceeded at interpolation timepoint(s) {interp_ts[outoftol].values}. Use the `tol` param to adjust the tolerance or exclude these interpolation timepoint(s).'
        raise ValueError(msg) from None
    try:
        assert(np.all(np.diff(meas_ts) > 0))
    except AssertionError:
        msg = 'The time column of `meas_df` must be increasing.'
        raise ValueError(msg) from None
    meas_cols = [c for c in meas_df.columns if c != meas_ts.name]
    try:
        results = {}
        for col in meas_cols:
            results[col] = np.interp(interp_ts, meas_ts, meas_df[col])
    except TypeError:
        msg = f"Could not interpolate column `{col}` from `meas_df`. Specify a subset of the dataframe that does not include it, e.g. `meas_df=df[['tcol', 'measurecol']]`."
        raise TypeError(msg) from None
    if interp_df is not None:
        newdf = pd.DataFrame(results, index=interp_ts.index)
        df = pd.concat([interp_df, newdf], axis='columns')
    else:
        try:
            tcolname = interp_ts.name
        except AttributeError:
            tcolname = 'tcol'
        df = pd.DataFrame({tcolname: interp_ts} | results)
    return df

def loadsig(path, chansel=[], offset=0.0, duration=None, rate=None, dtype=np.float32):
    """
    Load signal(s) from an audio file.
    
    By default audio samples are returned at the same sample rate as the input file.
    and channels
    are returned along the first dimension of the output array `y`.

    Parameters
    ==========

    path : string, int, pathlib.Path, soundfile.SoundFile, or file-like object
        The input audio file.

    chansel : int, list of int (default [])
        Selection of channels to be returned from the input audio file, starting with 0 for the first channel. For empty list [], return all channels in order as they appear in the input audio file. This parameter can be used to select channels out of order, drop channels, and repeat channels.

    offset : float (default 0.0)
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    rate : number > 0 [scalar]
        target sampling rate. 'None' returns `y` at the file's native sampling rate.

    dtype : numeric type (default float32)
        data type of **y**. No scaling is performed when the requested dtype differs from the native dtype of the file. Float types are usually in the range [-1.0, 1.0), and integer types usually make use of the full range of integers available to their size, e.g. int16 may be in the range [-32768, 32767].

    Returns
    =======

    *y : varying number of 1d signal arrays `y`
        Each channel is returned as a separate 1d array in the output list. The number of arrays is equal to the number of channels in the input file by default. If **chansel** is specified, then the number of 1d arrays is equal to the length of **chansel**.    

    rate : number > 0 [scalar]
        sampling rate of **y** arrays

    
    Example
    =======
    Load a stereo audio file, report the sampling rate of the file, and plot the left channel.  Note, this will produce an error with a one channel file. **left** and **right** are one-dimensional arrays of audio samples.

    >>> left, right, fs = loadsig('stereo.wav')
    >>> print(fs)
    >>> plt.plot(left);

    Load a wav file that has an unknown number of channels, downsampling to 12 kHz sampling rate.  Use **len(chans)** to determine how many channels there are in the file, and plot the last channel. **chans** is a two dimensional matrix with audio signals in the columns of the matrix.

    >>> *chans, fs = loadsig('threechan.wav', rate = 12000)
    >>> print(len(chans))      # the number of channels
    >>> plt.plot(chans[-1])    # plot the last of the channels
    
    """
    
    y, rate = librosa.load(
        path, sr=rate, mono=False, offset=offset, duration=duration, dtype=dtype
    )
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    if chansel == []:
        chansel = np.arange(y.shape[0], dtype=np.int16)
    return [ *list(y[chansel, :]), rate ]

