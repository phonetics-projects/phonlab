import pandas as pd
import numpy as np
import srt
from parselmouth.praat import call as pcall

def _df_to_praat_short_label_str(df, lblcol, t1col, t2col=None, fmt=None):
    """
    Return a string representing the labels of a tier in praat_short format
    from a dataframe.
    """

    if fmt is None:
        ts = df[t1col].astype(str)
    else:
        ts = df[t1col].map(fmt.format)
    if t2col is not None and fmt is None:
        ts = ts.str.cat(df[t2col].astype(str), sep='\n')
    elif t2col is not None:
        ts = ts.str.cat(df[t2col].map(fmt.format), sep='\n')
    return '\n'.join(
        ts.str.cat(
            df[lblcol].fillna('').astype(str) \
                .str.replace('"', '""', regex=False) \
                # ^|$ alone does not match twice on empty strings
                # Also, .str.replace doesn't seem to work with
                # beginning/end of string unless you capture, e.g. (^)
                # and we just use .replace instead.
                .replace('^', '"', regex=True) \
                .replace('$', '"', regex=True),
            sep='\n'
        )
    )

def _df_to_praat_long_label_str(df, lblcol, t1col, t2col=None, fmt=None):
    """
    Return a string representing the labels of a tier in praat_long format
    from a dataframe.
    """

    intvl = 'intervals [{}]:\n            '
    ts = pd.Series(np.arange(1, len(df)+1)).map(intvl.format)

    t1lbl = '{} = '.format('number' if t2col is None else 'xmin')
    if fmt is None:
        t1s = df[t1col].astype(str)
    else:
        t1s = df[t1col].map(fmt.format)
    ts = ts.str.cat(t1s.replace('^', t1lbl, regex=True))
    if t2col is not None:
        if fmt is None:
            t2s = df[t2col].astype(str)
        else:
            t2s = df[t2col].map(fmt.format)
        ts = ts.str.cat(
            t2s.replace('^', 'xmax = ', regex=True),
            sep='\n            '
        )
    lbl = '            {} = "'.format('mark' if t2col is None else 'text')
    return '\n        '.join(
        ts.str.cat(
            df[lblcol].fillna('').astype(str) \
                .str.replace('"', '""', regex=False) \
                # ^|$ alone does not match twice on empty strings
                # Also, .str.replace doesn't seem to work with
                # beginning/end of string unless you capture, e.g. (^)
                # and we just use .replace instead.
                .replace('^', lbl, regex=True) \
                .replace('$', '"', regex=True),
            sep='\n'
        )
    )

def _df_to_praat_short_tier(df, xmin, xmax, tname, lblcol, t1col,
    t2col=None, fmt=None):
    """
    Return a string representing the a tier defined in a dataframe in
    praat_short format.
    """

    return '\n'.join(
        [
            '"IntervalTier"' if t2col is not None else '"TextTier"',
            '"' + tname + '"',
            xmin,
            xmax,
            str(len(df)),
            _df_to_praat_short_label_str(df, lblcol, t1col, t2col, fmt)
        ]
)

def _df_to_praat_long_tier(idx, df, xmin, xmax, tname, lblcol, t1col,
    t2col=None, fmt=None):
    """
    Return a string representing a tier defined in a dataframe in
    praat_long format.
    """

    tclass = '"IntervalTier"' if t2col is not None else '"TextTier"'
    tier = '''    item [{}]:
        class = {}
        name = "{}"
        xmin = {}
        xmax = {}
        intervals: size = {}
        {}'''.format(
        idx, tclass, tname, xmin, xmax, str(len(df)),
        _df_to_praat_long_label_str(df, lblcol, t1col, t2col, fmt)
    )
    return tier

def _praat_short_preamble(xmin, xmax, tiercnt):
    """
    Preamble of a short Praat textgrid.
    """
    # xmin and xmax should already be strings
    return '''File type = "ooTextFile"
Object class = "TextGrid"

{}
{}
<exists>
{}'''.format(xmin, xmax, str(tiercnt))

def _praat_long_preamble(xmin, xmax, tiercnt):
    """
    Preamble of a long Praat textgrid.
    """
    # xmin and xmax should already be strings
    return '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = {}
xmax = {}
tiers? <exists>
size = {}
item []:'''.format(xmin, xmax, str(tiercnt))

def _df_degap(df, t1fld, t2fld, lblfld, start, end, fill):
    start = float(start)
    end = float(end)
    if end > df[t2fld].astype(float).iloc[-1] and end != np.inf:
        endfill = end
    else:
        endfill = df[t2fld].astype(float).iloc[-1]
    t1ser = df[t2fld].astype(float)
    t2ser = df[t1fld].astype(float).shift(-1, fill_value=endfill)
    if df[t1fld].iloc[0] > start:
        t1ser = pd.concat([pd.Series(start), t1ser])
        t2ser = pd.concat([pd.Series(df[t1fld].iloc[0]), t2ser])
    gapdf = pd.concat({
        t1fld : t1ser,
        t2fld : t2ser
    }, axis='columns')
    gapdf[lblfld] = fill
    gapdf = gapdf[gapdf[t1fld] != gapdf[t2fld]]
    return pd.concat(
        [df, gapdf],
        axis='rows'
    ).sort_values(t1fld).reset_index(drop=True)

def df_to_tg(dfs, tiercols, ts=['t1', 't2'], start=0.0, end=None, tgtype='short',
    codec='utf-8', fmt=None, fill_gaps='', outfile=None):
    """
Convert one or more dataframes to a Praat textgrid.

Each input dataframe represents a textgrid tier. Each dataframe row
represents a label. There must be a column in each dataframe to provide
1) the label text content; 2) the label start time for an IntervalTier or
point time for a PointTier (`t1`); and 3) the label end time for an
IntervalTier (`t2`).

If the `t1` and `t2` columns are numeric types, they are converted to `str`
type without any special formatting, unless the `fmt` parameter is used.

*The dataframes are converted to labels as-is. No sorting is performed
before creating the textgrid.*

Parameters
----------

dfs : dataframe or list of dataframes
    The input dataframes of labels. Each df represents a separate textgrid tier.

tiercols: str or dict or list of str/dict
    The column name in each dataframe in `dfs` that contains the label content. If the
    name is provided as a `str`, that name will also be the tier's name in the output
    textgrid. If a different name for the tier is desired, use a single-element `dict`
    to map the column name to the textgrid tier name, e.g. `{'text': 'word'}` maps the
    'text' column of a dataframe to a textgrid tier named 'word'. For multiple dataframes
    use a list of single-element dicts, e.g. `[{'text': 'word'}, {'text': 'phone'}, 'context']`
    to map the 'text' column of the first dataframe to a tier named 'word' and the 'text'
    column of the second dataframe to a tier named 'phone'. The third dataframe in
    this example has a column named 'context' that will also be the name of the
    textgrid tier.

ts : list of str or list of list of str (default=['t1', 't2'])
    The column names in each dataframe in `dfs` that hold the start and end
    times of the labels. For Point tiers Use `None` as the second value.
    If this value is a simple list, then all dataframes must be of the same
    Interval/Point type with the same names for the time columns.
    For a mix of Interval and Point tier types, or if time column names vary
    among the dataframes, use a list of two-element lists to specify the
    column names for each dataframe.

start : num or None (default=0.0)
    The start time of the textgrid. If `None`, the start time will be the
    minimum label time value among all the dataframes.

end : num or None (default=None)
    The end time of the textgrid. If `None`, the end time will be the
    maximum label time value among all the dataframes.

tgtype : str, default='short'
    The Praat textgrid output type. Must be one of 'short' or 'long'.

codec : str (default 'utf-8')
    The codec used to write the textgrid (e.g. 'utf-8', 'ascii').

fmt : str or None (default None)
    The format string to apply to all time columns, as used by the
    `format <https://docs.python.org/3/library/stdtypes.html#str.format>`
    built-in method, for example, '0.4f' for four-digit floating point.

fill_gaps : str or None (default '' empty string)
    When `fill_gaps` is not None, new labels will be inserted into IntervalTier
    outputs where consecutive dataframe rows are not contiguous in time (rows
    in which the end time of one row is less than the start time of the next row).
    The string value of `fill_gaps` is used as the text content of the inserted
    labels.

outfile : file path, optional
    If provided, write the textgrid to `outfile` and return `None` instead of
    the textgrid content.

Returns
-------

tg : str or None
    The textgrid output as a `str`. If `outfile` is specified, then `None` is
    returned instead.

Example
-------

.. code-block:: Python

    wddf = pd.DataFrame({                   # create example dataframes
        'word': ['', 'a', 'word'],
        't1': [0.0, 0.1, 0.23647890019],
        't2': [0.1, 0.2, 0.3],
    })
    ptdf = pd.DataFrame({      # a point tier
        'pt': ['pt1', 'pt2'],
        't1': [0.05, 0.15],
    })
    ctxdf = pd.DataFrame({
        'ctx': ['nonspeech', 'speech'],
        't1': [0.0, 0.1],
        't2': [0.1, 0.3]
    })

    # Single tier textgrid.
    phon.df_to_tg(wddf, tiercols='word', outfile='word.TextGrid')

    # Single tier textgrid where the column name doesn't match the tier name.
    phon.df_to_tg(
        ctxdf,
        tiercols={'ctx': 'context'},
        outfile='ctx.TextGrid'
    )

    # Two-tier textgrid. One tier name matches the column name and one does not.
    phon.df_to_tg(
        [wddf, ctxdf],
        tiercols=['word', {'ctx': 'context'}],
        outfile='wordctx.TextGrid'
    )

    # Three-tier textgrid of two interval tiers and one point tier. The
    # label content is in the 'word', 'pt', and 'ctx' columns, and the
    # textgrid tiernames will be 'word', 'pointevent', and 'context'.
    phon.df_to_tg(
        [wddf, ptdf, ctxdf],
        tiercols=['word', {'pt': 'pointevent'}, {'ctx': 'context'}],
        ts=[['t1', 't2'], ['t1', None], ['t1', 't2']],
        outfile='wordptctx.TextGrid'
    )

    # Specify numeric output to four decimal places.
    phon.df_to_tg(wddf, 'word', fmt='.4f', outfile='wordt1str.TextGrid')

    """

    # Process params.
    # Coerce to list of dataframes
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Coerce to list of dicts
    if not isinstance(tiercols, list):
        tiercols = [tiercols]
    if len(tiercols) == 1:
        tiercols = tiercols * len(dfs)
    tiercols = [t if isinstance(t, dict) else {t: t} for t in tiercols]

    # Coerce to list of lists
    if not isinstance(ts[0], list):
        ts = [ts] * len(dfs)

    # Find max/min times.
    if start is not None:
        xmin = start
    else:
        xmin = min([df[tcols[0]].min() for df, tcols in zip(dfs, ts)])
    maxcols = [
        t1col if t2col is None else t2col for t1col, t2col in ts
    ]
    if end is not None:
        xmax = end
    else:
        xmax = max([df[col].max() for df, col in zip(dfs, maxcols)])

    # Create TextGrid preamble.
    if tgtype != 'long':
        tg = _praat_short_preamble(xmin, xmax, len(dfs))
    else:
        tg = _praat_long_preamble(xmin, xmax, len(dfs))

    # Prep the `fmt` string, if needed.
    if fmt is not None and not fmt.startswith('{:'):
        fmt = '{:' + fmt + '}'

    # Convert xmin and xmax to (formatted) strings.
    if fmt is None:
        xmin = str(xmin)
        xmax = str(xmax)
    else:
        xmin = fmt.format(xmin)
        xmax = fmt.format(xmax)

    for df, colmap, (t1col, t2col) in zip(dfs, tiercols, ts):
        tiercol, tiername = list(colmap.items())[0]
        try:
            if len(df) > 1:
                assert((df[t1col].diff().iloc[1:] > 0).all())
                if t2col is not None:
                    assert((df[t2col].diff().iloc[1:] > 0).all())
        except AssertionError:
            raise RuntimeError(
                'Dataframe labels not sorted by time or duplicate times found.'
            ) from None
        try:
            if t2col is not None:
                assert(((df[t2col] > df[t1col])).all())
        except AssertionError:
            raise RuntimeError(
                'Interval label end values must be greater than start values in all rows.'
            ) from None
        try:
            # Every t1 must be >= the preceding t2.
            if t2col is not None:
                assert(
                    (df[t1col].shift(-1) >= df[t2col]).iloc[:-1].all()
                )
        except AssertionError:
            raise RuntimeError(
                'Dataframe interval labels cannot overlap in time. The start time of a row cannot be less than the end time of the preceding row.'
            ) from None

        if fill_gaps is not None and t2col is not None:
            df = _df_degap(
                df,
                t1fld=t1col,
                t2fld=t2col,
                lblfld=tiercol,
                start=xmin,
                end=xmax,
                fill=fill_gaps
            )
        if tgtype != 'long':
            tiertext = _df_to_praat_short_tier(df, xmin, xmax, tiername, tiercol, t1col, t2col, fmt)
        else:
            tiertext = _df_to_praat_long_tier(df, xmin, xmax, tiername, tiercol, t1col, t2col, fmt)
        tg += f'\n{tiertext}'
    if outfile is not None:
        with open(outfile, 'w', encoding=codec) as out:
            out.write(tg)
        return None
    else:
        return tg

def tg_to_df(tg, tiersel=[], names=None):
    '''
Read a Praat textgrid and return its tiers as a list of dataframes.

Parameters
----------

tg : path-like
    Filepath of input textgrid.

tiersel : list of str or int
    Selection of tiers to include in the output list, identified by tier name (`str`) or `0`-based integer index (`int`). If `[]` then all textgrid tiers are returned. Tiers can be selected in a different order than they appear in the textgrid.

names : None, str, or list of str (default None)
    Names of the label content columns in the output dataframes. If `None`, then the textgrid tier name is used as the column. If `str` then the same column name will be used for all dataframes. If list, then one name must be supplied for each tier selected by `tiersel`.

Returns
-------

tiers : list of dataframes
    Textgrid tiers are returned as a list of dataframes for each tier, in the order selected by `tiersel`. The time columns of each dataframe are named `t1` and `t2` for label start and end times of interval tiers, or `t1` for the timepoints of point tiers. The textgrid tier's name is used as the name of the column containing the label content unless column names are provided by `names`. If `tiers` is an empty list `[]` then all textgrid tiers are returned in the list of dataframes.

Example
-------

In this example we have the name of an existing Praat Textgrid file, and use **tg_to_df()** to read the textgrid into a set of dataframes (one for each tier of the textgrid file.

.. code-block:: Python

    textgrid_name = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid'
    phdf, wddf = phon.tg_to_df(textgrid_name, tiersel=['phone', 'word'])
    phdf.head()

.. figure:: images/tg_to_df.png
    :scale: 50 %
    :alt: The first few rows of the phones dataframe (phdf) given by tg_to_df()
    :align: center

    The first few rows of the phones dataframe (phdf) given by `tg_to_df()`

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
Add shifted versions of a dataframe column to provide context within rows. For example, if you have a dataframe of phone labels you can use this function to add the preceding/following phone context to each row.

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

Example
-------
In this example we have the name of an existing textgrid, read it into a Pandas dataframe with `phon.tg_to_df()` and
then with `phon.add_context()` add two context columns to the dataframe, one for the previous phone, and one for the following phone.

.. code-block:: Python

    textgrid_name = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid'

    phdf, wddf = phon.tg_to_df(textgrid_name, tiersel=['phone', 'word'])
    phdf = phon.add_context(phdf,'phone',nprev=1,nnext=1)
    phdf.head()


.. figure:: images/add_context.png
    :scale: 50 %
    :alt: The first few rows of the phones dataframe (phdf) given by add_context()
    :align: center

    The first few rows of the phones dataframe (phdf) given by `add_context()`
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
# TODO: add tolerance and overwrite params
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

Example
-------
In this example we have the name of an existing Praat TextGrid file, we read it into Pandas DataFrames with `phon.tg_to_df()`, and then merge two of the dataframes into a single larger dataframe that has all of the 
information that was in them using `phon.merge_tiers()`.  The `inner` dataframe is the one with intervals/events that are inside the intervals/events in the `outer` dataframe.  In this case the intervals in the 'phone' tier are contained in the intervals in the 'word' tier.

.. code-block:: Python

    textgrid_name = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid'
    phdf, wddf = phon.tg_to_df(textgrid_name, tiersel=['phone', 'word'])
    tgdf = phon.merge_tiers(inner_df=phdf, outer_df=wddf, suffixes=['', '_wd'])
    tgdf.head()

.. figure:: images/merge_tiers.png
    :scale: 50 %
    :alt: The first few rows of the combined dataframe given by merge_tiers()
    :align: center

    The first few rows of the combined dataframe given by `merge_tiers()`    
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

def adjust_boundaries(inner_ts, outer_ts, tolerance):
    """
Compare two Series and return the closest match of the second found in the first.

Two annotation tiers may be expected to have a strictly hierarchical relationship
where the boundaries should exactly align, e.g. the left boundary of a word tier aligns
with the left boundary of a phone (and right boundaries should also align). If the
annotations were not created carefully and do not match exactly, this function can
be used to adjust values of one (the `outer_ts`) to match a value found in the
other (the `inner_ts`). For example, the `outer_ts` value could be the left boundaries
of a series of words, and the `inner_ts` value could be the left boundaries of a series
of phones.

Normally the `outer_ts` series has a one-to-many relationship with the `inner_ts` series,
and the `inner_ts` series has a many-to-one relationship with the `outer_ts`. In the
preceding discussion, words contain multiple phones.

Parameters
==========

outer_ts : Series of num
    A series of time values that correspond to outer_ts boundaries.

inner_ts : Series of num
    A series of time values that correspond to inner_ts boundaries.

tolerance : num
    Maximum distance from outer_ts to inner_ts value for inexact matches.

Returns
=======
mod_outer_ts: array
    A modified numpy array of time values of `outer_ts` in which each value is an exact
match of a value in `inner_ts`.

Raises
======

    ValueError: A ValueError is raised if one or more boundaries are not within tolerance. A list of the values from `outer_ts` that are out of tolerance is included as the second value of the Exception object's `args` attribute. An error message is the first value of `args`.


Examples
========

Read phone and word tiers from a textgrid.

.. code-block:: Python

    [phdf, wddf] = phon.tg_to_df(tgpath, tiersel=['phone', 'word'])


    # Adjust word 't1' values up to 5 ms.
    try:
       wddf['t1'] = phon.adjust_boundaries(wddf['t1'], phdf['t1'], tolerance=0.005)
    except ValueError as e:
       badt = ', '.join([f'{t:0.4f}' for t in e.args[1]])
       msg = f"Word-phone boundary mismatch greater than {tolerance} in {tgpath}. " \
             f"Bad word boundary found at time(s) {badt}."
       raise ValueError(msg) from None
"""

    idx = pd.Index(inner_ts).get_indexer(outer_ts, method='nearest', tolerance=tolerance)
    try:
        return inner_ts[idx].values
    except KeyError:
        raise ValueError(
            'Boundaries out of tolerance in `outer_ts`.',
            [outer_ts[i] for i in np.where(idx == -1)[0]]
        )

def explode_intervals(divs, ts=['t1', 't2'], df=None, prefix='obs_'):
    '''
    Divide a series of time intervals into subintervals and explode into long format,
    with one row per subinterval timepoint. An interval [2.0, 3.0] divided into two
    subintervals, for example, produces three output rows for the times corresponding
    to 0%, 50%, and 100% of the interval: 2.0, 2.5, 3.0.

    The subinterval divisions can be specified as an integer number of subdivisions,
    or as a list of interval proportions in the range [0.0, 1.0]. For `int` the number
    of timepoints produced is the number of subintervals + 1, and for a list of
    proportions one timepoint is produced for each element of the list.

Parameters
----------

divs : int, list of float in range [0.0, 1.0]
    The subintervals to include. If `int`, the number of equal-duration subintervals
    each interval will be divided into. If a list, the values should be in the range
    [0.0, 1.0] and express proportions of the duration of each interval for which
    subinterval timepoints will be created. For example, `[0.25, 0.50, 0.75]` yield
    timepoints at 25%, 50%, and 75% of each input interval.

ts : list of str or list of numeric scalar/list/Series/arrays
    If list of `str`, these are the names of time columns in the `df` dataframe. The
    first name defines the start time of the interval to be subdivided, and the second
    name defines the end time. For numeric values, provided as a scalar, list,
    `pd.Series`, or `np.array`, the first scalar/list/Series/array provides the start
    times, and the second provides the end times.

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

Example
-------
In this example we have a dataframe produced by `phon.tg_to_df()`, and `phon.merge_tiers()` which has columns for each `phone` and it's starting and ending times (t1,t2).  We use the Pandas function `query` to get a subset dataframe that just has vowels in it, and then use `phon.explode_intervals()` to add new rows specifying the time points at 20%, 50% and 80% of the way through each vowel.

.. code-block:: Python

    vowels = ['ay', 'eh', 'iy', 'aa', 'aw']
    vdf = tgdf.query(f'phone in {vowels}').copy()  # make a dataframe that just has vowels
    vdf = phon.explode_intervals([0.2,0.5, 0.8], ts=['t1', 't2'], df=vdf) # get times for observations
    vdf.head()


.. figure:: images/explode_intervals.png
    :scale: 50 %
    :alt: The first few rows of a 'vowels' dataframe with observation points added by `phon.explode_intervals()`
    :align: center

    The first few rows of a 'vowels' dataframe with observation points added by `phon.explode_intervals()`    
    '''
    # TODO: test different kinds of indexes in input dataframe, e.g. MultiIndex
    try:
        t1col = np.array(ts[0], ndmin=1) if df is None else df[ts[0]]
        t2col = np.array(ts[1], ndmin=1) if df is None else df[ts[1]]
    except KeyError:
        msg = 'The `ts` values must be column names in `df` if `df` is not None. If `df` is None, `ts` values can be numeric.'
        raise ValueError(msg) from None
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
        divs = np.array(divs)
        try:
            assert((divs.min() >= 0.0) & (divs.max() <= 1.0))
        except AssertionError:
            msg = 'When `divs` is specified as a list, the list elements must specify proportions of the interval and be in the range [0.0, 1.0], e.g. [0.25, 0.50, 0.75] for 25%, 50%, 75% timepoints in the interval.'
            raise ValueError(msg) from None
        obs_t = (
            np.expand_dims((t2col - t1col), axis=1) * np.expand_dims(divs, axis=0)
        ) + np.expand_dims(t1col, axis=1)
        obs_id = np.tile(divs, len(t1col))

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

def interpolate_measures(meas_df, meas_ts, interp_df=None, interp_ts=None, tol=None, overwrite=False):
    '''
    Interpolate measurements from an analysis dataframe consisting of a time-based
    column and one or more columns containing measurement values. Linear interpolation
    of measurement values is performed for times specified by the time column of
    another dataframe or from an array or list of times.

    This function provides an interface to `numpy.interp()` in order to 
    add acoustic or articulatory measurements from a dataframe that has measurements at 
    monotonically increasing timepoints through the whole file (like F0 measurements at 
    5 ms intervals for example) to a dataframe that has target locations at which we 
    would like to extract measurements (like vowel midpoints, for example).

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

overwrite : bool (default False)
    If True, overwrite existing measurements in `interp_df` from columns of the
    same names in `meas_df`. If False, an error is raised when column names
    overlap. Measurement columns from `meas_df` that do not overlap `interp_df`
    are always added as new columns.

Returns
-------

df : dataframe
    The output dataframe of measurements. If `interp_df` is a dataframe, then interpolated
    values from the measurement columns of `meas_df` are concatenated as new columns to
    `interp_df` and returned. Otherwise, a dataframe of interpolation times and corresponding
    measurement values is returned. If `interp_ts` has an index, that index is used as the
    returned dataframe's index, and a default index is assigned otherwise.

Example
-------
TextGrid information is in a dataframe `vdf`, which has a column `obs_t` of times at which we 
would like to have formant measurements from the data in file 'im_twelve.csv' (produced by 
`phon.track_formants()`).  The function `phon.interpolate_measures()` extracts data from the 
formants dataframe and adds measurements at the desired observation times in the textgrid dataframe.

.. code-block:: Python

    fmtsdf = pd.read_csv('im_twelve.csv')  # read in the csv of formants measurements

    vdf = phon.interpolate_measures(
        meas_df=fmtsdf[['sec','F1', 'F2', 'F3', 'F4']],  # meas_ts and cols to interpolate only
        meas_ts='sec',        # time index in the measurements dataframe
        interp_df=vdf,       # textgrid dataframe
        interp_ts='obs_t',  # target observation times in the textgrid
        overwrite=True
    )
    vdf.head()


.. figure:: images/interpolate_measures.png
    :scale: 50 %
    :alt: The first few rows of a 'vowels' dataframe with formant measurements added by `phon.interpolate_measures()`
    :align: center

    The first few rows of a 'vowels' dataframe with with formant measurements added by `phon.interpolate_measures()`        
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
        if not overwrite:
            overlaps = set(interp_df.columns) & set(meas_cols)
            assert(len(overlaps) == 0)
    except AssertionError:
        msg = f'Found overlap of columns in `interp_df` and `meas_df`. To overwrite the measurements in the {overlaps} column(s) in `interp_df` set the parameter `overwrite=True`. If you wish to keep the existing measurements and add new columns from `meas_df` you can `rename` the column(s) in the input dataframes so that they do not match.'
        raise ValueError(msg) from None
    try:
        results = {}
        for col in meas_cols:
            results[col] = np.interp(interp_ts, meas_ts, meas_df[col])
    except TypeError:
        msg = f"Could not interpolate column `{col}` from `meas_df`. Specify a subset of the dataframe that does not include it, e.g. `meas_df=df[['tcol', 'measurecol']]`."
        raise TypeError(msg) from None
    if interp_df is not None:
        df = interp_df.assign(**results)
    else:
        try:
            tcolname = interp_ts.name
        except AttributeError:
            tcolname = 'tcol'
        df = pd.DataFrame({tcolname: interp_ts} | results)
    return df

def srt_to_df(srtfile, verbose=True):
    '''
    Read subtitles in an .srt file and return as a dataframe.

    The dataframe is checked for overlapping subtitle texts, and a warning is
    issued if any overlaps are found.

Parameters
----------

srtfile : pathlike
    Input `.srt` file path as a Path object or string.

verbose : bool (default True)
    If True, print informational messages.

Returns
-------

df : dataframe
    The output dataframe with time columns `t1` and `t2` that indicate start and end times
    of subtitle content, which is in the `text` column.
    '''
    # Read subtitles in .srt file to make a list of dicts containing subtitle content.
    subtitles = []
    with open(srtfile, 'r') as fh:
        for s in srt.parse(fh):
            text = s.content
            sdict = {
                't1': s.start.total_seconds(),
                't2': s.end.total_seconds(),
                'text': text,
            }
            subtitles.append(sdict)

    # Return a dataframe from the list of subtitle dicts.
    return pd.DataFrame(subtitles)

def split_speaker_df(df, textcol='text', ts=['t1', 't2'], sep=None, ffill=True, include=[], exclude=[], as_dict=True, verbose=True):
    '''
    Split speaker identifier from the text contained in a dataframe column, and
    add speaker as new column.

    To help guard against the misparsing of speaker identifiers, an error is raised
    if any speaker identifiers are found in the dataframe that are not explicitly
    listed in the `include` and `exclude` parameters.

Parameters
----------

df : dataframe
    Input dataframe of speaker utterances.

textcol : str
    Name of the column in `df` that contains utterance content. Speaker identifiers
    are split off from the values in this column, e.g. 'Speaker1: Some utterance' yields
    'Speaker1' and 'Some utterance' as the new `speaker` and `textcol` columns.

ffill : bool (default True)
    If True, `df` rows which have no `speaker` value (i.e. do not contain `sep` and
    cannot be split) inherit `speaker` from the immediately preceding row.

ts : list of str (default ['t1', 't2'])
    The names of the start and end time columns in the `df` dataframe. The
    first name defines the start time of the interval, and the second
    name defines the end time.

include : list of str (default [])
    List of speaker identifiers and associated rows to include in the return value.
    **Hint:** If you want to construct a list of possible speaker identifiers by
    integer you can use a list comprehension. For example, the list comprehension
    `includelist = [f'Speaker-{n}' for n in range(3)]` creates a list of three
    speakers: `['Speaker-0', 'Speaker-1', 'Speaker-2']`.

exclude : list of str (default [])
    List of speaker identifiers and associated rows to exclude from the return value.

sep : str
    String on which to split `textcol` into `speaker` and `utterance`.

as_dict : bool (True)
    If True, return value is a dict with speaker identifiers as keys. The values are
    dataframes of utterance rows for that speaker. If False, return original
    dataframe with new `speaker` column.

verbose : bool (default True)
    If True, print informational messages.

Returns
-------

df : dataframe or dict of dataframes
    If `as_dict` is True, a dict of dataframes is returned in which the keys are speaker
    identifiers and the values are the dataframes of utterances by the speaker. If
    `as_dict` is False, then a single dataframe is returned with the speaker identifiers
    in a new column named `speaker` added to the input dataframe, and with the speaker
    identifiers removed from `textcol`.
    '''
    newcols = pd.DataFrame(
        [
            (s[0], s[1]) if len(s) == 2 else (None, s[0]) for s in df[textcol].str.split(sep)
        ],
        columns=['speaker', textcol]
    )
    if ffill is True:
        newcols['speaker'] = newcols['speaker'].ffill()
    else:
        newcols['speaker'] = newcols['speaker'].fillna('*')
    newcols['speaker'] = newcols['speaker'].astype('category')
    unrecognized = set(newcols['speaker'].cat.categories) - set(include + exclude)

    try:
        assert(len(unrecognized) == 0)
    except AssertionError:
        msg = f'Found {len(unrecognized)} speaker(s) not in include/exclude lists. (Possible misplaced separator character "{sep}".) : "' + \
               ', '.join(unrecognized) + '"\n'
        raise ValueError(msg) from None

    # Replace textcol and add speaker col
    df = pd.concat([df.drop(textcol, axis='columns'), newcols], axis='columns')

    t1, t2 = ts[0], ts[1]

    # Probably not necessary to sort, but we do it in case the .srt is weird.
    sortcols = ['speaker', t1] if 'speaker' in df.columns else [t1]
    df = df.sort_values(sortcols)

    # Limit speakers to specific individuals found in `include`.
    df = df.query(f'speaker in {include}')

    # Groupby speaker
    grouper = 'speaker' if 'speaker' in df.columns else lambda x: '*'
    spgroups = df.groupby(grouper, observed=True)

    # Test for overlaps
    for spkr, gdf in spgroups:
        overlaps = gdf[t2].shift(1) > gdf[t1]
        if overlaps.any():
            sys.stderr.write(f'WARNING: Found time overlaps that need to be corrected in "{srtfile}" for speaker "{spkr}" at time(s):\n')
            for row in gdf[overlaps].itertuples():
                sys.stderr.write(f'{pd.to_timedelta(getattr(row, t1), unit="s")}: {row.text}\n')
            sys.stderr.write('\n')
        else:
            if verbose:
                sys.stdout.write(f'No overlaps found for "{spkr}".\n')

    if as_dict is True:
        return {k: group for k, group in spgroups}
    else:
        return df.sort_values(t1)
