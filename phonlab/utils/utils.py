# Utility functions.
__all__=['get_timestamp_now', 'cp_backup', 'dir2df', 'match_tokentimes', 'adjust_boundaries']

import os
import pandas as pd
import numpy as np
import re
import fnmatch
import shutil
from datetime import datetime
import dateutil


def get_timestamp_now():
    """
    Create a timestamp for an acquisition, using current local time.

    Returns
    =======

    timestamp, offset : tuple(str, str)
        A tuple of strings representing the datetime in YYYY-MM-DDTHHMMSS format and the timezone offset from UTC, e.g. '-0700'. 
    """

    # Regex that matches a timezone offset at the end of an acquisition
    # directory name.
    utcoffsetre = re.compile(
        r'(?P<offset>(?P<sign>[-+])(?P<hours>0\d|1[12])(?P<minutes>[012345]\d))'
    )
    ts = datetime.now(dateutil.tz.tzlocal()) \
             .replace(microsecond=0) \
             .isoformat() \
             .replace(":","")
    m = utcoffsetre.search(ts)
    utcoffset = m.group('offset')
    ts = utcoffsetre.sub('', ts)
    return (ts, utcoffset)

def cp_backup(fname, bkdir=None, hidden=True):
    """Make a backup copy of the file `fname` and return the name of the copied file. 

By default the copy will have the same name as `fname` with '.' prepended and a suffix of the form '.N', where N is an integer. Multiple calls to this function result in increasing values of N, starting with '1'.

Parameters
==========

fname : str
    The name of the file to be copied.

bkdir : str (default: None)
    By default, the backup file will be written in the same directory as the source file. If `bkdir` is provided, the backup file will be written to that path instead.  A `FileNotFoundError` will be thrown if the backup directory does not exist.

hidden : bool (default: True)
    If True, prepend '.' to the backup filename, resulting in a 'hidden' file. If not True, do not prepend anything.

Returns
=======

dst : str
    The name of the copied backup file.
    """
    
    if bkdir is None:
        bkdir = os.path.dirname(fname) if os.path.dirname(fname) != '' else '.'
    basename = os.path.basename(fname)
    cpname = '.' + basename if hidden is True else basename

    # Get extension integers from backups that already exist and find max N.
    rgx = re.compile(
        fnmatch.translate(cpname).replace(r'\Z', r'\.[0-9]+\Z')
    )
    Ns = np.array([
        int(os.path.splitext(f)[1].lstrip('.')) \
            for f in os.listdir(bkdir) if rgx.fullmatch(f)
    ], dtype=int)
    maxN = 0 if len(Ns) == 0 else Ns.max()

    cpname += '.{:d}'.format(maxN + 1)
    dst = os.path.join(bkdir, cpname)
    shutil.copyfile(fname, dst)
    return dst


def dir2df(dirname, addcols=[], fnpat=None, dirpat=None, sentinel='.bad.txt',
to_datetime=True, dotfiles=False, dotdirs=False, **kwargs):
    """Recursively collect the filenames in a directory tree using os.walk() and store as rows of a DataFrame.

    'Hidden' files and directories (those with names that start with '.') are ignored by default. dir2df() will also not descend into a directory tree that contains a sentinel file (default name '.bad.txt').

    Additional parameters can be used to filter which filepaths to include in the output, and also to add additional file metadata.

Parameters
==========

    dirname : str
        Top-level directory name for filename search.

    fnpat : str, re, default = None
        Regular expression pattern that defines the filenames to return. The only filenames in the result set will be those that return a match  for `re.search(fnpat, filename)`. If you use named captures in `fnpat`, new columns of dtype 'Categorical' will appear in the output that correspond to the capture group contents. If you need to use a flag with your pattern, you can use a precompiled regex for the value of `fnpat`. For example, you can do case-insensitive matching of '.wav' and '.WAV' files with `re.compile(r'\.wav$', re.IGNORECASE)`.

    dirpat : str, re, default = None
        Like `fnpat`, only applied against the relative path in dirname. Relative paths that do not match `dirpat` will be skipped.

    sentinel : str, default '.bad.txt'
        Name of the sentinel file, which marks a directory tree to be ignored. No filenames from the directory containing the sentinel file will be included in the output, nor will any filenames from any of its subdirectories. If the value of `sentinel` is '' or None, the sentinel file check will not be performed.

    addcols : str, list of str, default = []
        One or more additional columns to include in the output. Possible names and values provided are:
        'dirname': the user-provided top-level directory
        'barename': the filename without path or extension
        'ext': the filename extension
        'mtime': the last modification time of the file
        'bytes': the size of the file in bytes

    The 'mtime' column is cast to Pandas Timestamps automatically unless `to_datetime` is False. Resolution of the time-based stats is dependent on your platform; see the os.stat() documentation.

    to_datetime : boolean (default True)
        If True, 'mtime' stats will be converted from Unix epoch to datetime. If False, the values will not be converted.

    dotfiles : boolean (default False)
        If True, include filenames beginning with `.` in the output. Otherwise, omit these names.

    dotdirs : boolean (default False)
        If True, descend into directories with names that begin with `.`. If False, do not descend into these directories.

    kwargs : various
        Remaining kwargs are passed to os.walk(). If not used, then os.walk() will be called with default kwargs. Note that using os.walk(topdown=False) is not compatible with `dotdirs=False`.


Returns
=======

    fnamedf : DataFrame
        Pandas DataFrame with filenames recorded in rows.

Examples
========

>>> xraydir = '/media/sf_Downloads/xray_microbeam_database'
>>> df = phon.dir2df(xraydir, dirpat='^JW6', fnpat='*.wav',
>>>             addcols=['dirname', 'barename'])
>>> df.head()

    """
    
    # Coerce addcols to list if passed as single string.
    try:
        assert isinstance(addcols, basestring) # Python 2
        addcols = [addcols]
    except AssertionError:
        pass   # Should be a list already.
    except NameError:
        try:
            assert isinstance(addcols, str) # Python 3
            addcols = [addcols]
        except AssertionError:
            pass   # Should be a list already.

    if 'dirname' in addcols:
        firstcols = ['dirname', 'relpath', 'fname']
        addcols[:] = [c for c in addcols if c != 'dirname']
    else:
        firstcols = ['relpath', 'fname']
    mdcols = []  # Names of additional metadata columns from named captures.
    if dirpat is not None:
        dirpat = re.compile(dirpat)
        dirmdcols = list(dirpat.groupindex.keys())
        for c in dirmdcols:
            try:
                assert(c not in firstcols + addcols)
            except AssertionError:
                msg = 'Named group {:} masks another output column.'.format(c)
                raise RuntimeError(msg)
        mdcols += dirmdcols
    if fnpat is not None:
        fnpat = re.compile(fnpat)
        fnmdcols = list(fnpat.groupindex.keys())
        for c in fnmdcols:
            try:
                assert(c not in firstcols + addcols + mdcols)
            except AssertionError:
                msg = 'Named group {:} masks another output column.'.format(c)
                raise RuntimeError(msg)
        mdcols += fnmdcols

    stats = {'bytes': 'st_size'} if 'bytes' in addcols else {}
    if 'mtime' in addcols:
        stats['mtime'] = 'st_mtime'

    if dotdirs is False:
        try:
            assert(kwargs['topdown'] is True)
        except AssertionError:
            msg = 'topdown=False not compatible with dotdirs=False'
            raise RuntimeError(msg)
        except KeyError:
            pass

    recs = []
    for root, dirs, files in os.walk(dirname, **kwargs):
        if sentinel in files:
            dirs[:] = []  # Do not descend into subdirectories.
            continue      # Do not include files in this directory.
        relpath = os.path.relpath(root, dirname)
        dircols = []
        if dirpat is not None:
            dirm = dirpat.search(relpath)
            if dirm is None:
                continue # Do not include files in this directory.
            # Add named capture groups and replace unmatched optional
            # named captures with empty string.
            dircols = [
                '' if s is None else s for s in dirm.groupdict().values()
            ]
        for name in files:
            if (dotfiles is False) and (name[0] == '.'):
                continue
            fncols = []
            if fnpat is not None:
                fnm = fnpat.search(name)
                if fnm is None:
                    continue # Do not include this file.
                # Add named capture groups and replace unmatched optional
                # named captures with empty string.
                fncols = [
                    '' if s is None else s for s in fnm.groupdict().values()
                ]
            reccols = []
            if stats != {}:
                st = os.stat(os.path.join(root, name))
            if ('barename' in addcols) or ('ext' in addcols):
                (barename, ext) = os.path.splitext(name)
            for col in firstcols + addcols:
                if col in ['bytes', 'mtime']:
                    reccols += [getattr(st, stats[col])]
                elif col == 'relpath':
                    reccols += [relpath]
                elif col == 'fname':
                    reccols += [name]
                elif col == 'dirname':
                    reccols += [dirname]
                elif col == 'barename':
                    reccols += [barename]
                elif col == 'ext':
                    reccols += [ext]
            recs.append(reccols + dircols + fncols)

        # Change dirs in-place to prevent os.walk() from descending into
        # '.' directories.
        if dotdirs is False:
            dirs[:] = [d for d in dirs if not d[0] == '.']

    df = pd.DataFrame(
        recs,
        columns=firstcols + addcols + mdcols
    )
    df = df.sort_values(
        by=['relpath', 'fname'], axis='rows'
    ).reset_index(drop=True)
    if len(df) > 0:
        if 'dirname' in firstcols:
            df.dirname = df.dirname.astype('category')
        df.relpath = df.relpath.astype('category')
        df.fname = df.fname.astype('category')
        if 'barename' in addcols:
            df.barename = df.barename.astype('category')
        if 'ext' in addcols:
            df.ext = df.ext.astype('category')
        # Cast named captured columns to Categorical.
        df = df.astype({c: 'category' for c in mdcols})
        if to_datetime is True and 'mtime' in df.columns:
            df.loc[:, 'mtime'] = pd.to_datetime(df.loc[:, 'mtime'], unit='s')
    return df

def match_tokentimes(tokentimes, targettimes, mininc=1, return_warnidx=False):
    """
Find the closest matches in `targettimes` for every time in `tokentimes` and
return their indexes into `targettimes`.

Parameters
==========

tokentimes : 1d array of shape (n times) (or list, scalar), or 2d array of shape
    (m tokens, n times)
    The times to match against `targettimes`. These could be, for example, the
    times of events in an audio file for which you would like to find acoustic
    measurements. If the array is 2d, then tokens are assumed to lie along
    axis 0 (the first axis). For 2d arrays a warning is emitted if each
    consecutive returned index within a token does not increase from the one
    before it by at least `mininc`, i.e. if the token duration is too short.

targettimes : 1d array
    The times to search for the closest match. These could be, for example,
    the times corresponding to an arrray of pitch measurements or spectral
    slices in a spectrogram.

mininc : integer (optional; 1)
    The minimum increment between consecutive indexes within a token.
    If `tokentimes` is 1d, then this parameter is ignored. Use `None` to
    turn off this check.

    See `return_warnidx` for returning an index that identifies the tokens
    that emitted a warning.

return_warnidx : Bool (optional; False)
    If True, return a boolean index that can be used to select the tokens
    in `tokentimes` that cause a warning to be emitted.

Returns
=======

tidx : ndarray of integers in the shape of `tokentimes.shape`
    The integer indexes of the values in `targettimes` that are closest to each
    value of `tokentimes`, arranged in the same shape as `tokentimes`.

(tidx, warnidx) : (ndarray of integers, 1d boolean array)
    A tuple is returned when `return_warnidx=True`. In addition to `tidx`,
    `warnidx` is also returned, which contains a boolean index that has a
    True value for every token that causes a warning to be emitted.
    """
    
    tokentimes = np.array(tokentimes) # Cast in case list or scalar.
    flattokentimes = np.ravel(tokentimes)
    tidx = \
        np.argmin(
            np.abs(
                targettimes - \
                np.broadcast_to(
                    flattokentimes,
                    (len(targettimes), len(flattokentimes))
                ).transpose()
            ),
            axis=1
        ).reshape(tokentimes.shape)
    if len(tokentimes.shape) == 2:
        warnidx = \
            np.any(
                np.diff(tidx, axis=1) < mininc,
                axis=1
            )
        if np.any(warnidx):
            num = tokentimes.shape[collapse]
            sys.stderr.write(f'''
Short duration tokens were detected. {np.count_nonzero(warnidx)} token(s) were found
that are too short to be divided into {num} indexes from `targettimes` (the second parameter).

This warning may occur if `tokentimes` is not the right shape and tokens lie along the
second axis, in which case you must tranpose `tokentimes` so that tokens are on the first
axis.

This warning may also occur if the times in `tokentimes` don't really represent tokens, in
which case disable this warning.

To disable this check and warning, use `mininc=None`.

Debugging hint
--------------

To see the tokens that emitted a warning, use `return_warnidx=True` to return an additional
boolean array that contains a True value for any token that emitted a warning, e.g.

# tokentimes is 2d; tokens on axis 0 (default)
>> (tidx, noninc) = match_times(tokentimes, targettimes, return_warnidx=True)
>> tokentimes[noninc]    # Tokens that emit a warning

# tokentimes is 2d; tokens on axis 1
>> (tidx, noninc) = match_times(tokentimes, targettimes, tokenaxis=1, return_warnidx=True)
>> tokentimes[:,noninc]    # Tokens that emit a warning
            ''')
    if return_warnidx is True:
        return (tidx, warnidx)
    else:
        return tidx

def adjust_boundaries(parent, child, tolerance):
    """
Compare two Series and return the closest match of the second found in the first.

Two annotation tiers may be expected to have a strictly hierarchical relationship
where the boundaries should exactly align, e.g. the left boundary of a word tier aligns
with the left boundary of a phone (and right boundaries should also align). If the
annotations were not created carefully and do not match exactly, this function can
be used to adjust values of one (the `parent`) to match a value found in the
other (the `child`). For example, the `parent` value could be the left boundaries
of a series of words, and the `child` value could be the left boundaries of a series
of phones.

Normally the `parent` series has a one-to-many relationship with the `child` series,
and the `child` series has a many-to-one relationship with the `parent`. In the
preceding discussion, words contain multiple phones.

Parameters
==========

parent : Series of num
    A series of time values that correspond to parent boundaries.

child : Series of num
    A series of time values that correspond to child boundaries.

tolerance : num
    Maximum distance from parent to child value for inexact matches.

Returns
=======
mod_parent: array
    A modified numpy array of time values of `parent` in which each value is an exact
match of a value in `child`.

Raises
======

A ValueError is raised if one or more boundaries are not within tolerance. A
list of the values from `parent` that are out of tolerance is included as the
second value of the Exception object's `args` attribute. An error message is
the first value of `args`.
    
Examples
========

    Read phone and word tiers from a textgrid.
    
    >>> [phdf, wddf] = audiolabel.read_label(tgpath, from_type='praat')
    
    Adjust word 't1' values up to 5 ms. 
    
    >>> try:
    >>>    wddf['t1'] = phon.adjust_boundaries(wddf['t1'], phdf['t1'], tolerance=0.005)
    >>> except ValueError as e:
    >>>    badt = ', '.join([f'{t:0.4f}' for t in e.args[1]])
    >>>    msg = f"Word-phone boundary mismatch greater than {tolerance} in {tgpath}. " \
    >>>          f"Bad word boundary found at time(s) {badt}."
    >>>    raise ValueError(msg) from None
      """
    
    idx = pd.Index(child).get_indexer(parent, method='nearest', tolerance=tolerance)
    try:
        return child[idx].values
    except KeyError:
        raise ValueError(
            'Boundaries out of tolerance in `parent`.',
            [parent[i] for i in np.where(idx == -1)[0]]
        )
