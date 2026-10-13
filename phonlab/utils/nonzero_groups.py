import numpy as np


def nonzero_groups(a, minlen=1, containing=None):
    '''Return the runs of consecutive non-zero elements of a one-dimensional array.

**nonzero_groups** is like numpy's ``nonzero()``, but instead of one array holding the indexes
of all of the non-zero elements it returns a tuple of arrays, one for each run of consecutive
non-zero elements. Only one-dimensional input is supported.

Parameters
----------

a : array_like
    The one-dimensional array to search. As in ``np.nonzero()``, every element that is truthy
    counts as non-zero (NaN included), and the masked elements of a masked array count as zero.

minlen : int, default = 1
    The minimum length of a run. Runs with fewer than **minlen** elements are left out.

containing : int or array_like of int, default = None
    One or more indexes into **a**. If given, only the runs that contain at least one of these
    indexes are returned. The indexes must be integers and non-negative; an index that is
    outside **a**, or that falls on a zero element, matches no run. **minlen** is applied as
    well.

Returns
-------

runs : tuple of arrays
    The indexes into **a** of each run of consecutive non-zero elements, in order. The array
    for a run that occupies ``a[start:stop]`` is ``np.arange(start, stop)``. The tuple is empty
    if there are no runs.

Examples
--------

.. code-block:: Python

    a = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    phon.nonzero_groups(a)
    # (array([1, 2]), array([4]), array([7, 8, 9]))

    phon.nonzero_groups(a, minlen=2)
    # (array([1, 2]), array([7, 8, 9]))

    phon.nonzero_groups(a, containing=[4, 8])
    # (array([4]), array([7, 8, 9]))

**Post-processing a voicing decision.** :func:`phonlab.VAD` returns one row per 5 ms frame, with
a boolean **voiced** column and the time at the middle of the frame in **sec**. The runs of voiced
frames are the voiced stretches of the recording. A frame or two of voicing inside a voiceless
stop is common, so **minlen** is a convenient filter; here it drops stretches shorter than 20 ms
(4 frames). The indexes in a run are row numbers of the dataframe, and the interval a run covers
extends half a frame step before its first frame and after its last one:

.. code-block:: Python

    x, fs = phon.loadsig('im_twelve.wav', chansel=[0])
    df = phon.VAD(x, fs)
    runs = phon.nonzero_groups(df.voiced, minlen=4)
    for run in runs:
        print(f'voiced {df.sec[run[0]] - 0.0025:.3f}-{df.sec[run[-1]] + 0.0025:.3f} s')
    # voiced 0.023-0.253 s
    # voiced 0.352-0.797 s
    # voiced 0.998-1.297 s
    # voiced 1.393-1.573 s
    # voiced 1.643-1.978 s

To write the stretches to a TextGrid tier, put their times in a dataframe and use
:func:`phonlab.df_to_tg`:

.. code-block:: Python

    voicing = pd.DataFrame({'t1': [df.sec[run[0]] - 0.0025 for run in runs],
                            't2': [df.sec[run[-1]] + 0.0025 for run in runs],
                            'voicing': 'voiced'})
    phon.df_to_tg(voicing, 'voicing', end=len(x) / fs, fmt='0.4f',
                  outfile='im_twelve_voicing.TextGrid')

**containing** finds the voiced stretch around a landmark such as a vowel midpoint from a
TextGrid. Convert the time to a frame index first. The result is empty if the landmark falls on
an unvoiced frame, as it does in the middle of a voiceless stop:

.. code-block:: Python

    i = np.searchsorted(df.sec, 0.487)                  # the frame at 0.487 s, in the vowel of 'twelve'
    runs = phon.nonzero_groups(df.voiced, containing=i)
    if runs:
        run = runs[0]
        print(f'in a voiced stretch from {df.sec[run[0]]:.3f} to {df.sec[run[-1]]:.3f} s')
    # in a voiced stretch from 0.355 to 0.795 s

To clean up the decision itself, drop the short voiced stretches and then bridge the short gaps
between the ones that remain. The gaps are the runs of the negated array; a gap that touches
either end of the recording is leading or trailing silence rather than a gap, so it is left alone:

.. code-block:: Python

    voiced = df.voiced.to_numpy()
    clean = np.zeros_like(voiced)
    for run in phon.nonzero_groups(voiced, minlen=4):     # drop voiced stretches shorter than 20 ms
        clean[run] = True
    for gap in phon.nonzero_groups(~clean):                # bridge unvoiced gaps shorter than 20 ms
        if len(gap) < 4 and gap[0] > 0 and gap[-1] < len(clean) - 1:
            clean[gap] = True
    df['voiced'] = clean
    '''
    mask = np.ma.getmask(a)                     # nomask unless `a` is a masked array
    a = np.asarray(a).astype(bool)
    if a.ndim != 1:
        raise ValueError(f'nonzero_groups() expects a one-dimensional array, got {a.ndim} dimensions')
    if mask is not np.ma.nomask:
        a &= ~mask                              # masked elements count as zero, as in np.nonzero()

    # Pad with False at both ends so that runs touching either end of `a` also produce a
    # transition. The transitions then alternate: start of a run, end of a run, start, end, ...
    padded = np.concatenate(([False], a, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]    # a run occupies a[start:stop]

    keep = (stops - starts) >= minlen
    if containing is not None:
        idx = np.asarray(containing).reshape(-1)
        if idx.size and not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f'containing must be an integer index or indexes, got dtype {idx.dtype}')
        # The run that contains each index, if any: the last run that starts at or before the
        # index, provided that run has not ended yet.
        k = np.searchsorted(starts, idx, side='right') - 1
        found = k >= 0
        found[found] = idx[found] < stops[k[found]]
        wanted = np.zeros(starts.size, dtype=bool)
        wanted[k[found]] = True
        keep &= wanted

    return tuple(np.arange(start, stop) for start, stop in zip(starts[keep], stops[keep]))
