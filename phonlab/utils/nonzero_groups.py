import numpy as np


def nonzero_groups(a, minlen=1, containing=None):
    '''Return the runs of consecutive non-zero elements of a one-dimensional array.

**nonzero_groups** is like numpy's ``nonzero()``, but instead of one array holding the indexes
of all of the non-zero elements it returns a tuple of arrays, one for each run of consecutive
non-zero elements. Only one-dimensional input is supported.

Parameters
----------

a : array_like
    The one-dimensional array to search. It is converted with ``np.asarray(a).astype(bool)``,
    so every element that is truthy counts as non-zero (NaN included).

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

Report the stretches of at least 50 ms in a frame-by-frame voicing decision **voiced** that was
made every 5 ms:

.. code-block:: Python

    for run in phon.nonzero_groups(voiced, minlen=10):
        print(f'voiced from {run[0] * 0.005:.3f} to {(run[-1] + 1) * 0.005:.3f} s')
    '''
    a = np.asarray(a).astype(bool)
    if a.ndim != 1:
        raise ValueError(f'nonzero_groups() expects a one-dimensional array, got {a.ndim} dimensions')

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
