import numpy as np

def nonzero_groups(a, minlen=1, include_any=None):
    '''Return groups of indexes where elements are non-zero. This function
is similar to numpy's `nonzero()`, but instead of returning a single array of
indexes, a tuple of consecutive groups of indexes is returned. Unlike
`nonzero()` only 1d arrays are supported.

Parameters
----------

a : array
The input array to search. The `dtype` must be boolean or coerceable to
boolean.

minlen : int
The minimum length of a run in order to be included in the output.
Runs with fewer than `minlen` elements are excluded.

include_any : scalar or array_like, optional
One or more indexes that the output runs must include. Any runs that
contain at least one of the indexes provided by `include_any` are
in the output. Runs that do not contain at least one of the indexes
are excluded. This parameter does not guarantee that all

Returns
-------

runs : tuple of arrays
Tuple of arrays in which the contents of each array are the indexes
into `a` that represent a run of consecutive `True` values.

'''
    # Coerce to array of (0, 1) values.
    a = np.asanyarray(a).astype(bool).astype(int)

    # By taking the diff of `a` we detect transitions between True
    # and False values. A value of `1` marks a transition from False
    # to True, `-1` marks a transition from True to False, and `0`
    # indicates a steady state from the previous True or False value.
    # `0` is padded to the edges of `a` so that we can detect
    # transitions to/from True values at the beginning and end.
    try:
        states = np.diff(a, prepend=0, append=0)
    except TypeError:  # For older numpy
        states = np.diff(np.hstack([0, a, 0]))

    # Indexes of transitions to true and false.
    true_starts = (states == 1).nonzero()[0]
    false_starts = (states == -1).nonzero()[0]
    try:
        assert(np.all(true_starts < false_starts))
    except:
        raise RuntimeError('Unexpected state in nonzero_groups().')
    if include_any is not None:
        runs = tuple((
            np.arange(tidx, fidx) \
                for tidx, fidx in zip(true_starts, false_starts) \
                    if (fidx - tidx) >= minlen \
                        and np.any(np.isin(np.arange(tidx, fidx), include_any)) 
        ))
    else:
        runs = tuple((
            np.arange(tidx, fidx) \
                for tidx, fidx in zip(true_starts, false_starts) \
                    if (fidx - tidx) >= minlen
        ))
    return runs
