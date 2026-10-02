import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from scipy.linalg import solve_banded
from scipy.optimize import minimize_scalar
import scipy.sparse as sp


# Bounds on log10(s). They come from limits on the "average leverage" h of the smoother (see
# Garcia 2010): h must lie between 1e-6 and 0.99. For 1-D data the bounds work out to this:
def _s_bounds(h_min=1e-6, h_max=0.99):
    def bound(h):
        return np.sqrt((((1 + np.sqrt(1 + 8 * h ** 2)) / 4 / h ** 2) ** 2 - 1) / 16)
    return bound(h_max), bound(h_min)

_S_MIN, _S_MAX = _s_bounds()
_LOGS_MIN, _LOGS_MAX = np.log10(_S_MIN), np.log10(_S_MAX)

_MAX_REFITS = 8       # most times s is re-estimated while the filled-in data are being refined
_REFIT_TOL = 0.05     # stop refining when log10(s) changes by less than this
_REFIT_WINDOW = 1.0   # after the first estimate, only search this far (in log10 units) around it
_S_DIRECT_MAX = 1e6   # above this the banded system is too ill-conditioned to solve directly
_BOUND_WARN = 0.1     # warn if the best log10(s) is this close to either bound


def smooth1d(y, s=None, isrobust=False, W=None, sd=None, weightstr='bisquare', nS0=10, verbose=False):
    '''Robust spline smoothing of one-dimensional data.  After Damien Garcia's smoothn.

**smooth1d** is a fast, automatic, and robust discretized smoothing spline for a one-dimensional
signal. It gives results that are very close to those of :func:`phonlab.smoothn` for 1-D data,
but it is simpler and faster, and it never changes its input.


Parameters
----------

y : array
    A uniformly-sampled one-dimensional array of data to be smoothed (a numpy array, a
    pandas Series, or a masked array). Non-finite data (NaN or Inf) and masked values are treated
    as missing.

s : float, default = None
    The larger **s** is, the smoother the output will be. If **s** is omitted it is automatically
    determined using the generalized cross-validation (GCV) method. If given it must be a positive
    number.

isrobust : boolean, default = False
    Use the robust algorithm to minimize the influence of outlying data. The data are smoothed,
    outliers are down-weighted based on how far they are from the first smooth, and then the
    data are smoothed again.

W : array, default = None
    An array of weights (non-negative real values) the same length as **y**.

sd : array, default = None
    An array of standard deviations used to compute the weights as W = 1/sd**2. Points where
    sd is not positive get a weight of zero. If **sd** is given, **W** is ignored.

weightstr : string, default = 'bisquare'
    The function used to calculate robust weights from the residuals. Options are
    "bisquare", "cauchy" and "talworth".

nS0 : int, default = 10
    The number of values of **s** that are tried (evenly spaced in log10(s)) when searching for
    the best **s**, before the search is refined.

verbose : boolean, default = False
    Print some information about the search for **s**.


Returns
-------

z : array of float
    The smoothed version of **y**.

s : float
    The smoothing parameter that was used. This is the value of **s** that was passed in, or the one
    that was found by GCV.

exitflag : boolean
    True means that the estimate of **s** converged, False means that it was still changing when
    the iteration limit was reached.


Note
----

When the weights are not all equal (because of missing data, weights, or robust weighting), the
smooth is found by solving a banded system of equations, which gives the exact answer in one step.
This is different from :func:`phonlab.smoothn`, which finds approximately the same smooth by iterating
until the change is smaller than a tolerance, so the results can differ slightly.

References
----------

D. Garcia (2010) Robust smoothing of gridded data in one and higher dimensions with missing values.
`Computational Statistics & Data Analysis` **54** , 1187-1178. http://www.biomecardio.com/publis/csda10.pdf

Examples
--------

.. code-block:: Python

    x = np.linspace(0,100,2**5)
    y = np.cos(x/10)+(x/50)**2 + np.random.random_sample(len(x))/2
    y[[14, 17, 20]] = [2, 2.5, 3]

    z,s,e = phon.smooth1d(y)                 # Regular smoothing
    zr,sr,e = phon.smooth1d(y,isrobust=True) # Robust smoothing

    print(f'regular smoothing factor = {s:.3f}, robust smoothing factor = {sr:.3f}')
    '''
    y = _as_float_vector(y)
    n = len(y)
    finite = np.isfinite(y)
    if s is not None and not s > 0:
        raise ValueError(f"s must be a positive number, not {s}")
    if n < 2:
        return y.copy(), s, True

    w = _make_weights(n, W, sd) * finite
    if not np.any(w > 0):
        raise ValueError("there are no finite values with positive weight to smooth")

    weightstr = weightstr.lower()
    y0 = np.where(finite, y, 0.0)   # arbitrary values for missing data
    nof = int(finite.sum())         # number of finite data points

    # initial guess for the smooth, with missing values linearly interpolated
    idx = np.arange(n)
    ok = w > 0
    z = np.interp(idx, idx[ok], y0[ok])

    s_fixed = s    # None means that s is found automatically
    z, s, exitflag = _smooth(y0, w, nof, s_fixed, z, nS0, verbose)

    if isrobust:
        h = np.sqrt(1 + 16. * s)         # average leverage
        h = np.sqrt(1 + h) / np.sqrt(2) / h
        w = w * _robust_weights(y0 - z, finite, h, weightstr)
        if np.any(w > 0):
            z, s, exitflag = _smooth(y0, w, nof, s_fixed, z, nS0, verbose, near=np.log10(s))

    if s_fixed is None:
        if abs(np.log10(s) - _LOGS_MIN) < _BOUND_WARN:
            warnings.warn(f"smooth1d: s = {s:.3g}: the lower bound for s has been reached. "
                          "Pass s as an argument if required.", stacklevel=2)
        elif abs(np.log10(s) - _LOGS_MAX) < _BOUND_WARN:
            warnings.warn(f"smooth1d: s = {s:.3g}: the upper bound for s has been reached. "
                          "Pass s as an argument if required.", stacklevel=2)
    return z, float(s), exitflag


def _as_float_vector(y):
    """Return a new 1-D float array from an array, Series or masked array. Masked values become NaN."""
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if isinstance(y, np.ma.MaskedArray):
        y = np.ma.filled(y.astype(float), np.nan)
    y = np.array(y, dtype=float)    # always a copy, so the caller's data are never changed
    if y.ndim != 1:
        raise ValueError(f"smooth1d only smooths one-dimensional data, but y has {y.ndim} dimensions")
    return y


def _make_weights(n, W, sd):
    """Weights (scaled so the largest is 1) from either sd or W; all ones if neither is given."""
    if sd is not None:
        sd = _as_float_vector(sd)
        w = np.zeros(n)
        pos = sd > 0
        w[pos] = 1. / sd[pos] ** 2
    elif W is not None:
        w = _as_float_vector(W)
    else:
        return np.ones(n)
    if len(w) != n:
        raise ValueError(f"the weights have {len(w)} values but y has {n}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must all be finite and >= 0")
    wmax = w.max()
    return w / wmax if wmax > 0 else w


@lru_cache(maxsize=8)
def _eigenvalues_sq(n):
    """Squared eigenvalues of the second-difference matrix (with reflecting ends), which is
    diagonalized by the DCT. They determine how strongly each frequency is smoothed."""
    lam = 2. * (np.cos(np.pi * np.arange(n) / n) - 1.)
    return lam ** 2


@lru_cache(maxsize=8)
def _penalty_bands(n):
    """The squared second-difference matrix L'L, which has five nonzero diagonals, stored in the
    banded form that scipy.linalg.solve_banded wants (row 2 is the main diagonal).
    (This is faster here than solveh_banded, which makes use of the symmetry.)"""
    main = np.full(n, -2.)
    main[[0, -1]] = -1.
    L = sp.diags([np.ones(n - 1), main, np.ones(n - 1)], [-1, 0, 1], format='csr')
    L2 = L @ L
    ab = np.zeros((5, n))
    for k in range(-2, 3):      # diagonal k is stored in row 2 - k
        d = L2.diagonal(k)
        if k >= 0:
            ab[2 - k, k:] = d
        else:
            ab[2 - k, :n + k] = d
    return ab


def _smooth(y, w, nof, s, z, nS0, verbose, near=None):
    """Smooth y with weights w. If s is None it is found by GCV. z is a starting guess for the
    smooth, which is used to fill in the places where w < 1 when choosing s. near is an earlier
    estimate of log10(s) to search around, instead of searching the whole range."""
    uniform = bool(np.all(w == 1))
    if s is not None:
        return _solve(y, w, s, uniform, z), s, True

    exitflag = False
    logs = near
    for it in range(_MAX_REFITS):
        # Garcia's approach: with weights, GCV is evaluated on the data with the points that are
        # down-weighted replaced by the current smooth.
        filled = w * (y - z) + z
        logs_new = _best_logs(dct(filled, norm='ortho'), y, w, nof, nS0, near=logs)
        converged = uniform or (logs is not None and abs(logs_new - logs) < _REFIT_TOL)
        logs = logs_new
        s = 10 ** logs
        z = _solve(y, w, s, uniform, z)
        if verbose:
            print(f"smooth1d: pass {it + 1}, s = {s:.4g}")
        if converged:
            exitflag = True
            break
    return z, s, exitflag


def _solve(y, w, s, uniform, z0):
    """The smooth: z minimizes sum(w * (y - z)**2) + s**2 * sum((second difference of z)**2).
    z0 is a starting guess, which is only used when s is too large to solve directly."""
    n = len(y)
    if uniform:   # every sample has equal weight, so the DCT diagonalizes the problem
        gamma = 1. / (1. + s ** 2 * _eigenvalues_sq(n))
        return idct(gamma * dct(y, norm='ortho'), norm='ortho')
    if s > _S_DIRECT_MAX:
        return _solve_iterative(y, w, s, z0)
    ab = s ** 2 * _penalty_bands(n)
    ab[2] += w
    return solve_banded((2, 2), ab, w * y)


def _solve_iterative(y, w, s, z, tol=1e-8, maxit=1000):
    """The same smooth as _solve(), for very large s, found with Garcia's iteration: smooth the
    data with the missing/down-weighted points replaced by the current estimate, until it stops
    changing. A relaxation factor of 1.75 speeds up the convergence."""
    gamma = 1. / (1. + s ** 2 * _eigenvalues_sq(len(y)))
    relax = 1.75
    for _ in range(maxit):
        z_new = relax * idct(gamma * dct(w * (y - z) + z, norm='ortho'), norm='ortho') + (1 - relax) * z
        done = np.linalg.norm(z_new - z) <= tol * np.linalg.norm(z_new)
        z = z_new
        if done:
            break
    return z


def _gcv(logs, dcty, y, w, nof):
    """Generalized cross-validation scores for an array of log10(s) values."""
    n = len(y)
    s = 10 ** np.atleast_1d(logs)[:, None]
    gamma = 1. / (1. + s ** 2 * _eigenvalues_sq(n))                # shape (len(logs), n)
    if w.sum() / n > 0.9:   # (nearly) all of the data are equally weighted: no inverse DCT needed
        rss = np.sum((dcty * (gamma - 1.)) ** 2, axis=1)
    else:
        yhat = idct(gamma * dcty, norm='ortho', axis=1)
        rss = np.sum(w * (y - yhat) ** 2, axis=1)
    tr_h = gamma.sum(axis=1)
    return rss / nof / (1. - tr_h / n) ** 2


def _best_logs(dcty, y, w, nof, nS0, near=None):
    """Find the log10(s) that minimizes the GCV score. Unless there is a previous estimate (near),
    a coarse grid search finds the right neighborhood. Then a bounded 1-D minimization refines it."""
    if near is None:
        grid = np.linspace(_LOGS_MIN, _LOGS_MAX, nS0)
        best = int(np.argmin(_gcv(grid, dcty, y, w, nof)))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, nS0 - 1)]
    else:
        lo, hi = max(near - _REFIT_WINDOW, _LOGS_MIN), min(near + _REFIT_WINDOW, _LOGS_MAX)
    res = minimize_scalar(lambda p: _gcv(p, dcty, y, w, nof)[0], bounds=(lo, hi),
                          method='bounded', options={'xatol': 1e-4})
    return res.x


def _robust_weights(r, finite, h, wstr):
    """Weights for robust smoothing, based on the studentized residuals r."""
    mad = np.median(np.abs(r[finite] - np.median(r[finite])))   # median absolute deviation
    if mad == 0:     # residuals are (nearly) all identical: nothing can be called an outlier
        return np.ones_like(r)
    u = np.abs(r / (1.4826 * mad) / np.sqrt(1 - h))
    if wstr == 'cauchy':
        wt = 1. / (1 + (u / 2.385) ** 2)
    elif wstr == 'talworth':
        wt = (u < 2.795).astype(float)
    else:                        # bisquare
        wt = (1 - (u / 4.685) ** 2) ** 2 * ((u / 4.685) < 1)
    wt[np.isnan(wt)] = 0.
    return wt
