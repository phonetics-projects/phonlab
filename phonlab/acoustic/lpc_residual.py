import numpy as np
from math import comb
from librosa import util, lpc
from scipy.signal import windows, lfilter

from ..utils.prep_audio_ import prep_audio
from numba import njit
from ..utils.deprecation import renamed_kwargs
from ..utils.validate import positive_step, step_samples

@njit(cache=True)
def overlap_add(frames, hop_length, window_vec):
    """
    Perform Overlap-Add reconstruction from 2D frames to a 1D signal.

Parameters
==========
    frames : ndarray
        A 2D array of shape (n_frames, frame_length)

    hop_length : int
        Number of samples to shift for each frame

    window_vec : 1D array (frame_length) 
        use scipy.signal.windows to get this vector with a statement like 
        `win = windows.get_window('cosine', Nx=frames.shape[1], fftbins=False)`

Returns
=======
    y : ndarray
        The reconstructed 1D signal.
    """
    n_frames, frame_length = frames.shape
    final_length = (n_frames - 1) * hop_length + frame_length

    y = np.zeros(final_length, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        for j in range(frame_length):
            # Windowing and Adding performed in a single pass 
            # to maximize CPU cache efficiency
            y[start + j] += frames[i, j] * window_vec[j]
    return y


def _invalid_lpc(A, order):
    """True for rows of A (LPC polynomials [1, a1, ... ap]) that are non-finite or that
    cannot be minimum phase because some |a_k| exceeds the binomial coefficient C(p,k)."""
    bound = np.array([comb(order, k) for k in range(order + 1)], dtype=np.float64)
    return ~np.isfinite(A).all(axis=-1) | (np.abs(A) > bound).any(axis=-1)


def _window_sum(w, hop):
    """The sum of the window weights of the overlapping frames, in the middle of the signal (where it is constant)."""
    n = -(-len(w) // hop) * hop
    return np.pad(np.asarray(w, dtype=np.float64), (0, n - len(w))).reshape(-1, hop).sum(axis=0).max()


@renamed_kwargs(s='step', l='length')
@positive_step
def lpcresidual(y, fs, target_fs=16000, order = 18, length=0.04, step=0.005, window='cosine',
                gain='speech', normalize=True):
    """Compute the residual signal which results from filtering the input audio array **y** 
using LPC inverse filtering. This signal is useful in voice quality and periodicity routines, and
(with `gain=None`) as an excitation source for source-filter resynthesis.

The default LPC order, 18, is (target_fs/1000) + 2 for the default `target_fs` of 16000.

Each analysis frame (`length` seconds, every `step` seconds) is inverse filtered with its own LPC filter,
the filtered frames are weighted by `window` and overlap-added.  What the level of each frame is, in
comparison with the other frames, depends on `gain`.  Which is the right choice depends on what the
residual will be used for -- see `gain`.

Parameters
==========
    y : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **y**
    target_fs: int, default = 16000
        Algorithms from the covarep library of voice analysis routines require target_fs=16000
    order: integer, default = 18
        The "order" of the LPC analysis.  The number of coefficients to use in the LPC analysis.  
        The default value is that recommended by Drugman, Kane, and Gobl (2013) for voice quality
        analysis (fs/1000 + 2), with the caveat that a smaller number may be more appropriate for 
        voices with higher fundamental frequency.
    length: float, default = 0.04
        The duration of the LPC analysis window, 40 milliseconds
    step: float, default = 0.005
        The interval between successive frames in the LPC analysis, 5 milliseconds
    window : str, tuple, numeric, callable, list-like (default 'cosine')
        Window function applied to each (filtered) frame before the frames are overlap-added. This
        parameter is passed as the `window` parameter to scipy.signal's `get_window` function.
        `None` is the same as 'boxcar' (no weighting).
    gain : string {'speech', None}, default = 'speech'
        How the level of each frame of the residual is set.

        * ``'speech'`` -- each frame of the residual is scaled so that it has the same energy as the
          (unfiltered) frame of audio that it came from, as in the `lpcresidual.m` function of the COVAREP
          collection (Drugman et al.).  The level of the residual then follows the level of the speech,
          so the heights of its peaks can be compared from one frame to another.  This suits detection
          and measurement, such as `gci()`, `get_f0_srh()` and `h2h1()`, which use it.  The
          per-frame gain is not kept, so the residual can not be turned back into the audio with
          the LPC filters.
        * ``None`` -- no scaling: the residual is the LPC prediction error itself.  Filtering it (frame by
          frame) with the inverse of the LPC filters gives back the audio, so this is the choice if you
          use the residual as the source in a source-filter resynthesis.  Also use `normalize=False` in
          that case, to keep the level.
    normalize : boolean, default = True
        Divide the residual by its peak absolute value, so that the largest sample is 1.0.  If False, the
        residual keeps its level, which is that of the prediction error of the audio after it has been
        resampled and scaled by `prep_audio()` (see `gain`).

Returns
=======
    lpc_residual : ndarray
        A one-dimensional array -- the residual derived by inverse filtering the input 
        audio signal. It has the same number of samples as the input **y** array.
    fs : int
        The sampling rate of **lpc_residual**.  It will be the same as **target_fs**, which
        by default is 16000 Hz.

Raises
======
    ValueError
        if `gain` is not 'speech' or None.

    
    """
        
    if gain not in ('speech', None):
        raise ValueError(f"lpcresidual(): gain must be 'speech' or None, not {gain!r}")

    # 1. Resample and Pre-process
    x, fs = prep_audio(y, fs, target_fs=target_fs, preemph=0, fix_polarity=True)
    x = x.astype(np.float32) # Use float32 for speed and memory efficiency
    
    frame_length = int(fs * length)
    hop = step_samples(step, fs)

    # 2. Frame the signal
    # util.frame creates a 'view', which is O(1) memory and time.
    frames = util.frame(x, frame_length=frame_length, hop_length=hop, axis=0)
    
    # 3. Compute LPC Coefficients
    # librosa.lpc is generally efficient, we process frames in one go.
    A = lpc(frames, order=order) 

    # Burg's method in float32 can return garbage coefficients (non-finite, or finite but
    # huge) for frames that are badly conditioned, e.g. a low-F0 voice at fs=12000 with 
    # almost no energy above a few hundred Hz.  A single such frame makes the whole 
    # residual NaN (via overflow in the energy normalization below, when that was done over all frames at once).  A valid (minimum 
    # phase) LPC polynomial of order p has |a_k| <= C(p,k), so frames outside that bound 
    # are redone in float64, and if that also fails the frame is left unfiltered 
    # (A = [1, 0, 0, ...]).  Valid frames are not changed.
    bad = _invalid_lpc(A, order)
    if bad.any():
        A[bad] = lpc(frames[bad].astype(np.float64), order=order).astype(A.dtype)
        bad = _invalid_lpc(A, order)
        A[bad] = 0
        A[bad, 0] = 1

    # 4. Inverse Filtering (The Bottleneck)    

    # Pre-allocate output for frames
    inv = np.zeros_like(frames)
    
    # Applying the filter A[i] to frames[i]
    # A coefficients from librosa are [1, a1, a2, ...] 
    # The residual is calculated by filtering the frame with A
    for i in range(frames.shape[0]):
        # lfilter is significantly faster than fftconvolve for short FIR filters
        inv[i] = lfilter(A[i], [1.0], frames[i])

    # 5. Energy Normalization (each frame, if gain='speech')
    # The frame of the residual is scaled so that its energy is that of the frame of audio that it came from:
    # inv *= sqrt(sum(frame**2) / sum(inv**2)), as in COVAREP.  (Before 0.2.0 the sums were over all the frames at
    # once, which gave one number for the whole file, and the final peak normalization removed it:
    # it did nothing.)  The sums are in float64.  A frame with no energy in the residual is left as it is.
    if gain == 'speech':
        frame_energy = np.einsum('ij,ij->i', frames, frames, dtype=np.float64)
        inv_energy = np.einsum('ij,ij->i', inv, inv, dtype=np.float64)
        scale = np.ones(len(frames))
        ok = inv_energy > 0
        scale[ok] = np.sqrt(frame_energy[ok] / inv_energy[ok])
        inv = inv * scale[:, np.newaxis].astype(inv.dtype)

    # 6. Overlap-Add
    window = 'boxcar' if window is None else window
    w = windows.get_window(window, Nx=frames.shape[1], fftbins=False)
    lpc_resid = overlap_add(inv, hop, w)
    # The window weights add up to a constant (when the frames overlap enough) which is more than 1, so divide
    # by it: where there is no gain normalization, the residual is then at the level of the prediction error.
    lpc_resid /= _window_sum(w, hop)

    # 7. Final Normalization and Padding
    if normalize:
        max_val = np.max(np.fabs(lpc_resid))
        if max_val > 0:
            lpc_resid /= max_val

    # Pad to match original length
    if len(lpc_resid) < len(x):
        lpc_resid = np.pad(lpc_resid, (0, len(x) - len(lpc_resid)), mode='edge')
    else:
        lpc_resid = lpc_resid[:len(x)]
    
    return lpc_resid, fs
