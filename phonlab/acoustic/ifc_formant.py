import functools
import math

import numpy as np
from numba import njit, prange
from pandas import DataFrame
from scipy import signal

from ..utils.prep_audio_ import prep_audio

# constants
SR = 12000                     # sampling rate of the analysis
FMAX = 6000
FMIN = 200
frame_length = int(SR * 0.02)  # frame length 20 ms
half_frame = frame_length//2
step = int(SR * 0.01)          # number of samples between frames, 10 ms

# speaker parameters used in IFC tracking.  EG params[spkr]["fr"] are formant expectations for male if spkr == 0
params = [
    {
        'fr':np.array([ 60, 500, 1500, 2500, 3500]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5500,5500,4500]),
        'upper_bws': np.array([80,80,200]),
        'bands':np.array([[200,1100], [700, 2200], [2000,3500], [3000,4300]]),
        'spec_bounds':np.array([120,5500]),
        'du1':300, 'du2':600, 'dh1':200, 'dh2':500
    },
    {
        'fr':np.array([ 60, 620, 1860, 3100, 4340]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5580,5580]),  # stick with Ueda et al. (2007) on this.
        'upper_bws':np.array([3200,200]),
        'bands':np.array([[300,1300], [1000, 2500], [2500,4000], [3500,4800]]),
        'spec_bounds':np.array([250,5500]),
        'du1':400, 'du2':700, 'dh1':300, 'dh2':600
    },
    {
        'fr':np.array([ 60, 680, 2050, 3420, 4790]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([6000,5500]),
        'upper_bws':np.array([2700,2700]),
        'bands':np.array([[400,1600], [1300, 3000], [3000,4500], [3800,5500]]),
        'spec_bounds':np.array([300,5800]),
        'du1':500, 'du2':800, 'dh1':400, 'dh2':700
    }
]


# ------------ the compiled analysis of a frame -----------------------
#
#  The analysis of a frame (upper formant inverse filter, three passes over the three IFC blocks,
#  sorting of the formants) runs inside numba, and _ifc_frames() / _ifc_frames_parallel() run that
#  over all frames of a file.  The band amplitude ratios are computed for all frames at once
#  beforehand (_ifc_amplitude_ratios()).

@njit(cache=True, error_model='numpy')
def _inv_cascade(x, fr, bw, fs):
    '''A cascade of 3-tap inverse filters y[n] = c*(a*x[n-2] + b*x[n-1] + x[n]), one for each
    (frequency, bandwidth) pair in fr, bw, each stage started from rest; x is not modified.'''
    n = x.shape[0]
    src = x.copy()
    dst = np.empty(n, dtype=x.dtype)  # float32 in, float32 out (the arithmetic itself is double precision)
    for s in range(fr.shape[0]):
        exp1 = math.exp(-(math.pi/fs) * bw[s])
        a = exp1*exp1
        b = -2.0 * exp1 * math.cos((math.pi/fs * 2) * fr[s])
        c = 1.0/(1.0+a+b)
        dst[0] = c * src[0]
        if n > 1:
            dst[1] = c * (b * src[0] + src[1])
        for i in range(2, n):
            dst[i] = c*(a*src[i-2] + b*src[i-1] + src[i])
        src, dst = dst, src
    return src

@njit(cache=True, error_model='numpy')
def _ifc_filter(x, fs, n, f, b, f0, b0):
    '''Remove formant f (and, for n == 1, also the f0 notch) from x.'''
    if n == 1:
        return _inv_cascade(x, np.array((f, f0)), np.array((b, b0)), fs)
    else:
        return _inv_cascade(x, np.array((f,)), np.array((b,)), fs)

@njit(cache=True, error_model='numpy')
def _zc2(y, fs, loop, f_no, in_freq, du1, du2):
    '''Estimate the dominant frequency in y from the weighted mean of its zero-crossing period
    distribution (after Ueda et al.'s 2008 C code).  The input frequency (in_freq) is not used if
    loop==0.

    Input
        y - one frame of inverse filtered audio
        fs - the waveform sampling frequency
        loop - which iteration in _ifcblock() are we in? - 0,1,2
        f_no - which formant are we measuring - 1,2,3,4
        in_freq - the previously found frequency of this formant (for this frame)
        du1, du2 - search window parameters for F1 (from params[spkr])
    '''
    n = y.shape[0]
    freq = in_freq
    ca = 0.3
    dd1 = 300.0
    dd2 = 3000.0
    if loop == 0:  # first loop in an IFC block, we calculate mean freq as a spectral parameter
        ca = 0.2
        dd1 = 400.0
        dd2 = 4000.0
        wv = 0.0
        for i in range(n-1):
            wv += y[i+1] * y[i]        # sum of products of successive samples
        sv = 0.0
        for i in range(n):
            sv += y[i] * y[i]          # sum of squares
        ww = wv/sv
        if ww > 1: ww = 1.0
        if ww < -1: ww = -1.0
        freq = math.acos(ww) * (1/(2*math.pi/fs))
    if loop > 1:
        ca = 0.4
    aa = 0.5 * (0.3+ca)

    m = 0.0                            # remove the mean
    for i in range(n):
        m += y[i]
    m = m / n
    ym = np.empty(n)
    for i in range(n):
        ym[i] = y[i] - m

    zci = np.empty(n, np.int64)        # indices of zero crossings
    L = 0
    for i in range(n-1):
        if ym[i] * ym[i+1] <= 0:
            zci[L] = i
            L += 1
    # zcp[k] = (zci[k+1]-zci[k]) - rrr[zci[k]] + rrr[zci[k+1]],  rrr[j] = |y[j]|/(|y[j]|+|y[j+1]|)
    # z2[k]  = fs/(zcp[k] + zcp[k+1])      -- the last zero crossing period is not used
    nz = L - 3
    if nz < 0:
        nz = 0
    z2 = np.empty(nz)
    if L >= 3:
        zcp = np.empty(L-2)
        for k in range(L-2):
            i0 = zci[k]
            i1 = zci[k+1]
            r0 = abs(ym[i0])/(abs(ym[i0]) + abs(ym[i0+1]))
            r1 = abs(ym[i1])/(abs(ym[i1]) + abs(ym[i1+1]))
            zcp[k] = float(i1 - i0) - r0 + r1
        for k in range(nz):
            z2[k] = fs/(zcp[k] + zcp[k+1])

    if f_no <= 1:   # F1
        al1 = (du2-du1)/1000
        b1 = du1 - al1*200
    elif f_no == 2 or f_no == 3:   # F2 and F3
        al1 = (dd2-dd1)/4300
        b1 = dd1 - al1*200
    else:   # F4
        al1 = 0.83721
        b1 = 400 - al1 * 200

    dep = freq * al1 + b1
    for it in range(3):
        dep = dep - (it * aa * dep)
        sw = 0.0
        swz = 0.0
        for k in range(nz):
            w = 1 - abs(z2[k] - freq)/dep
            if w < 0: w = 0.0
            if w > 1: w = 1.0
            sw += w
            swz += z2[k] * w
        freq = swz/(sw + 0.001)

    if freq < FMIN: freq = FMIN
    if freq > FMAX: freq = FMAX
    return freq

@njit(cache=True, error_model='numpy')
def _ifcblock(x, fs, nc, fc, bc, nd, fd, bd, f0, b0, du1, du2):
    '''Three passes of removing one of two formants (fc) and estimating the other (fd), and
    then removing fd and re-estimating fc.  Returns the new (fc, fd).'''
    for i in range(3):  # Ueda et al do this 3 times
        yy = _ifc_filter(x, fs, nc, fc, bc, f0, b0)
        fd = _zc2(yy, fs, i, nd, fd, du1, du2)
        yy = _ifc_filter(x, fs, nd, fd, bd, f0, b0)
        fc = _zc2(yy, fs, i, nc, fc, du1, du2)
    return fc, fd

@njit(cache=True, error_model='numpy')
def _sort4(a, b, c, d):
    '''Four values from lowest to highest.'''
    if a > b: a, b = b, a
    if c > d: c, d = d, c
    if a > c: a, c = c, a
    if b > d: b, d = d, b
    if b > c: b, c = c, b
    return a, b, c, d

@njit(cache=True, error_model='numpy')
def _ifc_frame(x, fs, fr, bws, upper_fs, upper_bws, du1, du2, spkr, r12, r23, r34):
    '''Find F1-F4 in one frame x (float32 or float64), given its band amplitude ratios.
    Returns (F1, F2, F3, F4), unrounded.'''
    y = _inv_cascade(x, upper_fs, upper_bws, fs)          # remove upper formants
    f0 = fr[0]       # a fixed low-frequency notch used in the inverse filters, not a measured f0
    f1 = fr[1]
    f2 = fr[2]
    f3 = fr[3]
    f4 = fr[4]
    b1 = bws[1]
    b2 = bws[2]
    b3 = bws[3]
    b4 = bws[4]

    for i in range(3):   # loop over main IFC blocks to find formant frequencies
        # estimate F2 and F3 first
        b0 = 200.0
        if r23 >= -20: b0 = 100*r23 + 2200
        y2 = _inv_cascade(y, np.array((f4, f0, f1)), np.array((b4, b0, b1)), fs)   # filter out all but f2 and f3
        f3, f2 = _ifcblock(y2, fs, 3, f3, b3, 2, f2, b2, f0, b0, du1, du2)   # returns (F3, F2)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

        # estimate F1 and F2 next
        if spkr == 0:   # longer vocal tract - remove f6 again
            y2 = _inv_cascade(y, np.array((5500.0, f3, f4)), np.array((200.0, b3, b4)), fs)
        else:
            y2 = _inv_cascade(y, np.array((f3, f4)), np.array((b3, b4)), fs)   # filter out all but f1 and f2
        b0 = 200.0
        if r12 >= -20: b0 = 100*r12 + 2200
        f1, f2 = _ifcblock(y2, fs, 1, f1, b1, 2, f2, b2, f0, b0, du1, du2)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

        # estimate F3 and F4
        b0 = 200.0
        if r34 >= -20: b0 = 100*r34 + 2200
        y2 = _inv_cascade(y, np.array((f1, f0, f2)), np.array((b1, b0, b2)), fs)   # filter out all but f3 and f4
        f3, f4 = _ifcblock(y2, fs, 3, f3, b3, 4, f4, b4, f0, b0, du1, du2)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

    return f1, f2, f3, f4

@njit(cache=True, error_model='numpy')
def _ifc_frames(X, fs, fr, bws, upper_fs, upper_bws, du1, du2, spkr, ratios):
    '''_ifc_frame() for every row of X (one frame per row); ratios has one column per frame.'''
    R = X.shape[0]
    out = np.empty((R, 4))
    for r in range(R):
        f1, f2, f3, f4 = _ifc_frame(X[r], fs, fr, bws, upper_fs, upper_bws, du1, du2,
                                    spkr, ratios[0, r], ratios[1, r], ratios[2, r])
        out[r, 0] = f1
        out[r, 1] = f2
        out[r, 2] = f3
        out[r, 3] = f4
    return out

@njit(cache=True, error_model='numpy', parallel=True)
def _ifc_frames_parallel(X, fs, fr, bws, upper_fs, upper_bws, du1, du2, spkr, ratios):
    '''_ifc_frames() with the frames spread over all cores.'''
    R = X.shape[0]
    out = np.empty((R, 4))
    for r in prange(R):
        f1, f2, f3, f4 = _ifc_frame(X[r], fs, fr, bws, upper_fs, upper_bws, du1, du2,
                                    spkr, ratios[0, r], ratios[1, r], ratios[2, r])
        out[r, 0] = f1
        out[r, 1] = f2
        out[r, 2] = f3
        out[r, 3] = f4
    return out

@functools.lru_cache(maxsize=None)
def _ifc_filterbank(speaker, fs):
    '''The four band-pass filters (second-order sections) of params[speaker]["bands"] at sampling
    rate fs.  Designing them costs a few ms, so they are built once per (speaker, fs).'''
    return tuple(signal.butter(12, b, fs=fs, btype='bandpass', output='sos')
                 for b in params[speaker]["bands"])

_RATIO_BLOCK = 4096   # frames per block in _ifc_amplitude_ratios()

def _ifc_amplitude_ratios(X, fs, spkr):
    '''The relative amplitude ratios in the formant spectral regions of every frame.  X has one frame
    per row; each row is passed through the upper formant inverse filter and then through the four
    band-pass filters (one sosfiltfilt() call per band over a block of frames).  The frames are done
    in blocks of _RATIO_BLOCK because sosfiltfilt() works in double precision and returns an array
    that size, so the memory needed would otherwise grow with the length of the audio.

    Returns an array of shape (3, n_frames): r12, r23, r34 per frame, the ratios in dB of the
    energy in the F2 and F1, F3 and F2, and F4 and F3 bands.
    '''
    par = params[spkr]
    upper_fs = par["upper_fs"].astype(np.float64)
    upper_bws = par["upper_bws"].astype(np.float64)
    filterbank = _ifc_filterbank(spkr, fs)
    ratios = np.empty((3, X.shape[0]))
    for start in range(0, X.shape[0], _RATIO_BLOCK):
        B = X[start:start+_RATIO_BLOCK]
        Y = np.empty(B.shape, dtype=B.dtype)
        for i in range(B.shape[0]):
            Y[i] = _inv_cascade(B[i], upper_fs, upper_bws, int(fs))
        rms = np.zeros((len(filterbank), B.shape[0]))
        for idx, sos in enumerate(filterbank):
            y = signal.sosfiltfilt(sos, Y, axis=-1)
            rms[idx] = np.sqrt(np.sum(y**2, axis=-1)/B.shape[1])
        rms = np.maximum(rms, np.finfo(rms.dtype).tiny)
        ratios[0, start:start+B.shape[0]] = 20*np.log10(rms[1]/rms[0])   # r12
        ratios[1, start:start+B.shape[0]] = 20*np.log10(rms[2]/rms[1])   # r23
        ratios[2, start:start+B.shape[0]] = 20*np.log10(rms[3]/rms[2])   # r34
    return ratios

def _ifc_analyze(y, fs, speaker, parallel):
    '''Run the compiled analysis on every frame of y (a one-dimensional array that has already been
    resampled to 12 kHz and preemphasized).  The frames are a view of y, not a copy, and the
    arithmetic is done in the dtype of y: float32 for audio from prep_audio(), or float64 if y has
    been converted.

    Returns the frame center samples and an array with F1-F4 for each of them (unrounded).
    '''
    y = np.ascontiguousarray(y)
    centers = np.arange(half_frame, len(y)-frame_length, step)
    if len(centers) == 0:
        return centers, np.empty((0, 4))
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::step][:len(centers)]
    par = params[speaker]
    ratios = _ifc_amplitude_ratios(frames, fs, speaker)
    run = _ifc_frames_parallel if parallel else _ifc_frames
    res = run(frames, int(fs),
              par["fr"].astype(np.float64), par["bws"].astype(np.float64),
              par["upper_fs"].astype(np.float64), par["upper_bws"].astype(np.float64),
              float(par["du1"]), float(par["du2"]), int(speaker), ratios)
    return centers, res

def ifc_warmup_numba(verbose=True):
    """
    Pre-compile Numba functions for IFC_formants() with minimal data.

    Parameters
    ==========
    verbose: string, default='True'
    
    """
    if verbose:
        print("Warming up Numba (compiling IFC analysis functions)...",end="",flush=True)
    
    y = np.random.randn(1200).astype(np.float32)  # the dtype that IFC_formants() gives the compiled functions
    _ifc_analyze(y, 12000, 1, parallel=False)
    _ifc_analyze(y, 12000, 1, parallel=True)

    if verbose:
        print(" done")


def IFC_formants(x, fs, preemphasis = 0.94, speaker = 0, 
                 parallel = True, verbose=False):
    """Computes vowel formant values in audio of speech using the Inverse Filter Control method.

The function uses the Watenabe/Ueda Inverse Filter Control (IFC) method of vowel formant tracking (Watenabe, 2001; Ueda et al., 2007) to estimate the first four vowel formants (the resonant frequencies of the vocal tract) in 20 ms frames at 10 ms intervals.  The method can be quite a bit more accurate than LPC.  The analysis of the frames is accelerated by numba just in time (JIT) compilation into machine code, and the frames are analyzed in parallel on all available cores.  Therefore, the first call to `IFC_formants()` will be slower than subsequent calls, because of the time needed to compile (or load from the cache) the compiled functions.

The function only measures formants.  For f0, voicing, and amplitude see `get_f0()`, `get_rms()`, and related functions.

Parameters
==========
x : array
    a one-dimensional array of audio samples

fs : int
    the sampling rate of the audio in **x**.  The audio will be resampled to 12,000 Hz for analysis.

preemphasis : float, default = 0.94
    value of a high frequency preemphasis (spectral tilt) factor (0-1).

speaker : int, default = 0
    set formant expectations for IFC based on the approximate size of the speaker's vocal tract.

      * 0 = long vocal tract, male speaker
      * 1 = medium, female speaker
      * 2 = small, a child.

parallel : boolean, default = True
    True means the frames are analyzed on all cores.  False uses one core.

verbose : boolean, default = False
    True means print progress or warning messages

Returns
=======
df : dataframe
    a pandas dataframe with formant measurements at 0.01 sec intervals.

Raises
======
ValueError
    if `speaker` is not 0, 1, or 2.

Note
====

The columns in the output dataframe are:
    * sec - midpoint time of the frame
    * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.

References
==========

A. Watenabe (2001) `Formant estimation method using inverse-filter control`. IEEE Trans. Speech Audio Processing, **9**, 317-326.

Y. Ueda, T. Hamakawa, T. Sakata, S. Hario, & A. Watanabe (2007) `A real-time formant tracker based on the inverse filter control method`. Acoustical Science and Technology,  **28** (4), 271–274. https://doi.org/10.1250/ast.28.271

Examples
========

Use Inverse Filter Control to track formants in a file

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'sf3_cln.wav'
    x,fs = phon.loadsig(example_file, chansel=[0])
    fmtsdf = phon.IFC_formants(x, fs, speaker=1)

    ret = phon.sgram(x,fs, tf=6000, cmap="Reds")  # plot the spectrogram

    dot_color = "dodgerblue"
    sns.pointplot(fmtsdf,x='sec',y='F1',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F2',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F3',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F4',linestyle='none',native_scale=True,marker=".",color=dot_color)

"""

    if speaker not in (0, 1, 2):
        raise ValueError(f"IFC_formants(): unknown speaker {speaker!r}; expected 0, 1, or 2")

    if verbose:
        print(f"IFC_formants(), with speaker set to {speaker}")

    y, fs = prep_audio(x, fs, target_fs = SR,  # downsample
                       pre = preemphasis,  # apply preemphasis
                       scale = True, verbose=verbose)  
 
    centers, res = _ifc_analyze(y, fs, speaker, parallel)
    if len(centers) == 0:
        return DataFrame(np.empty((0, 5)), columns=("sec","F1","F2","F3","F4"))

    formants = np.empty((len(centers), 5))
    formants[:, 0] = centers/fs
    formants[:, 1:] = np.round(res, 3)

    return DataFrame(formants, columns=("sec","F1","F2","F3","F4"))
