import functools
import math
import warnings

import numpy as np
from numba import njit, prange
from pandas import DataFrame
from scipy import signal

from ..utils.prep_audio_ import prep_audio
from ..utils.deprecation import renamed_kwargs
from ..utils.validate import positive_step, step_samples

# constants
SR = 12000                     # sampling rate of the analysis for speech (scaled by ref_f1 / REF_F1, see below)
FMAX = 6000
FMIN = 200
REF_F1 = 500.0                 # the F1 (Hz) that SR and the internal constants of _zc2() / _ifc_frame() are sized for
FRAME_LENGTH = 0.02            # default analysis frame, seconds (20 ms; see IFC_formants())
STEP = 0.01                    # default interval between frames, seconds (10 ms)
MIN_FRAME_SAMPLES = 120        # shorter frames have too few samples for the band-pass filters' padding (10 ms at 12 kHz)

# Speaker parameters used in IFC tracking.  params[0] is the male preset, params[1] female, params[2] child.
# The keys (all frequencies in Hz):
#   fr          five expected frequencies: [f0 notch, F1, F2, F3, F4]
#   bws         the matching five inverse-filter bandwidths
#   upper_fs    resonances above F4 that are removed before tracking, and
#   upper_bws   their bandwidths (one per entry of upper_fs)
#   bands       four [low, high] bands whose level ratios set the f0 notch width in each IFC block
#   du1, du2    search-window parameters for F1 in _zc2()
#   extra_upper_fs, extra_upper_bws   resonances removed a second time before the F1/F2 block
#               (the male preset removes 5500 Hz again; the others nothing) - optional, default none
#   fmin, fmax  the frequency range a formant estimate is clamped to - optional, default 200 and 6000
#               scaled by ref_f1 / 500 (so 14-420 Hz for ref_f1 = 35)
#   ref_f1      the F1 the tracker is sized for - optional, default fr[1].  The analysis rate, the
#               internal search windows and the default frame length all scale with ref_f1 / 500
#               (see IFC_formants()).  The presets all set it to 500, the speech value.
#   spec_bounds, dh1, dh2   reserved, not used by the tracker
# A caller can pass a dict with these keys as `speaker` instead of a preset number (see IFC_formants()).
PARAM_DEFAULTS = {'extra_upper_fs': np.array([], dtype=np.float64), 'extra_upper_bws': np.array([], dtype=np.float64)}
params = [
    {
        'fr':np.array([ 60, 500, 1500, 2500, 3500]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5500,5500,4500]),
        'upper_bws': np.array([80,80,200]),
        'bands':np.array([[200,1100], [700, 2200], [2000,3500], [3000,4300]]),
        'spec_bounds':np.array([120,5500]),
        'du1':300, 'du2':600, 'dh1':200, 'dh2':500, 'ref_f1':REF_F1,
        'extra_upper_fs':np.array([5500.0]), 'extra_upper_bws':np.array([200.0])   # remove f6 again before F1/F2
    },
    {
        'fr':np.array([ 60, 620, 1860, 3100, 4340]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5580,5580]),  # stick with Ueda et al. (2007) on this.
        'upper_bws':np.array([3200,200]),
        'bands':np.array([[300,1300], [1000, 2500], [2500,4000], [3500,4800]]),
        'spec_bounds':np.array([250,5500]),
        'du1':400, 'du2':700, 'dh1':300, 'dh2':600, 'ref_f1':REF_F1
    },
    {
        'fr':np.array([ 60, 680, 2050, 3420, 4790]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([6000,5500]),
        'upper_bws':np.array([2700,2700]),
        'bands':np.array([[400,1600], [1300, 3000], [3000,4500], [3800,5500]]),
        'spec_bounds':np.array([300,5800]),
        'du1':500, 'du2':800, 'dh1':400, 'dh2':700, 'ref_f1':REF_F1
    }
]


_ARRAY_KEYS = {'fr': (5,), 'bws': (5,), 'upper_fs': None, 'upper_bws': None, 'bands': (4, 2),
               'spec_bounds': (2,), 'extra_upper_fs': None, 'extra_upper_bws': None}
_SCALAR_KEYS = ('du1', 'du2', 'dh1', 'dh2', 'fmin', 'fmax', 'ref_f1')
_REQUIRED_KEYS = ('fr', 'bws', 'upper_fs', 'upper_bws', 'bands', 'du1', 'du2')
_OPTIONAL_KEYS = ('extra_upper_fs', 'extra_upper_bws', 'fmin', 'fmax', 'ref_f1', 'spec_bounds', 'dh1', 'dh2')


def resolve_speaker_params(speaker):
    """Return a complete, validated parameter dict for IFC tracking.

    `speaker` is 0, 1 or 2 (a preset), or a dict of parameters in Hz with the keys 'fr', 'bws', 'upper_fs', 'upper_bws', 'bands', 'du1' and 'du2'
    (see the comment above `params`); 'extra_upper_fs', 'extra_upper_bws', 'fmin', 'fmax' and
    'ref_f1' are optional and default to none, 200 * ref_f1 / 500, 6000 * ref_f1 / 500 and fr[1].  Arrays are converted to float64 and
    shapes are checked.  The returned dict is a fresh copy.
    """
    if isinstance(speaker, dict):
        unknown = set(speaker) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)
        if unknown:
            raise ValueError(f"IFC speaker params: unknown key(s) {sorted(unknown)}; "
                             f"allowed: {sorted(_REQUIRED_KEYS + _OPTIONAL_KEYS)}")
        missing = set(_REQUIRED_KEYS) - set(speaker)
        if missing:
            raise ValueError(f"IFC speaker params: missing key(s) {sorted(missing)}")
        full = {**PARAM_DEFAULTS, 'spec_bounds': np.array([0.0, 0.0]), 'dh1': 0.0, 'dh2': 0.0, **speaker}
    elif speaker in (0, 1, 2):
        full = {**PARAM_DEFAULTS, **params[speaker]}
    else:
        raise ValueError(f"IFC_formants(): unknown speaker {speaker!r}; expected 0, 1, 2 or a dict of parameters")
    out = {}
    for k, shape in _ARRAY_KEYS.items():
        a = np.asarray(full[k], dtype=np.float64)
        if shape is not None and a.shape != shape:
            raise ValueError(f"IFC speaker params: {k} must have shape {shape}, got {a.shape}")
        out[k] = np.ascontiguousarray(a)
    if full.get('ref_f1') is None:
        full['ref_f1'] = out['fr'][1]
    if not full['ref_f1'] > 0:
        raise ValueError("IFC speaker params: ref_f1 must be positive")
    if full.get('fmin') is None:
        full['fmin'] = FMIN * full['ref_f1'] / REF_F1
    if full.get('fmax') is None:
        full['fmax'] = FMAX * full['ref_f1'] / REF_F1
    for k in _SCALAR_KEYS:
        out[k] = float(full[k])
    for a, b in (('upper_fs', 'upper_bws'), ('extra_upper_fs', 'extra_upper_bws')):
        if out[a].shape != out[b].shape:
            raise ValueError(f"IFC speaker params: {a} and {b} must have the same length")
    if not (0 < out['fmin'] < out['fmax']):
        raise ValueError("IFC speaker params: need 0 < fmin < fmax")
    nyq = analysis_rate(out['ref_f1']) / 2
    # the band-pass filters are designed with butter(), which needs 0 < edge < Nyquist; the inverse
    # filters are fine anywhere in 0 <= f <= Nyquist (the child preset puts one at 6000 Hz)
    how = (f"half the analysis rate for ref_f1 = {out['ref_f1']:g} Hz (12 * ref_f1); either lower the offending "
           f"frequencies or set 'ref_f1' explicitly higher than fr[1] to raise the analysis rate")
    if not (0 < out['bands'].min() and out['bands'].max() < nyq):
        raise ValueError(f"IFC speaker params: band edges must lie in 0 < f < {nyq:g} Hz "
                         f"(got {out['bands'].min():.0f}-{out['bands'].max():.0f}), {how}")
    top = max(out['fr'].max(), out['upper_fs'].max(initial=0.0), out['extra_upper_fs'].max(initial=0.0))
    if not (out['fr'].min() >= 0 and top <= nyq):
        raise ValueError(f"IFC speaker params: inverse filter frequencies must lie in 0 <= f <= {nyq:g} Hz "
                         f"(got up to {top:.0f}), {how}")
    return out


def analysis_rate(ref_f1=REF_F1):
    """The sampling rate (Hz) the audio is resampled to for a speaker with F1 expectation `ref_f1`:
    12000 Hz at the speech default of 500 Hz, in proportion otherwise.  The tracker's inverse
    filters are 3-tap difference filters whose coefficients are set by the formant frequency
    relative to the sampling rate; at a rate far above the formants they become ill-conditioned
    (a 35 Hz resonance at 12 kHz is a near-exact second difference with a gain of nearly 3000),
    so the rate has to follow the speaker's frequency range."""
    return int(round(SR * ref_f1 / REF_F1))


def ifc_speaker_params(speaker=0):
    """Return the IFC tracking parameters for a speaker as a plain dict, ready to edit and pass
    back to IFC_formants() as `speaker`.

Parameters
==========
speaker : int or dict, default = 0
    a preset number (0 = male, 1 = female, 2 = child) or a dict of parameters as accepted by
    IFC_formants(); a dict is validated and completed.

Returns
=======
dict
    every key IFC_formants() uses, with the optional ones filled in ('extra_upper_fs',
    'extra_upper_bws', 'fmin', 'fmax', 'ref_f1'); arrays are lists and scalars floats, so the
    dict can be edited, serialized or compared directly.  Keys the tracker does not use
    ('spec_bounds', 'dh1', 'dh2') are left out.

Example
=======
Start from the male preset and move the F1 expectation and search window for a speaker with
a long vocal tract::

    p = phon.ifc_speaker_params(0)
    p['fr'][1] = 400
    df = phon.IFC_formants(x, fs, speaker=p)

See the `speaker` argument of IFC_formants() for what each key does.
    """
    full = resolve_speaker_params(speaker)
    return {k: (full[k].tolist() if isinstance(full[k], np.ndarray) else float(full[k]))
            for k in _REQUIRED_KEYS + ('extra_upper_fs', 'extra_upper_bws', 'fmin', 'fmax', 'ref_f1')}


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
def _zc2(y, fs, loop, f_no, in_freq, du1, du2, fmin, fmax, fsc):
    '''Estimate the dominant frequency in y from the weighted mean of its zero-crossing period
    distribution (after Ueda et al.'s 2008 C code).  The input frequency (in_freq) is not used if
    loop==0.

    Input
        y - one frame of inverse filtered audio
        fs - the waveform sampling frequency
        loop - which iteration in _ifcblock() are we in? - 0,1,2
        f_no - which formant are we measuring - 1,2,3,4
        in_freq - the previously found frequency of this formant (for this frame)
        du1, du2 - search window parameters for F1 (from the speaker params)
        fmin, fmax - the estimate is clamped to this range
        fsc - ref_f1 / 500: scales the fixed, speech-sized search window constants below
    '''
    n = y.shape[0]
    freq = in_freq
    ca = 0.3
    dd1 = 300.0 * fsc
    dd2 = 3000.0 * fsc
    if loop == 0:  # first loop in an IFC block, we calculate mean freq as a spectral parameter
        ca = 0.2
        dd1 = 400.0 * fsc
        dd2 = 4000.0 * fsc
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
        al1 = (du2-du1)/(1000*fsc)
        b1 = du1 - al1*(200*fsc)
    elif f_no == 2 or f_no == 3:   # F2 and F3
        al1 = (dd2-dd1)/(4300*fsc)
        b1 = dd1 - al1*(200*fsc)
    else:   # F4
        al1 = 0.83721
        b1 = (400 - al1 * 200) * fsc

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

    if freq < fmin: freq = fmin
    if freq > fmax: freq = fmax
    return freq

@njit(cache=True, error_model='numpy')
def _ifcblock(x, fs, nc, fc, bc, nd, fd, bd, f0, b0, du1, du2, fmin, fmax, k):
    '''Three passes of removing one of two formants (fc) and estimating the other (fd), and
    then removing fd and re-estimating fc.  Returns the new (fc, fd).'''
    for i in range(3):  # Ueda et al do this 3 times
        yy = _ifc_filter(x, fs, nc, fc, bc, f0, b0)
        fd = _zc2(yy, fs, i, nd, fd, du1, du2, fmin, fmax, k)
        yy = _ifc_filter(x, fs, nd, fd, bd, f0, b0)
        fc = _zc2(yy, fs, i, nc, fc, du1, du2, fmin, fmax, k)
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
def _ifc_frame(x, fs, fr, bws, upper_fs, upper_bws, du1, du2, extra_fs, extra_bws, fmin, fmax, k,
               r12, r23, r34):
    '''Find F1-F4 in one frame x (float32 or float64), given its band amplitude ratios.
    extra_fs/extra_bws are removed again before the F1/F2 block (may be empty).
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
        b0 = 200.0 * k
        if r23 >= -20: b0 = (100*r23 + 2200) * k
        y2 = _inv_cascade(y, np.array((f4, f0, f1)), np.array((b4, b0, b1)), fs)   # filter out all but f2 and f3
        f3, f2 = _ifcblock(y2, fs, 3, f3, b3, 2, f2, b2, f0, b0, du1, du2, fmin, fmax, k)   # returns (F3, F2)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

        # estimate F1 and F2 next
        # filter out all but f1 and f2 (and, for a long vocal tract, remove the extra upper resonances again)
        y2 = _inv_cascade(y, np.concatenate((extra_fs, np.array((f3, f4)))),
                          np.concatenate((extra_bws, np.array((b3, b4)))), fs)
        b0 = 200.0 * k
        if r12 >= -20: b0 = (100*r12 + 2200) * k
        f1, f2 = _ifcblock(y2, fs, 1, f1, b1, 2, f2, b2, f0, b0, du1, du2, fmin, fmax, k)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

        # estimate F3 and F4
        b0 = 200.0 * k
        if r34 >= -20: b0 = (100*r34 + 2200) * k
        y2 = _inv_cascade(y, np.array((f1, f0, f2)), np.array((b1, b0, b2)), fs)   # filter out all but f3 and f4
        f3, f4 = _ifcblock(y2, fs, 3, f3, b3, 4, f4, b4, f0, b0, du1, du2, fmin, fmax, k)
        f1, f2, f3, f4 = _sort4(f1, f2, f3, f4)

    return f1, f2, f3, f4

@njit(cache=True, error_model='numpy')
def _ifc_frames(X, fs, fr, bws, upper_fs, upper_bws, du1, du2, extra_fs, extra_bws, fmin, fmax, k, ratios):
    '''_ifc_frame() for every row of X (one frame per row); ratios has one column per frame.'''
    R = X.shape[0]
    out = np.empty((R, 4))
    for r in range(R):
        f1, f2, f3, f4 = _ifc_frame(X[r], fs, fr, bws, upper_fs, upper_bws, du1, du2,
                                    extra_fs, extra_bws, fmin, fmax, k, ratios[0, r], ratios[1, r], ratios[2, r])
        out[r, 0] = f1
        out[r, 1] = f2
        out[r, 2] = f3
        out[r, 3] = f4
    return out

@njit(cache=True, error_model='numpy', parallel=True)
def _ifc_frames_parallel(X, fs, fr, bws, upper_fs, upper_bws, du1, du2, extra_fs, extra_bws, fmin, fmax, k, ratios):
    '''_ifc_frames() with the frames spread over all cores.'''
    R = X.shape[0]
    out = np.empty((R, 4))
    for r in prange(R):
        f1, f2, f3, f4 = _ifc_frame(X[r], fs, fr, bws, upper_fs, upper_bws, du1, du2,
                                    extra_fs, extra_bws, fmin, fmax, k, ratios[0, r], ratios[1, r], ratios[2, r])
        out[r, 0] = f1
        out[r, 1] = f2
        out[r, 2] = f3
        out[r, 3] = f4
    return out

@functools.lru_cache(maxsize=None)
def _ifc_filterbank(bands, fs):
    '''The four band-pass filters (second-order sections) for `bands` (a tuple of (low, high)
    pairs) at sampling rate fs.  Designing them costs a few ms, so they are built once per
    (bands, fs).'''
    return tuple(signal.butter(12, b, fs=fs, btype='bandpass', output='sos') for b in bands)

_RATIO_BLOCK = 4096   # frames per block in _ifc_amplitude_ratios()

def _ifc_amplitude_ratios(X, fs, speaker):
    '''The relative amplitude ratios in the formant spectral regions of every frame.  X has one frame
    per row; each row is passed through the upper formant inverse filter and then through the four
    band-pass filters (one sosfiltfilt() call per band over a block of frames).  The frames are done
    in blocks of _RATIO_BLOCK because sosfiltfilt() works in double precision and returns an array
    that size, so the memory needed would otherwise grow with the length of the audio.

    Returns an array of shape (3, n_frames): r12, r23, r34 per frame, the ratios in dB of the
    energy in the F2 and F1, F3 and F2, and F4 and F3 bands.
    '''
    upper_fs = speaker["upper_fs"]
    upper_bws = speaker["upper_bws"]
    filterbank = _ifc_filterbank(tuple(tuple(b) for b in speaker["bands"]), fs)
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

def _ifc_analyze(y, fs, speaker, parallel, frame_length=FRAME_LENGTH, step=STEP):
    '''Run the compiled analysis on every frame of y (a one-dimensional array that has already been
    resampled to the analysis rate `fs` and preemphasized) with the resolved speaker params `speaker`
    (from resolve_speaker_params()), in frames of
    `frame_length` seconds every `step` seconds.  The frames are a view of y, not a copy, and the
    arithmetic is done in the dtype of y: float32 for audio from prep_audio(), or float64 if y has
    been converted.

    Returns the frame center samples and an array with F1-F4 for each of them (unrounded).
    '''
    y = np.ascontiguousarray(y)
    flen = int(round(fs * frame_length))
    hop = step_samples(step, fs)
    centers = np.arange(flen // 2, len(y) - flen, hop)
    if len(centers) == 0:
        return centers, np.empty((0, 4))
    frames = np.lib.stride_tricks.sliding_window_view(y, flen)[::hop][:len(centers)]
    ratios = _ifc_amplitude_ratios(frames, fs, speaker)
    run = _ifc_frames_parallel if parallel else _ifc_frames
    res = run(frames, int(fs), speaker["fr"], speaker["bws"], speaker["upper_fs"], speaker["upper_bws"],
              speaker["du1"], speaker["du2"], speaker["extra_upper_fs"], speaker["extra_upper_bws"],
              speaker["fmin"], speaker["fmax"], speaker["ref_f1"] / REF_F1, ratios)
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
    speaker = resolve_speaker_params(1)
    _ifc_analyze(y, 12000, speaker, parallel=False)
    _ifc_analyze(y, 12000, speaker, parallel=True)

    if verbose:
        print(" done")


@renamed_kwargs(preemphasis='preemph')
@positive_step
def IFC_formants(x, fs, preemph = 0.94, speaker = 0, 
                 parallel = True, verbose=False, frame_length=None, step=None):
    """Computes vowel formant values in audio of speech using the Inverse Filter Control method.

The function uses the Watanabe/Ueda Inverse Filter Control (IFC) method of vowel formant tracking (Watanabe, 2001; Ueda et al., 2007) to estimate the first four vowel formants (the resonant frequencies of the vocal tract) in 20 ms frames at 10 ms intervals (by default; see `frame_length`).  The method can be quite a bit more accurate than LPC.  The analysis of the frames is accelerated by numba just in time (JIT) compilation into machine code, and the frames are analyzed in parallel on all available cores.  Therefore, the first call to `IFC_formants()` will be slower than subsequent calls, because of the time needed to compile (or load from the cache) the compiled functions.

The function only measures formants.  For f0, voicing, and amplitude see `get_f0()`, `get_rms()`, and related functions.

Parameters
==========
x : array
    a one-dimensional array of audio samples

fs : int
    the sampling rate of the audio in **x**.  The audio will be resampled to 12,000 Hz for analysis
    (or to 12000 * ref_f1 / 500 Hz for a custom speaker, see `speaker`).

preemph : float, default = 0.94
    value of a high frequency preemphasis (spectral tilt) factor (0-1).

speaker : int or dict, default = 0
    set formant expectations for IFC based on the approximate size of the speaker's vocal tract.

      * 0 = long vocal tract, male speaker
      * 1 = medium, female speaker
      * 2 = small, a child.

    A dict gives the parameters directly, in Hz, for a speaker the presets do not fit.
    `ifc_speaker_params(n)` returns preset n as such a dict to start from, e.g.
    `dict(phon.ifc_speaker_params(0), fr=[60, 450, 1400, 2400, 3400])`.

    Required keys:

      * 'fr' - five expected frequencies: the f0 notch and F1-F4
      * 'bws' - the five inverse-filter bandwidths that go with 'fr'
      * 'upper_fs', 'upper_bws' - resonances above F4 that are removed before tracking (may be
        empty lists)
      * 'bands' - four [low, high] frequency bands; the level ratios between neighboring bands
        set the width of the f0 notch
      * 'du1', 'du2' - the F1 search window

    Optional keys:

      * 'extra_upper_fs', 'extra_upper_bws' - resonances removed again before the F1/F2 block
        (default none; the male preset uses 5500 Hz)
      * 'fmin', 'fmax' - the range a formant estimate is clamped to (default 200 and 6000 Hz,
        scaled by ref_f1 / 500)
      * 'ref_f1' - the F1 the tracker is sized for, see below (default 'fr'[1])

    'ref_f1' is for vocal tracts far outside the human range.  Besides the speaker parameters,
    the tracker has fixed quantities that Watanabe and Ueda sized for adult speech, i.e. for an
    F1 near 500 Hz: the 12 kHz analysis rate (the 3-tap inverse filters become ill-conditioned
    when the formants are far below it), the widths of the windows over the zero-crossing period
    distribution that each formant estimate is averaged over, the bandwidth of the f0 notch, and
    the 20 ms frame.  They all scale by ref_f1 / 500, where 'ref_f1' is the F1 of a neutral
    vowel of your speaker (about c / 4L for a tract of length L: 35 Hz for 2.5 m); it defaults
    to the F1 expectation 'fr'[1], so the 'ref_f1' key only needs to be given when the two should
    differ.  For the 2.5 m example the analysis then runs at 840 Hz in 290 ms frames.  The presets
    all set 500: at speech scale the difference between a 500 and a 680 Hz F1 is immaterial, at
    35 Hz it is decisive.  In effect the tracker is applied to the audio as if its frequency
    axis were stretched by 500 / ref_f1, while everything the caller gives and gets back stays
    in real Hz and seconds.

parallel : boolean, default = True
    True means the frames are analyzed on all cores.  False uses one core.

verbose : boolean, default = False
    True means print progress or warning messages

frame_length : float, default = None
    duration of the analysis frame in seconds.  With the default None:

      * for a preset speaker (0, 1, 2) the frame is 20 ms, the value the method was developed
        with for speech;
      * for a dict it is `0.02 * 500 / ref_f1`, i.e. the same 20 ms unless 'ref_f1' (or 'fr'[1],
        its default) is far from 500 Hz, in which case the frame is lengthened in proportion:
        0.29 s for a 35 Hz speaker.

    The reason a frame has to scale with the speaker: each frame is analyzed on its own, from the
    distribution of zero-crossing periods of the inverse-filtered waveform, so it has to hold a
    few glottal periods -- with speech F0 above about 100 Hz a 20 ms frame holds two or more.
    To choose a value yourself, a rule of thumb is `frame_length = 3 / F0_min`, where F0_min is
    the lowest F0 expected (0.3 s for F0 down to 10 Hz); since vocal tract length and F0 scale
    together across speakers this usually agrees with the default.  Longer frames average over
    formant movement, so do not go beyond what the F0 requires.  The minimum is 120 samples at
    the analysis rate (10 ms for speech).

    A warning is issued when the given `frame_length` is more than a factor of two away from the
    default for the speaker, or when it holds too few or far too many periods of the f0 notch
    'fr'[0] (`frame_length * fr[0]` outside 0.6-6; the presets give 0.02 * 60 = 1.2).

step : float, default = None
    interval between frames in seconds.  When not given it is `frame_length / 2` (10 ms for the
    default frame), which keeps the 50% overlap of the original method.  As in every phonlab
    function, a `step` that is passed must be a positive number of seconds.

Returns
=======
df : dataframe
    a pandas dataframe with formant measurements every `step` seconds (0.01 by default).

Raises
======
ValueError
    if `speaker` is not 0, 1, 2 or a complete, valid dict of parameters (unknown or missing
    keys, wrong array shapes, or frequencies beyond the 6 kHz Nyquist rate of the analysis),
    or if `frame_length` is shorter than 120 samples at the analysis rate or `step` is not
    positive.

Warns
=====
UserWarning
    if `frame_length` looks mismatched with the speaker parameters' `ref_f1` or f0 notch
    `fr[0]` (see `frame_length`).

Note
====

The columns in the output dataframe are:
    * sec - midpoint time of the frame
    * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.

References
==========

A. Watanabe (2001) `Formant estimation method using inverse-filter control`. IEEE Trans. Speech Audio Processing, **9**, 317-326.

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

    speaker = resolve_speaker_params(speaker)       # a preset number or a dict -> the full, validated dict
    sr = analysis_rate(speaker['ref_f1'])
    expected = FRAME_LENGTH * REF_F1 / speaker['ref_f1']    # the default frame (20 ms for speech), scaled to this speaker
    if frame_length is None:
        frame_length = expected
    if step is None:
        step = frame_length / 2
    min_frame = MIN_FRAME_SAMPLES / sr
    if not (frame_length >= min_frame and step > 0):
        raise ValueError(f"IFC_formants(): frame_length must be at least {min_frame:.3g} s ({MIN_FRAME_SAMPLES} samples "
                         f"at the {sr} Hz analysis rate) and step positive, got {frame_length!r}, {step!r}")
    periods = frame_length * speaker['fr'][0]              # notch periods per frame; 1.2 for the presets at 20 ms
    if not (0.5 <= frame_length / expected <= 2):
        warnings.warn(f"IFC_formants(): frame_length {frame_length:g} s does not match ref_f1 = {speaker['ref_f1']:g} Hz, "
                      f"which calls for a frame near {expected:.3g} s", stacklevel=2)
    elif speaker['fr'][0] > 0 and not (0.6 <= periods <= 6):
        warnings.warn(f"IFC_formants(): frame_length {frame_length:g} s holds {periods:.2g} periods of the f0 notch "
                      f"fr[0] = {speaker['fr'][0]:g} Hz; the frame should hold a few glottal periods "
                      f"(about 1.2 notch periods, i.e. frame_length near {1.2 / speaker['fr'][0]:.3g} s)",
                      stacklevel=2)

    if verbose:
        print(f"IFC_formants(), ref_f1 {speaker['ref_f1']:g} Hz, "
              f"frame {frame_length*1000:g} ms every {step*1000:g} ms, analysis rate {sr} Hz")

    y, fs = prep_audio(x, fs, target_fs = sr,  # resample to the analysis rate
                       preemph = preemph,  # apply preemphasis
                       scale = True, verbose=verbose)  
 
    centers, res = _ifc_analyze(y, fs, speaker, parallel, frame_length, step)
    if len(centers) == 0:
        return DataFrame(np.empty((0, 5)), columns=("sec","F1","F2","F3","F4"))

    formants = np.empty((len(centers), 5))
    formants[:, 0] = centers/fs
    formants[:, 1:] = np.round(res, 3)

    return DataFrame(formants, columns=("sec","F1","F2","F3","F4"))
