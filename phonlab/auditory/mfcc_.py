import numpy as np
from pandas import DataFrame
from scipy.fft import dct, rfft
from scipy.signal import get_window

from ..utils.prep_audio_ import prep_audio
from ..utils.validate import positive_step, step_samples
from .mel_sgram import linear_to_mel_weight_matrix

# The log of a mel filter output is limited to this, so that a silent frame has a finite value (a power of 1e-10
# is -100 dB re the full scale of the audio after prep_audio())
LOG_FLOOR = 1e-10

# the most elements (frames times FFT points) that are transformed at once
CHUNK_ELEMENTS = 4_000_000


def _deltas(c, width=2):
    """Regression (delta) coefficients of the columns of `c` (frames by coefficients): the least squares slope of each
    coefficient over `2*width + 1` frames.  The first and last frames are repeated at the edges."""
    padded = np.pad(c, ((width, width), (0, 0)), mode='edge')
    k = np.arange(1, width + 1)
    num = sum(i * (padded[width + i:len(padded) - width + i] - padded[width - i:len(padded) - width - i]) for i in k)
    return num / (2 * np.sum(k ** 2))


@positive_step
def mfcc(x, fs, n_mfcc=12, c0=True, length=0.025, step=0.01, preemph=0.97, n_mels=40, fmin=0.0, fmax=None,
         lifter=0, window='hamming', delta=0, cmn=False):
    """Compute mel frequency cepstral coefficients (MFCC), which describe the shape of the spectrum of each frame of audio in a few numbers.

    The steps are the usual ones (Davis & Mermelstein, 1980):

    1. the audio is scaled by `prep_audio()` to have its peak at -1 dBFS (so that `c0` can be compared between recordings), and pre-emphasized by `preemph`.  The sampling rate is not changed.
    2. it is divided into frames of `length` seconds `step` seconds apart, and each frame is multiplied by a `window`
    3. the power spectrum of each frame is found with an FFT, divided by the sum of the squared window so that the level does not depend on the window length
    4. the power spectrum is passed through a bank of `n_mels` triangular filters that are evenly spaced on the mel scale (see `linear_to_mel_weight_matrix()`), each filter having a peak weight of 1
    5. the natural log of the power at the output of each filter is taken
    6. a discrete cosine transform (DCT-II, scaled to be orthonormal, as in `scipy.fft.dct(norm='ortho')`) of the log filter outputs gives the cepstral coefficients, of which the first `n_mfcc` + 1 are kept.

    A frame of digital silence has the log of the filter outputs set to `LOG_FLOOR` (-100 dB re the peak of the audio).

    The zeroth coefficient, `c0`, is proportional to the mean of the log filter outputs, so it is a measure of the level of the frame.  `c1` is large when the spectrum slopes (the lower mel filters have more energy than the higher ones, or the reverse), `c2` is larger when there is more energy in both the lowest and the highest filters than in the middle ones, and so on, with finer detail in the higher coefficients.

    Parameters
    ==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    n_mfcc : int, default = 12
        The number of cepstral coefficients `c1` ... `cN` to return, not counting `c0`.  It must be smaller than `n_mels`.
    c0 : boolean, default = True
        If True the zeroth coefficient, `c0` (the level of the frame), is also returned.  The tensorflow and librosa
        functions return `c0` and 12 more coefficients when they are asked for 13.
    length : float, default = 0.025
        The length (in seconds) of the analysis window.  The default is 25 milliseconds.
    step : float, default = 0.01
        The time (in seconds) between the starts of adjacent analysis windows.  The default is 10 milliseconds.
    preemph : float, default = 0.97
        High frequency preemphasis (a value between 0 and 1) applied to the audio before the analysis, which compensates for the
        spectral tilt of speech.  0 means none.
    n_mels : int, default = 40
        The number of mel filters.
    fmin : float, default = 0.0
        The lowest frequency (in Hz) of the filterbank.  The lower edge of the lowest filter.
    fmax : float, default = None
        The highest frequency (in Hz) of the filterbank, which can not be higher than the Nyquist frequency, `fs`/2, which is the default.
    lifter : float, default = 0
        If it is greater than 0 the coefficients `c1` ... `cN` are multiplied by `1 + (lifter/2) * sin(pi * n/lifter)`, as in HTK, which
        gives the higher coefficients a larger share (a value of 22 is often used with 12 coefficients). 0 means no liftering.
    window : string, default = 'hamming'
        The name of the window (see `scipy.signal.get_window()`) that is applied to each frame.  The window is symmetric (as in HTK).
    delta : int {0, 1, 2}, default = 0
        With 1 the delta coefficients (the slope of each coefficient over five frames, in units of coefficient per frame) are
        also returned, in columns called `d_c1` ..., and with 2 the delta-deltas, in columns called `dd_c1` ... are returned also.
    cmn : boolean, default = False
        If True the mean of each of the coefficients `c1` ... `cN` over all of the frames is subtracted (cepstral mean normalization).  This reduces the effect of a constant
        change of the spectrum from the recording channel (the microphone and the room) on the coefficients.  `c0` is left alone.

    Returns
    =======
    df : pandas DataFrame
        A dataframe with one row for each analysis frame.

    Note
    ====
    The columns in the returned dataframe are:
        * sec - time at the midpoint of each frame, in seconds
        * c0 - the zeroth coefficient, the level of the frame (if `c0` is True)
        * c1 - cN - the cepstral coefficients (if `lifter` is greater than 0 they are liftered)
        * d_c0, d_c1 ... - the delta coefficients (if `delta` is 1 or 2)
        * dd_c0, dd_c1 ... - the delta-delta coefficients (if `delta` is 2)

    The names of the columns of the coefficients are those that `mfcc_to_df()` gives to the MFCC values of a Praat object.  The coefficients are
    not the same as Praat's, however, which uses a different filterbank and scaling, but they are similar in what they show.

    This differs from the log mel spectrogram of `compute_mel_sgram()`, which has a window that is fixed at 64 ms, a log of the amplitude
    rather than the power, and 80 filters between 80 and 7600 Hz.  If you want mfccs of a different kind of spectrum, such as one made by `cepstral_smooth()`,
    use `scipy.fft.dct(log_mel_spectrum, type=2, norm='ortho')`.

    References
    ==========
    S. Davis, P. Mermelstein (1980) Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. `IEEE Transactions on Acoustics, Speech and Signal Processing` **28** (4), 357-366.

    S. Young et al. (2006) `The HTK Book`.  Cambridge University Engineering Department.  (The lifter and the delta coefficients.)

    Example
    =======

    .. code-block:: Python

        x,fs = phon.loadsig("sf3_cln.wav", chansel=[0])
        df = phon.mfcc(x, fs)                       # c0 - c12 every 10 ms
        df2 = phon.mfcc(x, fs, delta=2, cmn=True)   # with delta and delta-delta coefficients, and mean normalization

        plt.imshow(df.drop(columns='sec').T, origin='lower', aspect='auto',
                   extent=(df.sec.min(), df.sec.max(), 0, 13), cmap='Greys')
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("mfcc(): x must be a one-dimensional array of audio samples")
    if length <= 0:
        raise ValueError(f"mfcc(): length must be positive, not {length}")
    if n_mfcc < 1 or n_mfcc >= n_mels:
        raise ValueError(f"mfcc(): n_mfcc ({n_mfcc}) must be at least 1 and smaller than n_mels ({n_mels})")
    if delta not in (0, 1, 2):
        raise ValueError(f"mfcc(): delta must be 0, 1 or 2, not {delta!r}")
    if lifter < 0:
        raise ValueError(f"mfcc(): lifter must not be negative, not {lifter}")

    # (no tiny noise in digital silence, so the same audio always gives the same values: a silent frame is at the LOG_FLOOR)
    x, fs = prep_audio(x, fs, target_fs=None, preemph=preemph, scale=True, add_tiny_noise=False)

    frame_length = int(round(length * fs))
    hop = step_samples(step, fs)
    if len(x) < frame_length:
        raise ValueError(f"mfcc(): the audio ({len(x)} samples) is shorter than one {frame_length}-sample frame")
    nframes = 1 + (len(x) - frame_length) // hop

    nfft = int(2 ** np.ceil(np.log2(frame_length)))
    w = get_window(window, frame_length, fftbins=False)
    fmax = fs / 2 if fmax is None else fmax
    filters = linear_to_mel_weight_matrix(n_mels, nfft // 2 + 1, fs, fmin, fmax)   # (frequency, filter)

    # frames are processed in chunks, to limit the size of the arrays for a long recording
    log_mel = np.empty((nframes, n_mels))
    chunk = max(1, CHUNK_ELEMENTS // nfft)
    for a in range(0, nframes, chunk):
        idx = np.arange(a, min(a + chunk, nframes))
        frames = x[idx[:, None] * hop + np.arange(frame_length)[None, :]] * w
        power = np.abs(rfft(frames, nfft, axis=-1)) ** 2 / np.sum(w ** 2)
        log_mel[idx] = np.log(np.maximum(power @ filters, LOG_FLOOR))

    c = dct(log_mel, type=2, norm='ortho', axis=1)[:, :n_mfcc + 1]    # c0, c1, ... cN

    if lifter > 0:
        n = np.arange(1, n_mfcc + 1)
        c[:, 1:] *= 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
    if cmn:
        c[:, 1:] -= c[:, 1:].mean(axis=0)

    names = [f"c{i}" for i in range(n_mfcc + 1)]
    cols = [c]
    if delta >= 1:
        d = _deltas(c)
        cols.append(d)
        names += [f"d_{n}" for n in names[:n_mfcc + 1]]
        if delta == 2:
            cols.append(_deltas(d))
            names += [f"dd_{n}" for n in names[:n_mfcc + 1]]
    values = np.hstack(cols)
    if not c0:    # drop c0, and its deltas
        keep = [i for i, n in enumerate(names) if not n.endswith('c0')]
        values, names = values[:, keep], [names[i] for i in keep]

    sec = (np.arange(nframes) * hop + frame_length / 2) / fs
    df = DataFrame(values, columns=names)
    df.insert(0, 'sec', sec)
    return df
