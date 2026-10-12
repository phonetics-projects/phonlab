import numpy as np
from scipy.signal import windows
from numpy.lib.stride_tricks import sliding_window_view


def compute_mt_sgram(x, fs, w=0.008, s=0.001, NW=4, order=12, low_bias=True, scale='psd'):
    """Compute a multitaper spectrogram from an input waveform array of samples.

    This is a multitaper version of `compute_sgram()`.  In each analysis frame the (mean-removed) audio is multiplied by each of a set of orthogonal Slepian (DPSS) tapers, computed with `scipy.signal.windows.dpss()`, and the power spectra of the tapered frames are averaged, weighted by the eigenvalue (spectral concentration) of each taper.  This gives a low-variance spectral estimate from a short window.

    Parameters
    ==========
    x : ndarray
        a one-dimensional array of audio samples
    fs : integer
        The sampling frequency of the audio samples in `x`
    w : float, default = 0.008
        Length in seconds of the analysis window.  As in `compute_sgram()`, w = 0.008 gives a wideband spectrogram, and w = 0.04 a narrow band spectrogram.
    s : float, default = 0.001
        The time (in seconds) between adjacent spectra in the spectrogram.
    NW : float, default = 4
        The time-bandwidth product of the tapers.  The half-bandwidth of the analysis is NW/w Hz, and up to 2*NW tapers are used. Larger values produce smoother spectra with poorer frequency resolution.
    order : integer or None, default = 13
        This parameter determines the number of points in the FFT analysis that produces the spectrogram.  The number of points is 2**order and should be larger than the number of points in the analysis window (which is w*fs); the window is zero padded to this length.  If `order` is `None` there is no zero padding, and the FFT is as long as the analysis window.
    low_bias : boolean, default = True
        If True, only use the tapers that have better than 90% spectral concentration within the bandwidth.
    scale : string, default = 'psd'
        If 'psd', the spectrogram is a power spectral density (power per Hz). If 'nitime', it has the scaling of `nitime.algorithms.multi_taper_psd()` with its default `Fs`, which differs from 'psd' by a constant factor of 2*pi/fs.

    Returns
    =======
    t : ndarray
        Array of segment times (the center of each analysis frame).
    f : ndarray
        Array of sample frequencies.
    Sxx : ndarray
        The multitaper power spectrogram on the decibel scale (10 * log10(power)), with shape (len(f), len(t)), as returned by `compute_sgram()`.

    Example
    =======

    .. code-block:: Python

         x, fs = phon.loadsig("sf3_cln.wav", chansel=[0])
         t, f, Sxx = phon.compute_mt_sgram(x, fs, w=0.008, s=0.001)
         plt.pcolormesh(t, f, Sxx, cmap='Greys')
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array of audio samples")
    if scale not in ('psd', 'nitime'):
        raise ValueError("scale must be 'psd' or 'nitime'")

    nperseg = int(round(w * fs))
    step = max(1, int(round(s * fs)))
    if nperseg < 2 or len(x) < nperseg:
        raise ValueError("x must contain at least one analysis window of w seconds")
    nfft = nperseg if order is None else max(2**order, nperseg)  # the window is never truncated

    tapers, eigvals = windows.dpss(nperseg, NW, Kmax=int(2 * NW), sym=True, return_ratios=True)
    if low_bias:
        keep = eigvals > 0.9
        tapers, eigvals = tapers[keep], eigvals[keep]

    # PSD = sum_k(eig_k * |X_k|^2) / sum_k(eig_k), doubled at the one-sided frequencies
    # (tapers have unit energy, so no extra window normalization is needed).
    norm = fs if scale == 'psd' else 2 * np.pi
    weights = eigvals / eigvals.sum() / norm
    nfreq = nfft // 2 + 1
    onesided = np.full(nfreq, 2.0)
    onesided[0] = 1.0
    if nfft % 2 == 0:
        onesided[-1] = 1.0

    frames = sliding_window_view(x, nperseg)[::step]
    nframes = frames.shape[0]
    Sxx = np.empty((nfreq, nframes))

    chunk = max(1, int(2e6 // (nperseg * len(eigvals))))  # bound memory use
    for i in range(0, nframes, chunk):
        fr = frames[i:i + chunk]
        fr = fr - fr.mean(axis=-1, keepdims=True)
        spec = np.fft.rfft(fr[:, None, :] * tapers[None, :, :], n=nfft, axis=-1)
        psd = np.einsum('k,fkn->fn', weights, spec.real**2 + spec.imag**2)
        Sxx[:, i:i + chunk] = (psd * onesided).T

    Sxx = 10 * np.log10(np.maximum(Sxx, np.finfo(Sxx.dtype).tiny))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    ts = (np.arange(nframes) * step + nperseg / 2) / fs
    return (ts, f, Sxx)


def tandem_spectrum(Sxx):
    """Average the frames of a spectrogram into one stabilized "tandem" spectrum.

    The frames are averaged on the power scale (not the dB scale), which is the appropriate way to average spectra.  This is typically used with frames from `compute_mt_sgram()` that are closely spaced in time, to further reduce the variance of the spectral estimate in a quasi-stationary sound like a fricative.

    Parameters
    ==========
    Sxx : ndarray
        A spectrogram on the decibel scale (10 * log10(power)) with shape (frequency, time), like the one returned by `compute_mt_sgram()`.

    Returns
    =======
    spec : ndarray
        The average spectrum, on the decibel scale, with shape (frequency,).
    """
    Sxx = np.asarray(Sxx, dtype=float)
    if Sxx.ndim != 2:
        raise ValueError("Sxx must be a two-dimensional (frequency, time) spectrogram")
    power = np.mean(10 ** (Sxx / 10), axis=1)
    return 10 * np.log10(np.maximum(power, np.finfo(power.dtype).tiny))
