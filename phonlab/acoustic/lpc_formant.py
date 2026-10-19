import numpy as np
from librosa import util, lpc, frames_to_time
from scipy import signal
from pandas import DataFrame

from ..utils.prep_audio_ import prep_audio
from ..acoustic.choose_order_ import choose_order

# constants
SR = 12000                     # sampling rate of the analysis
WINDOW_STD = 1/4.5             # standard deviation of the Gaussian analysis window, as a fraction of l


def _analysis_window(frame_length, fs, l):
    '''The Gaussian window applied to each frame.  Its standard deviation is defined in time, as
    WINDOW_STD * l seconds, and converted to samples.  With WINDOW_STD = 1/4.5 the window reaches
    all the way to the ends of the frame (the ends are 2.25 standard deviations from the center,
    where the weight is 0.08) and is about as wide (FWHM 0.52 of the frame) as a Hamming window of
    the same length, but without Hamming's sidelobes.'''
    return signal.windows.gaussian(frame_length, WINDOW_STD * l * fs, sym=True).astype(np.float32)  # as the audio



def LPC_formants(x, fs, preemphasis = 0.94, order = -1, max_bw = 800, s = 0.01, l = 0.03, quiet = True):
    """Computes vowel formant values in audio of speech using linear predictive coding (LPC).

The function uses the Librosa implemenation of Burg's (1975) method for linear predictive coding and then polynomial root solving of the LPC coefficients to find the vowel formant frequencies (the resonant frequencies of the vocal tract) and their bandwidths in overlapping frames (by default 30 ms long, at 10 ms intervals).

The function only measures formants and their bandwidths.  For f0, voicing, and amplitude see `get_f0()`, `get_rms()`, and related functions.

Parameters
==========
x : array
    a one-dimensional array of audio samples

fs : int
    the sampling rate of the audio in **x**.  The audio will be resampled to 12,000 Hz for analysis, so choose the order parameter with this in mind.

preemphasis : float, default = 0.94
    value of a high frequency preemphasis (spectral tilt) factor (0-1).

order : int, default = -1
    the default value, -1 tries several values of lpc_order and determines the best value for this audio file (see `choose_order()`).  You can also specify the order by setting this parameter to a positive integer (12 or 14 for a male speaker, 10 or 12 for a female speaker, 8 or 10 for a child).

max_bw : float, default = 800
    the widest bandwidth (Hz) that a resonance can have and still be counted as a formant.  A wide bandwidth criterion (about 800 Hz) is needed to capture fast moving formants.  Resonances with a wider bandwidth are rejected: the formant frequency is NaN (and NaN sorts to the end of the frame's formants).

s : float, default = 0.01
    the interval (in seconds) between successive formant measurements.

l : float, default = 0.03
    the duration (in seconds) of analysis frames.  This is also the frame duration that `choose_order()` uses when `order` is -1.  Each frame is multiplied by a Gaussian window whose standard deviation is `l`/4.5, so the window spans the whole frame, with the ends at 2.25 standard deviations from the center (a weight of 0.08, like a Hamming window).  The window scales with `l`.

quiet : boolean, default = True
    False means print progress or warning messages

Returns
=======
df : dataframe
    a pandas dataframe with formant and bandwidth measurements at 0.01 sec intervals.

Raises
======
ValueError
    if `x` is shorter than one frame of `l` seconds (after resampling to 12,000 Hz), or if `s` or `l` is not positive.

Note
====

The columns in the output dataframe are:
    * sec - midpoint time of the frame
    * F1-n - the lowest vowel 'formants' - vocal tract resonances.
    * BW1-n - the bandwidths (Hz) of the formants F1-n, calculated from the radius of the LPC polynomial roots.  If fewer than n resonances have a bandwidth below `max_bw`, the remaining formants and their bandwidths are NaN.
    
References
==========

J. P. Burg (1975) `Maximum entropy spectral analysis`, Ph.D. dissertation, Dep. Geophys., Stanford Univ., Stanford, CA.

Examples
========

Track formants in a file with LPC analysis, which by default will try to pick the best `order` for the speaker.

.. code-block:: Python
    
    example_file = AUDIO / 'sf3_cln.wav'
    x,fs = phon.loadsig(example_file, chansel=[0])

    fmtsdf = phon.LPC_formants(x, fs,order=-1)

    vaddf = phon.VAD(x,fs)  # mark frames as voiced or voiceless
    fmtsdf = phon.align_timeseries(fmtsdf, vaddf, ts='sec', other_columns=['voiced'])
    fmtsdf.loc[fmtsdf["voiced"] == False, ("F1","F2","F3","F4")] = np.nan # ignore voiceless frames

    ret = phon.sgram(x,fs, tf=6000,cmap="Blues")  # plot the spectrogram

    dot_color = "red"
    sns.pointplot(fmtsdf,x='sec',y='F1',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F2',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F3',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F4',linestyle='none',native_scale=True,marker=".",color=dot_color)

.. figure:: images/LPC_formants.png
   :scale: 50%
   :alt: a spectrogram with the estimated vowel formants marked with blue dots 
   :align: center

   Plotting the formants found by `track_formants(... method='ifc')`, on the spectrogram of the utterance.

"""

    if s <= 0 or l <= 0:
        raise ValueError(f"LPC_formants(): s ({s}) and l ({l}) must be positive")

    if not quiet: print(f"LPC_formants(), with order set to {order}")

    x, fs = prep_audio(x, fs, target_fs = SR,  # downsample
                       pre = preemphasis,  # no preemphasis
                       scale = True, quiet = quiet) 
    
    frame_length = int(fs*l)  # samples in a frame
    step = int(fs*s)          # number of samples between frames
    if len(x) < frame_length:
        raise ValueError(f"LPC_formants(): audio is shorter than one {frame_length}-sample frame")

    if order < 0:  # guess the correct LPC order
        order, time = choose_order(x, fs, l)
        if not quiet: print(f"Selected LPC order is: {order}, measured at time {time:.3f}")

    # LPC coefficients for each (windowed) frame of the whole file
    frames = util.frame(x, frame_length=frame_length, hop_length=step, axis=0)
    frames = np.multiply(frames, _analysis_window(frame_length, fs, l))
    t = frames_to_time(range(frames.shape[0]), sr=fs, hop_length=step, n_fft=frame_length)
    A = lpc(frames, order=order, axis=-1)

    nb = A.shape[0]  # the number of frames (or blocks) in the LPC analysis
    nf = int((order-2)/2)  # the number of formants that will be computed
    formants = np.empty((nb, nf))
    bandwidths = np.empty((nb, nf))

    for i in range(nb):  # find formants from LPC coefficients (A) using root solving
        roots = np.roots(A[i,:])  # solve for the roots of the polynomial
        roots = roots[np.imag(roots)>0]  # only keep one of each conjugate pair; real roots are not resonances
        fr = np.angle(roots) * fs/(2 * np.pi)  # calculate formant frequencies from them
        bw = -(fs/np.pi) * np.log(abs(roots))  # calculate bandwidths from them

        wide = bw>max_bw
        fr[wide] = np.nan  # reject formants with too wide bandwidth, NaN sorts to the end of the array
        bw[wide] = np.nan  # and don't report the bandwidth of a rejected formant
        srt = np.argsort(fr)  # get the sort order so you can apply it more than one array
        n = min(nf, len(fr))  # a frame with several real roots can have fewer than nf resonances
        formants[i,:] = np.nan
        bandwidths[i,:] = np.nan
        formants[i,:n] = fr[srt][:n]  # lowest to highest, NaN last
        bandwidths[i,:n] = bw[srt][:n]  # use the same sort order with the bandwidth array

    columns = ['sec'] + [f"F{i}" for i in range(1, nf+1)] + [f"BW{i}" for i in range(1, nf+1)]
    return DataFrame(np.column_stack((t, formants, bandwidths)), columns=columns)
