import numpy as np
import pandas as pd

from .amp_env import amplitude_envelope
from .cepstral import CPP

# Low frequency band (Hz) whose amplitude envelope is a predictor of voicing, and the logistic
# regression coefficients of the voicing decision.  Both were fit by examples/train_vad.py against
# EGG-derived voicing.
AMP_BAND = (80, 800)
VAD_COEFS = {'intercept': -8.96, 'amp': 0.0733, 'cpp': 0.5347}


def band_db(x, fs, sec, bounds, target_fs=12000):
    """Amplitude envelope level (dB) in the band `bounds` = (low edge, high edge) in Hz, at times `sec`.

    The envelope (from `amplitude_envelope()` of peak-normalized audio) is floored at -80 dB re its peak
    (and never below -200 dB, so that a silent signal gives a finite value).
    """
    env, fs_env = amplitude_envelope(x, fs, bounds=list(bounds), target_fs=target_fs)
    db = 20 * np.log10(np.maximum(env, max(np.max(env) * 1e-4, 1e-10)))
    return db[np.minimum((np.asarray(sec) * fs_env).astype(int), len(db) - 1)]


def vad_features(y, fs, sec=None, f0_range=[63, 400], s=0.005):
    """Measure the predictors used by `VAD()`: `amp` and `cpp`.

    `CPP()` supplies `cpp` and `f0` at its own frame times.  They are put on the times in `sec` (by
    linear interpolation for `cpp` and the nearest frame for `f0`), which by default are spaced `s` seconds
    apart over the span of the `CPP()` frames.  Passing `sec` is how the training script measures the
    predictors at exactly the times of the EGG measurements.
    """
    c = CPP(y, fs, norm=False, f0_range=list(f0_range))
    t = c['sec'].to_numpy()
    if sec is None:
        sec = np.arange(t[0], t[-1], s)
    sec = np.asarray(sec)

    nearest = np.clip(np.searchsorted(t, sec), 1, len(t) - 1)
    nearest = np.where(np.abs(t[nearest - 1] - sec) <= np.abs(t[nearest] - sec), nearest - 1, nearest)
    df = {'sec': sec, 'f0': c['f0'].to_numpy()[nearest],
          'cpp': np.interp(sec, t, c['cpp'].to_numpy()),
          'amp': band_db(y, fs, sec, AMP_BAND)}
    return pd.DataFrame(df)


def VAD(y, fs, f0_range=[63, 400], s=0.005):
    """Voice activity detection: the probability that each frame of audio is voiced.

    The voicing decision is a logistic regression on two measurements made in each frame: the level of
    the amplitude envelope in a low frequency band (80-800 Hz), and cepstral peak prominence (`cpp`).
    The coefficients were trained against EGG voicing by `examples/train_vad.py`.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **y**
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.
        s : float, default = 0.005
            "Hop" interval between successive analysis windows (seconds).

    Returns
    =======
        df : pandas DataFrame
            measurements at `s` second intervals.

    Note
    ====
    The columns in the returned dataframe are:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency in Hz - from `phon.CPP()`
        * cpp - cepstral peak prominence (dB) - from `phon.CPP()`
        * amp - amplitude envelope (dB) in the low frequency band (80-800 Hz)
        * probv - probability of voicing
        * voiced - a boolean, true if probv > 0.5
    """
    df = vad_features(y, fs, f0_range=f0_range, s=s)
    z = VAD_COEFS['intercept'] + VAD_COEFS['amp'] * df['amp'] + VAD_COEFS['cpp'] * df['cpp']
    df['probv'] = 1 / (1 + np.exp(-z))
    df['voiced'] = df['probv'] > 0.5
    return df
