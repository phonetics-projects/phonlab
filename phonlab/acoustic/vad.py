import numpy as np

from .amp_env import amplitude_envelope
from .cepstral import CPP

# Low frequency band (Hz) whose amplitude envelope is a predictor of voicing, and the logistic
# regression coefficients of the voicing decision.  Both were fit by examples/train_vad.py against
# EGG-derived voicing.
AMP_BAND = (80, 800)
VAD_COEFS = {'intercept': -4.559, 'amp': 0.1848, 'cpp': 0.4548}


def band_db(x, fs, sec, bounds, target_fs=12000):
    """Amplitude envelope level (dB) in the band `bounds` = (low edge, high edge) in Hz, at times `sec`.

    The envelope (from `amplitude_envelope()` of peak-normalized audio) is floored at -80 dB re its peak
    (and never below -200 dB, so that a silent signal gives a finite value).
    """
    env, fs_env = amplitude_envelope(x, fs, bounds=list(bounds), target_fs=target_fs)
    db = 20 * np.log10(np.maximum(env, max(np.max(env) * 1e-4, 1e-10)))
    return db[np.minimum((np.asarray(sec) * fs_env).astype(int), len(db) - 1)]


def vad_features(y, fs, f0_range=[63, 400], s=0.005):
    """Measure the predictors used by `VAD()`: `cpp` (from `CPP()`) and `amp` (from `band_db()`).

    The dataframe from `CPP()` (columns `sec`, `f0` and `cpp`) gets an `amp` column measured at the same
    times.  `CPP()` is called without smoothing (smooth=0), which costs about a quarter as much as the
    default and predicts voicing as well (see examples/compare_cpp_smooth.py).  Its frames are `s`
    seconds apart, rather than the 2 ms that smoothing would give.
    """
    df = CPP(y, fs, norm=False, smooth=0, s=s, f0_range=list(f0_range))
    df['amp'] = band_db(y, fs, df['sec'], AMP_BAND)
    return df


def VAD(y, fs, f0_range=[63, 400], s=0.005):
    """Voice activity detection: the probability that each frame of audio is voiced.

    The voicing decision is a logistic regression on two measurements made in each frame: the level of
    the amplitude envelope in a low frequency band (80-800 Hz), and cepstral peak prominence (`cpp`).
    The coefficients were trained against EGG voicing by `examples/train_vad.py`.

    Training results found agreement between VAD() and egg_to_oq() of 92.2% on [-voice]
    decisions and 95.6% on [+voice] decisions.  

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
