import librosa
import numpy as np

def loadsig(path, chansel=[], offset=0.0, duration=None, fs=None, dtype=np.float32):
    """
    Load signal(s) from an audio file.

    By default audio samples are returned at the same sample rate as the input file.
    and channels
    are returned along the first dimension of the output array `y`.

Parameters
==========

    path : string, int, pathlib.Path, soundfile.SoundFile, or file-like object
        The input audio file.

    chansel : int, list of int (default [])
        Selection of channels to be returned from the input audio file, starting with 0 for the first channel. For empty list [], return all channels in order as they appear in the input audio file. This parameter can be used to select channels out of order, drop channels, and repeat channels.

    offset : float (default 0.0)
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    fs : number > 0 [scalar]
        target sampling rate. 'None' returns `y` at the file's native sampling rate.

    dtype : numeric type (default float32)
        data type of **y**. No scaling is performed when the requested dtype differs from the native dtype of the file. Float types are usually in the range [-1.0, 1.0), and integer types usually make use of the full range of integers available to their size, e.g. int16 may be in the range [-32768, 32767].

Returns
=======

    ys : list of 1d signal arrays `y` (plus `fs`)
        Each channel is returned as a separate 1d array in the output list. The number of arrays is equal to the number of channels in the input file by default. If **chansel** is specified, then the number of 1d arrays is equal to the length of **chansel**. Technically, the last value of the list is `fs`, see below.

    fs : number > 0 [scalar]
        sampling rate of the **y** arrays

Example
=======
Load a stereo audio file, report the sampling rate of the file, and plot the left channel.  Note, this will produce an error with a one channel file. **left** and **right** are one-dimensional arrays of audio samples.

.. code-block:: Python

    left, right, fs = loadsig('stereo.wav', chansel=[0,1])
    print(fs)
    plt.plot(left);
    
To load a one channel (mono) file, you can do this:

.. code-block:: Python

    x,fs = loadsig('mono.wav',chansel=[0])
    print(fs,len(x))
    plt.plot(x)

In this example we load channels from a wav file that has an unknown number of channels, downsampling to 12 kHz sampling rate. Use **len(chans)** to determine how many channels there are in the file, and plot the last channel. **chans** is a list of 1d audio signal arrays. You can `pop` the sample rate parameter off the list.

.. code-block:: Python

    chans = loadsig('threechan.wav', fs=12000)
    fs = chans.pop()     # Remove sample rate from end of the list of channels
    print(len(chans))      # the number of channels
    plt.plot(chans[-1])    # plot the last of the channels

    
    """

    y, fs = librosa.load(
        path, sr=fs, mono=False, offset=offset, duration=duration, dtype=dtype
    )
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    if chansel == []:
        chansel = np.arange(y.shape[0], dtype=np.int16)
    return [ *list(y[chansel, :]), fs ]


def channels_are_duplicates(chan_a, chan_b, atol=1e-6, rtol=1e-5, corr_thresh=0.999, rel_diff_thresh=0.01):
    """
    Determine whether two audio channels carry effectively the same signal.

    A mono recording saved as "stereo" is usually bit-identical across
    channels (no lossy step involved), so an exact/near-exact check via
    np.allclose catches that case cheaply. Channels that passed through a
    lossy codec (e.g. mid-side-coded MP3/AAC) won't be bit-exact even when
    they originated from the same mono source, so we fall back to a
    correlation + relative-RMS-difference check for near-duplicates.

Parameters
==========

    chan_a, chan_b : ndarray
        two arrays of audio samples
    
    atol: float (default = 1e-6)
        absolute difference tolerance passed to np.allclose()

    rtol: float (default = 1e-5)
        relative difference tolerance passed to np.allclose()

    corr_thresh: float (default=0.999)
        fall back correlation threshold for near-duplicates

    rel_diff_thresh: float (default=0.01)
        fall back relative RMS difference threshold for near-duplicates

Returns
=======

    boolean, True if chan_a and chan_b are duplicates of each other.

    
    """
    if chan_a.shape != chan_b.shape:
        return False
    if np.allclose(chan_a, chan_b, rtol=rtol, atol=atol):
        return True
    if np.std(chan_a) == 0 or np.std(chan_b) == 0:
        return np.allclose(chan_a, chan_b, atol=atol)
    corr = np.corrcoef(chan_a, chan_b)[0, 1]
    diff_rms = np.sqrt(np.mean((chan_a - chan_b) ** 2))
    sig_rms = np.sqrt(np.mean(chan_a ** 2))
    return corr >= corr_thresh and diff_rms / sig_rms < rel_diff_thresh
