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

