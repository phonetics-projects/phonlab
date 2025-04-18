__all__=['loadsig','get_signal']

import numpy as np
import librosa
from scipy.signal import resample

def loadsig(path, chansel=[], offset=0.0, duration=None, rate=None, dtype=np.float32):
    '''
    Load signal(s) from an audio file via `librosa.load`.
    
    This function does *not* follow librosa's default practices of (1) resampling
    to a standard sample rate; or (2) conversion of multichannel audio to mono.
    By default audio samples are returned at the same rate as the input file, and channels
    are returned along the first dimension of the output array `y`.

    Parameters
    ----------

    path : string, int, pathlib.Path, soundfile.SoundFile, audioread object, or file-like object
    The input audio file.

    chansel : int, list of int (default [])
    Selection of channels to be returned from the input audio file, starting
    with `0` for the first channel. For empty list `[]`, return all channels
    in order as they appear in the input audio file. This parameter can be used to
    select channels out of order, drop channels, and repeat channels.

    offset : float (default 0.0)
    start reading after this time (in seconds)

    duration : float
    only load up to this much audio (in seconds)

    rate : number > 0 [scalar]
    target sampling rate. 'None' returns `y` at the file's native sampling rate.

    dtype : numeric type (default float32)
    data type of `y`. No scaling is performed when the requested dtype differs from
    the native dtype of the file. Float types are usually scaled to the range `[-1.0, 1.0)`,
    and integer types potentially make use of the full range of integers available to
    their size, e.g. `int16` may be in the range `[-32768, 32767]`.

    Returns
    -------

    y : np.ndarray [shape=(n,) or (..., n)]
    audio time series. Multichannel is supported.

    rate : number > 0 [scalar]
    sampling rate of `y` 
    '''
    y, rate = librosa.load(
        path, sr=rate, mono=False, offset=offset, duration=duration, dtype=dtype
    )
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)
    if chansel == []:
        chansel = np.arange(y.shape[0], dtype=np.int16)
    return *list(y[chansel, :]), rate


def get_signal(sig, fs = 22050, fs_in=22050, chan = 0, pre = 0, outtype = "float", quiet = False):
    """ A utility function to prepare an audio waveform for acoustic analysis.  Calls Librosa.load() if a file name is passed in `signal`, and conditions the audio array according to the paramters.  If `signal` is an array of samples, you should pass in the sampling rate of the audio in `fs_in`.
    
Parameters
==========
    sig : string or array
        Either a path to a sound file to read, or a numpy array with audio samples in it. If the audio (either from the soundfile or in the array) is stereo, only one channel will be kept.

    fs : int, default = 22050
        The desired sampling rate of the audio samples that will be returned by the function.  Set fs=None if you are opening a sound file and want to use the native sampling rate of the file.
     
    fs_in : int, default=22050
        The sampling rate of the sound if **sig** is an array.  This parameter is ignored if **sig** is a filename.

    chan : int, default = 0
        which channel of a stereo file to keep - default is 0 (the left channel)

    preemphasis : float, default = 0
        how much high frequency preemphasis to apply (between 0 and 1).

    outtype : string {"float", "int"), default = "float"
        In some cases (like for IFC formant tracking) we want the audio in integer format. 


Returns
=======
    y : ndarray
        a 1D numpy array with audio samples 
    
    fs : int
        the sampling rate of the audio in **y** - should match parameter **fs**

Raises
======
    OSError 
        if the sound file can't be opened

Example
=======
Open a sound file and use the existing (native) sampling rate of the file.

>>> x,fs = phon.get_signal("sound.wav", pre=1,fs=None)

    """
    
    if type(sig) == np.ndarray:
        x = sig
        if fs==None: 
            if not quiet: print('fs is being set to 22050, was None')
            fs=22050  # None 
        if (fs_in != fs):  # resample to 'fs' samples per second
            if not quiet: print(f'Resampling from {fs_in} to {fs}')
            resample_ratio = fs/fs_in
            new_size = int(len(x) * resample_ratio)  # size of the downsampled version
            x = resample(x,new_size)  # now sampled at desired sampling freq
    else:  # sig is a file name
        try:
            x, fs = librosa.load(sig, sr=fs, mono=False, dtype=np.float32)  # read waveform
        except OSError:
            print('cannot open sound file: ', sig)
    
    if len(x.shape) == 2:  # if this is a stereo file, use one of the channels
            if not quiet: print(f'Stereo file, using channel {chan}')
            x = x[:,chan]
    if (np.max(x) + np.min(x)) < 0:  x = -x   #  set the polarity of the signal
    if outtype == "int":  
        x = np.rint(32000 * (x/np.max(x))).astype(np.float64)  
    if (pre > 0): y = np.append(x[0], x[1:] - pre * x[:-1])  # apply pre-emphasis
    else: y = x
    
    return y,fs
