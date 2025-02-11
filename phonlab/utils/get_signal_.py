__all__=['get_signal']

import numpy as np
import librosa
from scipy.signal import resample


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
            x, fs = librosa.load(sig, sr=fs,dtype=np.float32)  # read waveform
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
