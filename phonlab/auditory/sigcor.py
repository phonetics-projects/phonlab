__all__=["sigcor_noise"]

import librosa
import numpy as np
from ..utils.get_signal_ import get_signal

def sigcor_noise(sig, flip_rate = 0.5, start=0, end = -1, fs_in = 22050, fs = 22050, chan=0):
    """Add signal correlated noise to an audio file. 
    
    The function takes a filename and returns a numpy array that contains the signal 
    with added signal correlated noise.  This by done by flipping the polarity of samples 
    randomly. Note that flip_rate of 0 means no change, and 1 means flip the polarity of all of the 
    samples, 0.5 means flip 1/2 of the samples (imagine flipping a coin for each sample, heads 
    leave it as it was, tails multiply it by -1).  So, the maximum "noise" is with flip_rate = 0.5.
        
    Parameters
    ----------
        sig : Path, or ndarray
            Either a string containing the full path to an audio file (wav or mp3), or an array of audio samples.  If **sig** is an array, be sure to specify the sampling rate in fs_in.
        flip_rate : float, 0 <= flip_rate <= 1.0, default = 0.5 
            determines the proportion of samples to flip (0.5 gives maximum noise)    
        start : float, default = 0
            the time (in seconds) at which to start adding noise (default is 0)
        end : float, default = -1
            the time (in seconds) at which to stop adding noise (default is -1, apply to the end of the audio).
        fs_in : int
            if **sig** is an array, fs_in is the sampling frequency of the sound to be manipulated
        fs : int, default = 22050
            is the desired sampling frequency of the output array
        chan : int, default = 0
            if **sig** is a stereo wav file, select a channel (0 = left, 1 = right)
            
    Returns
    -------
        signal : ndarray
            a 1D array that has the altered audio from filename
            
        fs : float
            the sampling rate of the signal (default is 22050)
            
    Notes
    -----
    librosa.load() is used to open the audio file.  It is resampled to a sampling rate of 22050Hz, and 
    converted to mono (if not mono already) by adding the left and right channels.
        
    """
    x,fs = get_signal(sig, fs=fs, fs_in=fs_in, pre=0, chan = chan)
    
    start = int(start*fs)
    end = int(end*fs)
        
    # this buffer randomly has 1 (don't flip) or -1 (flip) for each sample in the signal
    flip_buffer = np.array([1. if q>flip_rate else -1. for q in np.random.rand(x.size)])
    flip_buffer[:start] = [1.]  # don't flip from 0 to start
    if end>0:
        flip_buffer[end:] = [1.]  # don't flip from "end" to signal end

    return x*flip_buffer, fs
    
