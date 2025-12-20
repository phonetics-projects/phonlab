import numpy as np
from ..utils.prep_audio_ import prep_audio

def sigcor_noise(x,fs, flip_rate = 0.5, start=0, end = -1):
    """Add signal correlated noise to an audio file. 
    
    The function takes a filename and returns a numpy array that contains the signal 
    with added signal correlated noise.  This by done by flipping the polarity of samples 
    randomly. Note that flip_rate of 0 means no change, and 1 means flip the polarity of all of the 
    samples, 0.5 means randomly flip the polarity of 1/2 of the samples (imagine flipping a coin for 
    each sample, heads leave it as it was, tails multiply it by -1).  So, the maximum "noise" is 
    with flip_rate = 0.5.
        
    Parameters
    ----------
        x : ndarray
            An one-dimensional array of audio samples. 
        fs : int
            the sampling frequency of the audio samples in **x**
        flip_rate : float, 0 <= flip_rate <= 1.0, default = 0.5 
            determines the proportion of samples to flip (0.5 gives maximum noise)    
        start : float, default = 0
            the time (in seconds) at which to start adding noise (default is 0)
        end : float, default = -1
            the time (in seconds) at which to stop adding noise (default is -1, apply to the end of the audio).
           
    Returns
    -------
        y : ndarray
            A one-dimensional array derived from **x**
            
        fs : float
            the sampling rate of **y**


    Example
    -------
    Open a file and add signal correlated noise to the section between 1.2 and 1.5 seconds.

    .. code-block:: Python
    
         x,fs = phon.loadsig("sf3_cln.wav",chansel=[0]) 
         y,fs = phon.sigcor_noise(x,fs,flip_rate=0.4,start=1.2,end=1.5)
    """    
    start = int(start*fs)
    end = int(end*fs)
        
    # this buffer randomly has 1 (don't flip) or -1 (flip) for each sample in the signal
    flip_buffer = np.array([1. if q>flip_rate else -1. for q in np.random.rand(x.size)])
    flip_buffer[:start] = [1.]  # don't flip from 0 to start
    if end>0:
        flip_buffer[end:] = [1.]  # don't flip from "end" to signal end

    return x*flip_buffer, fs
    
