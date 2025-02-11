__all__=['amplitude_envelope']

import numpy as np
import scipy
from ..utils.get_signal_ import get_signal

def amplitude_envelope(sig, bounds = [], fs_in=22050, fs=22050, chan = 0, cutoff=30, order=2 ):
    """ Get the amplitude envelope of an audio signal.  
    
    This routine rectifies the audio signal and then low pass filters it.  This is different from calculating an RMS amplitude contour because it gives an amplitude measurement at each point in the audio rather than measuring amplitude in discrete windows.  The filter is scipy.signal.sosfiltfilt()- a very stable filter that introduces no phase shift in the ouput waveform.

    If `bounds` is supplied (eg. `bounds = [100,2600]`) then a bandpass filter with the low and high frequency edges specified in `bounds` will be applied.  

    Parameters
    ==========
    sig : path or ndarray
        The name of a wav file, or audio samples in a 1-D array
    bounds : list, default = []
        a list (length 2) with the low and high edges of a bandpass filter
    fs_in : int, default = 22050
        if sig is an array, fs_in is the sampling frequency of the sound to be filtered
    fs : int, default = 22050
        is the desired sampling frequency of the output array
    chan : int, default = 0
        if sig is stereo, select a channel (0 = left, 1 = right)
    cutoff : float, default = 30
        cutoff freq. for the low-pass envelope filter
    order : int, default = 2
        order of the low-pass envelope filter
    
    Returns
    =======
    y : ndarray
        an array with the amplitude envelope, the array has the same number of samples as x
    fs : int
        the sampling frequency of the audio in y

    Example
    =======

    >>> hband, fs = phon.amplitude_envelope("sf3_cln.wav",bounds=[3000,8000])
    >>> lband, fs = phon.amplitude_envelope("sf3_cln.wav",bounds=[120,3000])
    >>> diff = lband-hband
    >>> time_axis = np.arange(len(diff))/fs
    >>> ax1,f,t,Sxx = phon.sgram("sf3_cln.wav")  # plot the spectrogram
    >>> ax2 = ax1.twinx()
    >>> ax2.plot(time_axis,diff, color = "red")  # add scaled diff function
    >>> ax2.axhline(0) 

    .. figure:: images/amp_env.png
       :scale: 90 %
       :alt: a spectrogram with a red line marking the balance of energy in low and high frequency bands
       :align: center

       Plotting the difference between the amplitude envelope in the low frequencies [120,3000] versus the amplitude envelope in high frequencies [3000,8000].

       ..


    """
    
    x, fs = get_signal(sig,chan = chan, fs = fs, fs_in = fs_in, pre=0, quiet = True)

    if bounds:
        coefs = scipy.signal.butter(8, bounds, fs=fs, btype='bandpass', output='sos')
        x = scipy.signal.sosfiltfilt(coefs, x) 
    
    coefs = scipy.signal.butter(order, cutoff, fs=fs, btype='lowpass', output='sos')
    y = scipy.signal.sosfiltfilt(coefs, np.abs(x))
    return y, fs