import numpy as np
import scipy
from ..utils.prep_audio_ import prep_audio

def amplitude_envelope(x, fs, bounds = [], target_fs=22050, cutoff=30, order=2 ):
    """ Get the amplitude envelope of an audio signal.  
    
    This routine rectifies the audio signal and then low pass filters it.  This is different from calculating an RMS amplitude contour because it gives an amplitude measurement at each point in the audio rather than measuring amplitude in discrete windows.  The filter is scipy.signal.sosfiltfilt()- a very stable filter that introduces no phase shift in the ouput waveform.

    If `bounds` is supplied (eg. `bounds = [100,2600]`) then a bandpass filter with the low and high frequency edges specified in `bounds` will be applied.  

    Parameters
    ==========
    x : ndarray
        A 1-D array of audio samples
    fs : int
        the sampling frequency of the audio in **x** (in Hz)
    bounds : list, default = []
        a list (length 2) with the low and high edges of a bandpass filter.  By default the function calculates the amplitude envelope for the full band from 0Hz to 11025 Hz (or whatever is the Nyquist frequency (1/2 of the target sampling rate).
    target_fs : int, default = 22050
        the desired sampling frequency of the output array
    cutoff : float, default = 30
        cutoff freq. for the low-pass envelope filter.
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
    This example shows the use of `phon.amplitude_envelope` to calculate the amplitude contour in two frequency bands, and then the difference between them. This plot of the difference in the balance of energy between high and low frequency interestingly seems to segment the audio into vocalic and non-vocalic regions. 
    
    .. code-block:: Python

         x,fs = phon.loadsig(example_file,chansel=[0])
         hband, fs = phon.amplitude_envelope(x,fs,bounds=[3000,8000])
         lband, fs = phon.amplitude_envelope(x,fs,bounds=[120,3000])
         diff = lband-hband
         
         time_axis = np.arange(len(diff))/fs
         ax1,f,t,Sxx = phon.sgram(x,fs)  # plot the spectrogram
         ax2 = ax1.twinx()
         ax2.plot(time_axis,diff, color = "red")  # add scaled diff function
         ax2.axhline(0) 

    .. figure:: images/amp_env.png
       :scale: 90 %
       :alt: a spectrogram with a red line marking the balance of energy in low and high frequency bands
       :align: center

       Plotting the difference between the amplitude envelope in the low frequencies [120,3000] versus the amplitude envelope in high frequencies [3000,8000].


    """
    y = np.copy(x)
    y, fs = prep_audio(y,fs, target_fs = target_fs, pre=0, quiet = True)

    if bounds:
        coefs = scipy.signal.butter(8, bounds, fs=fs, btype='bandpass', output='sos')
        y = scipy.signal.sosfiltfilt(coefs, y) 
    
    coefs = scipy.signal.butter(order, cutoff, fs=fs, btype='lowpass', output='sos')
    y = scipy.signal.sosfiltfilt(coefs, np.abs(y))
    return y, fs
