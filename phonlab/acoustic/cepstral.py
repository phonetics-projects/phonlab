from ..utils.prep_audio_ import prep_audio
import numpy as np
from librosa import util
from scipy import fft
from pandas import DataFrame
from scipy.signal import windows,filtfilt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter

def compute_cepstrogram(x,fs, dBscale=True, l= 0.04, s=0.005):
    '''Compute a `cepstrogram` of an audio signal.  Cepstral analysis was introduced by Bogert et al. (1963).  
Sounds (such as voiced vowels) that have a harmonic structure, with peaks of amplitude at integral multiples of
the fundamental frequency will have a strong peak in the cepstrum at the period of the fundamental.  The cepstrum
is computed as the spectrum of the spectrum to discover the harmonic structure of the spectrum.  

Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    dBscale : boolean, default = True
        Scale the cepstrum in dB
    l : float, default = 0.04
        Length of analysis windows.  The default is 40 milliseconds.
    s : float, default = 0.005
        Step size, of hops between analysis windows. The default is 5 milliseconds.  

Returns
=======
    quef : ndarray
        A one dimensional numpy array with Quefrency values (in ms)
    sec : ndarray
        A one dimensional numpy array with frame time points (in seconds)
    Sxx : ndarray
        A two dimensional numpy array with cepstral magnitudes at these timepoints and quefrencies.


References
==========
    B. P. Bogert, M. J. R. Healy, and J. W. Tukey, (1963) `The Quefrency Alanysis [sic] of Time Series for Echoes: Cepstrum, Pseudo Autocovariance, Cross-Cepstrum and Saphe Cracking, `Proceedings of the Symposium on Time Series Analysis` (M. Rosenblatt, Ed) Chapter 15, 209-243. New York: Wiley.

    '''
    frame_length = int(l*fs)
    step = int(s*fs)
    half_frame = round(frame_length/2)
    frame_length = half_frame * 2 + 1    # odd number in frame
    NFFT = int(2**(np.ceil(np.log(frame_length)/np.log(2))))
    quef = (np.array(range(NFFT//2))/fs *1000)  # the quefrecy axis of the cepstra (in ms)
    frames = util.frame(x,frame_length=frame_length, hop_length=step,axis=0)

    nb = frames.shape[0]
    f = frames.shape[1]
    w = windows.hann(frame_length)
    
    Sxx = 10 * np.log10(np.abs(fft.rfft(w*frames,NFFT))**2)
    Sxx2 = np.abs(fft.rfft(Sxx,NFFT))**2   # spectrum of the spectrum -- cepstrum
    if (dBscale):
        Ceps = 10 * np.log10(Sxx2[:,:-1])
    else:
        Ceps = np.log(Sxx2[:,:-1])
        
    ts = (np.array(range(nb)) * step + half_frame)/fs

    return(quef, ts, Ceps)

def CPP(x,fs, target_fs = 16000, smooth=2, norm=True, dBscale=True, f0_range = [60,400], l= 0.04, s=0.005):
    '''Measure Cepstral Peak Prominence - an acoustic measure that has been shown to be highly correlated with 
perceived breathy vocal quality (Hillenbrand & Houde, 1996).  This implementation drew inspiration and ideas from 
John Kane's cpp() matlab function in the `covarep` repository.

Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    target_fs : int, default = 16000
        Sampling rate for the analysis algorithm.
    smooth : float, default = 2
        The sigma value of a Gaussian filter that smooths the cepstrogram in both frequency and time
    norm : boolean, default = True
        Flag to request that the cepstral peak prominence be normalized for overall amplitude, using a linear fit 
        to the cepstrum and measuring the height of the peak above the fit line.
    dBscale : boolean, default = True
        Scale the cepstrum in dB
    f0_range : an array of two numbers, default=[50,500]
        The expected range for f0.
    l : float, default = 0.04
        Length of analysis windows.  The default is 40 milliseconds.
    s : float, default = 0.005
        Step size, of hops between analysis windows. The default is 5 milliseconds.  If smoothing==True, the step size
        is reduced to 0.002 (2 millisecond).
    return_Sxx : boolean, default = False
        flag to request to return the cepstrogram (produced by `compute_cepstrogram()`) instead of returning an analysis 
        dataframe.  If smoothing is requested the smoothed cepstrogram is returned.  The return values (instead of the 
        dataframe described below) are then:
        
            * quef - A one dimensional numpy array with Quefrency values (in ms)
            * sec - A one dimensional numpy array with frame time points (in seconds)
            * Sxx - A two dimensional numpy array with cepstral magnitudes at these timepoints and quefrencies.

Returns
=======
    df: pandas DataFrame  
            measurements at 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame in seconds
    * f0 - estimate of the fundamental frequency in Hz
    * cpp - cepstral peak prominence in dB

Note
====
The default parameter values are an implementation of Hillenbrand and Houde (1996), for the measurement of breathy voice.  
Lower values of cepstral peak prominence are associated with breathy phonation.  Following H&H, the algorithm 
uses smoothing both across time and quefrency to improve the identification of the cepstral peak that corresponds to the 
period of the voicing fundamental.


References
==========
    J. Hillenbrand, R.A. Houde (1996) Acoustic Correlates of Breathy Vocal Quality: Dysphonic Voices and Continuous Speech `Journal of Speech and Hearing Science Research`, 39, 311-321.

    B. P. Bogert, M. J. R. Healy, and J. W. Tukey, (1963) The Quefrency Alanysis [sic] of Time Series for Echoes: Cepstrum, Pseudo Autocovariance, Cross-Cepstrum and Saphe Cracking, `Proceedings of the Symposium on Time Series Analysis` (M. Rosenblatt, Ed) Chapter 15, 209-243. New York: Wiley.

Example
=======
This example plots the cepstral peak prominence through the "I'm twelve" example recording.  Note the increase in breathiness (decrease in cpp in each of the first two words "I'm" and "twelve".

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/im_twelve.wav'
    x,fs = phon.loadsig(example_file,chansel=[0])

    cppdf   = phon.CPP(x,fs)
    cpp,s,_ = phon.smoothn(cppdf.cpp,s=35,isrobust=True)  # smooth the cpp curve

    ret = phon.sgram(x,fs,cmap="Blues") # draw the spectrogram from the array of samples
    ax2 = ret[0].twinx()
    ax2.plot(cppdf.sec, cpp, 'r-')
    plt.ylabel("Cepstral Peak Prominence (dB)")

.. figure:: images/cpp.png
    :scale: 40 %
    :alt: Spectrogram, with cepstral peak prominence curve overlaid.
    :align: center  

    '''
    y,fs = prep_audio(x, fs, target_fs=target_fs, pre=0.0, quiet=True)  # resample to 16kHz

    if smooth: s = 0.002  # faster framerate if smoothed

    quef,sec,Sxx = compute_cepstrogram(y, fs, dBscale, l, s)
    
    if smooth:
        Sxx = gaussian_filter(Sxx,sigma = smooth,truncate=3)
        
    Sxx = np.nan_to_num(Sxx) # replaces NaN with 0

    sT = int(np.round(fs/f0_range[1]))  # the shortest expected pitch period
    lT = int(np.round(fs/f0_range[0])) # the longest expected pitch period
    
    cp = np.argmax(Sxx[:,sT:lT],axis=-1) + sT
    cpp = np.max(Sxx[:,sT:lT],axis=-1)
    f0 = 1/(cp/fs)

    if norm:
        # hard coding here the range for the linear regression CPP normalization
        sT = int(np.round(fs/500)) # was [300,60] in Hillenbrand & Houde
        lT = int(np.round(fs/50)) # was [300,60] in Hillenbrand & Houde
        X = np.array(np.arange(sT,lT,1))  # line fitting in the f0 region
        X = np.reshape(X,(len(X),1))
        reg =LinearRegression().fit(X,y=Sxx[:,sT:lT].T)  # vectorized regressions
        p = np.diag(reg.predict(np.reshape(cp,(-1,1))))
        cpp = cpp-p
            
    return DataFrame({'sec': sec, 'f0':f0, 'cpp':cpp})
