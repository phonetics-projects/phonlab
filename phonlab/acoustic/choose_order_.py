import numpy as np
from librosa import util, lpc
from scipy.signal import lfilter,windows
from ..acoustic.amp_env import amplitude_envelope

def choose_order(x,fs,l=0.03, base = 'BIC', verbose=False):
    '''The most impactful parameter in LPC formant tracking is the 'order',the number of LPC coefficients to use in calculating a fit to the spectrum.  Many authors propose a rule of thumb having to do with the expected number of formants in the analyzed frequency range: namely that the order should 2*nf + 2 (two times the number of formants expected in the frequency range of the analysis [0, fs/2] plus 2). 
    
This function instead tests a number of different values of LPC order and returns the order value that results in the lowest Bayesian Information Criterion (BIC).  The proceedure is to (1) select a 30 ms (by default) frame of audio waveform samples around the time of the peak amplitude in the band-limited (120Hz to 3000Hz) amplitude envelope of `x`, (2) then calculate LPC coeffients and BIC for each of several lpc orders (from fs/1000 - 2 to fs/1000 + 6) for that frame, and then (3) return the best order based on how well the LPC analysis fits the audio spectrum.
    
If `base` is 'BIC', the function returns the lpc order that produces the lowest Bayesian Information Criterion of the LPC residual as a function of the number of LPC coefficients.  

If 'base' is 'coef', the function returns the lpc order that maximizes the first LPC coefficient, as a measure of the LPC filter gain.

Parameters
----------
    x : ndarray
        a one-dimensional numpy arry with audio waveform samples 
    fs : int
        the sampling frequency of *x*
    l : float (default=0.03)
        the duration (in seconds) of the analysis frame (default=0.03)
    base : string (default = "BIC")
        see above.
    verbose : boolean (default = False)
        if True, print messages about the operation of the function
        
Returns
-------
    lpc_order : int
        the best fitting `order` term for LPC analysis
    time : float
        the time (in seconds) of the analysis frame.
    '''

    # ---- 1. bandpass from 120 to 3000
    vowel_band, fs = amplitude_envelope(x,fs,target_fs=fs,bounds=[120,3000])

    # ----- 2. take a frame of data at the peak in the bandpassed signal
    i = np.argmax(vowel_band) # find the location of the peak in the bandpassed signal
    time_of_frame = i/fs
    frame_length = int(fs*l)
    half_frame = frame_length//2
    myframe = x[i-half_frame:i+half_frame]

    # ---- 3. find the best LPC order for this frame
    os = (fs//1000) - 2  # starting possible order
    if os<=6: os = 8
    oe = (fs//1000) + 7  # ending possible order
    BICmin = 1e10
    Amax = -1e10 
    for o in range(os,oe,2):
        A = lpc(myframe, order=o)  # Calculate lpc coefficients
        resid = lfilter(A,[1.0],myframe) # get residual signal
        sigma2 = np.mean(abs(resid)**2) #  error variance 
        n= frame_length-o
        BIC = (n*np.log(sigma2)) + (o * np.log(n))
        if BIC < BICmin:  
            BICmin = BIC
            BICorder = o
        A1 = A[1]
        if A1 > Amax:
            Amax = A1
            Aorder = o    
        if verbose:
            print(f"{o}: BIC = {BIC:.1f}, A1 = {A1:.3f}, BIC order={BICorder}, coef order={Aorder}")
    if (base == "coef"):
        order = Aorder
    else: 
        order = BICorder
    return order, time_of_frame
