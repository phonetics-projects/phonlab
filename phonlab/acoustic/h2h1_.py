import numpy as np
from scipy.signal import iirpeak,filtfilt,windows, convolve, find_peaks
from scipy import fft
from ..utils.prep_audio_ import prep_audio

def h2h1(resid,fs,f0df,f0med=None,use_ac=True,length=None):
    '''Estimate the amplitude ratio of harmonic 2 and harmonic 1, in decibels.  This function 
uses the method described by Drugman, Kane & Gobl (2012), and was inspired by the matlab
function `get_creak_h2h1()` published in the Covarep Matlab repository of speech analysis software.  
The h2h1 measurements given by this function are correlated with those produced by `phonlab.get_f0_acd()`
but the measurements from `h2h1()` should be used when testing for creaky voice.  

The method here is to:
    * apply a narrow (150Hz) filter centered on the f0med, to the LPC residual (see `phonlab.lpcresidual()`) 
    * produce an autocorrelation of each frame
    * produce a spectrum of the autocorrelation function
    * based on the measured F0 for the frame, pick the spectral peaks closest to F0 and 2*F0
    * and report the amplitude difference between them (the H2-H1 amplitude difference).

Drugman et al. (2012) reported that H2H1 > 0dB is a reliable indicator of creaky voice.


Parameters
==========
    resid : ndarray
        A one-dimensional array of lpc residual samples, as produced by `phon.lpcresidual()`
    fs : int
        the sampling rate of the audio in **resid**.
    f0df : Dataframe
        A Pandas dataframe produced by one of the get_f0 functions 
    f0med : float, default = None
        Normally the median f0 is computed from the f0 values in f0df with the 
        line: `f0med = np.nanmedian(f0df.f0)`, but this parameter lets the user supply an 
        estimate of the median f0 explicitly.
    use_ac : boolean, default = True
        Following Drugman et al. (2012), the function by default finds H1 and H2 in the spectrum 
        of the autocorrelation function.
    length : float, default = None
        The length (in seconds) of the analysis window, which is centered on the time `sec` of each row of `f0df`.
        By default it is twice the first `sec` in `f0df`, which is the window length of the f0 function when `f0df`
        starts with its first frame (e.g. 0.05 for `get_f0_acd()`).  If you give `h2h1()` a dataframe that does not start at the first frame (for
        example one that has been filtered to the voiced frames), pass the window length of the f0 analysis here.
        
Returns
=======
    df : Dataframe
        The input dataframe `f0df` is returned with a new column `h2h1`.  There is one measurement for each row
        of `f0df`, made in a window centered on its `sec`.  The rows can be in any order and need not be evenly spaced, and
        the index of `f0df` can be anything, so the dataframe can be filtered before it is passed in.  The h2h1 of a row is NaN if no
        spectral peak is found.

References
==========
    T. Drugman, J. Kane, C. Gobl (2012) Resonator-based Creaky Voice Detection. INTERSPEECH 2012, ISCA's 13th Annual Conference Portland, OR, USA, September 9-13, 2012

Example
=======
    .. code-block:: Python
    
        example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'
        x,fs = phon.loadsig(example_file,chansel=[0])

        x,fs = phon.prep_audio(x, fs, target_fs=16000, preemph = 0)  # resample, scale
        resid,fs = phon.lpcresidual(x,fs)
        f0df = phon.get_f0_acd(x,fs)

        df = phon.h2h1(resid,fs,f0df)  # <- using information from lpcresidual(), and get_f0_acd()

        # ---- Now plot the results -------
        ret = phon.sgram(x,fs,cmap="Grays") # draw a spectrogram of the sound
        ax2 = ret[0].twinx() 
        ax2.plot(df.sec,df.h2h1,'bd')
        ax2.axhline(0,color="red")  # Drugman et al. use 0dB as the threshold for creaky voice

    .. figure:: images/h2h1.png
        :scale: 33 %
        :alt: a spectrogram with a trace of the h2h1 ratio calculated by phon.h2h1()
        :align: center

        A trace of the h2h1 ratio as calculated from the LPC residual signal by phon.h2h1().  The red
        line at 0 dB marks a proposed threshold for classifying vocal fold vibration as creaky versus modal.

        
    '''
    x,fs = prep_audio(resid, fs, target_fs = 16000, 
                      scale = True, verbose=False)

    # The analysis window is centered on the time of each row of the f0 dataframe, so no assumption is made
    # about the step between rows, or about the first row being the first frame of the f0 analysis.
    sec = f0df['sec'].to_numpy(dtype=float)
    if np.isnan(sec).any():
        raise ValueError("h2h1(): the `sec` column of f0df has missing values")
    if length is None:
        # The first frame of the f0 analysis is centered half a window from the start of the audio.
        if not 0.005 <= sec[0] <= 0.1:
            raise ValueError(f"h2h1(): the window length cannot be found from the first time in f0df ({sec[0]} s), "
                             "which is not the center of the first f0 analysis frame. Give the `length` of the f0 analysis window.")
        half_frame = int(round(sec[0] * fs))
    elif length <= 0:
        raise ValueError(f"h2h1(): length must be positive, not {length}")
    else:
        half_frame = round(int(fs * length)/2)
    frame_length = half_frame * 2 + 1    # odd number in frame  
    centers = np.round(sec * fs).astype(int)   # sample at the center of each frame

    if f0med==None:
        f0med = np.nanmedian(f0df.f0)

    bandwidth = 150  # narrow band around the median f0
    b,a = iirpeak(f0med,f0med/bandwidth,fs=fs)
    res_n = filtfilt(1,a,x)  # narrowband filtered residual (x is `resid` at 16000 Hz, the rate `a` was designed for)
    res_n = res_n/np.max(res_n)

    # one window of frame_length samples centered on each time in f0df (zeros where the window is outside the audio)
    padded = np.pad(res_n, half_frame)
    frames = padded[np.clip(centers[:,None] + np.arange(frame_length)[None,:], 0, len(padded)-1)]
    F0 = np.nan_to_num(f0df.f0.to_numpy(dtype=float),nan=f0med).astype(int)  # f0 values passed into the function

    # ------- autocorrelations of all of the frames in the file -----------
    if use_ac:    
        N = 1024
        while (frame_length+frame_length//2 > N): 
            N = N * 2  # increase fft size if needed

        Sxx = fft.fft(windows.hann(frame_length)*frames,N)
        ra = fft.fft(np.square(np.abs(Sxx)),N)  # matrix of autocorrelations
        ra = np.divide(ra.T,np.max(ra,axis= -1)).T  # frame by frame normalization

        spec = fft.fft(windows.blackmanharris(N)*ra,fs)  # spectrum of the autocorrelation
    else:
        spec = fft.fft(windows.hann(frame_length)*frames,fs) # spectrum of the filtered residual
        
    spec = np.abs(spec[:,0:int(fs/2)-1])
    spec = np.divide(spec.T,np.sqrt(np.sum(spec**2,axis=-1))).T
    spec = 20 * np.log10(np.maximum(spec, np.finfo(spec.dtype).tiny))
    h2h1 = np.full(F0.shape[0], np.nan)
    #h2h1 = np.diagonal(spec[:,F0*2]) - np.diagonal(spec[:,F0])  
    for n in range(F0.shape[0]):
        peaks,props = find_peaks(spec[n,0:1000],distance = max(F0[n]/2, 1))
        if len(peaks) == 0:   # e.g. a window with no audio in it
            continue
        peak1 = int(np.argmin(np.fabs(peaks-F0[n])))
        peak2 = int(np.argmin(np.fabs(peaks-(F0[n]*2))))
        h2h1[n] = spec[n,peaks[peak2]] - spec[n,peaks[peak1]]

    df = f0df.copy()
    df['h2h1'] = h2h1
    return df
