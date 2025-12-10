import numpy as np
from scipy.signal import windows, find_peaks, spectrogram, peak_prominences, fftconvolve
from scipy import fft
from librosa import feature, util, lpc
from pandas import DataFrame
from scipy import linalg

from ..utils.prep_audio_ import prep_audio
from ..acoustic.lpc_residual import lpcresidual

np.seterr(divide="ignore")

def get_rms(y, fs, s=0.005, l=0.04, scale=False):
    """Measure the time-varying root mean square (RMS) amplitude of the signal in **y**.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        scale : boolean, default = False
            optionally scale the rms amplitude to maximum peak.

    Returns
    =======
        df: pandas DataFrame
            There are two columns in the returned frame - sec, rms.

    """

    # constants and global variables
    frame_length = int(fs * l) 
    half_frame = frame_length//2
    frame_length = half_frame * 2 + 1    # odd number in frame  
    step = int(fs * s)  # number of samples between frames

    rms = feature.rms(y=y,frame_length=frame_length, hop_length=step,center=False)[0]
    if scale:
        rms = 20*np.log10(rms/np.max(rms),where=np.where(rms>0,True,False,out=np.zeros(rms.shape)))
    else:
        rms = 20*np.log10(rms,where=np.where(rms>0,True,False,out=np.zeros(rms.shape)))

    nb = rms.shape[0]  # the number of frames
    sec = (np.array(range(nb)) * step + half_frame).astype(int)/fs

    return DataFrame({'sec': sec, 'rms':rms})


def get_f0(y, fs, f0_range = [63,400], s= 0.005):
    """Track the fundamental frequency of voicing (f0), using autocorrelation

    This function implements the autocorrelation method described in Boersma (1993). where 
    the autocorrelation function is normalized by the autocorrelation of the 
    analysis window. The raw best fit frame-by-frame values are returned -- that is, this 
    function does not follow Boersma (1993) in using the Viterbi algorithm to choose the 
    optimal path among f0 candidates.

    The Harmonics-to-Noise ratio (HNR) in each frame is estimated from the peak of the 
    autocorreation function (c) as `10 * log10(c/(1-c))`.

    Probability of voicing is given from a logistic regression formula using `rms` and `c` 
    to predict the voicing state as determined by EGG data using the function 
    `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. 
    The prediction of the EGG voicing decision was about 86% correct.


    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.
        s : float, default = 0.005
            "Hop" interval between successive analysis windows. The default is 5 milliseconds

    Returns
    =======
        df: pandas DataFrame  
            measurements at 5 msec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - peak normalized rms amplitude in the band from 0 to fs/2
        * c - value of the peak autocorrelation found in the frame
        * HNR - an estimate of the harmonics to noise ratio
        * probv - probability of being voiced
        * voiced - a boolean, true if probv > 0.5

    References
    ==========
    P. Boersma (1993) Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. `Institute of Phonetic Sciences, Amsterdam University, Proceedings`. **17**, 97-110.


    .. figure:: images/get_f0_B93.png
        :scale: 33 %
        :alt: a spectrogram with three pitch traces compared - get_f0_acd, get_f0_B93, praat
        :align: center

        Comparing the f0 found by `phon.get_f0()` plotted in blue, and the f0 values found by `parselmouth` `to_Pitch()`, plotted with chartreuse dots, and the f0 values found by get_f0_acd, plotted in orange.  The traces are offset from each other by 10Hz so they can be seen.

    """
    x, fs = prep_audio(y, fs, target_fs = None, pre = 0.0, quiet=True)  
    
    # ---- setup constants and global variables -----
    step = int(fs * s)  # number of samples between frames
    s_lag = int((1/f0_range[1])*fs) # shortest lag
    l_lag = int((1/f0_range[0])*fs) # longest lag
    frame_length = int(l_lag * 3)  # room for 3 periods (6 for HNR)
    half_frame = frame_length//2
    frame_length = half_frame * 2 + 1    # odd number in frame  
    N = 1024
    while (frame_length+frame_length//2 > N): N = N * 2  # increase fft size if needed

    # ----- split into frames --------
    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    nb = frames.shape[0]
    f = frames.shape[1]  # number of frequency steps
    
    # ----- Hann window and its autocorrelation ----------
    w = windows.hann(frame_length)
    s = fft.fft(w,N) + 0.000001
    rw = fft.fft(np.square(np.abs(s)),N)
    rw = rw/np.max(rw)

    # ------- autocorrelations of all of the frames in the file -----------
    Sxx = fft.fft(w*frames,N)+0.000001
    ra = fft.fft(np.square(np.abs(Sxx)),N)
    ra = np.divide(ra.T,np.max(ra,axis= -1)).T  # frame by frame normalization

    # ------ normalized autocorrelations ------------
    rx = ra/rw

    # ------ find best lag in each frame -------
    lag = np.array([s_lag + np.argmax(rx[i,s_lag:l_lag]) for i in range(nb)])

    # ---- compute columns for Dataframe -------
    sec = (np.array(range(nb)) * step + half_frame)/fs
    f0 = 1/(lag/fs)  # convert lags into f0
    rms = 20 * np.log10(np.sqrt(np.sum(np.square(np.abs(Sxx)),axis=-1))) 
    c = np.array([np.abs(np.max(rx[i,s_lag:l_lag])) for i in range(nb)])
    c = np.where(c>=1,0.999,c)
    HNR = 10 * np.log10(c/(1-c),where=np.where(c<1,True,False),out=np.zeros(c.shape))

    # voicing decision
    odds = np.exp(-2.25 + (0.26*rms) + (3.39*c))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    Voiced = probv > 0.5
    #Voiced = ((rms - np.mean(rms)) + HNR) > 4

    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'c':c, 'HNR': HNR, 'probv': probv, 'voiced': Voiced})


def SRH(Sxx,fs,f0_range):  
    ''' test all of the f0 values (integers) between the min and max of the range
    and choose the one with the greatest sum of residual harmonics in each frame of
    the spectrogram.
    '''

    nb = Sxx.shape[0]
    T = Sxx.shape[1]/fs  # interval between frequency steps
    max_harmonic = 7
    f0 = np.empty(nb)
    SRHval = np.empty((nb))
    plus = np.empty((f0_range[1]-f0_range[0],max_harmonic-1),dtype=np.int16)
    minus = np.empty((f0_range[1]-f0_range[0],max_harmonic-1),dtype=np.int16)
    
    # build matrix of frequency indices for each harmonic of each possible f0
    for f in range(f0_range[0],f0_range[1]):
        fT = f*T
        for k in range(1,max_harmonic):
            plus[f-f0_range[0],k-1] = int(fT*k)  # indeces in the spectrum
            minus[f-f0_range[0],k-1] = int(fT*(k+0.5))

    for n in range(nb): 
        S = Sxx[n]
        srh = np.sum(S[plus],axis=-1) - np.sum(S[minus],axis=-1)
        max_srh = np.max(srh)
        idx_srh = np.argmax(srh)
        f0[n] = f0_range[0]+idx_srh
        SRHval[n] = max_srh
    return f0,SRHval


def get_f0_srh(y, fs, f0_range = [60,400], isResidual = False, l = 0.06, s=0.005, vthresh=0.07):
    """Track the fundamental frequency of voicing (f0), using a frequency domain method.

This function is an implementation of Drugman and Alwan's (2011) "Summation of 
Residual Harmonics" (SRH) method of pitch tracking.  The signal is downsampled to 
16 kHz, and inverse filtered with LPC analysis to remove the influence of vowel 
formants. Then harmonics are found in the spectrum of the residual signal.
Drugman and Alwan found that this technique provides an estimate of F0 that is 
robust when the audio signal is corrupted by noise. 

The f0 range is adaptively adjusted.

Probability of voicing is given from a logistic regression formula using `rms` and `srh` 
trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` 
over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing 
decision was about 83% correct.

Parameters
==========
    y : string or ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    f0_range : list of two integers, default = [60,400]
        The lowest and highest values to consider in pitch tracking. This algorithm is quite sensitive to the values given in this setting.
    isResidual : Boolean, default = False
        If the input array is a residual signal
    s : float, default = 0.005
        "Hop" interval between successive analysis windows. The default is 5 milliseconds

Returns
=======
    df : pandas DataFrame  
        measurements at 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame
    * f0 - estimate of the fundamental frequency
    * rms - rms amplitude in the band from 0 to 5 kHz
    * srh - value of SRH (normalized sum of the residual harmonics)
    * probv - estimated probability of voicing
    * voiced - a boolean decision based on the srh value (see Drugman and Alwan)

References
==========

T. Drugman, A. Alwan (2011) Joint robust voicing detection and pitch estimation based on residual harmonics. 'ISCA (Florence, Italy)' pp. 1973ff

    """
    x,fs = prep_audio(y, fs, target_fs=16000, pre = 0, quiet=True)  # resample, preemphasis

    frame_length = int(fs * l)  # frame size in samples
    half_frame = frame_length//2
    frame_length = half_frame * 2 + 1    # odd number in frame  
    step = int(fs * s)  # number of samples between frames
    
    # ----- get rms amplitude from audio wav -------------
    rms = feature.rms(y=x,frame_length=frame_length,hop_length=step,center=False)
    rms = 20 * np.log10(rms[0]) 
    
    # ---- get the f0 from the sum of the residual harmonics (srh) -------------
    if isResidual:
        resid = x
    else:
        resid,fs = lpcresidual(x,fs)  # get the lpc residual signal

    w = windows.hamming(frame_length)
    frames = util.frame(resid,frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc
    
    Sxx = np.abs(np.fft.rfft(frames,2**14))+0.00001      # spectrogram of the residual
    Sxx = np.divide(Sxx.T,linalg.norm(Sxx,axis=-1)).T    # amplitude normalized

    f0,SRHval =  SRH(Sxx,fs,f0_range)
    F0med = int(np.nanmedian(np.where(SRHval<0.1,f0,np.nan)))

    oldF0med,iters = 0, 0  # iterate once to adjust the pitch range
    while F0med != oldF0med and iters<1 and np.max(SRHval) > 0.1:
        oldF0med = F0med
        iters += 1
        f0_range[1] = int(F0med) + 100  # only adjusting the top end of the range
        f0,SRHval =  SRH(Sxx,fs,f0_range)  # recalculate
        F0med = int(np.nanmedian(np.where(SRHval<0.1,f0,np.nan)))

    # ---------- get voicing decisions --------------
    odds = np.exp(1.65 + (0.15*rms) + (15.26*SRHval))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    # vthresh = 0.06               # Drugman et al voicing decision
    #if np.std(SRHval) > 0.05: vthresh = vthresh*1.2 
    #voiced = np.where(SRHval > vthresh,True,False)
    
    # ---- get the time at the center of each frame ---------------
    sec = (np.array(range(frames.shape[0])) * step + half_frame).astype(int)/fs
   
    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'srh':SRHval, 'probv': probv, 'voiced': voiced})


def get_f0_ac(y, fs, f0_range = [60,400], l=0.05, s=0.005):
    """Track the fundamental frequency of voicing (f0), using autocorrelation.

    This function implements a simple autocorrelation method of pitch tracking with no filtering prior to calculating the autocorrelation. Probability of voicing is given from a logistic regression formula using `rms` and `c` trained to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 88% correct.

    Parameters
    ==========
        y : ndarray
            A one-dimensional array of audio samples
        fs : int
            Sampling rate of **x**
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.
        l : float, default = 0.05
            Length of the pitch analysis window in seconds. The default is 50 milliseconds.  
        s : float, default = 0.005
            "Hop" interval between successive analysis windows. The default is 5 milliseconds  
            
    Returns
    =======
        df: pandas DataFrame  
            measurements at 5 msec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - peak normalized rms amplitude in the band from 0 to fs/2
        * c - value of the peak autocorrelation found in the frame
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5
    """
    x, fs = prep_audio(y, fs, target_fs = 32000, pre = 0, quiet=True)  

    short = fs//f0_range[1]  # period of highest allowable frequency - shortest lag
    long = fs//f0_range[0]  # period of lowest allowable frequency - longest lag
    
    frame_length = int(fs * l)  # room for periods
    half_frame = frame_length//2
    frame_length = half_frame * 2 + 1    # odd number in frame  

    step = int(fs * s)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step,center=False)
    rms = 20*np.log10(rms[0]/np.max(rms[0]))

    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis

    sec = (np.array(range(nb)) * step + half_frame).astype(int)/fs

    f0 = np.empty(nb)
    c = np.empty((nb))

    # N/(N-t)  
    ac_norm = [frame_length/(frame_length - k) for k in range(frame_length)]
    
    for i in range(nb): 
        cormat = np.correlate(frames[i], frames[i], mode='full') # autocorrelation 
        ac = cormat[cormat.size//2:] # the autocorrelation is in the last half of the result
        ac = ac*ac_norm
        idx = np.argmax(ac[short:long]) + short # index of peak correlation (in range lowest to highest)
        f0[i] = 1/(idx/fs)      # converted to Hz
        if (ac[0]<= 0) | (ac[idx] <= 0):
            c[i] = 0
        else:
            c[i] = np.sqrt(ac[idx]) / np.sqrt(ac[0])
    
    HNR = 10 * np.log10(c/(1-c),where=np.where(c<1,True,False),out=np.zeros(c.shape))

    odds = np.exp(0.48 + (0.19*rms) + (5.44*c))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec, 'f0':f0, 'rms':rms, 'c':c, 'HNR':HNR, 'probv': probv, 'voiced':voiced})

def f0_from_harmonics(f_p,i,h,nh):  
    ''' Assign harmonic numbers to the peaks in f_p -- this function is used in get_f0_acd
    
        f_p: an array of peak frequencies
        i: the starting peak to look at (0,n)
        h: the starting harmonic number to assign to this peak (1,n-1)
    '''
    Np = len(f_p)  # number of peaks
    m = np.zeros(Np)
    f0 = []
    m[i] = h
    f0 = np.append(f0, f_p[i]/h)  # f0 if peak i is harmonic h
    thresh = 0.055 * f0[0]  # 5.5% of the f0 value
    ex = 0  # number of harmonics over h=11

    for j in range(i+1,Np):  # step through the spectral peaks
        lowest_deviation = 1000
        best_f0 = np.nan
        for k in range(h+1,nh+1):  # step through harmonics
            test_f0 = f_p[j]/k
            deviation = abs(test_f0 - f0[0])
            if deviation < lowest_deviation: # pick the best harmonic number for this peak
                lowest_deviation = deviation
                best_f0 = test_f0
                best_k = k
        if lowest_deviation < thresh:  # close enough to be a harmonic
            m[j] = best_k
            f0 = np.append(f0,best_f0)
            if (h>11): ex = ex + 1
            h=h+1
    C = ((h-1) + (Np - ex))/ np.count_nonzero(m)

    mean_f0 = np.average(f0,weights=np.arange(len(f0))+1)
    return C,mean_f0 
    
def get_f0_acd(y, fs,  f0_range=[60,400], l=0.05, s=0.005, prom=14, min_height = 0.6, test_time=-1):
    """Track the fundamental frequency of voicing, using a frequency domain method.

This function implements the 'approximate common denominator" algorithm proposed by Aliik, Mihkla and Ross (1984), which was an improvement on the method proposed by Duifuis, Willems and Sluyter (1982).  The algorithm finds candidate harmonic peaks in the spectrum, and chooses a value of f0 that best predicts the harmonic pattern.  One feature of this method is that it reports a voice quality measure (the difference in the amplitudes of harmonic 1 and harmonic 2).

Probability of voicing is given from a logistic regression formula using `rms`, the `h2h1` ratio, and Duifuis et al.'s harmonicity criterion `C`  to predict the voicing state as determined by EGG data using the function `phonlab.egg_to_oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The prediction of the EGG voicing decision was about 86% correct.
Aliik et al. (1984) used a cutoff of C < 3.5 as a voicing threshold.

Parameters
==========
    y : ndarray
        A one-dimensional array of audio samples
    fs : int
        the sampling rate of the audio in **y**.
    f0_range : a list of two integers, default=[60,400]
        The lowest and highest values to consider in pitch tracking. The algorithm is not particularly sensitive to this parameter, but it can be useful in avoiding pitch-halving or pitch-doubling.
    l : float, default = 0.05
        Length of the pitch analysis window in seconds. The default is 50 milliseconds.  
    s : float, default = 0.005
        "Hop" interval between successive analysis windows. The default is 5 milliseconds
    prom : numeric, default = 14 dB
        In deciding whether a peak in the spectrum is a possible harmonic, this prominence value is passed to scipy.find_peaks().  A larger value means that the spectral peak must be more prominent to be considered as a possible harmonic peak, and thus the algorithm is less likely to report pitch values when the parameter is given a high value.  In general, 20 is a high value, and 3 is low.
    min_height: numeric, default = 0.6
        As a proportion of the range between the lowest amplitude in the spectrum and the highest, only peaks above `min_height` will be considered to be harmonics. The value that is passed to find_peaks() is: `amplitude_min + min_height*(amplitude_range)`. 

Returns
=======
    df: pandas DataFrame  
        measurements at 5 msec intervals.

Note
====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - rms amplitude in a low frequency band from 0 to 1200 Hz
        * h1h2 - the difference in the amplitudes of the first two harmonics (H1 - H2) in dB
        * C - harmonicity criterion (lower values indicate stronger harmonic pattern)
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5

References
==========

J. Allik, M. Mihkla, J. Ross (1984) Comment on "Measurement of pitch in speech: An implementation of Goldstein's theory of pitch perception" [JASA 71, 1568 (1982)].  `JASA` 75(6), 1855-1857.

H. Duifhuis & L.F. Willems (1982) Measurement of pitch in speech: An implementation of Goldstein's theory of pitch perception.  `JASA` 71(6), 1568-1580.

Example
=======

.. code-block:: Python
    
    example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'stereo.wav'

    x,fs = phon.loadsig(example_file, chansel=[0])
    f0df = phon.get_f0_acd(x,fs,prom=18)

    snd = parselmouth.Sound(str(example_file)).extract_left_channel()  # create a Praat Sound object
    pitch = snd.to_pitch()  # create a Praat pitch object
    f0df2 = phon.pitch_to_df(pitch)  # convert it into a Pandas dataframe

    ret = phon.sgram(x,fs,cmap='Grays') # draw a spectrogram of the sound

    f0_range = [60,400]

    ax1 = ret[0]  # get the plot axis
    ax2 = ax1.twinx()  # and twin it for plotting f0
    ax2.plot(f0df2.sec,f0df2.f0, color='chartreuse',marker="s",linestyle = "none")
    ax2.plot(f0df.sec,f0df.f0, color='dodgerblue',marker="d",linestyle = "none")  
    ax2.set_ylim(f0_range)
    ax2.set_ylabel("F0 (Hz)", size=14)
    for item in ax2.get_yticklabels(): item.set_fontsize(14)

.. figure:: images/acd_pitch_trace.png
    :scale: 33 %
    :alt: a spectrogram with a pitch trace calculated by get_f0_acd
    :align: center

    Comparing the f0 found by `phon.get_f0_acd()` plotted with blue diamonds, and the f0 
    values found by `parselmouth` `to_Pitch()`, plotted with chartreuse dots.

    """
    nh = 6  # maximum number of harmonics to consider
    down_fs = nh*400  # down sample frequency
    
    x, fs = prep_audio(y, fs, target_fs = down_fs, pre=0,quiet=True)  # no preemphasis
    
    N = 1024    # FFT size

    frame_len = int(fs*l)  # 40 ms frame
    half_frame = frame_len//2
    frame_length = half_frame * 2 + 1    # odd number in frame  
    step = int(fs*s)  # stride between frames
    noverlap = frame_len - step   # points of overlap between successive frames

    while (frame_len > N): N = N * 2  # increase fft size if needed
    w = windows.hamming(frame_len)
    f,ts,Sxx = spectrogram(x,fs=fs,noverlap = noverlap, window=w, nperseg = frame_len, 
                              nfft = N, scaling = 'spectrum', mode = 'magnitude', detrend = 'linear')
    Sxx = Sxx + 0.0001
    rms = 20 * np.log10(np.sqrt(np.divide(np.sum(np.square(Sxx),axis=0),len(f)))) 
    Sxx = 20 * np.log10(Sxx)

    nb = len(ts)  # the number of frames in the spectrogram
    f0 = np.full(nb,np.nan)  # array filled with nan
    h1h2 = np.full(nb,np.nan)        # array filled with nan
    c = np.full(nb,11.0)      # default value of c is 5.0
        
    min_dist = int(f0_range[0]/(fs/N)) # min distance btw harmonics
    max_dist = int(f0_range[1]/(fs/N))
    dist = int((min_dist + max_dist)/2)

    ## temp 
    if test_time>0:
        i_test = np.argmin(np.fabs(test_time-ts)) # the ts that is closest to this
    else: 
        i_test = -1
    ## temp
    
    for idx in range(nb):
        spec = Sxx[:,idx]
        height = np.min(spec) + min_height * np.abs(np.max(spec)-np.min(spec))  # required height of a peak
        peaks,props = find_peaks(spec, height = height, prominence=prom, distance = min_dist, wlen=dist)

        if len(peaks)>2:  # we did find some harmonics?
            for p in range(3):  # for each of the first three spectral peaks
                for h in range(1,5): # treat it as one of the first four harmonics
                    if (h==p*2): break
                    C,_f0 = f0_from_harmonics(f[peaks],p,h,nh)
                    
                    if idx==i_test: 
                        print(f'_f0: {_f0:0.1f}, peak: {p}, harmonic: {h}, C: {C:0.2f}')
                        
                    if (f0_range[0] < _f0) & (_f0 < f0_range[1]) & (C < c[idx]):
                        c[idx] = C      
                        i_f0 = np.argmin(np.fabs(_f0 - f)) # f index that is closest to f0
                        i_2f0 = np.argmin(np.fabs((2 * _f0) - f)) # closest to 2f0 for (h1h2)
                        h1h2[idx] = spec[i_f0] - spec[i_2f0]
                        f0[idx] = _f0

        if idx==i_test:  # show diagnostic info, only at a target frame
            if len(peaks)<nh: # the highest harmonic number to consider
                n = len(peaks)
            else:
                n = nh
            plt.plot(f,spec)
            plt.vlines(f[peaks[0:n]],np.min(spec),np.max(spec))
            plt.axhline(height)
            print("number of peaks: ",len(peaks), "start time: ", ts[0])
            print(f'min_dist = {min_dist}, max_dist = {max_dist}, down_fs={fs}, len(f)={len(f)}, N={N}')
            print(f"median difference between adjacent peaks {np.median(np.ediff1d(f[peaks[0:n]])):0.2f}")
            print(f"frequency of the lowest peak {f[peaks[0]]:0.2f}")
            print(f'mean prominence: {np.mean(props["prominences"][0:n]):0.3f} mean peak: {np.mean(props["peak_heights"][0:n]):0.3f}')
            print(f"height = {height:0.2f},max={np.max(spec):0.2f}, min={np.min(spec):0.2f}, c={best_c}")
            print(f"time = {test_time}, f0 = {f0[idx]:0.2f}, h1h2 = {h1h2[idx]:0.2f}")


    odds = np.exp(8.32 + (0.149*rms) - (0.616*c) - (0.0062*h1h2))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': ts, 'f0':f0, 'rms':rms, 'h1h2':h1h2, 'C':c, 'probv': probv, 'voiced':voiced})

