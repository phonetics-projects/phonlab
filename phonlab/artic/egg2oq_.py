import scipy.signal
import numpy as np
import librosa
import pandas as pd

def egg_to_oq(x, fs, hop_dur = 0.005, f0_range = [60,400],
           hp_cut = 70, hp_order = 8, threshold=0.43):
    """Get glottal open quotient from electroglottography data
    
    Extract fundamental frequency of voice (f0) and glottal open quotient (OQ) 
    at equally spaced intervals in a recording of electroglottography (a stereo 
    file).  OQ is calcuated with the 'hybrid' method (Herbst, 2020). Preprocessing 
    filtering method follows Rothenberg (2002).
    

    Parameters
    ==========
        
        x : ndarray
            A one-dimensional array of samples in an electroglottograph waveform
        fs : int
            the sampling rate of **x**
        hop_dur : float, default = 0.005
            interval in seconds between frames (0.005 seconds = 5 milliseconds)
        f0_range : list of two integers, default = [63,400]
            The lowest and highest allowed values of f0.
        hp_cut : int, default = 70 
            highpass filter cut frequency (in Hz) (Rothenberg, 2002)
        hp_order : int, default = 8
            highpass filter order (Rothenberg, 2002)
        threshold : float, default = 0.43
            threshold between 0 and 1, on the EGG signal for glottal opening instant (Herbst, 2020)
        
    Returns
    =======
        df : DataFrame
            Open quotient and f0 results in a Pandas dataframe.

    Note
    ====
        The returned Pandas dataframe has columns:
        
            * sec - midpoint times (seconds) of the analysis frames in f0 and OQ
            * OQ - the glottal open quotient as a function of time.
            * f0 - estimates of the fundamental frequency of voicing as a function of time

    References
    ==========

    C. T. Herbst (2020) Electroglottography − An Update. *Journal of Voice* , **34** (4), pp. 503-526.

    M. Rothenberg (2002) Correcting low-frequency phase distortion in electroglottograph waveforms. *Journal of Voice* , **16** (1), 32-36. 
    

    Example
    =======

    .. code-block:: Python
    
     file = "F1_bha24_1.wav"  # a stereo file with audio in channel 0 and egg in 1
     egg,audio,fs = phon.loadsig(file, chansel=[1,0])  # fool with `chansel`
     oqdf = phon.egg2oq(egg,fs)  # return open quotient data

    See the example ipython notebook for the code used to generate this figure

    .. figure:: images/egg2oq.png
       :scale: 100 %
       :alt: a figure showing acoustic waveform, EGG signal, and calculated OQ and f0
       :align: center

       Calculate Open Quotient and f0 from ElectroGlottoGraphic data

       ..

        
    """
    window_length = (1.0/f0_range[0]) * 1.5 # add 25% for alignment?
    win = int(window_length*fs)  # window for the longest period 
    hop = int(hop_dur*fs)  # 5ms hop
 
    # highpass filter the egg signal
    coefs = scipy.signal.butter(hp_order, hp_cut, fs=fs, btype='highpass', output='sos')
    egg = scipy.signal.sosfiltfilt(coefs, x)
    degg = np.gradient(egg)  # differential of the egg

    # scale the filtered egg and degg to (0,1)  -- for long files do this locally?
    #  using pandas rolling_max

    egg_s = pd.Series(egg)
    degg_s = pd.Series(degg)

    rw = int(1.5*fs)
    if len(egg) > fs*5:  # only do this if the duration is greater than 5 second5
        eggmax = egg_s.rolling(rw,min_periods=10).max()  # max in a rolling window
        deggmax = degg_s.rolling(rw,min_periods=10).max()
        eggmin = egg_s.rolling(rw,min_periods=10).min()  # min in a rolling window
        deggmin = degg_s.rolling(rw,min_periods=10).min()
        eggmax[eggmax/np.max(egg) < 0.05] = np.max(egg)  # require some minimal egg activity
    
    else:  # no need for local max and min, just take the whole signal
        eggmax = np.max(egg)
        eggmin = np.min(egg)
        deggmax = np.max(degg)
        deggmin = np.min(degg)

    
    egg = (egg - eggmin)/(eggmax-eggmin)  # range normalization
    degg = (degg - deggmin)/(deggmax-deggmin)

    # get peaks in the degg waveform - the glottal closing instants (gci)
    # minimum spacing between peaks (distance) is shortest possible period
    peak_sep = 1.0/f0_range[1]
    degg_peaks = scipy.signal.find_peaks(degg,distance=peak_sep*fs, height=0.5) 
    glottal = np.zeros(degg.size)
    glottal[degg_peaks[0]]=1  # closing times (peaks returns indices of peaks)
    for i in range(egg.size-1):  # opening times (threshold method)
        if egg[i]>threshold and egg[i+1]<threshold:
            glottal[i] = -1

    frames = librosa.util.frame(glottal, frame_length=win, hop_length=hop, axis=0)
    frame_rate = 1/hop_dur
    
    sec = np.array([(i/frame_rate)+(window_length/2) for i in range(frames.shape[0])])  # get times of frames
    f0 = np.full(frames.shape[0],np.nan)
    OQ = np.full(frames.shape[0],np.nan)
    
    for k in range(frames.shape[0]):
    
        gci = np.argwhere(frames[k,:] == 1.0).flatten()  # glottal closure instances
        goi = np.argwhere(frames[k,:] == -1.0).flatten()  # glottal opening instances

        if (gci.size<2):  # nothing to look at
            continue
        if (goi.size<2):  # nothing to look at
            continue
            
        period = gci[1]-gci[0]  # interval between closures
        if goi[0] > gci[0] and goi[0]<gci[1]:  # find an appropriate opening instant
            op = goi[0]
        elif goi[1] > gci[0] and goi[1]<gci[1]: 
            op = goi[1]
        else:
            continue
        
        OQ[k] = (gci[1]-op)/period
        f0[k] = 1/(period/fs)   # may as well calculate this while we are here

    df = pd.DataFrame({'sec':sec,'OQ':OQ,'f0':f0})

    return df
