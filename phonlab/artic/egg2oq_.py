import scipy.signal
import numpy as np
import librosa
import pandas as pd

def egg_to_oq(x, fs, hop_dur = 0.005, f0_range = [60,400],
           hp_cut = 70, hp_order = 8, threshold=0.43, norm_window=0.7, center=True,
           egg_norm_window=None, floor=True, peak_height=0.5, diagnostics=False):
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
        norm_window : float, default = 0.7
            Duration in seconds of the window over which the differentiated EGG is range-normalized to (0,1), so that
            glottal closing instants are the peaks above `peak_height`.  The window slides along the signal.  A shorter
            window follows changes in the strength of the signal (and of the closing peaks, which are smaller at low f0)
            more closely.  If the signal is no longer than the window, the whole signal is used.
        center : boolean, default = True
            If True the normalization window is centered on each sample; if False it covers the preceding `norm_window`
            seconds.
        egg_norm_window : float, default = None
            Window duration in seconds for range-normalizing the EGG signal itself (for the glottal opening instants).
            By default it is the same as `norm_window`.
        floor : float, True or None, default = True
            Where the range of the differentiated EGG in the normalization window is less than this proportion of its
            largest range anywhere in the signal, no closing instants are found.  This keeps the normalization from
            magnifying noise or hum in stretches with no voicing.  True is the same as 0.05.  None or False turns the floor off.
        peak_height : float, default = 0.5
            A closing instant is a peak in the normalized differentiated EGG which is higher than this.
        diagnostics : boolean, default = False
            If True, the returned dataframe has extra columns that show what the analysis found in
            each frame, which can explain why a frame has no OQ and f0 (see the Note below).
        
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
            * voiced - True if the frame's window has two glottal closing instants that are one period apart (a period
              between 1/f0_range[1] and 1/f0_range[0] seconds).  Unlike OQ this does not require opening instants.

        A frame has OQ and f0 only if its window (1.5 periods of the lowest f0) contains two glottal closing
        instants (peaks in the differentiated EGG), two glottal opening instants (downward crossings of
        `threshold`) and an opening instant between the two closing instants.  When `diagnostics` is True
        these columns are added, with values measured in the same window:

            * n_gci - the number of glottal closing instants found
            * n_goi - the number of glottal opening instants found
            * degg_max - the highest value of the normalized differentiated EGG (a closing instant needs a peak above 0.5)
            * egg_min, egg_max - the lowest and highest values of the normalized EGG (an opening instant needs a downward crossing of `threshold`)
            * egg_range - the range of the filtered EGG used to normalize it at the center of the window, in the units of **x**
            * fail - why the frame has no OQ: 'few_gci', 'few_goi', 'no_opening' (no opening instant between the
              closing instants), or '' if the frame has OQ

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

    def local_range(sig, window):
        """max and min of sig in a rolling window (seconds), or over all of it if it is not longer than the window"""
        rw = int(window*fs)
        if len(sig) > rw:
            r = pd.Series(sig).rolling(rw, min_periods=10, center=center)
            return r.max().to_numpy(), r.min().to_numpy()
        return np.max(sig), np.min(sig)

    eggmax, eggmin = local_range(egg, norm_window if egg_norm_window is None else egg_norm_window)
    deggmax, deggmin = local_range(degg, norm_window)
    if np.ndim(eggmax) > 0:
        eggmax = np.where(eggmax/np.max(egg) < 0.05, np.max(egg), eggmax)  # require some minimal egg activity

    egg_range = np.asarray(eggmax - eggmin)   # local scale of the EGG, kept for diagnostics

    egg = (egg - eggmin)/(eggmax-eggmin)  # range normalization
    drange = deggmax - deggmin
    degg = (degg - deggmin)/drange
    if floor:
        floor = 0.05 if floor is True else floor
        degg = np.where(drange < floor*np.nanmax(drange), 0.0, degg)   # nothing to see in a stretch with little EGG activity

    # get peaks in the degg waveform - the glottal closing instants (gci)
    # minimum spacing between peaks (distance) is shortest possible period
    peak_sep = 1.0/f0_range[1]
    degg_peaks = scipy.signal.find_peaks(degg,distance=peak_sep*fs, height=peak_height) 
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
    voiced = np.zeros(frames.shape[0],dtype=bool)
    
    if diagnostics:
        n_gci = np.zeros(frames.shape[0], dtype=int)
        n_goi = np.zeros(frames.shape[0], dtype=int)
        fail = np.full(frames.shape[0], '', dtype=object)

    for k in range(frames.shape[0]):
    
        gci = np.argwhere(frames[k,:] == 1.0).flatten()  # glottal closure instances
        goi = np.argwhere(frames[k,:] == -1.0).flatten()  # glottal opening instances

        if diagnostics:
            n_gci[k], n_goi[k] = gci.size, goi.size
        voiced[k] = gci.size >= 2 and np.any(np.diff(gci) <= fs/f0_range[0])  # two closures, a possible period apart

        if (gci.size<2):  # nothing to look at
            if diagnostics: fail[k] = 'few_gci'
            continue
        if (goi.size<2):  # nothing to look at
            if diagnostics: fail[k] = 'few_goi'
            continue
            
        period = gci[1]-gci[0]  # interval between closures
        if goi[0] > gci[0] and goi[0]<gci[1]:  # find an appropriate opening instant
            op = goi[0]
        elif goi[1] > gci[0] and goi[1]<gci[1]: 
            op = goi[1]
        else:
            if diagnostics: fail[k] = 'no_opening'
            continue
        
        OQ[k] = (gci[1]-op)/period
        f0[k] = 1/(period/fs)   # may as well calculate this while we are here

    df = pd.DataFrame({'sec':sec,'OQ':OQ,'f0':f0,'voiced':voiced})

    if diagnostics:
        dframes = librosa.util.frame(degg, frame_length=win, hop_length=hop, axis=0)
        eframes = librosa.util.frame(egg, frame_length=win, hop_length=hop, axis=0)
        center = np.minimum(np.arange(frames.shape[0]) * hop + win // 2, egg.size - 1)
        df['n_gci'] = n_gci
        df['n_goi'] = n_goi
        df['degg_max'] = np.nanmax(dframes, axis=1)
        df['egg_min'] = np.nanmin(eframes, axis=1)
        df['egg_max'] = np.nanmax(eframes, axis=1)
        df['egg_range'] = egg_range if egg_range.ndim == 0 else egg_range[center]
        df['fail'] = fail

    return df
