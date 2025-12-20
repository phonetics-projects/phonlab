import numpy as np
from scipy.signal import windows, filtfilt, buttord, butter, sosfiltfilt, argrelmax, argrelmin, find_peaks
from pandas import DataFrame
from ..utils.prep_audio_ import prep_audio
from ..acoustic.lpc_residual import lpcresidual


def get_mbs(x,fs, f0median, width = 1.4):
    ''' filter with a blackman window, to smooth glottal pulses to something more sinusoidal.

    width is the size of the window in number of glottal pulses.  Drugman and Dutoit used 1.75,
    1.4 seems to be better for higher pitched voices without harming performance with lower 
    voices.
    '''
    
    # calculate the MeanBased Signal - smooth with a window that is 1.4 glottal pulses wide
    t0mean = round(fs/f0median)  # expected number of samples per glottal cycle
    halfL = round((0.5*width)*t0mean)     # using a factor of 1.4 (0.7*2) instead of 1.75
    w = windows.blackman(2*halfL +1)  # the length of the smoothing filter is critical
    mbs = filtfilt(w,1,x)  # smooth audio with blackman window, for mean-based signal

    # remove lowfrequency oscillation in the mbs
    n,Wp = buttord(50,30,3,60,fs=fs)
    sos = butter(n, Wp, 'hp', output='sos',fs=fs)
    mbs = sosfiltfilt(sos, mbs)  # highpass filter the mean-based signal
    mbs = mbs/np.max(np.abs(mbs))

    return mbs

# GCI function

def gci_sedreams(x,fs,f0median=200,order=None,cthresh=0.0):
    '''Identify Glottal Closure instances (GCI) using Drugman & Dutoit's (2009) `sedreams` algorithm which 
was published as a Matlab function in the Covarep repository.   The function also returns f0 and vocal jitter 
based on the derived GCI estimates.  Two parameters are given here which were not a part of the original 
implementation.  Based on a comment in D&D(2009), the function includes a process to choose different LPC 
orders for different f0medians.  

    .. code-block:: Python

        if order==None:
            if f0median<190: order = 18
            elif f0median<250: order = 16
            elif f0median<300: order = 14
            else: order = 12

If you don't explicitly specify and LPC order one will be chosen for you.  This differs from D&D in that 
they just used order=18 for everything.  This doesn't seem to change much in the operation of this function.

The other additional parameter is a threshold for peaks in the residual function. Changing this parameter 
changes the output of the function quite a lot.  The algorithm looks for a peak in a temporal span 
of the residual in each presumed glottal cycle and reports the location of that peak 
as the location of the glottal closing instant, and height of the peak as the SOE (strength of excitation).  
With the **cthresh** parameter the user is given the option of disregarding intervals that have 
weak evidence of a glottal closure.  D&D used cthresh=0, resulting
in a lot of apparent jitter in regions where the actual f0 is much lower than f0median.  If you add a threshold, the algorithm rejects candidate GCIs for which there is weak evidence of glottal closure.  This may be a mistake for breathy voice.  Further testing is needed.

Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        ampling rate of **x**
    f0median : float
        The median of the fundamental frequency of voicing in **x**. This value can be estimated 
        by the user, but it is best to measure the median of values given by a pitch tracker.
    order : int, default = None
        By default the order used in LPC analysis (to calculate the LPC residual signal) is guessed
        based on the value of F0median.  Drugmand & Dutoit (2009) used order=18.
    cthresh: float, default = 0.1
        A peak in the LPC residual must be greater than this thresholf value in 
        in order to be considered a glottal closure instant.  The residual is amplitude normalized 
        to the range [0,1], so 0.1 is 10% of the amplitude range.

Returns
=======
    df : pandas DataFrame
        See the note below
    mbs : ndarray
        A waveform the same length as the input **x** containing the "Mean Based Signal"
    resid : ndarray
        A waveforem the same length as the input **x** containing the LPC residual signal.

Note
====
The columns in the returned DataFrame are:
    * sec - the time points (in seconds) of each glottal closure instant. These are the GCI times.
    * f0 - pitch of voicing, which is `1/t`, where `t` is the glottal period - the duration between adjacent GCI times.
    * jitter - the relative discrepancy between adjacent glottal period durations (call them t1 and t2). The value is calculated as `j = 2 * |0.5 - t1/(t1+t2)|`. This results in values on a scale from 0 (adjacent periods (t) are equal in duration), to 1 (the theoretical limit of jitter).  A value of 0.5 means that t1 is twice (or 1/2) as long as t2.  This corresponds to 1/2 of Praat's "local jitter" measurement (which is scaled from 0 to 2).
    * soe - strength of excitation is the height of the LPC residual peak for each GCI

References
==========
    T. Drugman, T. Dutoit (2009) Glottal Closure and Opening Instant Detection from Speech Signals, Interspeech09, Brighton, U.K.

    T. Drugman, M. Thomas, J. Gudnason, P. Naylor, T. Dutoit (2012) Detection of Glottal Closure Instants from Speech Signals: a Quantitative Review, IEEE Transactions on Audio, Speech and Language Processing, vol. 20, Issue 3, pp. 994-1006.


Example
=======

The example here shows glottal closure instances for a small 40 millisecond window in one of the exaample audio files.

    .. code-block:: Python
    
        example_file = importlib.resources.files('phonlab') / 'data/example_audio/im_twelve.wav'
        x,fs = phon.loadsig(example_file, fs=16000,chansel=[0])
        f0df = phon.get_f0_B93(x,fs)
        f0med = np.nanmedian(f0df.f0)

        gci_df, mbs,resid = gci_sedreams(x,fs,f0med, cthresh = 0.01)

        times = np.array(range(len(x)))/fs  # construct a time axis for plotting
        start = 1.8                         # choose a chunk
        end = start+0.04
        s = int(start*fs)
        e = int(end*fs)
        plt.plot(times[s:e], x[s:e]+0.75)   # plot the sound wave (centered on y = 0.75)
        plt.plot(times[s:e], mbs[s:e])      # the meanbased signal
        plt.plot(times[s:e], resid[s:e],'c-')  # the lpc residual
        plt.vlines(np.ma.masked_outside(gci_df.sec, start,end),0.4,1.1,color='red')
        plt.axhline(0.1)   
        plt.axhline(-0.1)

The figure here shows the derived waves used in finding GCIs.  In the top trace, the sound wave is shown in blue and the glottal closure instants (GCIs) are shown with red vertical lines.  In the bottom trace, the mean-based signal is shown in orange and the LPC residual is in cyan.  Horizontal lines show the threshold (cthresh=0.1) for considering a peak in the residual to be a possible GCI.


    .. figure:: images/gci.png
       :scale: 50 %
       :alt: Results of the above code showing the derived waves used in finding GCIs.
       :align: center

       
    '''
    
    y,fs = prep_audio(x,fs,target_fs=16000,pre=0,quiet=True)

    if order==None:
        if f0median<190: order = 18
        elif f0median<250: order = 16
        elif f0median<300: order = 14
        else: order = 12
            
    mbs = get_mbs(y,fs,f0median)

    # get the locations of the peaks and valleys in the mbs
    imax = argrelmax(mbs)[0]  # find peaks in the mean-based signal
    imin = argrelmin(mbs)[0]  # and valleys
    while imax[0]<imin[0]:  # insure that sequences of max and min are as expected.
        imax = np.delete(imax,0)
    while imin[-1]>imax[-1]:
        imin = np.delete(imin,-1)

    resid,fs = lpcresidual(y,fs,order = order)
    
    # get some big residual peaks -- to find an estimate for where to find the GCI in the MBS
    rp,_ = find_peaks(resid,height=0.3)
    relpos = np.empty(0)

    for k in range(len(rp)):
        q = np.argmin(np.fabs(imin-rp[k]))
        if mbs[imax[q]]>0.3:
            rel = (rp[k] - imin[q])/(imax[q] - imin[q])
            if rel>0 and rel < 1: 
                relpos = np.append(relpos,rel)
    if len(relpos)==0:
        ratioGCI = 0.5
    else:
        ratioGCI = np.median(relpos)  # a reasonable expectation of where the GCI will be
    
    gci = np.full(len(imin),np.nan)  # initialize with nan values
    soe = np.full(len(imin),np.nan)  # initialize with nan values

    idx = 0
    for k in range(len(imin)):
        t = imax[k] - imin[k]
        start = imin[k] + round((ratioGCI - 0.3)*t)
        if start < 1: start = 1
        if start > len(resid): break
        stop = imin[k] + round((ratioGCI + 0.3)*t)
        if stop > len(resid): stop = len(resid)
        if np.max(resid[start:stop]) > cthresh:  # threshold to posit glottal closure
            i = np.argmax(resid[start:stop])
            soe[k] = np.max(resid[start:stop])
            gci[k] = start + i -1
    soe = soe[~np.isnan(soe)]
    gci = gci[~np.isnan(gci)]  # remove rows that didn't get filled
    gci = (gci)/fs
    t1 = gci[1:]-gci[:-1]  # duration between adjacent GCI
    t1 = np.pad(t1,(0,1),mode='edge')  # repeat the last one
    f0 = 1/t1
    t2 = gci[2:]-gci[:-2] # duration between intervals of 2 GCI
    t2 = np.pad(t2,(0,2),mode='edge') # repeat the last two
    jitter = 2 * np.fabs(0.5 - (t1/t2)) # 0 = identical adjacent periods, 1 = one is twice as long as the other

    df = DataFrame({'sec': gci, 'f0':f0, 'jitter':jitter, 'soe':soe})
    
    return df,mbs,resid
