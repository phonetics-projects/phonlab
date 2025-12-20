from ..utils.prep_audio_ import prep_audio

import numpy as np
from librosa import util
from scipy.interpolate import interp1d
from numpy.fft import rfft, rfftfreq
from pandas import DataFrame
import matplotlib.pyplot as plt

def _maxarg(x,axis = -1):
    try:
        idx = np.argmax(x,axis=axis)
        val = np.max(x,axis=axis)
    except:
        idx = 0
        val = np.nan
        
    return idx,val

def get_f0_shs(x,fs, f0_range=[50,400], l=0.06, s=0.005, shr_threshold = 0.3, target_time=None):
    ''' A pitch determination function implementing the subharmonic summation (SHS) algorithm proposed by Dik Hermes (1988), with a method of finding the relative amplitude of a subharmonic pitch (if there is evidence of a subharmonic) as suggested by Xuejing Sun (2002). The method is good at tracking pitch changes when the voice goes into creak, and is relatively insensitive to the setting of the f0_range parameter.  If you find that there is pitch halving, try increasing the shr_threshold, and if you find pitch doubling try decreasing shr_threshold and/or the top end of the f0_range.

    
Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    f0_range : an array of two numbers, default=[50,400]
        The expected range for f0.  There is rarely any need to adjust the lower value.
    l : float, default = 0.06
        Length of analysis windows.  The default is 60 milliseconds.
    s : float, default = 0.005
        Step size, of hops between analysis windows. The default is 5 milliseconds.
    shr_threshold : float, default = 0.3
        A value which determines how sensitive the algorithm is in deciding that a subharmonic 
        component is present in the voicing spectrum. 
    target_time : float, default = None
        If a time value (in seconds) is given, a diagnostic figure comparable to Sun's Figure 1 
        will be produced for the frame at time target_time.

Returns
=======
    df: pandas DataFrame  
        measurements at (by default) 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame in seconds
    * f0 - estimate of the fundamental frequency in Hz
    * rms - rms amplitude in the frequency band from 0 to fs/2
    * shr - The Subharmonic-to-harmonic Ratio.  If the ratio is low the subharmonic energy is close to that of the harmonic.
    
References
==========
    D. Hermes (1988) Measurement of pitch by subharmonic summation. `J. Acoust. Soc. Am.`, 83(1),257-264.
    
    X. Sun (2002) Pitch determination and voice quality analysis using subharmonic-to-harmonic ratio. `Proceedings of ICASSP2002` I-333 - I-336.

Example
=======

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'
    x,fs = phon.loadsig(example_file,chansel=[0])
    ttime = 2.101

    df = phon.get_f0_shs(x,fs, shr_threshold=0.2, target_time=2.1)

This example shows diagnostic plots from the get_f0_shs() function. The top left panel shows the spectrum at a particular time in the audio file on a log2 scale.  The top right and bottom left panels show harmonic and subharmonic summation spectra which are used to produce the difference spectrum in the bottom right.  The maximum of the difference spectra is taken as the pitch in this frame.
 
.. figure:: images/shs.png
    :scale: 60 %
    :alt: Diagnostic plots of the log spectrum to 1200Hz, and the harmonic and submonic spectra, and in the bottom left panel shows the difference between them.
    :align: center  

    Diagnostic plots from shr_pitch().


.. figure:: images/shs_pitch.png
    :scale: 35 %
    :alt: Pitch trace produced by the get_f0_shs() function.
    :align: center  

    Pitch trace produced by get_f0_shs() function.

    '''
    y,fs = prep_audio(x,fs,target_fs=16000,quiet=True)

    top_freq = f0_range[1]*2.5
    
    frame_length = round(l*fs)
    half_frame = frame_length//2
    step = round(s*fs)

    if target_time != None:  # ---- for the diagnostic print -------
        fn = int(((target_time * fs)-half_frame)/step) # frame number
    else:
        fn = None
    
    NFFT = int(2**(np.ceil(np.log(frame_length*(2))/np.log(2))))
    
    freq = rfftfreq(NFFT, d=1./fs)
    limit = np.argmin(np.abs(freq-top_freq))+1  
    freq = freq[1:limit]   # frequency values in the linear spectrum
    logf = np.log2(freq)   # frequency values in the log2 spectrum
    minbin = logf[-1] - logf[-2]  # minimum step size on the log freq scale


    ilogf = np.arange(logf[0],logf[-1],minbin)  # equal spacing on the log scale
    ifreq = 2**ilogf

    maxlogf = np.log2(f0_range[1])
    minlogf = np.log2(f0_range[0])
    upperbound = np.argmin(np.abs(ilogf-maxlogf))
    lowerbound = np.argmin(np.abs(ilogf-minlogf))

    N = int(top_freq/f0_range[0])  # number of harmonics to consider
    N = N - N%2                    # make sure it is an even number

    frames = util.frame(y,frame_length=frame_length, hop_length=step,axis=0)
    nb = frames.shape[0]
    f = frames.shape[1]  # number of frequency steps
    
    w = np.blackman(frame_length)
    S = np.abs(rfft(w*frames,NFFT))  # spectrogram below topfreq
    Sxx = S[:,1:limit]
    rms = 20 * np.log10(np.sqrt(np.sum(np.square(np.abs(Sxx)),axis=-1))) 
    interp_function = interp1d(logf, Sxx) 
    logS = interp_function(ilogf)
    W = 0.5 + (1/np.pi)*np.arctan(5*(ilogf-np.log2(60)))  # W function (Hermes, 1988) to damp freqs below 60
    logS = W * logS
    #logS -= np.min(logS,axis=0)
    s_points = logS.shape[1]

    starts = np.round(np.log2(range(1,N+1,1))/minbin).astype(np.int16) 
    starts2 = np.round(np.log2(np.arange(1.5,N+1,1))/minbin).astype(np.int16) 
    
    shift_mat = np.zeros((nb,s_points,N))
    shift_mat2 = np.zeros((nb,s_points,N))

    for i in range(N):
        shift_mat[:,:s_points-starts[i],i] = logS[:,starts[i]:]
        shift_mat2[:,:s_points-starts2[i],i] = logS[:,starts2[i]:]
    Hxx = np.sum(shift_mat,axis=-1)  # subharmonic summation
    Pxx = np.sum(shift_mat2,axis=-1)  # shifted subharmonic summation
    Dxx = Hxx - Pxx
    
    idx1 = np.argmax(Dxx[:,lowerbound:upperbound],axis= -1) 
    idx1 += lowerbound
    mag1 = np.max(Hxx[:,idx1],axis=-1)
    mag1 = np.where(mag1>0,mag1,0.00001) # avoid 0/0 in SHR calculation
    
    # consider the possibility that idx1 is a subharmonic - look for harmonic peak at log2(f)+log2(1)
    step1 = round(np.log2(2 - 0.0625)/minbin) # search bounds for later, with subharmonics
    step2 = round(np.log2(2 + 0.0625)/minbin)
    s = idx1 + step1 # where to find the possible harmonic peak
    e = idx1 + step2
    top = min(s_points,upperbound)
    e = np.where(e>top,top,e)

    idx2,mag2 = np.array([_maxarg(Hxx[i,s[i]:e[i]]) for i in range(nb)]).T
    idx2 = (idx2 + s).astype(np.int16)

    idx2 = np.where(s>upperbound,idx1,idx2)
    mag2 = np.where(s>upperbound,mag1,mag2)
    mag2 = np.where(mag2<=0,mag1,mag2)

    SHR = (mag1-mag2)/(mag1+mag2)
    SHR = np.where(SHR==0,np.nan,SHR)
    f0idx = np.where(SHR<shr_threshold,idx2,idx1)
    F0 = ifreq[f0idx]
    
    ts = (np.array(range(nb)) * step + half_frame)/fs  # time axis for output

    if fn!=None:
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
        ax1.plot(ilogf,logS[fn,:],color="orange")
        ax2.plot(ilogf[lowerbound:upperbound],Hxx[fn,lowerbound:upperbound],color='black')
        ax3.plot(ilogf[lowerbound:upperbound],Pxx[fn,lowerbound:upperbound],color='black')
        ax3.plot(ilogf[lowerbound:upperbound],Hxx[fn,lowerbound:upperbound],color='black',linestyle="dotted")
        ax4.plot(ilogf[lowerbound:upperbound],Dxx[fn,lowerbound:upperbound],color='blue')

        ax2.axvline(ilogf[idx1[fn]],color="green",linestyle="dotted")
        ax2.axvline(ilogf[idx2[fn]],color="red",linestyle="dotted")
        ax3.axvline(ilogf[idx1[fn]],color="green",linestyle="dotted")
        ax3.axvline(ilogf[idx2[fn]],color="red",linestyle="dotted")
        ax4.axvline(ilogf[idx1[fn]],color="green",linestyle="dotted")
        ax4.axvline(ilogf[idx2[fn]],color="red",linestyle="dotted")

        print(f"F0={F0[fn]:0.1f}, shr={SHR[fn]:0.3f}, N={N}")
        print(f"   idx1={ifreq[idx1[fn]]:0.1f}, idx2={ifreq[idx2[fn]]:0.1f}, shr = {SHR[fn]:0.3f}")
        print(f"   mag1={mag1[fn]:0.3f}, mag2={mag2[fn]:0.3f}")

    return DataFrame({'sec': ts, 'f0': F0, 'rms':rms, 'shr': SHR})

def _shr_pitch(x,fs, f0_range=[64,400], l=0.04, s=0.005, shr_threshold = 0.15, top_freq = 1000, target_time=None):
    '''An implementation of Xuejing Sun's (2002) SubHarmonic-to-Harmonic Ratio (SHR) pitch determination method.  
This method is a variant of Hermes' (1988) subharmonic summation method.  In addition to computing pitch, the 
function returns the SHR when there is a measureable subhamonic component, otherwise the value is NaN.  
Sun found that between 40% and 90% of all voiced frames (depending on the speaker) had no measurable SHR. 

The algorithm as implemented here is three times faster than the fastest of the other pitch tracking algorithms 
in `phonlab` -- (the autocorrelation pitch tracker in `phonlab.get_f0`, and the cepstral peak pitch tracker in `phonlab.CPP`.  
It produces very accurate pitch determination, if the top of the F0 pitch range is chosen appropriately for the talker.
    
Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    f0_range : an array of two numbers, default=[64,400]
        The expected range for f0.  Choose the top value deliberately for the speaker being analyzed.
    l : float, default = 0.04
        Length of analysis windows.  The default is 40 milliseconds.
    s : float, default = 0.005
        Step size, of hops between analysis windows. The default is 5 milliseconds.
    shr_threshold : float, default = 0.15
        A value which determines how sensitive the algorithm is in deciding that a subharmonic 
        component is present in the voicing spectrum. 
    top_frequency : int, default = 1000 (Hz)
        The spectrum will be limited to this top frequency
    target_time : float, default = None
        If a time value (in seconds) is given, a diagnostic figure comparable to Sun's Figure 1 
        will be produced for the frame at time target_time.

Returns
=======
    df: pandas DataFrame  
        measurements at (by default) 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame in seconds
    * f0 - estimate of the fundamental frequency in Hz
    * shr - The Subharmonic-to-harmonic Ratio.  If the ratio is low the subharmonic energy is close to that in the harmonic spectrum.
    
References
==========
    D. Hermes (1988) Measurement of pitch by subharmonic summation. `J. Acoust. Soc. Am.`, 83(1),257-264.
    
    X. Sun (2002) Pitch determination and voice quality analysis using subharmonic-to-harmonic ratio. `Proceedings of ICASSP2002` I-333 - I-336.

Example
=======

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'
    x,fs = phon.loadsig(example_file,chansel=[0])

    df = phon.shr_pitch(x,fs, f0_range=[50,300], target_time=1.33)


This example shows diagnostic plots from the shr_pitch() function. The top left panel shows the spectrum at a particular time in the audio file on a log2 scale.  The top right and bottom left panels show harmonic and subharmonic summation spectra which are used to produce the difference spectrum in the bottom right.  The maximum of the difference spectra is taken as the pitch in this frame.
 
.. figure:: images/shr.png
    :scale: 60 %
    :alt: Diagnostic plots of the log spectrum to 1000Hz, and the harmonic and submonic spectra, and in the bottom left panel shows the difference between them.
    :align: center  

    Diagnostic plots from shr_pitch().


.. figure:: images/shr_pitch.png
    :scale: 45 %
    :alt: Pitch trace produced by the shr_pitch() function.
    :align: center  

    Pitch trace produced by shr_pitch() function.

    '''
    y,fs = prep_audio(x,fs,target_fs=16000,quiet=True)

    frame_length = round(l*fs)
    half_frame = frame_length//2
    step = round(s*fs)

    if target_time != None:  # ---- for the diagnostic print -------
        fn = int(((target_time * fs)-half_frame)/step) # frame number
    else:
        fn = None
    
    NFFT = int(2**(np.ceil(np.log(frame_length*(1.5))/np.log(2))))
    
    freq = rfftfreq(NFFT, d=1./fs)
    limit = np.argmin(np.abs(freq-top_freq))+1  
    freq = freq[1:limit]   # frequency values in the linear spectrum
    logf = np.log2(freq)   # frequency values in the log2 spectrum
    minbin = logf[-1] - logf[-2]  # minimum step size on the log freq scale


    ilogf = np.arange(logf[0],logf[-1],minbin)  # equal spacing on the log scale
    ifreq = 2**ilogf

    maxlogf = np.log2(f0_range[1]/2)
    minlogf = np.log2(f0_range[0]/2)
    upperbound = np.argmin(np.abs(ilogf-maxlogf))
    lowerbound = np.argmin(np.abs(ilogf-minlogf))

    N = int(top_freq/f0_range[0])  # number of harmonics to consider
    N = N - N%2                    # make sure it is an even number

    frames = util.frame(y,frame_length=frame_length, hop_length=step,axis=0)
    nb = frames.shape[0]
    SHR = np.ones(nb) * np.nan
    F0 = np.ones(nb) * np.nan
    
    w = np.blackman(frame_length)
    S = np.abs(rfft(w*frames,NFFT))[:,1:limit]  # spectrogram below 1250Hz
    interp_function = interp1d(logf, S) 
    logS = interp_function(ilogf)
    W = 0.5 + (1/np.pi)*np.arctan(5*(ilogf-np.log2(60)))  # W function (Hermes, 1988) to damp freqs below 60
    logS = W * logS
    logS -= np.min(logS,axis=0)
    s_points = logS.shape[1]

    # 'odd' shift matrix log2(1,3,5,...), 'even' shift matrix log2(2,4,6...)
    odd_starts = np.round(np.log2(range(1,N,2))/minbin).astype(np.int16) 
    even_starts = np.round(np.log2(range(2,N+1,2))/minbin).astype(np.int16) 
    shift_mat_odd = np.zeros((nb,s_points,N//2))
    shift_mat_even = np.zeros((nb,s_points,N//2))

    for i in range(N//2):
        shift_mat_odd[:,:s_points-odd_starts[i],i] = logS[:,odd_starts[i]:]
        shift_mat_even[:,:s_points-even_starts[i],i] = logS[:,even_starts[i]:]
    Sub = np.sum(shift_mat_odd,axis=-1)
    Har = np.sum(shift_mat_even,axis=-1)
    DA = Har-Sub
    
    idx1,mag1 = _maxarg(DA[:,lowerbound:upperbound]) 
    idx1 += lowerbound

    # consider the possibility that idx1 is a subharmonic - look for harmonic peak at log2(f)+log2(1)
    step1 = round(np.log2(2 - 0.0625)/minbin) # search bounds for later, with subharmonics
    step2 = round(np.log2(2 + 0.0625)/minbin)
    s = idx1 + step1 # where to find the possible harmonic peak
    e = idx1 + step2
    #top = min(s_points,upperbound)
    e = np.where(e>s_points,s_points,e)
    
    idx2,mag2 = np.array([_maxarg(DA[i,s[i]:e[i]]) for i in range(nb)]).T
    idx2 = (idx2 + s).astype(np.int16)
    
    # if start > upperbound - then idx2,mag2 are invalid - f0 = ifreq[idx1]*2, SHR = 0
    # if mag1 < 0 - then f0 = np.nan, SHR = np.nan
    # if mag2 < 0 - then f0 = ifreq[idx1]*2, SHR = 0
   
    mask = (s<upperbound) & (mag1>0) & (mag2>0)   # constraints on calculating SHR
    SHR[mask] = (mag1[mask]-mag2[mask])/(mag1[mask]+mag2[mask])
    SHR = np.where(mag1<=0,np.nan,SHR)
    SHR = np.where(mag2<=0,0,SHR)
    F0 = np.where(mag1>0,ifreq[idx1]*2,np.nan)
    F0 = np.where(SHR<shr_threshold,ifreq[idx2]*2,F0)
    F0 = np.where(F0>f0_range[1],F0/2, F0)

    ts = (np.array(range(nb)) * step + half_frame)/fs  # time axis for output

    if fn!=None:
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
        ax1.plot(ilogf,logS[fn,:],color="orange")
        ax2.plot(ilogf,Har[fn,:],color='black')
        ax3.plot(ilogf,Sub[fn,:],color='black')
        ax3.plot(ilogf,Har[fn,:],color='gray',linestyle="dotted")
        ax4.plot(ilogf,DA[fn,:],color="dodgerblue")
        ax4.axvline(ilogf[lowerbound],linestyle="dashed")
        ax4.axvline(ilogf[upperbound],linestyle="dashed")
        ax4.axvline(ilogf[idx2[fn]],color="red",linestyle="dotted")
        ax4.axvline(ilogf[idx1[fn]],color="green",linestyle="dotted")
        print(f"SHR={SHR[fn]}, F0={F0[fn]}, N={N}, s={s[fn]}, e={e[fn]}, upperbound = {upperbound}")
        print(f"   (idx1={ifreq[idx1[fn]]}, mag1={mag1[fn]}), (idx2={ifreq[idx2[fn]]}, mag2={mag2[fn]})")

    return DataFrame({'sec': ts, 'f0': F0, 'shr': SHR})
