from ..utils.prep_audio_ import prep_audio

import numpy as np
from librosa import util
from numpy.fft import rfft, irfft
from pandas import DataFrame
import matplotlib.pyplot as plt

def HNR(x,fs, f0_range = [64,400], l= 0.06, s=0.005, target_time = None):
    '''Compute the Harmonics-to-Noise Ratio using the Cepstrum-Based method given by de Krom (1993).
The matlab implementation by Yen-Liang Shue (2009) which is a part of the VoiceSauce collection of 
software provided some valuable pointers in how to implement de Krom's method.  One difference between
the Shue implementation and this one is that here we use a fixed window size (specified with the input 
parameter **l** and by default 60 ms) where Shue's algorithm required that F0 estimates be provided as 
input and then HNR is calculated over windows of variable length, always long enough for 5 pitch periods
according to the estimated F0 for that location in the audio.

de Krom's (1993) method uses cepstral analysis to filter out the harmonic component of the spectrum, 
allowing the separate calculation of harmonic and non-harmonic energy. 

Parameters
==========
    x : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **x**
    f0_range : an array of two numbers, default=[64,400]
        The expected range for f0.
    l : float, default = 0.06
        Length of analysis windows.  The default is 60 milliseconds.
    s : float, default = 0.005
        Step size, of hops between analysis windows. The default is 5 milliseconds.
    target_time : float, default = None
        If a time value (in seconds) is given, diagnostic matplotlib plots of the spectrum and cepstrum 
        will be produced.
    
Returns
=======
    df: pandas DataFrame  
            measurements at (by default) 5 msec intervals.

Note
====
The columns in the returned dataframe are for each frame of audio:
    * sec - time at the midpoint of each frame in seconds
    * f0 - estimate of the fundamental frequency in Hz
    * hnr_500 - The harmonics-to-noise ratio in the spectrum below 500Hz
    * hnr_1500 - The harmonics-to-noise ratio in the spectrum below 1500Hz
    * hnr_2500 - The harmonics-to-noise ratio in the spectrum below 2500Hz


References
==========
    G. de Krom (1993) A Cepstrum-Based Technique for Determining a Harmonics-to-Noise Ratio in Speech Signals. `Jounral of Speech and Hearing Research` 36,254-266.


Example
=======

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'
    x,fs = phon.loadsig(example_file,chansel=[0])

    df = HNR(x,fs, target_time=2.1)  # get diagnostic plots for the spectrum at time 2.1 seconds.


This example shows diagnostic plots of the spectrum (top panel) and cepstrum (bottom panel) of a frame of audio
at time 2.1 seconds in the file 'sf3_cln.wav'. In the top panel we see the log magnitude spectrum in black, the noise component of the spectrum in orange, and the harmonic component in light blue.  In the bottom panel we see the cepstrum of the log magnitude spectrum.  The "Rahmonics" in black are removed (set to zero) giving the "liftered" cepstrum (in pink).  The FFT of this liftered cepstrum gives the noise spectrum (well, after some level correction).

.. figure:: images/HNR.png
    :scale: 50 %
    :alt: Diagnostic plots of the spectrum (top panel) and cepstrum (bottom panel) of a frame of audio.
    :align: center  


    '''
    x,fs = prep_audio(x,fs,target_fs=16000,quiet=True)
    
    frame_length = int(l*fs)
    step = int(s*fs)
    half_frame = frame_length//2
    frame_length = half_frame * 2 + 1    # odd number in frame
    #NFFT = int(2**(np.ceil(np.log(frame_length)/np.log(2)))*2)
    NFFT = frame_length
    
    frames = util.frame(x,frame_length=frame_length, hop_length=step,axis=0)

    nb = frames.shape[0]  # number of frames
    ts = (np.array(range(nb)) * step + half_frame)/fs  # time axis for output
    w = np.blackman(frame_length)
    S = 20 * np.log10(np.abs(rfft(w*frames,NFFT)))
    C = np.real(irfft(S,NFFT))  # spectrum of the spectrum 

    if f0_range[0]<f0_range[1]:
        f0_range = np.flip(f0_range)  # from [60,400] -> [400,60]
    T0_range = np.round(fs/np.array(f0_range)).astype(np.int32)
    T0 = np.argmax(C[:,T0_range[0]:T0_range[1]],axis=-1) + T0_range[0]  # F0 period
    F0 = fs/T0
    
    half_combwidth = 5
    C_cl = C.copy()
    maxindex = C_cl.shape[1]-1
    for j in range(nb):     # lifter out the harmonics
        for k in range(1,4):
            il = T0[j]*k - (half_combwidth+1)
            ir = T0[j]*k + half_combwidth
            if ir>maxindex: break
            C_cl[j,il:ir] = 0

    # -------- get noise spectrum from liftered cepstrum
    Nap = np.real(rfft(C_cl[:,:NFFT//2], NFFT)) 
    Ha = S - Nap          # estimate Harmonic spectrum

    # ------ Baseline correction ---------
    N = Nap
    for j in range(nb):
        Hdelta = F0[j]/fs * NFFT
        for f in np.arange(Hdelta+0.0001,NFFT//2-1,Hdelta):
            fend = round(f)
            fstart = np.ceil(f-Hdelta).astype(np.int32)
            N[j,fstart:fend] += np.min(Ha[j,fstart:fend])
    H = S - N  # harmonic spectrogram
    N = S - H  # noise spectrogram
    
    i500 = int(500/fs * NFFT)  # index at 500 Hz
    i1500 = int(1500/fs * NFFT)
    i2500 = int(2500/fs * NFFT)

    HNR500 = np.mean(H[:,:i500],axis=-1) - np.mean(N[:,:i500],axis=-1)
    HNR1500 = np.mean(H[:,:i1500],axis=-1) - np.mean(N[:,:i1500],axis=-1)
    HNR2500 = np.mean(H[:,:i2500],axis=-1) - np.mean(N[:,:i2500],axis=-1)


    # ------------- diagnostic plot ---------------
    if target_time != None:
        fig,[ax1,ax2] = plt.subplots(nrows=2,ncols=1)
        quef = (np.array(range(1,NFFT//2+1))/fs *1000)  # the quefrecy axis of the cepstra (in ms)
        freqs = np.array(range(1,NFFT//2+1)) * (fs/NFFT)

        fn = int(((target_time * fs)-half_frame)/step)
        print(f"{ts[fn]}, {T0[fn]/fs*5}, {F0[fn]}, {HNR500[fn]}, {HNR1500[fn]}")
        print(f"T0[fn] = {T0[fn]}, {quef[T0[fn]]}")
        ax1.plot(freqs[:NFFT//4],N[fn,:NFFT//4],color="orange")
        ax1.plot(freqs[:NFFT//4],H[fn,:NFFT//4],color="lightblue")
        ax1.plot(freqs[:NFFT//4],S[fn,:NFFT//4],color="black")

        ax2.plot(quef[2:NFFT//2],C[fn,2:NFFT//2],color='black')
        ax2.plot(quef[2:NFFT//2],C_cl[fn,2:NFFT//2],color='pink')
        ax2.axvline(quef[T0[fn]],color='black',alpha=0.2)

    return DataFrame({'sec': ts, 'f0':F0, 'hnr_500':HNR500, 'hnr_1500':HNR1500, 'hnr_2500':HNR2500})
