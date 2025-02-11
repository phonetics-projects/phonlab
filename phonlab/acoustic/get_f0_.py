__all__=['get_f0','get_f0_srh','get_f0_ac']

import numpy as np
import scipy.signal
from scipy import fft
import librosa
from pandas import DataFrame
from ..utils.get_signal_ import get_signal
  
def get_f0(sig, f0_range = [63,400], chan = 0, pre = 1.0, fs_in=12000):
    """Track the fundamental frequency of voicing (f0)

    The method in this function mirrors that used in track_formants().  LPC coefficients are calculated for each frame and the audio signal is inverse filtered with these, resulting in a quasi glottal waveform. Then autocorrelation is used to estimate the fundamental frequency.  Probability of voicing is given from a logistic regression formula using `rms` and `c` trained to predict the voicing state as determined by EGG data using the function `phonlab.egg2oq()` over the 10 speakers in the ASC corpus of Mandarin speech. The log odds of voicing in that training data was given by `odds = -4.31 + 0.17*rms + 13.29*c`, and probability of voicing is thus:  `probv = odds / (1 + odds)`.

    Parameters
    ==========
        sig : string or ndarray
            The name of a sound file, or an array of audio samples
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.
        chan : int, default = 0
            If the audio in sig is stereo, which channel should be analyzed?
        fs_in : int
            if sig is an array, pass the sampling rate, if sig is a file name this parameter is ignored.

    Returns
    =======
        df - a pandas dataframe  measurements at 0.01 sec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - estimate of the rms amplitude found with `librosa.feature.rms()`
        * c - value of the peak autocorrelation found in the frame
        * probv - estimated probability of voicing
        * voiced - a boolean, true if probv>0.5

    Example
    =======
    There is one lie in this example.  F0 measurements in nonsonorants were removed prior to actually making this plot.  See the examples folder for how that is done.
    
    >>> f0df = get_f0(x,fs_in=fs, f0_range= [63,400])
    >>>
    >>> ret = phon.sgram(x,fs_in = fs,cmap='Blues') # draw the spectrogram from the array of samples
    >>> ax1 = ret[0]  # the first item returned, is the matplotlib axes of the spectrogram
    >>> ax2 = ax1.twinx()
    >>> ax2.plot(f0df.sec,f0df.f0, 'go')  

    .. figure:: images/get_f0.png
       :scale: 90 %
       :alt: a 'bluescale' spectrogram with red dots marking the f0
       :align: center

       Marking the f0 found by `phon.get_f0()`

       ..

   """
    # constants and global variables
    frame_length_sec = 0.075
    step_sec = 0.01
    
    x, fs = get_signal(sig, fs = 12000, fs_in=fs_in, chan=chan, pre = 0)  # read waveform, no preemphasis

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = librosa.feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,1:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    if (pre > 0): x = np.append(x[0], x[1:] - pre * x[:-1])  # now apply pre-emphasis

    w = scipy.signal.windows.hamming(frame_length)
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    
    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    A = librosa.lpc(frames, order=14,axis=-1)  # get LPC coefs, can use a largish order
    
    f0 = np.empty(nb)
    c = np.empty((nb))

    th = fs//f0_range[1]
    tl = fs//f0_range[0]
    
    for i in range(nb): 
        y = np.convolve(frames[i],A[i,:])  #inverse filter with lpc coeffs
        cormat = np.correlate(y, y, mode='full') # autocorrelation 
        ac = cormat[cormat.size//2:] # the autocorrelation is in the last half of the result
        idx = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
        f0[i] = 1/(idx/fs)      # converted to Hz
        c[i] = np.sqrt(ac[idx]) / np.sqrt(ac[0])

    odds = np.exp(-4.2 + (0.17*rms[:nb]) + (13.2*c[:nb]))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec[:nb], 'f0':f0[:nb], 'rms':rms[:nb], 'c':c[:nb],
                    'probv': probv[:nb], 'voiced':voiced[:nb]})



def get_f0_srh(sig, f0_range = [60,400], chan = 0, pre = 0.94, fs_in=12000):
    # constants and global variables
    frame_length_sec = 0.1
    step_sec = 0.01
    
    x, fs = get_signal(sig, fs = 12000, fs_in=fs_in, chan=chan, pre = 0)  # read waveform, no preemphasis

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = librosa.feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,1:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    if (pre > 0): x = np.append(x[0], x[1:] - pre * x[:-1])  # now apply pre-emphasis

    w = scipy.signal.windows.hamming(frame_length)
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    
    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    A = librosa.lpc(frames, order=14,axis=-1)  
    
    f0 = np.empty(nb)
    c = np.empty((nb))

    for i in range(nb): 
        y = np.convolve(frames[i],A[i,:])  #inverse filter with lpc coeffs
        S = np.abs(np.fft.rfft(y,2**16)) # compute the power spectrum
        T = len(S)/fs
        srh_max = 0
        for f in range(f0_range[0], f0_range[1]): 
            fT = int(f*T)  # test this as frequency of H1
            for k in range(2,5):
                h = S[fT*k] - S[int(fT*(k-0.5))]
            srh = S[fT] + h
            if srh > srh_max:
                srh_max = srh
                f0[i] = f
                c[i] = srh

    odds = np.exp(-4.2 + (0.17*rms[:nb]) + (13.2*c[:nb]))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec[:nb], 'f0':f0[:nb], 'rms':rms[:nb], 'c':c[:nb],
                    'probv': probv[:nb], 'voiced':voiced[:nb]})


def get_f0_ac(sig, f0_range = [60,400], fs_in= -1,chan = 0):

    # constants and global variables
    frame_length_sec = 1/f0_range[0]
    step_sec = 0.005

    # read waveform, no preemphasis, up-sample
    x, fs = get_signal(sig, fs = 48000, fs_in=fs_in, chan=chan, pre = 0)  

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = librosa.feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,1:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    
    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    
    f0 = np.empty(nb)
    c = np.empty((nb))

    th = fs//f0_range[1]
    tl = fs//f0_range[0]
    
    for i in range(nb): 
        cormat = np.correlate(frames[i], frames[i], mode='full') # autocorrelation 
        ac = cormat[cormat.size//2:] # the autocorrelation is in the last half of the result
        idx = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
        f0[i] = 1/(idx/fs)      # converted to Hz
        c[i] = np.sqrt(ac[idx]) / np.sqrt(ac[0])

    odds = np.exp(-4.2 + (0.17*rms[:nb]) + (13.2*c[:nb]))  # logistic formula, trained on ASC corpus
    probv = odds / (1 + odds)
    voiced = probv > 0.5

    return DataFrame({'sec': sec[:nb], 'f0':f0[:nb], 'rms':rms[:nb], 'c':c[:nb],
                    'probv': probv[:nb], 'voiced':voiced[:nb]})






