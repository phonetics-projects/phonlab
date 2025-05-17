__all__=['get_f0','get_f0_srh','get_f0_cor', 'get_f0_acd']

import numpy as np
from scipy.signal import windows, find_peaks, spectrogram, peak_prominences
from scipy import fft
from librosa import feature, util, lpc
from pandas import DataFrame
from ..utils.get_signal_ import prep_audio
  
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
    
    x, fs = prep_audio(sig, fs = 12000, fs_in=fs_in, chan=chan, pre = 0)  # read waveform, no preemphasis

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,1:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    if (pre > 0): x = np.append(x[0], x[1:] - pre * x[:-1])  # now apply pre-emphasis

    w = windows.hamming(frame_length)
    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    
    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    A = lpc(frames, order=14,axis=-1)  # get LPC coefs, can use a largish order
    
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
    frame_length_sec = 0.1
    step_sec = 0.01
    
    x, fs = prep_audio(sig, fs = 12000, fs_in=fs_in, chan=chan, pre = 0)  # read waveform, no preemphasis

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,1:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    if (pre > 0): x = np.append(x[0], x[1:] - pre * x[:-1])  # now apply pre-emphasis

    w = windows.hamming(frame_length)
    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame, for lpc

    nb = frames.shape[0]  # the number of frames (or blocks) in the LPC analysis
    
    sec = np.array([(i * (step/fs)+(frame_length_sec/2)) for i in range(nb)])  
    A = lpc(frames, order=14,axis=-1)  
    
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


def get_f0_cor(sig, f0_range = [60,400], fs_in= -1,chan = 0):

    # constants and global variables
    frame_length_sec = 1/f0_range[0]
    step_sec = 0.005

    # read waveform, no preemphasis, up-sample
    x, fs = prep_audio(sig, fs = 48000, fs_in=fs_in, chan=chan, pre = 0)  

    frame_length = int(fs * frame_length_sec) 
    half_frame = frame_length//2
    step = int(fs * step_sec)  # number of samples between frames

    rms = feature.rms(y=x,frame_length=frame_length, hop_length=step)[0,0:-1] # get rms amplitude
    rms = 20*np.log10(rms/np.max(rms))

    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)    

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

def f0_from_harmonics(f_p,i,h):  
    ''' Assign harmonic numbers to the peaks in f_p -- this function is used in get_f0_acd
    
        f_p: an array of peak frequencies
        i: the starting peak to look at (0,n)
        h: the starting harmonic number to assign to this peak (1,n-1)
    '''
    N = len(f_p)  # number of peaks
    m = np.zeros(N)
    f0 = []
    m[i] = h
    f0 = np.append(f0, f_p[i]/h)  # f0 if peak i is harmonic h
    thresh = 0.05 * f0[0]  # 5% of the f0 value
    ex = 0  # number of harmonics over h=11

    for j in range(i+1,N):  # step through the spectral peaks
        lowest_deviation = 1000
        best_f0 = np.nan
        for k in range(h+1,7):  # step through harmonics
            test_f0 = f_p[j]/k
            deviation = abs(test_f0-f0[0])
            if deviation < lowest_deviation: # pick the best harmonic number for this peak
                lowest_deviation = deviation
                best_f0 = test_f0
                best_k = k
        if lowest_deviation < thresh:  # close enough to be a harmonic
            m[j] = best_k
            f0 = np.append(f0,best_f0)
            if (h>11): ex = ex + 1
            h=h+1
    C = ((h-1) + (N - ex))/ np.count_nonzero(m)
    
    return C,np.mean(f0)  #,np.int32(m)
    
def get_f0_acd(sig, f0_range = [60,500], fs_in= -1,chan = 0, crit_c = 3.5,prom=10, peak_height = 0.4):
    """Track the fundamental frequency of voicing (f0)

    The method in this function implements the 'approximate common denominator" algorithm proposed by Aliik, Mihkla and Ross (1984), which was an improvement on the method proposed by Duifuis, Willems and Sluyter (1982).  The method finds candidate harmonic peaks in the spectrum up to 6 times the highest possible F0 given in the **f0_range** parameter, and chooses a value of f0 that will give the best fit for the harmonic pattern.

    Parameters
    ==========
        sig : string or ndarray
            The name of a sound file, or an array of audio samples
        f0_range : list of two integers, default = [63,400]
            The lowest and highest values to consider in pitch tracking.
        chan : int, default = 0
            If the audio in sig is stereo, which channel should be analyzed?
        fs_in : int, default = -1
            if sig is an array, pass the sampling rate, if sig is a file name this parameter is ignored.
        crit_c: float, default = 3.5
            c, a spectral fit criterion is computed for each frame.  This critical value can be used to keep F0 
            measurements in frames that with **c** below the **crit_c** value.  

    Returns
    =======
        df - a pandas dataframe  measurements at 0.01 sec intervals.

    Note
    ====
    The columns in the returned dataframe are for each frame of audio:
        * sec - time at the midpoint of each frame
        * f0 - estimate of the fundamental frequency
        * rms - estimate of the rms amplitude in the downsampled spectrum (0-2400 Hz by default)
        * c - spectral fit criterion, smaller means the fit is better

    Example
    =======
    
    >>> f0df = get_f0_acd(x,fs_in=fs, f0_range= [63,400])
    >>>
    >>> ret = phon.sgram(x,fs_in = fs,cmap='Blues') # draw the spectrogram from the array of samples
    >>> ax1 = ret[0]  # the first item returned, is the matplotlib axes of the spectrogram
    >>> ax2 = ax1.twinx()
    >>> ax2.plot(f0df.sec,f0df.f0, 'go')  

    .. figure:: images/get_f0_acd.png
       :scale: 90 %
       :alt: a 'bluescale' spectrogram with red dots marking the f0
       :align: center

       Marking the f0 found by `phon.get_f0()`

       ..

    """
    down_fs = f0_range[1] * 6  # allow for 5 harmonics of the highest f0
    x, fs = prep_audio(sig, fs = down_fs, fs_in = fs_in, pre=0, chan=chan)  

    step_sec = 0.002
    N = 512    # FFT size

    frame_len = int(down_fs*0.04)  # forty ms frame
    step = int(down_fs*step_sec)  # stride between frames
    noverlap = frame_len - step   # points of overlap between successive frames

    while (frame_len > N): N = N * 2  # increase fft size if needed
    w = windows.hamming(frame_len)
    f,ts,Sxx = spectrogram(x,fs=fs,noverlap = noverlap, window=w, nperseg = frame_len, 
                              nfft = N, scaling = 'spectrum', mode = 'magnitude', detrend = 'linear')
    nb = len(ts)  # the number of frames in the spectrogram
    f0 = np.empty(nb)
    rms = np.empty(nb)
    c = np.empty((nb))
    rms = 20 * np.log10(np.sqrt(np.divide(np.sum(np.square(Sxx),axis=0),len(f)))) 
    
    Sxx = 20 * np.log10(Sxx)

    min_height = np.max(Sxx) - peak_height * np.abs(np.max(Sxx) - np.min(Sxx)) 
    min_dist = int(f0_range[0]/(down_fs/N)) # min distance btw harmonics
    max_dist = int(f0_range[1]/(down_fs/N)) 
    print(f'min_height = {min_height}, max = {np.max(Sxx)}, min = {np.min(Sxx)}')

    for idx in range(nb):
        peaks,props = find_peaks(Sxx[:,idx], height = min_height, 
                                 distance = min_dist, prominence=prom, 
                                 wlen=max_dist)
        if idx==45: 
            proms = peak_prominences(Sxx[:,idx], peaks, wlen=max_dist)
            print(props['peak_heights'],proms[0])
            
        c[idx] = 5
        f0[idx] = np.nan
        if len(peaks)>2:  # we did find some harmonics?
            for p in range(3):  # for each of the first three spectral peaks
                for h in range(1,4): # treat it as one of the first three harmonics
                    C,_f0 = f0_from_harmonics(f[peaks],p,h)
                    if C < c[idx]:  # keep the best peak/harmonic alignment
                        #print(f'frame {idx}, C = {C}')
                        c[idx] = C
                        if C<crit_c: 
                            f0[idx] = _f0
                        else:  
                            f0[idx] = np.nan
        
    return DataFrame({'sec': ts, 'f0':f0, 'rms':rms[:nb], 'c':c[:nb]})



