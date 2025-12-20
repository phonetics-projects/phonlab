import numpy as np
import librosa

def burst (x, fs, start, end):
    """Find a stop burst in a short interval of audio
    
This function takes an array of waveform samples and returns the location in time of the most prominent stop release burst in a segment of audio indicated by the `start` and `end` parameters.  Generally, `start` and `end` are taken from a TextGrid and define a short (50ms to 150ms) segment that has been labeled as containing a consonant of interest (a stop, fricative or nasal for example). In addition to the time of the strongest burst in the time-span, a burst strength score is also returned.

The algorithm looks at sudden peaks in the acoustic waveform that correspond to a large change 
in the mel frequency spectrum (a sudden increase in spectral energy) and scores each candidate with
a linear discriminant formula that was trained on TIMIT stop release bursts. The time and score of the winning candidate is 
returned.

Parameters
==========

x : ndarray
      a one-dimensional array of waveform samples

fs : number
      the sampling frequency of the audio in **x**.  The algorithm was trained on speech sampled at 16000 Hz, and will work best if you resample to 16000 with `phon.prep_audio()` before calling this function.  See the example.

start :   float
      the time in seconds at the start of the waveform chunk in x

end :   float
      the time in seconds at the end of the waveform chunk in x



Returns
=======

    b_time :  float
        the time in seconds of the most prominent burst in x between start and end
        
    b_score : float
        a measure of the prominence of the burst

Example
=======
   
In this example we open a sound file with `phon.loasig()`, prepare it for the burst detector with `phon.prep_audio()` and then search for the best candidate for a stop release burst in the interval from time 1.5 seconds to time 2.0 seconds.  The return value `b_time` is the location of the burst in seconds, and `b_score` is a measure of the burst prominence.  The `sgram()` plot in this example illustrates the use of `start` and `end` to produce a spectrogram of a specified portion of signal.

.. code-block:: Python

     x,fs = phon.loadsig("sf3_cln.wav", chansel=[0])
     x,fs = phon.prep_audio(x,fs, target_fs=16000, pre=0.94) 
    
     t1 = 1.5
     t2 = 2
    
     b_time, b_score = phon.burst(x,fs,t1,t2,fs)  # find a stop burst in the span from t1 to t2
    
     ax1,f,t,Sxx = phon.sgram(x,fs_in=fs,start=t1, end=t2)
     ax1.axvline(b_time,color="red")

.. figure:: images/burst.png
   :scale: 50 %
   :alt: a spectrogram with a red line marking the location of the burst
   :align: center

   Marking the burst found by `phon.burst()`

"""
    n_fft = 1024
    w_cands = np.zeros((3,2))  # [,0] are scores, and [,1] are times
    s_cands = np.zeros((3,2))  # [,0] are scores, and [,1] are times
    cand = {}  # keep candidates in a dictionary
    maxb = 0 
    burst_time = -1  
    mel_f = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=60, fmin=300, fmax=int(fs/2))

    step = int(fs*0.005)   # 5 ms between spectra

    y = x[int(start*fs):int(end*fs)]  # section of waveform to search

    if (np.max(y)+np.min(y) < 0): y = -y  # set polarity of the waveform

    # -------- find points in the waveform that are good burst candidates -------------
    for i in range(step,len(y)-2):  # loop over the waveform samples
        if ((y[i]< y[i+1]) and (y[i+1]>y[i+2])):  # A waveform peak is located at i+1
            ave = np.mean(np.fabs(y[i-step:i-1] - y[i-step+1:i]))  # average deviation across samples in this frame
            peak_size = np.fabs(y[i]-y[i+1])/ave  # normalized deviation at peak
            for l in range(3):  # keep top three candidates, with sharpest peaks
                if peak_size > w_cands[l,0]:
                    if (l<2):
                        w_cands[2,] = w_cands[1,]
                    if (l<1):
                        w_cands[1,] = w_cands[0,]
                    w_cands[l,0] = peak_size
                    w_cands[l,1] = np.float64((i)/fs)
                    break

    # --------- find points in the spectrogram that are good burst candidates --------------
    sgram = librosa.stft(y, n_fft=n_fft, hop_length=step, win_length=step)  # get the spectrogram
    melgram = np.log(mel_f.dot(np.abs(sgram))) # convolve the Mel filterbank over the spectrogram
    spec_diffs = np.sum(np.diff(melgram),axis=0)  # get summed difference between adjacent spectra

    for i in range(len(spec_diffs)):  # loop over spectra, a new one every 5 ms
        for l in range(3):
            if spec_diffs[i] > s_cands[l,0]:
                if (l<2):
                    s_cands[2,] = s_cands[1,]
                if (l<1):
                    s_cands[1,] = s_cands[0,]
                s_cands[l,0] = spec_diffs[i]
                s_cands[l,1] = np.float64((i+1)*step/fs)
                break

    # ----- use the waveform and spectrogram candidates to score and select the best burst candidate ------
    for w in range(3):
        for s in range(3):
            if np.fabs(w_cands[w,1] - s_cands[s,1]) < 0.004: # both waveform and spectrographic evidence
                cand[w]=s
    for (w,s) in cand.items():  
        try:
            b = -1.814 + 0.618*np.log(w_cands[w,0]) + 0.045*s_cands[s,0] # linear model trained on TIMIT
        except:
            b = 0

        if (b>maxb): 
            maxb = b
            burst_time = w_cands[w,1] + start
  
    return (burst_time,maxb)
