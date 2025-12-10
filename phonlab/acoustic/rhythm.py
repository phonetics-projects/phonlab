import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from librosa.util import frame
from librosa import frames_to_time
from ..utils.prep_audio_ import prep_audio

def rem_dc(x):
    '''Remove the DC component of a signal.'''
    return x - np.mean(x)

def bandpass(x, cutoffs, fs, order=3):
    '''Apply a butterworth bandpass filter.'''
    nyq = 0.5 * fs
    coefs = scipy.signal.butter(order, np.array(cutoffs) / nyq, btype='band',output='sos')
    return scipy.signal.sosfiltfilt(coefs, x)

def lowpass(x, cutoff, fs, order=3):
    '''Apply a butterworth lowpass filter.'''
    nyq = 0.5 * fs
    coefs = scipy.signal.butter(order, cutoff / nyq, btype='low',output='sos')
    return scipy.signal.sosfiltfilt(coefs, x)

def centerpass(x, center, bw, fs):
    '''Apply a 'center-pass' filter - a peak at center frequency with a bandwidth bw.'''
    nyq = 0.5 * fs
    w0 = center/nyq
    Q = w0 / (bw/nyq)
    coefs = scipy.signal.iirpeak(w0, Q,output='sos')  
    return scipy.signal.sosfiltfilt(coefs, x)

# this function takes a chunk of waveform and returns a frequency spectrum
# sec is the size of the chunk in seconds
def get_rhythm_spectrum(x,fs): 
    """Return a rhythm spectrum - ala Tilsen & Johnson, 2008

    This approach to measuring rythmicity was described by Tilsen and Johnson (2008).  The code here and in `rhythmogram` is a translation and extension of Tilsen's Matlab code.  This function takes a chunk of waveform (best if it is 4 seconds long) and computes a spectrum of the bandpass filtered amplitude envelope.  It finds periodicity in the amplitude envelope and returns a spectrum in the range from **f** to 7 Hz, tracking components in the amplitude envelope that repeat at low frequency.  The lowest frequency represented **f** is determined from the duration of the waveform in **x**, which should be at least 4 seconds for accurate detection of slow rhythmic components.s

    Parameters
    ==========
    x : ndarray
        A one dimensional array of audio samples - 
    fs : int
        The sampling frequency of x

    Returns
    =======
        freq : ndarray
            The frequency axis of the rhythm spectrum
        psd : ndarray 
            The amplitude values of the rhythm spectrum

    References
    ==========
    S. Tilsen & K. Johnson (2008) Low-frequency Fourier analysis of speech rhythm. `Journal of the Acoustical Society of America`, **124** (2), EL34-EL39.
    
    Example
    =======

    .. code-block:: Python
    
        x,fs = phon.loadsig("s09003.wav")
        slice = x[fs*3:fs*7]

        f,Sx = phon.get_rhythm_spectrum(slice,fs)
        plt.plot(f,Sx)

    .. figure:: images/get_rhythm_spectrum.png
       :scale: 40 %
       :alt: the spectrum of the amplitude envelope of a 4 second chunk of audio
       :align: center

       The spectrum of the amplitude envelope of a 4 second chunk of audio.  This section of speech in Spanish has an oscillation of about 3 Hz; regular repetition of amplitude peaks at intervals of about 330 msec.
         
    """
    
    ds_rate = 100
    ds_factor = fs//ds_rate
    npoints = 512
    chunk_size = len(x)/fs

    # Amplitude Envelope.
    x = bandpass(x, cutoffs=[100, 1500], fs=fs, order=2)
    x = lowpass(np.fabs(x), cutoff=10, fs=fs, order=6)

    # Downsample and remove DC component.
    y = rem_dc(scipy.signal.decimate(x, ds_factor, ftype = 'fir', zero_phase=True))
    y = y * scipy.signal.windows.tukey(len(y), 0.1)  # Shape downsampled signal with a Tukey window.
    norm_frame = y / np.sqrt(np.var(y))  # Normalize to unit variance.
    freq, powsd = scipy.signal.periodogram(norm_frame,   # find frequency spectrum of the amplitude envelope
                                           fs=ds_rate, nfft=npoints, scaling="spectrum")
    min_freq = 1/(chunk_size*(1/2))
    i = np.where((freq < 7) & (freq > min_freq))  # pick indices for spectrum less than lowpass Hz
    
    return (freq[i],powsd[i]) 
    
def rhythmogram(x,fs):
    """Return a rhythm spectrogram - ala Tilsen & Johnson, 2008

The approach to measuring rythmicity that is implemented by this function was described by Tilsen and Johnson (2008).  The code here and in `get_rhythm_spectrum` is a translation and extension of Tilsen's Matlab code.  This function takes a buffer of audio and computes spectra of the amplitude envelope of the bandpass [100 - 1500Hz] filtered signal. Spectra are computed twice a second (i.e. the step size is 0.5 seconds) over windows that are 4 seconds long. It finds periodicity in the amplitude envelope and returns a spectrogram with a frequency range from 1 to 7 Hz, tracking components that repeat at intervals of once per second down to a repetition rate of 140 ms.
    
Parameters
----------

    x : ndarray
        A 1-D array of audio samples
    fs : int
        the sampling frequency of the audio in **x** (in Hz)

Returns
-------
    f : ndarray, one-dimensional
        the frequency axis of the rhythmogram, in Hz
        
    t : ndarray, one-dimensional 
        the time axis of the rhythmogram, in seconds
        
    Sxx : ndarray, two-dimensional
        the amplidude values of the rhythmogram, axis 1 is time, axis 0 is frequency


References
----------
    
S. Tilsen & K. Johnson (2008) Low-frequency Fourier analysis of speech rhythm. `Journal of the Acoustical Society of America`, **124** (2), EL34-EL39.
    

Example
-------

.. code-block:: Python

         x,fs = phon.loadsig("s1202a.wav", chansel=[0])
         f,ts,Sxx = phon.rhythmogram(x,fs)  # calculate rhythm spectra over time
         
         m = np.mean(Sxx,axis=0)  # the mean spectrum of the file
         sd = np.std(Sxx,axis=0)  # the standard deviation of the spectrum
         Sxx_thresh = Sxx - (m + 0.5*sd)   # subtract a threshold to find "rhythmic" sections
        
         start = 136  # seconds, show the thresholded spectrogram for a 30 second interval
         end = 166
         s = np.int32((start-2) *2)  # start frame, two frames per second
         e = np.int32((end-2) *2) # end frame
         extent = (start,end,min(f),max(f))  # get the time and frequency values for figure.
         
         plt.imshow(Sxx_thresh.T[:,s:e], aspect='auto', extent = extent, 
               origin='lower',cmap="coolwarm",interpolation="spline36")
         plt.set(xlabel="Time (sec)", ylabel="Frequency (Hz)")

.. figure:: images/rhythmogram.png
       :scale: 40 %
       :alt: a time/frequency plot of low frequency energy in a 30 second long chunk of speech
       :align: center

       A time/frequency plot of low frequency energy in a 30 second long chunk of speech

    """
    
    # Amplitude Envelope.    
    y = bandpass(x, cutoffs=[100, 1500], fs=fs, order=2)
    y = lowpass(np.fabs(y), cutoff=10, fs=fs, order=6)

    # Downsample and remove DC component.
    fs_env = 100  # desired sampling frequency of the envelope
    ds_factor = fs//fs_env  # new rate will be 100 samples per second
    env = rem_dc(scipy.signal.decimate(y, ds_factor, ftype = 'fir', zero_phase=True))

    # divide into frames, window them
    flen_sec = 4
    hop_sec = 0.5
    frame_length = np.int32(flen_sec * fs_env)
    step = np.int32(hop_sec * fs_env)
    frames = frame(env, frame_length=frame_length, hop_length=step,axis=0) 
    t = frames_to_time(range(len(frames)), sr=fs_env, hop_length=step,n_fft = frame_length)
    w = scipy.signal.windows.tukey(frame_length,0.1)
    frames = np.multiply(frames,w)   # apply a Tukey window to each frame
    norm_frames = frames.T / np.sqrt(np.var(frames,axis=1))  # normalize the amplitudes in the frames
    freq, psd = scipy.signal.periodogram(norm_frames.T,fs=fs_env, nfft=512, scaling="spectrum")  # spectrum
    min_freq = 1/(flen_sec*(1/2))
    i = np.where((freq < 7) & (freq > min_freq))  # pick indices for spectrum less than lowpass Hz
    f = freq[i]
    Sxx = np.squeeze(psd[:,i])
    
    return f, t, Sxx  
 
