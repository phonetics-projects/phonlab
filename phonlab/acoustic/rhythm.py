__all__=["get_rhythm_spectrum","rhythmogram"]

import scipy.signal
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from librosa.util import frame
from librosa import frames_to_time

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
def get_rhythm_spectrum(x,fs,chunk_size = 4): 
    """Return a rhythm spectrum - ala Tilsen & Johnson, 2008

    This approach to measuring rythmicity was described by Tilsen and Johnson (2008).  The code here and in `rhythmogram` is a translation and extension of Tilsen's Matlab code.  This function takes a chunk of waveform (best if it is 4 seconds long) and computes a spectrum of the bandpass filtered amplitude envelope.  It finds periodicity in the amplitude envelope and returns a spectrum in the range from 1 to 7 Hz, tracking components that repeat at intervals of 1 second down to a repetition rate of 140 ms.

    Parameters
    ==========
    x : ndarray
        A one dimensional array of audio samples - 
    fs : int
        The sampling frequency of x
    chunk_size : float, default = 4
        The size (in seconds) of the audio clip in x

    Returns
    =======
        freq : ndarray
            The frequency axis of the rhythm spectrum
        psd : ndarray 
            The amplitude values of the rhythm spectrum

    References
    ==========
    Tilsen, S. & Johnson, K. (2008) Low-frequency Fourier analysis of speech rhythm. Journal of the Acoustical Society of America 124(2), EL34-EL39.
    
    Examples
    ========

    .. code-block:: Python
         f,Sx = rhythm.get_rhythm_spectrum(slice,fs,chunk_size)
         plt.plot(f,Sx)
         
    """
    
    ds_rate = 100
    ds_factor = fs//ds_rate
    npoints = 512

    # Amplitude Envelope.
    x = bandpass(x, cutoffs=[100, 1500], fs=fs, order=2)
    x = lowpass(np.fabs(x), cutoff=10, fs=fs, order=6)

    # Downsample and remove DC component.
    y = rem_dc(scipy.signal.decimate(x, ds_factor, ftype = 'fir', zero_phase=True))
    y = y * scipy.signal.tukey(len(y), 0.1)  # Shape downsampled signal with a Tukey window.
    norm_frame = y / np.sqrt(np.var(y))  # Normalize to unit variance.
    freq, powsd = scipy.signal.periodogram(norm_frame,   # find frequency spectrum of the amplitude envelope
                                           fs=ds_rate, nfft=npoints, scaling="spectrum")
    min_freq = 1/(chunk_size*(1/2))
    i = np.where((freq < 7) & (freq > min_freq))  # pick indices for spectrum less than lowpass Hz
    
    return (freq[i],powsd[i]) 
    
def rhythmogram(sig):
    """Return a rhythm spectrogram - ala Tilsen & Johnson, 2008

    This approach to measuring rythmicity was described by Tilsen and Johnson (2008).  The code here and in `get_rhythm_spectrum` is a translation and extension of Tilsen's Matlab code.  This function takes the name of an audio file and computes spectra of the bandpass filtered amplitude envelope. Spectra are computed twice a second (i.e. the step size is 0.5 seconds) over windows that are 4 seconds long. It finds periodicity in the amplitude envelope and returns a spectrogram with a frequency range from 1 to 7 Hz, tracking components that repeat at intervals of 1 second down to a repetition rate of 140 ms.
    
    Parameters
    ==========

    sig : string
        The name of an audio file

    Returns
    =======
    f : ndarray, one-dimensional
        the frequency axis of the rhythmogram, in Hz
        
    t : ndarray, one-dimensional 
        the time axis of the rhythmogram, in seconds
        
    Sxx : ndarray, two-dimensional
        the amplidude values of the array, axis 1 is time, axis 0 is frequency

    References
    ==========
    Tilsen, S. & Johnson, K. (2008) Low-frequency Fourier analysis of speech rhythm. Journal of the Acoustical Society of America 124(2), EL34-EL39.
    
    Examples
    ========

    .. code-block:: Python
    
         f,ts,Sxx = phon.rhythmogram("s1202a.wav")  # calculate rhythm spectra over time
         
         m = np.mean(Sxx,axis=0)  # the mean spectrum of the file
         sd = np.std(Sxx,axis=0)  # the standard deviation of the spectrum
         Sxx_thresh = Sxx - (m + 0.5*sd)   # subtract a threshold to find "rhythmic" sections
        
         start = 136
         end = 166
         s = np.int32((start-2) *2)  # start frame, two frames per second
         e = np.int32((end-2) *2) # end frame
         extent = (start,end,min(f),max(f))  # get the time and frequency values for figure.
         
         plt.imshow(Sxx_thresh.T[:,s:e], aspect='auto', extent = extent, 
               origin='lower',cmap="coolwarm",interpolation="spline36")
         plt.set(xlabel="Time (sec)", ylabel="Frequency (Hz)")

     .. figure:: images/rhythmogram.png
       :scale: 80 %
       :alt: a time/frequency plot of low frequency energy in a 30 second long chunk of speech
       :align: center

       A time/frequency plot of low frequency energy in a 30 second long chunk of speech

    """
    
    fs, x = scipy.io.wavfile.read(sig)
    x_t = np.arange(len(x)) / fs  # time axes
    print(f"file is {len(x)/fs} seconds long")

    # Amplitude Envelope.
    y = bandpass(x, cutoffs=[100, 1500], fs=fs, order=2)
    y = lowpass(np.fabs(y), cutoff=10, fs=fs, order=6)

    # Downsample and remove DC component.
    fs_env = 100  # desired sampling frequency of the envelope
    ds_factor = fs//fs_env  # new rate will be 100 samples per second
    env = rem_dc(scipy.signal.decimate(y, ds_factor, ftype = 'fir', zero_phase=True))
    ds_t = (np.arange(len(env)) / fs_env) + (0.5/fs_env)  # time axes

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
 
