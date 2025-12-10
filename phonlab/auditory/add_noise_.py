import numpy as np
import matplotlib.pyplot as plt
import random
import colorednoise as cn
import librosa
from importlib.resources import files as res_files
from ..utils.prep_audio_ import prep_audio

def peak_rms(y):
    """
Return the peak rms amplitude

The function uses the librosa.feature.rms to calculate an RMS contour from short time Fourier transforms taken from windows of 2048 samples with a step of 512 samples (the librosa.stft defaults).  This makes for different window lengths (in terms of seconds) depending on the sampling rate.  

Parameters
==========
    y : ndarray
        a one-dimensional array of audio waveform samples

Returns
=======
    float
        the maximum rms value in y

    """
    
    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S = S)

    return np.max(rms)

def add_noise(x, fs, noise_type="white", snr = 0, target_amp = -2):
    """
Add noise to audio

This function is partially adapted from matlab code written by Kamil Wojcicki, UTD, July 2011. It does the following:

    * pads the audio signal with 1/2 second of silence at the beginning and end
    * takes an audio file and mixes it with a noise (or a passed audio file) at a specified signal to noise ratio.
    * scales the peak intensity of the resulting mixed audio to prevent clipping
    * writes the resulting mixed audio as .wav files to an output directory


Parameters
==========
x : array
    A one-dimensional array of audio samples
fs : int
    the sampling frequency of the audio in **x**
   
noise_type : string, default = "white"
        The type of noise - one of "pink", "white", "brown", 'babble', 'party', or 'restaurant'.
        
snr : float, default = 0
        the signal to noise ratio in dB.  0 means that the signal peak RMS amplitude will be the same as the noise amplitude. Less than zero (e.g. -5) means that the signal amplitude will be lower than the noise, and greater than zero means that the signal amplitude will be greater than the noise amplitude.
        
target_amp : number, default = -2
        Scale the resulting signal (the result of adding the noise to the signal) so that the peak amplitude is target_amp relative to the maximum possible value.  Use a negative number to avoid clipping.  -2 means scale the resulting signal so that it is -2 dB below the maximum for digital audio files.

Returns
=======
    y : ndarray
        The result of adding noise to the signal
    fs : int
        The sampling rate of the signal in **y**

Raises
======
    ValueError 
        if the noise_type is not a valid type


Example
=======
This example adds white noise at a signal-to-noise ratio (SNR) of 3 dB

.. code-block:: Python

     x,fs = phon.loadsig("sf3_cln.wav",chansel=[0])
     y,fs = phon.add_noise(x,fs,"white",snr=3)
     phon.sgram(x,fs)

.. figure:: images/add_noise.png
   :scale: 90 %
   :alt: a spectrogram a speech sample buried in white noise
   :align: center

   The result of adding white noise.

    """
    # Valid options that can be passed to the `sox` `synth` effect.
    colored_noise = (
        'brown', 'pink', 'white'
    )
    # Names of files in the package data/noise directory. 
    pkg_noise = (
        'babble', 'party', 'restaurant'
    )

    signal_peak = peak_rms(x)
    
    pad = np.zeros(int(fs/2))  # number of points in 1/2 a second
    x = np.append(np.append(pad,x),pad) #add 500 ms of silence before/after signal, 
            # the stimulus will begin 500 ms after the onset of the noise after

    if noise_type in colored_noise:
        if (noise_type == 'pink'):
            beta = 1 # the exponent for pink noise
        elif (noise_type == 'white'):
            beta = 0 
        elif (noise_type == 'brown'):
            beta = 2
        noise_rate = fs  #sampling rate of the signal 
        noise = cn.powerlaw_psd_gaussian(beta, len(x))  #generate the noise samples
 
    elif noise_type in pkg_noise:  # noise is an audiofile
        noise_file = res_files('phonlab') / 'data' / 'noise' / f'{noise_type}.wav'
        noise, noise_rate = librosa.load(noise_file, sr = fs)  # resample to the rate of the signal
        
        #get length of signal and noise files
        s = len( x )
        n = len( noise )
    
        while ( s > n ):  # noise must be longer than signal
            noise = np.concatenate([noise,noise])  # rude way to grow the noise sample by doubling
            n= len(noise)
    
        # generate a random start location in the noise signal to extract a random section of it 
        r = random.randint(1,1+n-s)
        noise = noise[r:r+s]
    else:
        print(f"{noise_type} must be one of 'pink', 'white', 'brown', 'babble', 'party', or 'restaurant'")
        exit()
        
    noise_peak = peak_rms(noise)

    # scale the noise file w.r.t. to target at desired SNR level (arrays must be the same length)
    noise = noise / noise_peak * signal_peak / np.power(10.0, snr/20) # peak amp
    # or noise = noise / np.linalg.norm(noise) * np.linalg.norm(signal) / np.power(10.0,snr/20)  # whole file (Wojcicki)

    # mix the noise and audio files 
    mixed_audio = x + noise
    
    # calculate the gain needed to scale to the desired peak RMS level (-3dB usually, below max)
    current_peak = np.max(np.abs(mixed_audio))
    gain = np.power(10.0, target_amp/20.0) / current_peak
    
    return gain * mixed_audio, fs
