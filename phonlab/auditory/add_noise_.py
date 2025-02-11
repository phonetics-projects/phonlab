__all__ = ["peak_rms", "add_noise"]

import numpy as np
import matplotlib.pyplot as plt
import random
from ..utils.get_signal_ import get_signal


import colorednoise as cn  # installed from: https://github.com/felixpatzelt/colorednoise
# pip install colorednoise

import librosa


def peak_rms(y):
    """Return the peak rms amplitude

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

def add_noise(sig,noise_type, fs_in = 22050, fs = 22050, chan = 0, snr = 0, target_amp = -2):
    """Add noise to audio

    This function is partially adapted from matlab code written by Kamil Wojcicki, UTD, July 2011. It does the following:

    * pads the audio signal with 1/2 second of silence at the beginning and end
    * takes an audio file and mixes it with a noise (or a passed audio file) at a specified signal to noise ratio.
    * scales the peak intensity of the resulting mixed audio to prevent clipping
    * writes the resulting mixed audio as .wav files to an output directory


Parameters
==========
sig : Path or array
        the name of a .wav file, or an array of audio samples
        
noise_type : string
        The type of noise - one of "pink", "white", or "brown", or the name of a .wav file to be mixed with the signal_file.

fs_in : int, default = 22050
    If **sig** is an array, this is the sampling frequency of the audio in **sig**

fs : int, default = 22050
    The desired sampling frequency of the output signal

chan : int, default = 0 
    if the signal is stereo, select a channel for processing (0 = left, 1 = right)
    
snr : float, default = 0
        the signal to noise ratio in dB.  0 means that the signal peak RMS amplitude will be the same as the noise amplitude. Less than zero (e.g. -5) means that the signal amplitude will be lower than the noise, and greater than zero means that the signal amplitude will be greater than the noise amplitude.
        
target_amp : number, default = -2
        Scale the resulting signal (the result of adding the noise to the signal) so that the peak amplitude is target_amp relative to the maximum possible value.  Use a negative number to avoid clipping.  -2 means scale the resulting signal so that it is -2 dB below the maximum for digital audio files.

Returns
=======
    y : ndarray
        The result of adding noise to the signal
    fs : int
        The sampling rate of the signal in 'mixed'

Raises
======
    ValueError 
        if the noise_type is not a valid type


Example
=======
This example adds white noise at a signal-to-noise ratio (SNR) of 3 dB

>>> x,fs = phon.add_noise("sf3_cln.wav","white",snr=3)
>>> phon.sgram(x,fs_in=fs)

.. figure:: images/add_noise.png
   :scale: 90 %
   :alt: a spectrogram a speech sample buried in white noise
   :align: center

   The result of adding white noise.

   ..


    """

    x, fs = get_signal(sig,chan = chan, fs = fs, fs_in = fs_in, pre=0, quiet = True)

    signal_peak = peak_rms(x)
    
    pad = np.zeros(int(fs/2))  # number of points in 1/2 a second
    x = np.append(np.append(pad,x),pad) #add 500 ms of silence before/after signal, 
            # the stimulus will begin 500 ms after the onset of the noise after

    if not ".wav" in noise_type:  # not a stored sample of noise
        if (noise_type == 'pink'):
            beta = 1 # the exponent for pink noise
        elif (noise_type == 'white'):
            beta = 0 
        elif (noise_type == 'brown'):
            beta = 2
        else: 
            raise ValueError(f'\"{noise_type}\" is not a valid noise type')

        noise_rate = fs  #sampling rate of the signal 
        noise = cn.powerlaw_psd_gaussian(beta, len(x))  #generate the noise samples
 
    else:  # noise is a stored sample of background noise
        try:
            noise, noise_rate = librosa.load(noise_type)  # read the noise again each time the function is called
        except OSError:
            print('cannot open', noise_type)
        
        #get length of signal and noise files, return error and terminate if noise file is shorter than signal
        s = len( x )
        n = len( noise )
    
        while ( s>n ):  # signal is longer than noise
            noise = np.concatenate([noise,noise])  # rude way to grow the noise sample
            n= len(noise)
    
        # generate a random start location in the noise signal to extract a random section of it 
        r = random.randint(1,1+n-s)
        noise = noise[r:r+s]
        
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