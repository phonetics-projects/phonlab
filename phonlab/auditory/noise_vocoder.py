import numpy as np
import scipy
from ..utils.prep_audio_ import prep_audio
from ..acoustic.amp_env import amplitude_envelope


def design_filter(bounds, fs, order=4):
    ''' Use scipy.signal.butter() to design a Butterworth bandpass filter.

        inputs:
            -- bounds: the lower bound, upper bound of a bandpass filter
            -- fs: sampling frequency of the audio
        
        output:
            -- sos filter coefficients of the bandpass filter

    '''
    return scipy.signal.butter(order, bounds, fs=fs, btype='bandpass', output='sos')

def third_octave_bounds(cf):
    ''' Return the upper and lower frequency bounds for a 
        1/3 octave filter around a center frequency.
        
        input:
            -- cf: a center frequency (Hz)
    
        output:
            -- an array of two values, the upper and lower bound of the filter
    '''
    third_octave_ratio = 2 ** (1/6)  # 2 * 1/6 = 1/3
    return np.array([cf / third_octave_ratio, cf * third_octave_ratio])

def octave_spaced_frequencies(low, high):
    ''' Get octave spaced-center frequencies between low and high.  The first 
    is equal to 'low' and the remainder are twice the previous one on the list.
    
        inputs:
            -- low: the lowest center frequency
            -- high: the upper bound for center frequencies
        output:
            -- an array of center frequencies
    '''
    number_of_octaves = np.log2(high) - np.log2(low)
    return low * 2 ** np.arange(number_of_octaves)

def third_octave_bands(low = 100, high=5000):
    """Compute cutoff frequencies for 1/3 octave bands, the centers are spaced
    one octave apart

    Parameters
    ==========

    low : int, default = 100
        low frequency of the range to be covered by the channels (Hz)
    high : int, default = 5000
        high freq of range covered by the channels (at most fs/2 - 1)
        
    Returns
    =======
    bands : list 
        a list of band limits (low,high) for a bank of bandpass filters
    """

    center_frequencies = octave_spaced_frequencies(low, high)
    return[third_octave_bounds(cf) for cf in center_frequencies]

def shannon_bands(nc = 24, low = 70, high = 5000):
    """
    Split a frequency range (low, high) into nc (number of channels) frequency bands on log freq scale

    Parameters
    ==========
    nc : int, default = 24
        the number of frequency bands in the vocoded output
    low : int, default = 70
        low frequency of the range to be covered by the channels (Hz)
    high : int
        high freq of range covered by the channels (at most fs/2 - 1)
        
    Returns
    =======
    bands : list
        a list of band limits for a bank of bandpass filters
    """
    
    freqs = np.exp(np.linspace(np.log(low), np.log(high-1), num=nc+1))
    return [(freqs[i],freqs[i+1]) for i in range(nc)]


def apply_filterbank(x, bands, fs = 22050, order = 8):
    """ 
Apply a bank of bandpass filters
    
Filter using repeated calls to scipy.signal.sosfiltfilt(), once for each of a bank of 8th order Butterworth bandpass filters.  The output array has one copy of x for each of the bands listed in the input "bands" parameter.
        
Parameters
==========
    x : ndarray
        audio samples in a 1-D numpy array
    bands : list
         A list of c filter lower/upper cut off freq pairs (c,2)
    fs : int, default = 22050
        the sampling frequency of the sound to be filtered
    order : int, default=8
        the order of the bandpass filters (passed to scipy.signal.butter)
                
Returns
=======
    y : ndarray, shape(c,n)
        a 2-D array of x filtered by each band, y[0] is the input filtered by bands[0]
    """

    # if bands is not specified, complain
    
    n_samples = x.shape[-1]
    n_channels = len(bands)
    
    # this line makes coefficients for the filter bank
    filterbank = [design_filter(b, fs,order) for b in bands]
    y = np.zeros((n_channels, n_samples))
    for idx, coefs in enumerate(filterbank):
        y[idx] = scipy.signal.sosfiltfilt(coefs, x)  # filter each band
    return y

def vocode(x, fs, bands, target_fs = None):
    """ 
Noise vocoding - replace sound with bandpassed noise, using a bank of filters defined by bands.

This module is based on the excellent vocoder notebook published by Alexandre Chabot-Leclerc ([@AlexChabotL](http://twitter.com/alexchabotl)).   https://github.com/achabotl/vocoder.git

Keith Johnson added the "Shannon" vocoding scheme and altered some of the functions to be a little more general, and also converted the filtering to use sos coefficients, which improved the numerical stability of the code. 
        
Parameters
==========
x : ndarray
    A one-dimensional array of audio samples
fs : int
    the sampling frequency of the audio samples in **x**.
bands : list
    A list of n bandpass filter lower/upper cut off freqs (n,2), as returned by third_octive_bands() or shannon_bands()
target_fs : int, default None
    the desired frequency of the resulting vocoded signal.  **None** keeps the rate at the value passed in **fs**.
        
Returns
=======
y : ndarray
    an array of samples, the same length as **sig**
fs : int
    the sampling frequency of **y**

Example
=======

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'

    bands_third = phon.third_octave_bands(high=8000)  # define filter bank
    bands_shan = phon.shannon_bands(high=8000,nc=10)  # define filter bank

    x,fs = phon.loadsig(example_file,chansel=[0])   
    y,fs = phon.vocode(x, fs, bands_shan)  # use one of the filter banks

    phon.sgram(y,fs)

.. figure:: images/vocode.png
    :scale: 90 %
    :alt: a spectrogram of Shannon vocoded speech
    :align: center

    A spectrogram of Shannon vocoded speech


    """
    y, fs = prep_audio(x, fs, target_fs = target_fs, pre=0, quiet = True)

    n_channels = len(bands)
    n_samples = y.shape[-1]
    noise = np.random.randn(n_samples) 

    filtered_y = apply_filterbank(y, bands, fs)
    filtered_noise = apply_filterbank(noise, bands, fs)
    vocoded_noise = np.zeros((n_channels, n_samples))
    
    for idx, (x_band, noise_band) in enumerate(zip(filtered_y, filtered_noise)):
        envelope,fs = amplitude_envelope(x_band,fs,target_fs=None)
        vocoded_noise[idx] = envelope * noise_band
    return np.sum(vocoded_noise, axis=0), fs


    
