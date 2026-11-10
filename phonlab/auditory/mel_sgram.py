#import tensorflow as tf
from librosa.util import frame
import scipy.signal as signal
import numpy as np
from ..utils.deprecation import renamed_kwargs

def mel_to_Hz(m):
    """Converts frequencies in `m` from the mel scale to linear scale using the following formula::

    Hz(m) = 700 * (exp(m/1127) - 1)

  Args:
    m: An array of frequencies in the mel scale.

  Returns:
    A numpy array of the same shape and type as `m` containing linear
    scale frequencies in Hertz.
    """
    return 700 * (np.exp(m / 1127) - 1.0)

def Hz_to_mel(f):
    """Converts frequencies in `f` in Hertz to the mel scale with the following formula::
    
    mel(f) = 1127  * log(1.0 + f/700)

  Args:
    f: An array of frequencies in Hertz.

  Returns:
    A numpy array of the same shape and type of `f` containing
    frequencies in the mel scale.
    """
    return 1127  * np.log(1.0 + f/700)

# an adapted verison of the tensor-flow linear_to_mel_weight_matrix function

def _validate_arguments(num_mel_bins, sample_rate,
                        lower_edge_hertz, upper_edge_hertz):
    """Checks the inputs to linear_to_mel_weight_matrix."""
    if num_mel_bins <= 0:
        raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
    if lower_edge_hertz < 0.0:
        raise ValueError('lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' % (lower_edge_hertz, upper_edge_hertz))
    if sample_rate <= 0.0:
      raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
    if upper_edge_hertz > sample_rate / 2:
      raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                       'frequency (sample_rate / 2). Got %s for sample_rate: %s'
                       % (upper_edge_hertz, sample_rate))

def linear_to_mel_weight_matrix(num_mel_bins=80,
                                num_spectrogram_bins=1024,
                                sample_rate=16000,
                                lower_edge_hertz=80.0,
                                upper_edge_hertz=7600.0):
    """Returns a matrix to warp linear scale spectrograms to the `mel scale <https://en.wikipedia.org/wiki/Mel_scale>`_.  This is code adapted from the TensorFlow package.
    
Returns a weight matrix that can be used to re-weight a matrix containing `num_spectrogram_bins` linearly sampled frequency information from `[0, sample_rate / 2]` into `num_mel_bins` frequency information from `[lower_edge_hertz, upper_edge_hertz]` on the mel scale.

In the returned matrix, all the triangles (filterbanks) have a theoretical peak value of 1.0 when the center frequency of the mel band matches a spectrogram bin center frequency.

For example, the returned matrix `A` can be used to right-multiply a spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear scale spectrum values (e.g. STFT magnitudes) to generate a "mel spectrogram" `M` of shape `[frames, num_mel_bins]`.

      - `S` has shape [frames, num_spectrogram_bins]
      - `M` has shape [frames, num_mel_bins]
      - M = tf.matmul(S, A)

The matrix can be used with `numpy.dot` to convert a spectrogram of linear-scale spectral bins into a filtered spectrogram on the mel scale.

      - S has shape [..., num_spectrogram_bins].
      - M has shape [..., num_mel_bins].
      - M = np.dot(S, A)

Parameters
==========
    num_mel_bins: int. 
        How many bands in the resulting mel spectrum.
    num_spectrogram_bins: int 
        How many bins there are in the source spectrogram data, which is understood to be `fft_size // 2 + 1`, i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: float. 
        Samples per second of the input signal used to create the spectrogram. Used to figure out the frequencies corresponding to each spectrogram bin, which dictates how they are mapped into the mel scale.
    lower_edge_hertz: float. 
        Lower bound on the frequencies to be included in the mel spectrum. This corresponds to the lower edge of the lowest triangular band.
    upper_edge_hertz: float. 
        The desired top edge of the highest frequency band.
      
Returns
=======
    A numpy array of shape `[num_spectrogram_bins, num_mel_bins]`.

Raises
======
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency.

    """

    _validate_arguments(num_mel_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)

    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(0.0, nyquist_hertz, 
                                        num=num_spectrogram_bins)
    spectrogram_bins_mel = Hz_to_mel(linear_frequencies)

    band_edges_mel = np.linspace(Hz_to_mel(lower_edge_hertz),
                          Hz_to_mel(upper_edge_hertz),
                          num_mel_bins + 2)
    bands = frame(band_edges_mel, frame_length=3, hop_length=1,axis=0)

    mel_weights_matrix = np.zeros((num_mel_bins,num_spectrogram_bins))
    for idx,band in enumerate(bands):
        lower_slopes = (spectrogram_bins_mel - band[0]) / (band[1]-band[0])
        upper_slopes = (band[2]-spectrogram_bins_mel) / (band[2]-band[1])
        mel_weights_matrix[idx] = np.maximum(0.0,np.minimum(lower_slopes,upper_slopes))

    return(mel_weights_matrix.T)

@renamed_kwargs(step_sec='step')
def compute_mel_sgram(x,fs, step = 0.01):
    """Compute a Mel frequency spectrogram of the signal in **x**.  This function is adapted from the code example given in the documentation for the `tensorflow` function `mfccs_from_log_mel_spectrograms()`.

Parameters
==========

    x: ndarray
        A one-dimensional array of audio samples
    fs: int
        The sampling rate of the audio samples in **x**.  The `tensorflow` example
        assumed that fs=16000
    step: float, default = 0.01
        The step size between successive spectral slices.  The `tensorflow` example
        used t=0.016, 16 milliseconds.

Returns
=======
    sec : ndarray
        a one dimensional array of time values, the time axis of the spectrogram
    mel_f : ndarray
        a one dimensional array of mel frequency values - the frequency axis of the spectrogram
    mel_sgram: ndarray
        A two-dimensional (time,frequency) array of amplitude values.  The interval between 
        time slices is set by the **step** input parameter, by default 10 ms, and the 
        frequencies are evenly spaced on the mel scale from 80 to 7600 Hz in 80 steps.

Example
=======
This example uses the function to compute a log mel-frequency spectrogram, and then passes
that to the tensor flow function to compute mel-frequency cepstral coefficients from it.

.. code-block:: Python

    # Compute log_mel_spectrogram.
    sec, mel_f, mel_sgram = phon.compute_mel_sgram(x,fs,step=0.005)

    vmin = np.min(mel_sgram) + (np.max(mel_sgram)-np.min(mel_sgram))*0.2

    fig = plt.figure(figsize=(12, 4.5),dpi=72)
    ax1 = fig.add_subplot(111)

    extent = (min(sec),max(sec),min(mel_f),max(mel_f)) 
    ax1.imshow(mel_sgram.T, aspect='auto', interpolation='nearest', 
                    cmap='Grays', vmin=vmin,
                    extent = extent, origin='lower')
    ax1.grid(which='major', axis='y', linestyle=':')  
    ax1.set_xlabel("Time (sec)", size=14)
    ax1.set_ylabel("Frequency (mels)", size=14)
    plt.tick_params(labelsize=14)

.. figure:: images/mel_sgram.png
    :scale: 40 %
    :alt: a mel-frequency spectrogram
    :align: center

    The mel_sgram of the example audio file sf3_cln.wav - "cottage cheese with 
    chives is delicious"

    """
    frame_length_sec = 0.064
    fft_pow = 10

    frame_length = int(frame_length_sec*fs)
    hop = int(step*fs)
    fft_length = int(2**fft_pow)
    while fft_length < frame_length:
        fft_pow = fft_pow+1
        fft_length = 2**fft_pow
    noverlap = frame_length-hop

    f, t, Zxx = signal.stft(x, fs, window="hann", nperseg=fft_length,
                            noverlap=noverlap,padded=False)
   
    # Warp the linear scale spectra onto the mel-scale.
    num_freq_bins = len(f)
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    warping_matrix = linear_to_mel_weight_matrix(num_mel_bins, 
                    num_freq_bins, fs, lower_edge_hertz, upper_edge_hertz)
    
    mel_sgram = np.dot(abs(Zxx.T), warping_matrix)
    log_mel_sgram = np.log(mel_sgram + 1e-6)
    
    start,stop = Hz_to_mel(np.array([lower_edge_hertz, upper_edge_hertz]))
    mel_f = np.linspace(start,stop,num_mel_bins)
    
    return t, mel_f, log_mel_sgram
