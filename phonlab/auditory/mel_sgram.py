import tensorflow as tf
import numpy as np

def Hz_to_mel(f):
    return 1127  * np.log(1.0 + f/700)

def mel_to_Hz(m):
    return 700 * (np.exp(m / 1127) - 1.0)


def compute_mel_sgram(x,fs, s=0.01):
    """Compute a Mel frequenc spectrogram of the signal in **x**.  This function is 
slightly adapted from the code example given in the documentation for the `tensorflow`
function `mfccs_from_log_mel_spectrograms()`.

https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms

Parameters
==========

    x: ndarray
        A one-dimensional array of audio samples
    fs: int
        The sampling rate of the audio samples in **x**.  The `tensorflow` example
        assumed that fs=16000
    s: float, default = 0.01
        The step size between successive spectral slices.  The `tensorflow` example
        used t=0.016, 16 milliseconds.

Returns
=======
    mel_f : ndarray
        a one dimensional array of mel frequency values - the frequency axis of the spectrogram
    sec : ndarray
        a one dimensional array of time values, the time axis of the spectrogram
    mel_sgram: ndarray
        A two-dimensional (time,frequency) array of amplitufe values.  The intervals between 
        time slices is dependent on the **s** input parameter, by default 10 ms, and the 
        frequencies are evenly spaced on the mel scale from 80 to 7600 Hz in 80 steps.

Example
=======
This example uses the function to compute a log mel-frequency spectrogram, and then passes
that to the tensor flow function to compute mel-frequency cepstral coefficients from it.

.. code-block:: Python

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mel_f, sec, mel_sgram = phon.compute_mel_sgram(x,fs)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_sgram)[..., :13]

.. figure:: images/mel_sgram.png
    :scale: 40 %
    :alt: a mel-frequency spectrogram
    :align: center

    The mel_sgram of the example audio file sf3_cln.wav - "cottage cheese with 
    chives is delicious"

    """
    frame_length_sec = 0.064
    step_sec = s
    fft_pow = 10

    frame_length = int(frame_length_sec*fs)
    step = int(step_sec*fs)
    fft_length = int(2**fft_pow)
    while fft_length < frame_length:
        fft_pow = fft_pow+1
        fft_length = 2**fft_pow

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(x, frame_length=frame_length, 
                           frame_step=step,
                           fft_length=fft_length)
    sgram = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_freq_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    warping_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, 
                    num_freq_bins, fs, lower_edge_hertz, upper_edge_hertz)
    mel_sgram = tf.tensordot(sgram, warping_matrix, 1)
    mel_sgram.set_shape(sgram.shape[:-1].concatenate(warping_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_sgram = tf.math.log(mel_sgram + 1e-6).numpy()

    start,stop = Hz_to_mel(np.array([lower_edge_hertz, upper_edge_hertz]))
    mel_f = np.linspace(start,stop,num_mel_bins)

    sec = (np.array(range(log_mel_sgram.shape[0])) * step_sec + frame_length_sec/2)
    
    return mel_f, sec, log_mel_sgram


