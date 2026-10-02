
import warnings

import numpy as np
from scipy.signal import resample_poly

# peak amplitude (as a fraction of full scale, 1.0) that scale=True normalizes to: -1 dBFS
DEFAULT_PEAK = 0.89125


def prep_audio(x, fs, target_fs=32000, pre = 0, scale = True, 
               add_tiny_noise = True, outtype = "float", pad_to = 0.0,
               fix_polarity = False, quiet = True):
    """ Prepare an array of audio waveform samples for acoustic analysis. 
    
Parameters
==========
    x : array
        a one-dimensional numpy array with audio samples in it. 

    fs : int
          The sampling rate of the sound in **x**.
   
    target_fs : int, default=32000
        The desired sampling rate of the audio samples that will be returned by the function.  
        Set **target_fs = None** if you don't want to change the sampling rate.

    pre : float, default = 0
        how much high frequency preemphasis to apply (between 0 and 1).

    scale: boolean or float, default = True
        normalize the signal based on its absolute peak amplitude.
        **True** scales the peak to 0.89125 (-1 dBFS).
        **False** (or None) leaves the amplitude as it is.
        A number scales the peak to that many dB relative to full scale (dBFS), like sox's `gain -n`.
        For example, **scale = 0** puts the peak at full scale (amplitude 1.0) and **scale = -3** puts
        it at about 0.71. Full scale is an amplitude of 1.0 for float samples, and 32767 for 16 bit integers.
        Values above 0 produce samples outside of [-1, 1], which will be clipped (with a warning) if
        **outtype** is "int".

    add_tiny_noise: boolean, default = True
        replace any exact-zero samples (e.g. digital silence, or the samples added by `pad_to`) with a tiny
        bit of random noise, to avoid problematic waveforms with runs of zero amplitude. Samples that are
        already nonzero are left untouched, so real recordings are unaffected -- this only matters for
        synthetic or padded silence.

    pad_to: float, default = 0.0
        add samples so duration is a multiple of `pad_to`. For example, if the duration is 1.99 seconds 
        and `pad_to` is 0.1 then the signal will be padded to 2.0 seconds

    fix_polarity: boolean, default = False
        Apply a heuristic to ensure that positive pressure in an acoustic waveform is represented as a positive value.

    outtype : string {"float", "int"), default = "float"
        The "int" waveform is 16 bit integers - in the range from [-32768, 32767].
        The "float" waveform is 32 bit floating point numbers - in the range from [-1, 1].
        When converting to integers, samples outside of [-1, 1] are clipped and a warning is issued.


Returns
=======
    y : ndarray
        a 1D numpy array with audio samples 
    
    fs : int
        the sampling rate of the audio in **y**.

Note
====
By default, this function will return audio with a sampling rate of 32 kHz and scaled so that the peak amplitude is -1 dBFS (0.89125).

Example
=======
Open a sound file and prepare it for acoustic analysis.  By default, prep_audio() will 
resample the audio to a sampling rate of 32000, and scale the waveform to use the full range.
In this example, we have also asked the function to apply a preemphasis factor of 1 (about 6dB/octave).

.. code-block:: Python

    y,fs = phon.loadsig("sound.wav",chansel=[0])
    x,fs = phon.prep_audio(y, fs, pre=1)

Take the right channel, and resample to 16,000 Hz

.. code-block:: Python

    *chans,fs = phon.loadsig("sound.wav")
    print(f'the old sampling rate is: {fs}')
    y,fs = phon.prep_audio(chans[1],fs, target_fs=16000)
    print(f'the new sampling rate is: {fs}')

    """
    
    if target_fs == None:
        target_fs = fs
        
    if target_fs==fs:
        x2 = np.array(x, copy=True)  # always copy, so callers never get back a view of their input buffer
    else:
        if not quiet: 
            print(f'Prep Audio: Resampling from {fs} to {target_fs}')
        cd = np.gcd(fs,target_fs)   # common denominator   
        x2 = resample_poly(x,up=target_fs/cd, down=fs/cd)
            
    if fix_polarity:
        if (np.max(x2) + np.min(x2)) < 0:  x2 = -x2   #  set the polarity of the signal
        
    if scale is not None:
        if isinstance(scale, (bool, np.bool_)):   # scale=True or scale=False
            target_peak = DEFAULT_PEAK if scale else None
        else:                                     # scale is a number of dBFS
            target_peak = 10 ** (float(scale) / 20)
        if target_peak is not None:
            peak = np.max(np.abs(x2))
            if peak > 0:
                x2 = x2 / peak * target_peak
            # else: x2 is all zeros (a fully silent signal) -- dividing by peak here would be a
            # divide by zero, producing nan/inf that then propagates through everything downstream.
            # Leave x2 as is.

    if pad_to > 0:
        # Pad to an exact multiple of the frame length in *samples*, not just in time -- when
        # pad_to*target_fs isn't a whole number (e.g. 0.05 sec at 22050 Hz = 1102.5 samples),
        # padding to a time boundary can still leave a sample-count remainder for anything that
        # reshapes the output into frames of round(pad_to*target_fs) samples.
        frame_len = max(1, round(pad_to * target_fs))
        remainder = len(x2) % frame_len
        extra_samples = 0 if remainder == 0 else frame_len - remainder
        if extra_samples > 0:
            x2 = np.concatenate((x2, np.zeros(extra_samples, dtype=x2.dtype)))
        if not quiet:
            print(f"Prep Audio: Padding signal to a multiple of {pad_to} sec ({frame_len} samples), which involves adding {extra_samples} extra samples.")

    if (pre > 0):
        y = np.append(x2[0], x2[1:] - pre * x2[:-1])  # apply pre-emphasis
    else:
        y = x2

    if add_tiny_noise:
        # only exact-zero samples get jittered (digital silence, or the padding above) -- real
        # recordings essentially never contain literal zeros, so this leaves them bit-identical
        # across repeated calls instead of dithering every sample.
        zero_mask = (y == 0)
        n_zero = int(np.count_nonzero(zero_mask))
        if n_zero > 0:
            y[zero_mask] = (((np.random.rand(n_zero) - 0.5) * 0.00001).astype(y.dtype))
    
    if outtype == "int" or outtype == "int16":
        peak_out = np.max(np.abs(y)) if len(y) > 0 else 0.0
        if peak_out > 1.0:
            n_clipped = int(np.count_nonzero(np.abs(y) > 1.0))
            warnings.warn(
                f"prep_audio: {n_clipped} samples ({100 * n_clipped / len(y):.3g}%) exceed full scale "
                f"(peak = {20 * np.log10(peak_out):.2f} dBFS) and will be clipped when converting to "
                f"integers. Use a lower scale value or reduce the pre-emphasis to avoid clipping.",
                UserWarning, stacklevel=2)
            y = np.clip(y, -1.0, 1.0)
        y = np.rint(np.iinfo(np.int16).max * y).astype(np.int16)

    return y,target_fs
