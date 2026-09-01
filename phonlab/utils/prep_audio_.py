
import numpy as np
from scipy.signal import resample_poly


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

    scale: boolean, default = True
        scale the samples to use the full range for audio samples (based on the peak amplitude in the signal)

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


Returns
=======
    y : ndarray
        a 1D numpy array with audio samples 
    
    fs : int
        the sampling rate of the audio in **y**.

Note
====
By default, this function will return audio with a sampling rate of 32 kHz and scaled to be in the range from [1,-1]

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
        x2 = np.array(x, copy=True)  # always copy, so callers never get back a view of their input buffer
    else:  # resample to 'target_fs' samples per second
        if target_fs==fs:
            x2 = np.array(x, copy=True)  # always copy, so callers never get back a view of their input buffer
        else:
            if not quiet: 
                print(f'Prep Audio: Resampling from {fs} to {target_fs}')
            cd = np.gcd(fs,target_fs)   # common denominator   
            x2 = resample_poly(x,up=target_fs/cd, down=fs/cd)
        
    if fix_polarity:
        if (np.max(x2) + np.min(x2)) < 0:  x2 = -x2   #  set the polarity of the signal
    if (pre > 0): y = np.append(x2[0], x2[1:] - pre * x2[:-1])  # apply pre-emphasis
    else: y = x2
    if scale: y = y/np.max(y) * 0.9  # scale to about full range
    if pad_to > 0:
        # Pad to an exact multiple of the frame length in *samples*, not just in time -- when
        # pad_to*target_fs isn't a whole number (e.g. 0.05 sec at 22050 Hz = 1102.5 samples),
        # padding to a time boundary can still leave a sample-count remainder for anything that
        # reshapes the output into frames of round(pad_to*target_fs) samples.
        frame_len = max(1, round(pad_to * target_fs))
        remainder = len(y) % frame_len
        extra_samples = 0 if remainder == 0 else frame_len - remainder
        if extra_samples > 0:
            y = np.concatenate((y, np.zeros(extra_samples, dtype=y.dtype)))
        if not quiet:
            print(f"Prep Audio: Padding signal to a multiple of {pad_to} sec ({frame_len} samples), which involves adding {extra_samples} extra samples.")
    if add_tiny_noise:
        # only exact-zero samples get jittered (digital silence, or the padding above) -- real
        # recordings essentially never contain literal zeros, so this leaves them bit-identical
        # across repeated calls instead of dithering every sample.
        zero_mask = (y == 0)
        n_zero = int(np.count_nonzero(zero_mask))
        if n_zero > 0:
            y[zero_mask] = (((np.random.rand(n_zero) - 0.5) * 0.00001).astype(y.dtype))
    if outtype == "int":  y = np.rint(np.iinfo(np.int16).max * y).astype(np.int16)
    if outtype == "int16":  y = np.rint(np.iinfo(np.int16).max * y).astype(np.int16)

    return y,target_fs
