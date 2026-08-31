
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
        add a tiny bit of noise to the audio to avoid problematic waveforms with many samples at zero amplitude.

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
        x2 = x
    else:  # resample to 'target_fs' samples per second
        if target_fs==fs:
            x2 = x
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
    if add_tiny_noise:  y = y + (((np.random.rand(len(y)) - 0.5) * 0.00001).astype(y.dtype))
    if pad_to > 0: 
        # check whether any extra are needed
        extra_time = pad_to - ((len(y)/target_fs) % pad_to)
        extra_samples = int(extra_time * target_fs)
        y = np.concatenate((y,((np.random.rand(extra_samples) - 0.5) * 0.00001).astype(y.dtype))
        if not quiet:
            print(f"Prep Audio: Padding signal to {pad_to} sec, which involves adding {extra_samples} extra samples.")
    if outtype == "int":  y = np.rint(np.iinfo(np.int16).max * y).astype(np.int16)
    if outtype == "int16":  y = np.rint(np.iinfo(np.int16).max * y).astype(np.int16)

    return y,target_fs
