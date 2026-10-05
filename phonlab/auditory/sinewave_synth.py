import numpy as np
from ..acoustic.amp_env import amplitude_envelope
from ..acoustic.vad import VAD
from ..utils.prep_audio_ import prep_audio

def sine_synth(formant_data, x, fs, target_fs=16000, voiceless="copy",
               formant_amps=(0, -8, -12.5, -29)):
    """Produces 'sinewave speech' - an audio signal made up of time-varying sinusoidal waves at the frequencies of the vowel formants. 

Note
====
The input dataframe is usually produced by :func:`phonlab.track_formants` and in any case should have these columns:

            * sec : Time (seconds) of the frames. 
            * F1,F2,F3,F4:  Frequencies (Hz) of the lowest four vowel formants, in the frames. 
       

The frequencies of four sine wave components are given by the formant
estimates in the input file.  The relative amplitudes of the sine
waves (in dB, see `formant_amps`) simulate the amplitude relations
among the formants of natural speech, in which F1 is louder than F2,
F2 is louder than F3, and so on.  They are constants, not recalculated
frame by frame.  The sum of the sine waves is then multiplied by the
amplitude envelope of the original audio signal **x**, as calculated
by :func:`phonlab.amplitude_envelope`, so that the overall amplitude
contour of the sine wave speech follows that of the original speech.

The `voiceless` parameter controls what happens when there is no
voicing (fricatives, stop bursts, silence), as determined by
:func:`phonlab.VAD`.  By default (`voiceless = "copy"`) the original
audio is copied to the output in voiceless frames, and the sine wave
synthesis is used only in voiced frames.  The level of the sine waves
in voiced frames is set so that its RMS amplitude matches that of the
original audio in those frames, and the transitions between synthetic
and original audio are smoothed over about 3 ms.  Use `voiceless =
-48` to substantially reduce the amplitude of voiceless frames.

The output has the same duration as **x**, a sampling rate of
**target_fs**, and is scaled by :func:`phonlab.prep_audio` to its
default level (a peak amplitude of -1 dBFS, 0.89125).  Also, short
20ms on-ramp and off-ramp amplitude contours are applied to the
beginning and end of the audio.

Parameters
==========
    formant_data :  dataframe
        a pandas dataframe with speech analysis data as produced by phonlab.track_formants()

    x : ndarray
        the 1-D array of audio samples that was analyzed to produce formant_data. Its amplitude envelope is imposed on the synthesized signal.

    fs : number
        the sampling frequency of **x**.

    target_fs : number, default=16000
        the sampling frequency of the resulting sound wave.

    voiceless : string {"copy", "silent", "sinewave"} or number, default="copy"
        what to put in the voiceless frames:

        - **"copy"** - the original audio is copied into the output
        - **"silent"** - there is no audio in voiceless frames
        - **"sinewave"** - sinewave synthesis is used, just as in the voiced frames
        - **a number** - the original audio is copied, as in "copy", with its amplitude decreased or increased by this many dB.  For example, -6 reduces the amplitude of the voiceless frames by 6 dB.  "copy" is the same as 0.

    formant_amps : a sequence of four numbers, default=(0, -8, -12.5, -29)
        the amplitudes (in dB) of the sine waves at F1, F2, F3 and F4, relative to each other.

       
Returns
=======
        
    wav : ndarray
        a one-dimensional numpy array containing audio samples
    fs : number
        the sampling frequency of the audio samples in wav (the value of **target_fs**)


References
==========

R. E. Remez, P.E. Rubin, D.B. Pisoni & T.D. Carrell (1981) Speech perception without traditional speech cues. `Science` **212** (4497), 947–950. doi:10.1126/science.7233191


Example
=======
.. code-block:: Python

     x,fs = phon.loadsig("sf3_cln.wav",chansel=[0]) 
     fmtsdf = phon.track_formants(x,fs)    # track the formants
     y,fs = phon.sine_synth(fmtsdf,x,fs)   # produce the sinewave synthesis
     scipy.io.wavfile.write('sf3_cln_sinewave.wav', fs, y)  # save wav file

    """
    pifac = np.pi*2/target_fs
    
    step=formant_data.sec[1]-formant_data.sec[0]  # read step size from input
    nframes = len(formant_data)  # read file length from input
    npoints = int(np.round(step*target_fs))
    
    wav = np.zeros(npoints*nframes)  # allocate a waveform buffer

    formant_data.interpolate(inplace=True,limit_direction='both') # no NaNs
    formants = np.array((formant_data.F1,
                         formant_data.F2, 
                         formant_data.F3,
                         formant_data.F4))
    
    # relative amplitudes of the sine waves
    amps = 10**(np.asarray(formant_amps, dtype=float)/20)
    
    for f in range(formants.shape[0]):   # synthesize each formant
        rfreq = 0.0
        iwv = 0
        for y in range(1,nframes):      # synthesize frame by frame
            freq = formants[f,y-1]*pifac
            finc = ((formants[f,y]-formants[f,y-1])*pifac)/npoints
            
            for i in range(npoints):    # synthesize each point in the frame
                rfreq += freq
                wav[iwv] += np.sin(rfreq)*amps[f]
                freq += finc
                iwv += 1
    
    # impose the amplitude envelope of the original signal
    env, _ = amplitude_envelope(x, fs, target_fs=target_fs)
    out = np.zeros(len(env))
    start = int(np.round(formant_data.sec.iloc[0]*target_fs))  # time of the first analysis frame
    n = max(0, min(len(wav), len(out)-start))
    out[start:start+n] = wav[:n]
    wav = out * env
    
    # add rise time and decay time
    ns = int(0.02 * target_fs)  # number of samples in 20ms
    ramp = np.linspace(0, 1, ns)
    wav[:ns] *= ramp         # apply a short (20ms) rise time
    wav[-ns:] *= ramp[::-1]  # and a short (20ms) decay time
    
    if isinstance(voiceless, (int, float, np.number)) and not isinstance(voiceless, bool):
        voiceless_gain = 10**(voiceless/20)   # volume control (dB) for the copied voiceless audio
        voiceless = "copy"
    elif voiceless == "copy":
        voiceless_gain = 1.0
    elif voiceless not in ("silent", "sinewave"):
        raise ValueError('voiceless must be "copy", "silent", "sinewave" or a number (dB)')
    
    if voiceless != "sinewave":
        # voicing decision for each sample, from the nearest VAD frame
        vad = VAD(x, fs)
        vstep = vad.sec.iloc[1] - vad.sec.iloc[0]
        t = np.arange(len(wav))/target_fs
        idx = np.clip(np.round((t - vad.sec.iloc[0])/vstep).astype(int), 0, len(vad)-1)
        voiced = vad.voiced.to_numpy()[idx].astype(float)
        nsm = int(0.003 * target_fs)  # smooth the transitions (about 3 ms) to avoid clicks
        voiced = np.convolve(voiced, np.hanning(nsm + 2)[1:-1]/np.sum(np.hanning(nsm + 2)[1:-1]), mode='same')
        
        if voiceless == "silent":
            wav = voiced*wav
        else:
            # the original audio at the output sampling rate
            orig, _ = prep_audio(x, fs, target_fs=target_fs, scale=False, add_tiny_noise=False)
            orig = np.pad(orig, (0, max(0, len(wav)-len(orig))))[:len(wav)]
            
            # match the level of the synthetic and original audio in the voiced frames
            core = voiced > 0.99
            if np.any(core):
                wav *= np.sqrt(np.mean(orig[core]**2)/np.mean(wav[core]**2))
            wav = voiced*wav + (1 - voiced)*orig*voiceless_gain
    
    # scale to the default level, leaving the sampling rate and the silent samples alone
    wav, fs = prep_audio(wav, target_fs, target_fs=None, add_tiny_noise=False)
           
    return wav,fs
