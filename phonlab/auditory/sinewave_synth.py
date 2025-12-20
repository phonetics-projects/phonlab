import numpy as np

def sine_synth(formant_data,fs=16000):
    """
Produces 'sinewave speech' - an audio signal made up of time-varying sinusoidal waves at the frequencies of the vowel formants. 

Note
====
The input dataframe is usually produced by :func:`phonlab.track_formants` and in any case should have these columns:

            * sec : Time (seconds) of the frames. 
            * rms :  The RMS amplitude of each frame.
            * F1,F2,F3,F4:  Frequencies (Hz) of the lowest four vowel formants, in the frames. 
       

The overall amplitude contour of the sine wave speech waveform is determined by the RMS contour in the input file. The frequencies of four sine wave components are given by the formant estimates in the input file.

The amplitude of the sine wave speech waveform is scaled to use the full amplitude range available with 16 bit integer samples.  Also, short 20ms on-ramp and off-ramp amplitude contours are applied to the beginning and end of the audio.

This is a python translation of code that Keith Johnson got from Howard Nusbaum, via Alexander Francis, in 1998.
 
Parameters
==========
    formant_data :  dataframe
        a pandas dataframe with speech analysis data as produced by phonlab.track_formants()

    fs : number, default=16000
        the sampling frequency of the resulting sound wave.
       
Returns
=======
        
    wav : ndarray
        a one-dimensional numpy array containing audio samples
    fs : number
        the sampling frequency of the audio samples in wav


References
==========

R. E. Remez, P.E. Rubin, D.B. Pisoni & T.D. Carrell (1981) Speech perception without traditional speech cues. `Science` **212** (4497), 947–950. doi:10.1126/science.7233191


Example
=======
.. code-block:: Python

     x,fs = phon.loadsig("sf3_cln.wav",chansel=[0]) 
     fmtsdf = phon.track_formants(x,fs)    # track the formants
     x,fs = phon.sine_synth(fmtsdf,fs=fs)     # produce the sinewave synthesis
     librosa.output.write_wav('sf3_cln_sinewave.wav', x, fs)  # save wav file

    """
    pifac = np.pi*2/fs
    
    step=formant_data.sec[1]-formant_data.sec[0]  # read step size from input
    nframes = len(formant_data)  # read file length from input
    npoints = int(np.round(step*fs))
    
    wav = np.zeros(npoints*nframes)  # allocate a waveform buffer

    formant_data.interpolate(inplace=True,limit_direction='both') # no NaNs
    rms = formant_data.rms
    rms = (rms-np.min(rms))/(np.max(rms)-np.min(rms))
    formants = np.array((formant_data.F1,
                         formant_data.F2, 
                         formant_data.F3,
                         formant_data.F4))
    
    # TODO:  check that formant frequencies do not exceed fs/2
    #         if they do, consider increasing fs
        
    for f in range(formants.shape[0]):   # synthesize each formant
        rfreq = 0.0
        iwv = 0
        for y in range(1,nframes):      # synthesize frame by frame
            amp = rms[y-1]
            freq = formants[f,y-1]*pifac
            ainc = (rms[y]-rms[y-1])/npoints
            finc = ((formants[f,y]-formants[f,y-1])*pifac)/npoints
            
            for i in range(npoints):    # synthesize each point in the frame
                rfreq += freq
                #if (rfreq > 2*np.pi):  # is this "if" really necessary?
                #    rfreq -= 2*np.pi
                wav[iwv] += np.sin(rfreq)*amp
                amp += ainc
                freq += finc
                iwv += 1
    
    # add rise time and decay time
    ns = int(0.02 * fs)  # number of samples in 20ms
    fac = 0
    facinc = 1.0/ns  # ramp from 0 to 1
    for x in range(ns):   
        wav[x] *= fac    # apply a short (20ms) rise time
        wav[-x] *= fac  # and a short (20ms) decay time
        fac += facinc
    
    # scale 
    wav /= np.max(np.abs(wav))
           
    return wav,fs 
