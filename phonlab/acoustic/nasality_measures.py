import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.prep_audio_ import prep_audio
from ..acoustic.sgram_ import compute_sgram
from ..acoustic.rlpc import RLPC_formants
from ..acoustic.cepstral import cepstral_smooth
from ..third_party.robustsmoothing import smoothn

def nasality(x,fs, order=-1, analysis_band=(200,2500),  acoustic_spectrum = "CEP",
             smoothing_factor=None, preemphasis=0.96, length=0.04, step=0.005, lift=0.002, 
             target_fs=12000, slice_time=-1, verbose=False):

    '''This function implements an idea in Johnson (`Acoustic and Auditory Phonetics`) that 
    vowel nasality may be indicated by the spectral cosine difference (SCD) between the LPC 
    spectrum (which assumes non-nasality) and a spectrum with fewer assumptions (a narrow-band FFT 
    or a cepstrally smoothed spectrum).  
    The idea is that because nasality introduces antiformants and/or additional resonant peaks in 
    the spectrum of a vowel, nasalized vowels should have a greater difference between the LPC 
    spectrum and an acoustic spectrum than the SCD we will find in non-nasalized vowels.

    In this function phon.RLPC_formants() is used to to produce LPC spectra, phon.compute_sgram() 
    is used to compute narrow-band spectra, and phon.cepstral_smooth() 
    is used to produce cepstrally smoothed spectra. The function then reports the spectral 
    cosine difference (SCD) (in the specified analysis band) between the LPC spectrum and (depending
    on the these to ways of 
    estimating the smoothed spectrum (one which assumes an all-pole [non-nasal] filter and one 
    which does not).
    
    In addition the function returns two commonly used measures of vowel nasality: the bandwidth 
    of F1 from the RLPC function, and the difference between the amplitude of the harmonic 
    closest to F1 and the lowest harmonic. 
    
Parameters
==========
    x : ndarray
        a 1D array of audio samples
    fs : int
        the sampling rate of samples in x.  This will be resampled to 'target_fs' for analysis, 
        so choose the order paramter with this in mind.
    order : int (default = -1)
        Number of LPC coefficients, passed to RLPC_formant().  Pass a value of -1 to use the 
        function choose_order() to determine the best value for this parameter.  
        Or pass a positive integer.
    analysis_band : list, default = (200,2500)
        Compute the spectral difference over this frequency band (in Hz)
    acoustic_spectrum : string, default = "CEP"
        Which acoustic spectrum to compare with the LPC spectrum.  Allowable values are "FFT" for the 
        narrow-band FFT, or "CEP" for the cepstrally smoothed spectrum.  The default approach, using 
        the cepstrally smoothed spectrum seems to be less affected by voice pitch effects.
    smoothing_factor : float, default = None
        If the value of this parameter is less than zero, then no smoothing will be done.  Otherwise this
        is the `s` parameter to be passed to phon.smoothn() to robustly smooth the data.
        Note that passing `None` (the default action) will have smoothn() automatically determine a 
        smoothing factor for each trajectory
        and in verbose mode will report the value of the choosen smoothing factor.  
        See smoothn for more information.
    preemphasis : float, default = 0.96
        value of a preemphasis factor (0-1).
    length : float, default = 0.04
        duration (in seconds) of analysis frames, the default of 0.04 gives a narrowband spectrum with resolution appropriate for resolving harmonics.
    step : float, default = 0.005
        the interval (in seconds) between successive formant measurements.
    lift : float, default = 0.003
        The cepstral smoothing filtering coefficient (in seconds).  Compenents that have a period 
        longer than the `lift` value will be removed from the signal. This is used in cepstral 
        smoothing to remove source spectral characteristics and keep vocal tract filter effects. 
        The default value of 3 ms means that cepstral components lower than 333 Hz will be removed.  
        Smaller values of this parameter lead to more cepstral smoothing, and larger values lead to 
        less spectral cepstral smoothing.
    target_fs : int, default = 12000
        Sampling frequency at which the analysis will be done. The top frequency in the analysis is 
        thus the Nyquist frequency (1/2 of the target_fs).
    slice_time, float, default = -1
        If a time value greater than 0 is given, then a diagnostic plot will be produced for
        the spectral slice at the slice_time.
    verbose : boolean (default = False)
        Print diagnostic messages

Returns
=======
    df - a pandas dataframe with columns as described in the note below.

Note
====
    The columns in the output dataframe are:
        * sec - midpoint time of the frame 
        * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.
        * BW1-4 - bandwidths of the vowel formants
        * gain - the LPC gain (variance of the residual) for each frame.
        * BW1_nas - the smoothed BW1, bandwidth of F1. Larger values indicate more nasality.
        * a1h1 - the difference in amplitude of the harmonic closest to F1 and the lowest harmonic.  Smaller values indicate more nasality (lower amplitude F1).
        * SCD - the spectral cosine distance between the LPC spectrum and the acoustic spectrum in the analysis band.  Larger values indicate more nasality.


Example
=======

    This example returns nasality measurements in a dataframe, and plots example spectra at a particular slice time.  The Spectral Cosine Distance values are plotted on the spectrogram.

    .. code-block:: Python

        audio_dir = importlib.resources.files('phonlab') / 'data/example_audio'
        example_file = audio_dir / 'the_soviet_union.wav'
        x,fs = phon.loadsig(example_file,chansel=[0])

        st = 0.72

        nasdf = phon.nasality(x,fs, order=-1, slice_time=st, smoothing_factor=None, verbose=True)

        # --------  plots -----------
        ax,_ = phon.sgram(x,fs,tf=5000)
        ax.plot(nasdf.sec,nasdf.F1,'w.')
        if st >0: 
            ax.axvline(st,color='red')
        ax2 = ax.twinx()
        ax2.plot(nasdf.sec,nasdf.SCD,'mo')
 
    '''    
    top_freq = target_fs//2
    
    x,fs = prep_audio(x, fs, target_fs=target_fs, pre=preemphasis, quiet=not verbose)  # resample to target_fs

    # -------- get acoustic spectra -----------------
    if acoustic_spectrum == "FFT":
        if verbose:
            print(f"calling phon.compute_sgram()")
        t,freqS,Sxx = compute_sgram(x,fs,w=length,s=step)
        n = Sxx.shape[0]*2-1
        freq = freqS
    else:
        if verbose:
            print(f"calling phon.cepstral_smooth() with lift = {lift} and top_freq = {top_freq}")
        t, freq, Exx, freqS, Sxx, _, _ = cepstral_smooth(x, fs, s=step, l=length,  
                                             tf = top_freq, preemphasis = 0.0, lift=lift)
        n = Exx.shape[0]*2-1
        
    fstep = freqS[1]-freqS[0]   # step size in Hz of the Sxx spectrum     

    # --------- use RLPC_formants() to get LPC spectra and formants and bandwidths -------------
    if verbose:
        print(f"calling phon.RLPC_formants() with order = {order} and top_freq = {top_freq}")
    fmtdf, A =  RLPC_formants(x, fs, s=step, l=length, tf = top_freq, 
                                      preemphasis=0.0, order=order, verbose=verbose)
   
    gain = fmtdf.gain.to_numpy()
    Lxx = 10 * np.log10((gain)/np.abs(np.fft.rfft(A, n=n)).T)

    # -------- measure the difference between the smoothed spectrum and the LPC spectrum -------
    if verbose:
        print(f"computing spectral cosine distance (SCD) in band {analysis_band}")
        print(f"Lxx.shape = {Lxx.shape}")
        print(f"Sxx.shape = {Sxx.shape}")
    lowidx = np.argmin(abs(freq-analysis_band[0]))
    highidx = np.argmin(abs(freq-analysis_band[1]))

    LPC_frames = Lxx[lowidx:highidx, :]
    if acoustic_spectrum=="FFT":
        Env_frames = Sxx[lowidx:highidx, :]
    else:
        Env_frames = Exx[lowidx:highidx, :]
    numerator = np.sum(LPC_frames * Env_frames, axis=0)
    denominator = np.linalg.norm(LPC_frames, axis=0) * np.linalg.norm(Env_frames, axis=0)
    SCD = np.sqrt(2*(1-(numerator / denominator))) * 100  # cosine distance between spectra
    
    # ----------- measure a1h1 ---------------------- 
    a1_minus_h1 = np.zeros(Lxx.shape[1])
    for idx in range(Lxx.shape[1]):
        if fmtdf.F1[idx]==0:
            a1_minus_h1[idx] = np.nan
        else:
            peaksep = int(50//fstep)  # number of steps in xx Hz
            peaks,props = find_peaks(Sxx[:,idx], width = peaksep, distance = peaksep)
            h1 = Sxx[peaks[0],idx]  # amplitude of the first (lowest)  spectral peak
            a1idx = np.argmin(np.fabs(freqS[peaks]-fmtdf.F1[idx]))
            a1 = Sxx[peaks[a1idx],idx]  # amplitude of the peak closest to F1
            a1_minus_h1[idx] = a1 - h1   # difference in amp at F1 and at H1
    
    # ------- optionally smooth the data --------------------
    if smoothing_factor is None or smoothing_factor > 0:
        if verbose:
            print(f"Smoothing with phon.smoothn(): s = {smoothing_factor}")
        SCD,snas,e = smoothn(SCD, s=smoothing_factor, isrobust=True) # Robust smoothing
        BW1_nas,sbw1,e = smoothn(fmtdf.BW1, s=smoothing_factor, isrobust=True)
        a1h1,sa1h1,e = smoothn(a1_minus_h1, s=smoothing_factor, isrobust=True)
        if smoothing_factor is None and verbose:
            print(f"  Smoothing factors: ")
            print(f"     {sbw1:.3f} for BW1")
            print(f"     {sa1h1:.3f} for a1-h1")
            print(f"     {snas:.3f} for SCD")
    else:
        BW1_nas = fmtdf.BW1.to_numpy()
        a1h1 = a1_minus_h1

    # -------- optionally show a plot of the acoustic and LPC spectra ---------
    if slice_time > 0:  
        idx = np.argmin(abs(t-slice_time))  # find the closest index

        fig3 = plt.figure(figsize=(5, 4),dpi=150)
        ax3 = fig3.add_subplot(111)
        ax3.plot(freq, Lxx[:,idx])  # the LPC spectrum
        if acoustic_spectrum=="CEP":
            ax3.plot(freq, Exx[:,idx])  # the Cepstrum
        
        peaks,props = find_peaks(Sxx[:,idx], width = peaksep, distance=peaksep)
        h1 = Sxx[peaks[0],idx]
        a1idx = np.argmin(np.fabs(freqS[peaks]-fmtdf.F1[idx]))
        a1 = Sxx[peaks[a1idx],idx]
        print(f"F1 = {fmtdf.F1[idx]:.1f}")
        print(f"at {slice_time} ({idx}): SCD = {SCD[idx]:.3f}, a1-h1 = {a1h1[idx]:.3f}")

        ax3.plot(freqS, Sxx[:,idx],color="gray", alpha=0.4)
        ax3.plot(freqS[peaks[0]],h1, "rx")
        ax3.plot(freqS[peaks[a1idx]],a1,"rx")
        ax3.text(freqS[peaks[0]],h1, "h1")
        ax3.text(freqS[peaks[a1idx]],a1,"a1")
        
        ax3.axvline(analysis_band[0],color='gray', linestyle="dashed")
        ax3.axvline(analysis_band[1],color='gray', linestyle="dashed")

     # Create DataFrame
    data_dict = {'sec': t, "BW1_nas":BW1_nas, "a1h1" : a1h1, "SCD" : SCD}
    df = pd.merge(fmtdf,pd.DataFrame(data_dict), on='sec')
    
    return df
