import numpy as np
from scipy.signal import find_peaks,windows
import pandas as pd

from ..utils.signal import loadsig
from ..utils.prep_audio_ import prep_audio
from ..utils.align_series import align_timeseries
from ..acoustic.sgram_ import compute_sgram
from ..acoustic.rlpc import RLPC_formants
from ..acoustic.cepstral import _cepstral_smooth_band, _smooth_sgram
from ..acoustic.amp_env import amplitude_envelope
from ..acoustic.vad import VAD
from ..acoustic.display_wave_ import display_wave
from ..utils.smooth1d_ import smooth1d


BAND_STEP = 1.0   # Hz, spacing of the frequencies at which spectra are compared in the analysis band


def _spectral_distance(Sxx, A, fs, lift, freqs, chunk=2048):
    '''RMS difference (dB) between the LPC spectrum and the cepstrally smoothed spectrum, at frequencies `freqs` (Hz).

Both spectra are evaluated only at `freqs` (see cepstral._cepstral_smooth_band() and below), so there is no need
to compute full-length spectra for every frame.  The mean of each spectrum over the band is removed before they are
compared, so only spectral shape matters.  That also means the LPC gain and the window amplitude correction (a
constant for each frame) drop out and are not needed here; they are needed to draw the diagnostic plot where the
LPC and acoustic spectra are shown on the same dB scale.
'''
    Exx = _cepstral_smooth_band(Sxx, fs, lift, freqs)                       # (len(freqs), nframes)
    # |A(e^jw)| at the requested frequencies: the polynomial A has only order+1 coefficients, so this is a small
    # matrix product, which is much cheaper than a full-length FFT of A for every frame
    W = np.exp(-2j*np.pi*np.outer(np.arange(A.shape[1]), freqs)/fs)
    tiny = np.finfo(float).tiny
    SD = np.empty(A.shape[0])
    for a in range(0, A.shape[0], chunk):                                   # limits the size of the temporaries
        Lxx = -20*np.log10(np.maximum(np.abs(A[a:a+chunk] @ W).T, tiny))    # LPC spectrum, up to a constant
        Eb = Exx[:, a:a+chunk]
        SD[a:a+chunk] = np.sqrt(np.mean(((Lxx - Lxx.mean(axis=0)) - (Eb - Eb.mean(axis=0)))**2, axis=0))
    return SD


def _find_peaks_h1_a1(spectrum, peaksep):
    '''Find h1 (the lowest spectral peak) and a1 (the peak nearest F1) in a narrow-band dB spectrum.  
Only peaks within 35 dB of the highest point of the spectrum are considered.  Returns (peaks, minheight), 
where `peaks` is empty if there are no such peaks.'''
    minheight = np.max(spectrum) - 35
    peaks, _ = find_peaks(spectrum, width=peaksep, distance=peaksep, height=minheight)
    return peaks, minheight


def _slice_diagnostic_plot(x, fs, idx, step, length, lift, A_row, gain, F1, SD_i, a1h1_i, slice_time, analysis_band):
    '''Plot the LPC spectrum, the cepstrally smoothed spectrum and the narrow-band spectrum of frame `idx`, 
with h1 and a1 marked.  Only this one frame is analyzed, and its spectra are computed over the whole frequency range.  
`x` is the prepared audio that was analyzed by nasality().'''
    import matplotlib.pyplot as plt
    hop, win = int(fs*step), int(fs*length)
    _, freqS, Sxx = compute_sgram(x[idx*hop: idx*hop + win], fs, length, step)   # one frame
    freq, Exx, _ = _smooth_sgram(Sxx, fs, lift)
    Sxx, Exx = Sxx[:, 0], Exx[:, 0]
    n = len(Exx)*2 - 1

    # We use a window for FFT spectra, and don't for LPC, therefore an amplitude correction is needed
    window = windows.blackmanharris(win)
    amp_correction = 2.5 + 20 * np.log10(np.sum(window)/np.sqrt(np.sum(np.square(window))))
    tiny = np.finfo(float).tiny
    lpc_mag = np.abs(np.fft.rfft(A_row, n=n))
    Lxx = 20 * np.log10(np.maximum(np.sqrt(gain), tiny)/np.maximum(lpc_mag, tiny)) - amp_correction

    fig3 = plt.figure(figsize=(5, 4),dpi=150)
    ax3 = fig3.add_subplot(111)
    ax3.plot(freq, Lxx)  # the LPC spectrum
    ax3.plot(freq, Exx)  # the Cepstrum

    peaksep = int(50//(freqS[1]-freqS[0]))  # number of steps in 50 Hz
    peaks, minheight = _find_peaks_h1_a1(Sxx, peaksep)
    print(f"F1 = {F1:.1f}")
    print(f"at {slice_time} ({idx}): SD = {SD_i:.3f}, a1-h1 = {a1h1_i:.3f}")
    print(f"peak must be at least {minheight:.2f} dB to be h1")

    ax3.plot(freqS, Sxx, color="gray", alpha=0.4)
    if len(peaks) > 0:
        h1 = Sxx[peaks[0]]
        a1idx = np.argmin(np.fabs(freqS[peaks]-F1))
        a1 = Sxx[peaks[a1idx]]
        ax3.plot(freqS[peaks[0]],h1, "rx")
        ax3.plot(freqS[peaks[a1idx]],a1,"rx")
        ax3.text(freqS[peaks[0]],h1, "h1")
        ax3.text(freqS[peaks[a1idx]],a1,"a1")

    ax3.axvline(analysis_band[0],color='gray', linestyle="dashed")
    ax3.axvline(analysis_band[1],color='gray', linestyle="dashed")


def nasality(x,fs, order=-1, analysis_band=(100,1200),
             preemphasis = 0.96, length=0.03, step=0.005, lift=0.002, slice_time=-1, verbose=False,
             index_smoothing=24, calibration="fixed", voicing_threshold=0.5, reference=None):

    '''This function returns measures of nasality for voiced speech in a single (mono) recording: the 
    difference between the amplitude of the harmonic closest to F1 and the lowest harmonic (A1-H1), the 
    bandwidth of F1 from the RLPC function, and a spectral distance (SD) between an LPC spectrum and a 
    cepstrally smoothed acoustic spectrum that has fewer assumptions.

    The SD measure started from an idea in Johnson's `Acoustic and Auditory Phonetics`: an LPC spectrum 
    assumes an all-pole (non-nasal) vocal tract, and nasalization adds zeros and extra poles, so 
    nasalized vowels should show a larger distance between the LPC spectrum and the acoustic spectrum. 
    When SD was compared with nasalance measured with a nasometer (3 recordings, 2 talkers) the 
    relationship was the opposite: SD is LOWER in more nasal frames (Spearman rho about -0.55 to -0.6). 
    In oral vowels the acoustic spectrum has a deep trough between F1 and F2 that the LPC spectrum fills 
    in, and this mismatch is most of the SD. Nasalization lowers and broadens the F1 peak and fills the 
    trough, so the acoustic spectrum becomes smoother and closer to the LPC shape, and SD falls.

    Recommended use. Neither SD nor a1h1 alone predicts nasalance well (rho about 0.55 each). The best 
    predictor found is their average after robust smoothing and z-scoring each:

        nas_index = -( z(SD) + z(a1h1) ) / 2        (larger values = more nasal)

    which this function returns in the `nas_index` column. It correlated with nasalance at about 0.70 
    (0.65 in nasal murmurs, 0.5 in vowel-like frames; 0.66-0.73 across recordings). BW1_nas was a weaker 
    predictor (rho 0.3-0.45) and did not improve the index. With the default parameters, this index mostly 
    picks out nasal consonants and the nasalized portions of vowels. The index is relative, 
    not an estimate of nasalance in absolute terms. With calibration="recording" (the default) it is 
    z-scored within the voiced frames of the recording, so it is not comparable across talkers or sessions. 
    With calibration="fixed" it is scaled with reference means and standard deviations drawn from test 
    recordings (see `reference`), which keeps recordings comparable and needs no voice activity detection.

    In this function phon.RLPC_formants() is used to produce LPC spectra, and phon.compute_sgram() is used to 
    produce narrow-band spectra.  The cepstrally smoothed spectra (see phon.cepstral_smooth()) are computed only at 
    frequencies in the analysis band, which makes the function several times faster and use much less memory than 
    computing full spectra for every frame.  The diagnostic plot (`slice_time`) analyzes a single frame over the
    whole frequency range.

    Parameters
    ==========
        x : ndarray
            a 1D array of audio samples
        fs : int
            the sampling rate of samples in x.  The audio will be resampled to 12000 Hz for analysis, 
            so choose the order paramter with this in mind.
        order : integer (default = -1)
            Number of LPC coefficients, passed to RLPC_formant().  The default of -1 lets choose_order() 
            determine the value; values of 16-18 gave equivalent results in the nasometer comparison (larger 
            values, 22 and up, were worse).  
        analysis_band : list, default = (100,1200)
            Compute the spectral difference over this frequency band (in Hz)
        preemphasis : float, default = 0.96
            factor to tilt the spectrum up (emphasizing higher frequencies) [0-1].  Values of 0.9-0.98 gave
            similar results; with no pre-emphasis the gain-calibrated SD was much worse.
        length : float, default = 0.03
            duration (in seconds) of analysis frames, the default of 0.03 gives a narrowband spectrum with 
            resolution appropriate for resolving harmonics. 0.04 gave equivalent results.  
            Shorter windows (0.02-0.025) help SD in nasal murmurs but hurt a1h1 (harmonic resolution).
        step : float, default = 0.005
            the interval (in seconds) between successive formant measurements.
        lift : float, default = 0.002
            The cepstral smoothing filtering coefficient (in seconds).  Compenents that have a period 
            longer than the `lift` value will be removed from the signal. This is used in cepstral 
            smoothing to remove source spectral characteristics and keep vocal tract filter effects. 
            The default value of 2 ms means that cepstral components lower than 500 Hz will be removed.  
            Smaller values of this parameter lead to more cepstral smoothing, and larger values lead to 
            less spectral cepstral smoothing.  Values of 0.004 or more were much worse in the nasometer comparison.
        slice_time : float, default = -1
            If a time value greater than 0 is given, then a diagnostic plot will be produced for
            the spectral slice at the slice_time.
        verbose : boolean, default = False
            If true, print diagnostic messages.
        index_smoothing : float, default = 24
            The `s` parameter passed to phon.smooth1d() (robust smoothing) when smoothing SD and a1h1 
            before they are combined into `nas_index`. Values of 0 or less give no smoothing. 
            Smoothing matters: in the nasometer comparison the index correlated with nasalance at 0.57 
            without smoothing, 0.70 with s = 16 and 0.71 with s = 32-48. The SD and a1h1 columns 
            themselves are not smoothed.
        calibration : string, default = "recording"
            How the smoothed SD and a1h1 are z-scored to make `nas_index`. 
            
                * "recording": use the mean and standard deviation over the voiced frames of this recording, found with phon.VAD() (frames with probv > `voicing_threshold`). `nas_index` is NaN in other frames. 
                * "fixed": use the reference means and standard deviations in `reference`. phon.VAD() is not called, there is no `probv` column, and `nas_index` is computed for every frame (values in unvoiced frames are meaningless).
                
            The constants are only valid for the default analysis parameters (a warning is issued otherwise).
        voicing_threshold : float, default = 0.5
            Frames with VAD probv greater than this value are used as voiced frames in calibration="recording".
        reference : dict, default = None
            Reference (mean, standard deviation) pairs of the smoothed SD and a1h1 over voiced frames, used when
            calibration="fixed", as {"SD": (mean, sd), "a1h1": (mean, sd)}. The default reference values (reference=None)
            uses {"SD": (3.86, 1.26), "a1h1": (10.56, 5.28)}, pooled over three nasometer test recordings (2 talkers). 
            The per-recording means ranged from 3.2 to 4.3 for SD and from 9.2 to 12.4 for a1h1, so these constants 
            carry some talker and session variation.

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
            * a1h1 - the difference in amplitude of the harmonic closest to F1 and the lowest harmonic.  Smaller values indicate more nasality.
            * SD - the spectral distance between the LPC spectrum and the acoustic spectrum in the analysis band.  Smaller values indicate more nasality (see above).
            * probv - the probability of voicing from phon.VAD() (only with calibration="recording").
            * nas_index - the combined index -(z(SD) + z(a1h1))/2 described above; larger values indicate more nasality. NaN in unvoiced frames with calibration="recording".


    Example
    =======

        This example returns nasality measurements in a dataframe, and plots example spectra at a particular slice time.  The SD and a1h1 measures are plotted on the spectrogram.  Note that the French bon/beau example does not show a clear SD difference between the nasal and oral vowels (SD is about 2.0-2.1 in both), so it should not be read as a demonstration of the SD effect.

    .. code-block:: Python

        audio_dir = importlib.resources.files('phonlab') / 'data/example_audio'
        example_file = audio_dir / 'FR_bon_beau.wav'
        x,fs = loadsig(example_file,chansel=[0])

        st = 0.41

        nasdf = nasality(x,fs, slice_time=st, verbose=True)
        
        # --------  plot -----------
        ax = phon.sgram(x,fs,tf=5000)[0]
        ax.plot(nasdf.sec,nasdf.F1,'w.')
        ax.text(0.2,4500, "[ bõ ]",fontsize=18)
        ax.text(0.8,4500, "[ bo ]",fontsize=18)
        ax.text(1.4,4500, "[ bõ ]",fontsize=18)
        ax.text(2,4500, "[ bo ]",fontsize=18)
        if st >0: 
              ax.axvline(st,color='red')
        
        ax2 = ax.twinx()
        ax2.plot(nasdf.sec,nasdf.SD,'md')
        ax2.set_ylim(0,16)
        ax3 = ax.twinx()
        ax3.plot(nasdf.sec,nasdf.a1h1,color='dodgerblue',marker='o',linewidth=0)
        ax3.set_ylim(0,16)

    .. figure:: images/nasality.png
       :scale: 36 %
       :alt: Overlaid on a spectrogram are plotted nasality measures SD (magenta diamonds) and a1h1 (blue dots) for two repetitions of the French minimal pair <bon> and <beau>.
       :align: center
                                        
       Overlaid on a spectrogram are plotted nasality measures SD
       (magenta diamonds) and a1h1 (blue dots) for two repetitions of
       the French minimal pair <bon> and <beau>.  Frequency of F1 is
       plotted with small white dots.

    '''

    target_fs = 12000
    top_freq = target_fs//2

    if calibration not in ("recording", "fixed"):
        raise ValueError(f"nasality(): calibration must be 'recording' or 'fixed', not {calibration!r}")
    if calibration == "recording":
        vad = VAD(x, fs)    # voicing probability, from the audio at its original sampling rate
    
    x,fs = prep_audio(x, fs, target_fs=target_fs,  
                      scale=True, pre = preemphasis,
                      verbose = verbose)  # resample to target_fs

    # Pre-emphasis comes after scaling in prep_audio(), so scale the audio again.  This puts the narrow-band
    # spectra on the same amplitude scale as the audio that RLPC_formants() analyzes (it scales its input in the
    # same way), which the LPC gain relies on when the LPC and acoustic spectra are compared in the diagnostic plot.
    x,fs = prep_audio(x, fs, target_fs=fs, scale=True, pre=0.0)

    # -------- get narrow-band spectra (dB) -----------------
    if verbose:
        print(f"calling phon.compute_sgram() with length = {length} and step = {step}")
    t, freqS, Sxx = compute_sgram(x, fs, length, step)
    fstep = freqS[1]-freqS[0]   # step size in Hz of the Sxx spectrum     

    # --------- use RLPC_formants() to get LPC spectra and formants and bandwidths -------------
    if verbose:
        print(f"calling phon.RLPC_formants() with order = {order} and top_freq = {top_freq}")
    # this call is why RLPC is different from other xxx_formants() functions.  In this function
    # we need both the fmtdf and the A matrix
    fmtdf, A =  RLPC_formants(x, fs, s=step, l=length, tf = top_freq, 
                                      preemphasis=0.0, order=order, verbose=verbose)
    if Sxx.shape[1] != A.shape[0]:
        raise RuntimeError(f"nasality(): {Sxx.shape[1]} spectra but {A.shape[0]} LPC frames; analysis frames do not line up")

    # -------- measure the difference between the smoothed spectrum and the LPC spectrum -------
    # Only the analysis band is needed, so both spectra are computed only at frequencies in the band.
    band_freqs = np.arange(analysis_band[0], analysis_band[1], BAND_STEP)
    if verbose:
        print(f"computing spectral distance (SD) in band {analysis_band} at {len(band_freqs)} frequencies")
        print(f"Sxx.shape = {Sxx.shape}")
    SD = _spectral_distance(Sxx, A, fs, lift, band_freqs)
    
    # ----------- measure a1h1 ---------------------- 
    F1 = fmtdf.F1.to_numpy()
    a1_minus_h1 = np.full(Sxx.shape[1], np.nan)
    peaksep = int(50//fstep)  # number of steps in xx Hz
    for idx in range(Sxx.shape[1]):
        if np.isnan(F1[idx]) or F1[idx]<=0:  # no F1 in this frame
            continue
        peaks, _ = _find_peaks_h1_a1(Sxx[:,idx], peaksep)
        if len(peaks) == 0:                  # no peak is high enough to be h1
            continue
        h1 = Sxx[peaks[0],idx]  # amplitude of the first (lowest)  spectral peak
        a1idx = np.argmin(np.fabs(freqS[peaks]-F1[idx]))
        a1 = Sxx[peaks[a1idx],idx]  # amplitude of the peak closest to F1
        a1_minus_h1[idx] = a1 - h1   # difference in amp at F1 and at H1
 
    BW1_nas = fmtdf.BW1.to_numpy()
    a1h1 = a1_minus_h1

    # -------- optionally show a plot of the acoustic and LPC spectra ---------
    if slice_time > 0:
        idx = np.argmin(abs(t-slice_time))  # find the closest index
        _slice_diagnostic_plot(x, fs, idx, step, length, lift, A[idx], fmtdf.gain.iloc[idx], F1[idx],
                               SD[idx], a1h1[idx], slice_time, analysis_band)

     # Create DataFrame
    data_dict = {'sec': t, "BW1_nas":BW1_nas, "a1h1" : a1h1, "SD" : SD}
    df = pd.merge(fmtdf,pd.DataFrame(data_dict), on='sec')

    # -------- nas_index: smooth SD and a1h1, z-score them, and average (negated) ---------
    def _smooth(v):
        v = pd.Series(v).interpolate(limit_direction='both').to_numpy(dtype=float)   # fill frames with no F1
        if index_smoothing is None or index_smoothing <= 0:
            return v
        return smooth1d(v, s=index_smoothing, isrobust=True)[0]

    SD_s, a1h1_s = _smooth(df.SD), _smooth(df.a1h1)
    if calibration == "recording":
        df = align_timeseries(df, vad, ts='sec', other_columns=['probv'])
        voiced = (df.probv > voicing_threshold).to_numpy()
        if voiced.sum() < 10:
            import warnings
            warnings.warn("nasality(): fewer than 10 voiced frames found; nas_index is calibrated over all frames")
            voiced = np.ones(len(df), dtype=bool)
        ref = {"SD": (np.nanmean(SD_s[voiced]), np.nanstd(SD_s[voiced], ddof=1)),
               "a1h1": (np.nanmean(a1h1_s[voiced]), np.nanstd(a1h1_s[voiced], ddof=1))}
    else:
        ref = reference if reference is not None else {"SD": (3.86, 1.26), "a1h1": (10.56, 5.28)}
        if reference is None and (index_smoothing != 24 or tuple(analysis_band) != (100, 1200)
                                  or length != 0.03 or lift != 0.002 or preemphasis != 0.96):
            import warnings
            warnings.warn("nasality(): the default reference means and standard deviations were measured with the "
                          "default analysis parameters; nas_index may be mis-scaled with the parameters used here")
    nas_index = -((SD_s - ref["SD"][0]) / ref["SD"][1] + (a1h1_s - ref["a1h1"][0]) / ref["a1h1"][1]) / 2
    if calibration == "recording":
        nas_index[~voiced] = np.nan
    df["nas_index"] = nas_index
    
    return df

# --------- function phon.nasalance() ------------
def nasalance(filename, nasal_channel = 'left',
              diagnostic_plot = False,
              start=0, end=-1, step=0.005):
    """Nasalance from a stereo nasometer file (nasal in one channel,
    oral in the other).  The channels are scaled to -2 dBFS within
    channel, then the amplitude envelope within the frequency band
    from 200 to 1200 Hz is found for each channel and finally
    nasalance is calculated to fall within the range [-1,1] where
    negative values are more oral and positive values are more nasal.

     nasalance = (N/(N+O) - 0.5) * 2

    Parameters
    ==========
        filename: string
            the name of a stereo audio file.
        nasal_channel: string (default = "left")
            The channel ('left' or 'right') that contains the nasal microphone from the nasometer.
        diagnostic_plot: boolean (default = False)
            Whether to print a diagnostic plot showing the two channels (oral in gray, nasal in pink),
            and the computed nasalance trace.
        start: float (default = 0.0)
            The starting time for the diagnostic plot.
        end: float (default = -1.0)
            The end time of the diagnostic plot (-1 means to the end of the file)
        step: fload (default = 0.005)
            The intervals between sucessive nasalance measurements in the output dataframe.
    
    Returns
    =======
        df - a pandas dataframe with columns as described in the note below.

    Note
    ====
        The columns in the output dataframe are:
            * sec - midpoint time of the frame 
            * nasalance - median nasalance in the frame, from -1 (all oral) to +1 (all nasal)
            
    Frames are non-overlapping, `step` seconds long (default 5 ms).  A final partial frame is dropped.

    .. figure:: images/nasalance.png
        :scale: 36 %
        :alt: example diagnostic plot of a nasometer recording of the words 'night, bite, bine'.
        
        Example diagnostic plot of a nasometer recording of the words
        'night, bite, bine'.  The pink waveform is from the nasal
        microphone, the gray plot is from the oral microphone, and the
        blue trace is the computed nasalance.

    """
    # load all channels; loadsig() returns [chan0, chan1, ..., fs]
    chans = loadsig(filename)
    raw_fs = chans.pop()
    if len(chans) != 2:
        raise ValueError(f"nasalance(): '{filename}' has {len(chans)} channel(s); "
                         "a stereo (2-channel) nasometer file is required "
                         "(nasal in one channel, oral in the other)")

    # Assume nasal in the left channel
    if nasal_channel == 'left':
        nas, oro = chans[0], chans[1]
    else:
        nas, oro = chans[1], chans[0]

    target_fs = 12000

    # prepare the audio
    nas,fs = prep_audio(nas,raw_fs,target_fs=target_fs,scale=-2)
    oro,fs = prep_audio(oro,raw_fs,target_fs=target_fs,scale=-2)

    # get the amplitude envelope of the nasal and oral channels
    bounds = [200,1200]  # focus on the amplitude in the F1 band
    nas_amp, fs = amplitude_envelope(nas,fs,target_fs=fs, bounds=bounds)
    oro_amp, fs = amplitude_envelope(oro,fs,target_fs=fs, bounds=bounds)
    nas_amp += 0.01
    oro_amp += 0.01  # amplitude floor

    # Compute nasalance as the proportion
    nasalance = (((nas_amp)/(nas_amp+oro_amp)) -0.5) *2

    if diagnostic_plot:
        # Plot the waveforms of the two channels - nasal is red
        ax = display_wave(oro,fs,start=start,end=end,alpha=0.3)
        ax = display_wave(nas,fs,start=start,end=end,ax=ax,color="red",alpha=0.5)
        display_wave(nasalance,fs,start=start,end=end, ax = ax, color="blue")

    # median nasalance in non-overlapping frames of `step` seconds
    flen = int(round(step*fs))
    nframes = len(nasalance)//flen
    med = np.median(nasalance[:nframes*flen].reshape(nframes, flen), axis=1)
    sec = (np.arange(nframes) + 0.5) * flen / fs

    return pd.DataFrame({'sec': sec, 'nasalance': med})

