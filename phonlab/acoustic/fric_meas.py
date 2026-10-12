import numpy as np
from scipy.signal import find_peaks
from collections import namedtuple
from .multitaper_ import compute_mt_sgram, tandem_spectrum

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(np.maximum(x, np.finfo(float).tiny))
    else:
        np.log10(np.maximum(x, np.finfo(float).tiny), out)
        np.multiply(out, 10, out)

def hz2bark(hz):
    """
    Convert frequency in Hz to Bark using the Schroeder 1977 formula::

        bark(hz) = 7 * arcsinh(hz/650)

    Parameters
    ==========

    hz : scalar or array
        Frequency in Hz.

    Returns
    =======
        
    bark : scalar or array
        Frequency in Bark, in an array the same size as `hz`, or a scalar if `hz` is a scalar.
    """
    
    return 7 * np.arcsinh(hz/650)

def bark2hz(bark):
    '''
    Convert frequency in Hz to Bark using the Schroeder 1977 formula::

        Hz(b) = 650 * sinh(b/7)

    Parameters
    ----------

    bark: scalar or array
        Frequency in Bark.

    Returns
    -------
    scalar or array
        Frequency in Hz, in an array the same size as `bark`, or a scalar if `bark` is a scalar.
    '''
    return 650 * np.sinh(bark/7)


FricMeas = namedtuple('FricMeas', [
    'Fm', 'Am', 'AmpD', 'Fsec', 'Asec', 'mode',
    'COG', 'SD', 'Skew', 'Kurtosis', 'spec', 'freq'
])

def _measure(spec, freq, fspace):
    """Find the main spectral peaks and compute the spectral moments of a fricative spectrum.

    `spec` is the spectrum in dB, `freq` its frequency axis (Hz), and `fspace` the frequency spacing (Hz per point) used to map from frequency to array index.  Returns a `FricMeas` namedtuple.
    """
    bottom_freq = int(300/fspace)  # bottom of the search space (in points in the array)
    top_freq = int(16000/fspace)  # set frequency range for analysis -- up to 16kHz for peaks (11kHz for moments),
    top_freq = min(top_freq, len(freq))  # but never beyond the Nyquist freq (1/2 the sampling rate)
    spec_chunk = spec[bottom_freq:top_freq]
    min_dist = int(500/fspace)  # for peak picking - peaks must be at least this many points apart

    mode = '500/50%/8dB'
    height = 0.5  # proportion of amplitude range in fricative band
    min_height =  np.min(spec_chunk) + (np.max(spec_chunk)-np.min(spec_chunk))*height 
    min_prom = 8  # dB
    (peaks,prop) = find_peaks(spec_chunk, height=min_height, distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '500/50%/4dB'
        min_prom = 4  # dB
        (peaks,prop) = find_peaks(spec_chunk, height=min_height, distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '500/33%/3dB'
        height = 0.33
        min_height =  np.min(spec_chunk) + (np.max(spec_chunk)-np.min(spec_chunk))*height
        min_prom = 3  # dB
        (peaks,prop) = find_peaks(spec_chunk, height=min_height, distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = 'no peak found'
        peaks = np.append(peaks,[0])

    index_of_main_peak = bottom_freq + peaks[0]  # this is selecting the first peak in an array of peaks
    Fm = freq[index_of_main_peak]      # convert to Hz
    Am = spec[index_of_main_peak]      # get amplitude
    AmpD = Am - np.min(spec[bottom_freq:index_of_main_peak])

    Fsec = np.nan
    Asec = np.nan
    if len(peaks)>1:  
        index_of_second_peak = bottom_freq + peaks[1]
        Fsec = freq[index_of_second_peak]
        Asec = spec[index_of_second_peak]
        if Asec>Am: AmpD = Asec - np.min(spec[bottom_freq:index_of_main_peak])

    # -------- moments analysis -----------------------
    bottom_freq = int(300/fspace)
    top_freq = int(11000/fspace)  # set frequency range for the moments analysis -- 11kHz,
    top_freq = min(top_freq, len(freq))  # but never beyond the Nyquist freq (1/2 the sampling rate)
    f = freq[bottom_freq:top_freq] 
    temp = spec[bottom_freq:top_freq] - np.min(spec[bottom_freq:top_freq]) # make sure none are negative
    Ps = temp/np.sum(temp)  # scale to sum to 1
    COG = np.sum(Ps*f)  # center 
    dev = f-COG  # deviations from COG
    Var = np.sum(Ps * dev**2)  # second moment
    SD = np.sqrt(Var)  # Standard deviation
    Skew = np.sum(Ps * dev**3)  # third moment 
    Kurtosis = np.sum(Ps * dev**4)  # fourth moment

    # scaling recommended by Forrest et al. 1990
    Skew = Skew/np.sqrt(Var**3)
    Kurtosis = Kurtosis/(Var**2) - 3
    
    return FricMeas(
        Fm=Fm, Am=Am, AmpD=AmpD, Fsec=Fsec, Asec=Asec, mode=mode,
        COG=COG, SD=SD, Skew=Skew, Kurtosis=Kurtosis, spec=spec, freq=freq
    )


def fricative(x,fs,t,w=0.02,NW=4,order=12):
    """
    Measure fricative acoustic values, from a window (20 ms by default) centered on time `t`.

    In addition to reporting the spectral moments (COG, SD, skew, and kurtosis) of the multitaper spectrum (from `compute_mt_sgram()`, using DPSS tapers from `scipy.signal.windows.dpss()`), `fricative()` finds the "main peak" in the fricative spectrum.  For fricatives with a resonant cavity, this is the lowest resonance of the cavity.  
    For those with no resonance the main peak frequency may be more a property of the source function.
     
    The main peak heuristic is to start by looking the in the spectrum above 300 Hz, for the lowest frequency peak that is at least 50% of the amplitude range, separated from the nearest peak by at least 500Hz, and prominent by 8bB above the next nearest peak (see the **scipy.signal.find_peaks()** documentation).  If no peak is found, relax the prominence constraint, then the amplitude constraint, and then both. 

Parameters
==========

    x : ndarray
        a one-dimensional array of audio samples
    fs : int
        the sampling frequency of `x` (ideally should be at least 16000)
    t : float
        the time (in seconds) at which to take measurements (this is usually in a fricative, but doesn't have to be).
    w : float, default = 0.02
        the length (in seconds) of the analysis window, centered on `t`.
    NW : float, default = 4
        the time-bandwidth product of the multitaper analysis.  The bandwidth (smoothness) of the spectrum is 2*NW/`w` Hz (400 Hz with the defaults), and about 2*NW-1 tapers are averaged to stabilize the estimate.  To keep the same smoothness when `w` is changed, scale `NW` in proportion to `w`.
    order : integer or None, default = 12
        determines the number of points in the FFT of the spectrum (2**`order` points, at least as many as in the window).  The value, `None`, uses no zero padding, so the spectrum has a point every 1/`w` Hz (50 Hz with the default window).  A larger value, such as 12, gives a spectrum with finer frequency spacing (for peak finding), but no more frequency resolution.


Returns
=======
     
namedtuple :
    A `FricMeas` namedtuple of fricative measures is returned, with the following fields:

        Fm : float
            Frequency (in Hz) of the first main spectral peak.  A measure correlated with the length of the front tube.
        Am : float
            Amplitude (in dB) at Fm
        AmpD : float
            The difference in amplitude (dB) between the higher of Am or Asec and the minimum amplitude between 500Hz and Fm.  A measure of sibilance.
        Fsec : float 
            Frequency of the second major peak in the spectrum.  If the front tube is long there can be a second resonance. If there is no second major peak, this field's value is `None`
        Asec: float
            Amplitude (in dB) at Fsec. If there is no second major peak, this field's value is `None`
        mode : string
            a report on the peak finding parameters used
        COG : float
            center of gravity, the first moment of the spectrum.
        SD : float
            standard deviation, the second moment of the spectrum.
        Skew : float
            scaled third moment, skew
        Kurtosis : float
            scaled fourth moment, kurtosis
        spec : ndarray
            the multi-taper power spectrum at the midpoint (e.g. for use in plotting spectra)
        freq : ndarray
            the frequency scale of the spectrum (e.g. for use in plotting spectra)

Note
====

    The multitaper spectrum is not zero padded (it has a point every 1/(window length) Hz, 50 Hz for a 20 ms window).  For a spectrum with finer frequency spacing that is also averaged over several adjacent windows, see `tandem_fric()`.

    The multitaper spectrum is not zero padded (it has a point every 1/`w` Hz, or 50 Hz for the 20 ms window).  For a spectrum with finer frequency spacing that is also averaged over several adjacent windows to make it more stable, see `tandem_fric()`.

    The major peaks analysis implemented here draws on ideas from Shadle (2023) and Shadle et al. (2023).   Moments analysis was introduced for analysis of stop release burst spectra by Forrest et al. (1988).


References
==========

    K. Forrest, G. Weismer, P. Milenkovic, and R.N. Dougall (1988) Statistical analysis of word-initial voiceless obstruents: Preliminary data. `J. Acoust. Soc. Am.` 84(1), 115–123.

    C. Shadle (2023) Alternatives to moments for characterizing fricatives: Reconsidering Forrest et al. (1988). `J. Acoust. Soc. Am.` 153 (2): 1412–1426. https://doi.org/10.1121/10.0017231

    C. Shadle, W-R. Chen, L.L. Koenig, J.L. Preston (2023) Refining and extending measures for fricative spectra, with special attention to the high-frequency range. `J. Acoust. Soc. Am.` 154 (3): 1932–1944. https://doi-org.libproxy.berkeley.edu/10.1121/10.0021075



Example
=======

    This example returns fricative measurements from time 2.25 seconds in the 
    audio samples returned by `loadsig()`. The major peak and COG frequencies 
    are indicated in a plot of the spectrum.

    .. code-block:: Python
    
         x, fs = phon.loadsig("sf3_cln.wav")
         y, fs = phon.prep_audio(x, fs, target_fs=16000)
         fricm = phon.fricative(y, fs, 2.25)

         print(f"first major peak at {fricm.Fm:.1f}, Center of Gravity is {fricm.COG:.1f}")
         plt.plot(fricm.freq, fricm.spec)
         plt.axvline(fricm.Fm, color="red")
         plt.axvline(fricm.COG, color="green")

The figure here shows major peaks and COG in several different fricatives.

    .. figure:: images/fricative.png
       :scale: 50 %
       :alt: Marking major peaks and COG in fricative spectra.
       :align: center

       Marking major peaks (red lines) and COG (green line) in fricative spectra.

    """
    winsize = w   # window centered at midpoint (mp)
    
    i_center = int(t * fs)   # index of midpoint time: seconds to samples
    i_start  = int(i_center - winsize/2*fs)  # back half a window
    i_end = int(i_center + winsize/2*fs)     # forward half a window
    chunk = x[i_start:i_end]
    
    # one multitaper spectrum, by default with no zero padding (order=None), as in the nitime-based version.
    # scale='nitime' keeps the dB levels the same as in that version.
    _, freq, sgram = compute_mt_sgram(chunk, fs, w=len(chunk)/fs, s=winsize, NW=NW, order=order, scale='nitime')
    spec = sgram[:, 0]       # multi-taper spectrum (dB)
    fspace = (fs/2)/len(freq)  # frequency spacing in spectrum - map from frequency to array index
    
    return _measure(spec, freq, fspace)


def tandem_fric(x,fs,t,w=0.02,NW=4,total=0.04,step=0.002):
    """
    Measure fricative acoustic values, from a 40 ms stretch of audio (by default) centered on time `t`.

    In addition to reporting the spectral moments (COG, SD, skew, and kurtosis) of a "tandem" multitaper spectrum (multitaper spectra from `compute_mt_sgram()`, using DPSS tapers from `scipy.signal.windows.dpss()`, averaged across closely spaced windows by `tandem_spectrum()`), `tandem_fric()` finds the "main peak" in the fricative spectrum.  For fricatives with a resonant cavity, this is the lowest resonance of the cavity.  
    For those with no resonance the main peak frequency may be more a property of the source function.
     
    The main peak heuristic is to start by looking the in the spectrum above 300 Hz, for the lowest frequency peak that is at least 50% of the amplitude range, separated from the nearest peak by at least 500Hz, and prominent by 8bB above the next nearest peak (see the **scipy.signal.find_peaks()** documentation).  If no peak is found, relax the prominence constraint, then the amplitude constraint, and then both. 

Parameters
==========

    x : ndarray
        a one-dimensional array of audio samples
    fs : int
        the sampling frequency of `x` (ideally 32000, so that the high frequency energy of sibilants is included)
    t : float
        the time (in seconds) at which to take measurements (this is usually in a fricative, but doesn't have to be).
    w : float, default = 0.02
        the length (in seconds) of each multitaper analysis window.
    NW : float, default = 4
        the time-bandwidth product of the multitaper analysis.  The bandwidth (smoothness) of the spectrum is 2*NW/`w` Hz (400 Hz with the defaults), and about 2*NW-1 tapers are averaged to stabilize the estimate.  To keep the same smoothness when `w` is changed, scale `NW` in proportion to `w`.
    total : float, default = 0.04
        the total duration (in seconds) of audio, centered on `t`, that goes into the spectrum.  Windows of length `w` are spaced `step` apart across this duration and their power spectra are averaged (a "tandem" spectrum), which stabilizes peak finding.  It must be at least `w`.  If `total` equals `w` a single multitaper spectrum is used.  Because the windows are spread across `total`, a short fricative or a rapid transition will be blurred.
    step : float, default = 0.002
        the time (in seconds) between the starts of adjacent windows in the tandem spectrum.  With the defaults 11 spectra are averaged.  Steps much larger than 2 ms, or smaller `NW`, make the spectrum less stable.


Returns
=======
     
namedtuple :
    A `FricMeas` namedtuple of fricative measures is returned, with the following fields:

        Fm : float
            Frequency (in Hz) of the first main spectral peak.  A measure correlated with the length of the front tube.
        Am : float
            Amplitude (in dB) at Fm
        AmpD : float
            The difference in amplitude (dB) between the higher of Am or Asec and the minimum amplitude between 500Hz and Fm.  A measure of sibilance.
        Fsec : float 
            Frequency of the second major peak in the spectrum.  If the front tube is long there can be a second resonance. If there is no second major peak, this field's value is `None`
        Asec: float
            Amplitude (in dB) at Fsec. If there is no second major peak, this field's value is `None`
        mode : string
            a report on the peak finding parameters used
        COG : float
            center of gravity, the first moment of the spectrum.
        SD : float
            standard deviation, the second moment of the spectrum.
        Skew : float
            scaled third moment, skew
        Kurtosis : float
            scaled fourth moment, kurtosis
        spec : ndarray
            the multi-taper power spectrum at the midpoint (e.g. for use in plotting spectra)
        freq : ndarray
            the frequency scale of the spectrum (e.g. for use in plotting spectra)

Note
====

    The major peaks analysis implemented here draws on ideas from Shadle (2023) and Shadle et al. (2023).   Moments analysis was introduced for analysis of stop release burst spectra by Forrest et al. (1988).


References
==========

    K. Forrest, G. Weismer, P. Milenkovic, and R.N. Dougall (1988) Statistical analysis of word-initial voiceless obstruents: Preliminary data. `J. Acoust. Soc. Am.` 84(1), 115–123.

    C. Shadle (2023) Alternatives to moments for characterizing fricatives: Reconsidering Forrest et al. (1988). `J. Acoust. Soc. Am.` 153 (2): 1412–1426. https://doi.org/10.1121/10.0017231

    C. Shadle, W-R. Chen, L.L. Koenig, J.L. Preston (2023) Refining and extending measures for fricative spectra, with special attention to the high-frequency range. `J. Acoust. Soc. Am.` 154 (3): 1932–1944. https://doi-org.libproxy.berkeley.edu/10.1121/10.0021075



Example
=======

    This example returns fricative measurements from time 2.25 seconds in the 
    audio samples returned by `loadsig()`. The major peak and COG frequencies 
    are indicated in a plot of the spectrum.

    .. code-block:: Python
    
         x, fs = phon.loadsig("sf3_cln.wav")
         y, fs = phon.prep_audio(x, fs, target_fs=32000)
         fricm = phon.tandem_fric(y, fs, 2.25)

         print(f"first major peak at {fricm.Fm:.1f}, Center of Gravity is {fricm.COG:.1f}")
         plt.plot(fricm.freq, fricm.spec)
         plt.axvline(fricm.Fm, color="red")
         plt.axvline(fricm.COG, color="green")

The figure here shows major peaks and COG in several different fricatives.

    .. figure:: images/fricative.png
       :scale: 50 %
       :alt: Marking major peaks and COG in fricative spectra.
       :align: center

       Marking major peaks (red lines) and COG (green line) in fricative spectra.

    """
    if total < w:
        raise ValueError("total must be at least as long as w")
    n_chunk = int(round(total*fs))      # number of samples of audio analyzed
    i_start = int(round(t*fs)) - n_chunk//2   # centered on time t: seconds to samples
    i_end = i_start + n_chunk
    if i_start < 0 or i_end > len(x):
        raise ValueError(f"time t={t} is too close to the edge of the audio for a {total} s analysis")

    # multitaper spectra in closely spaced windows, averaged (on the power scale) into one stabilized spectrum
    # scale='nitime' keeps dB levels comparable with fricative()
    _, freq, sgram = compute_mt_sgram(x[i_start:i_end], fs, w=w, s=step, NW=NW, order=12, scale='nitime')
    spec = tandem_spectrum(sgram)   # tandem multitaper spectrum (dB)
    fspace = freq[1]-freq[0]  # frequency spacing in spectrum - map from frequency to array index

    return _measure(spec, freq, fspace)
