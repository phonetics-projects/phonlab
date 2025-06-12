__all__ = ['hz2bark', 'fricative']

import nitime.algorithms as tsa  # has the multitaper routine
import numpy as np
import scipy.signal as signal

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def hz2bark(hz):
    """
    Convert frequency in Hz to Bark using the Schroeder 1977 formula.

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

def fricative(x,fs,t):
    """
    Measure fricative acoustic values, from a 20 ms window, centered on time `t`.

    In addition to reporting the spectral moments (COG, SD, skew, and kurtosis) of the multitaper
    spectrum (from `nitime.algorithms.multi_taper_psd()`), `fricative()` finds the "main peak" in the 
    fricative spectrum.  For fricatives with a resonant cavity, this is the lowest resonance of the cavity.  
    For those with no resonance the main peak frequency may be more a property of the source function.
     
    The main peak heuristic is to start by looking the in the spectrum above 500 Hz, for the lowest frequency peak that 
    is at least 80% of the amplitude range, separated from the nearest peak by at least 500Hz, and prominent
    by 6bB above the next nearest peak (see the **scipy.signal.find_peaks()** documentation).  If no peak is found, 
    relax the amplitude constraint, then the prominence constraint, and then both. 


    Parameters
    ==========

    x : ndarray
        a one-dimensional array of audio samples
    fs : int
        the sampling frequency of `x` (ideally should be at least 16000)
    t : float
        the time (in seconds) at which to take measurements (this is usually in a fricative, but doesn't have to be).


    Returns
    =======
     
    Fm : float
        Frequency of the first main spectral peak in Hertz
    FmB : float
        Frequency of the main peak on the Bark scale
    Am : float
        Amplitude (in dB) at Fm
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
        the multi-taper power spectrum at the midpoint
    freq : ndarray
        the frequency scale of the spectrum


    Note
    ----

    The major peaks analysis implemented here draws on ideas from Shadle (2023) and Shadle et al. (2023).
    
    Christine H. Shadle; Alternatives to moments for characterizing fricatives: Reconsidering Forrest et al. (1988). J. Acoust. Soc. Am. 1 February 2023; 153 (2): 1412–1426. https://doi.org/10.1121/10.0017231

    Christine H. Shadle, Wei-Rong Chen, Laura L. Koenig, Jonathan L. Preston; Refining and extending measures for fricative spectra, with special attention to the high-frequency range. J. Acoust. Soc. Am. 1 September 2023; 154 (3): 1932–1944. https://doi-org.libproxy.berkeley.edu/10.1121/10.0021075



    Example
    =======

    This example returns fricative measurements from time 2.25 seconds in the 
    audio samples returned by `get_signal()`. The major peak and COG frequencies 
    are indicated in a plot of the spectrum.

    .. code-block:: Python
    
         x,fs = phon.prep_audio("sf3_cln.wav")
         Fm,FmB,Am,mode,COG,SD,Skew,Kurtosis,spec,freq = phon.fricative(x,fs,2.25)
        
         print(f"first major peak at {Fm:.1f}, Center of Gravity is {COG:.1f}")
         plt.plot(freq,spec)
         plt.axvline(Fm,color="red")
         plt.axvline(COG,color="green")

    .. figure:: images/fricative.png
       :scale: 60 %
       :alt: a fricative spectrum with vertical lines marking the main peak (red) and COG (green)
       :align: center

       Marking the major peak and COG in a fricaitve spectrum.

    """
    winsize = 0.02   # 20 ms window centered at midpoint (mp)
    
    i_center = int(t * fs)   # index of midpoint time: seconds to samples
    i_start  = int(i_center - winsize/2*fs)  # back 10 ms
    i_end = int(i_center + winsize/2*fs)     # forward 10 ms
    
    f, psd, nu = tsa.multi_taper_psd(x[i_start:i_end])  # get multi-taper spectrum
    
    spec = dB(psd)           # work with magnitude spectrum
    freq = (f/(2*np.pi))*fs   # frequency axis for the spectrum
    nyquist = fs/2
    fspace = nyquist/len(f)  # frequency spacing in spectrum - map from frequency to array index
    
    '''
    i1 = int(550/fspace)  # indexes of band edges
    i2 = int(3000/fspace)
    i3 = int(7000/fspace)
    i4 = int(nyquist/fspace)

    Ll = dB(np.sum(psd[i1:i2])/len(f[i1:i2]))  # level in low band 
    Lm = dB(np.sum(psd[i2:i3])/len(f[i2:i3]))  # level in mid band
    Lh = dB(np.sum(psd[i3:i4])/len(f[i3:i4]))  # level in high band
    '''
    
    bottom_freq = int(500/fspace)  # bottom of the search space
    top_freq = int(11000/fspace)  # set frequency range for analysis (11kHz is max
    if (nyquist < top_freq):      # but if sampling rate is less than 22kHz, the max
        top_freq = nyquist        # is reduced to the Nyquist freq (1/2 the sampling rate)
    min_dist = int(500/fspace)  # for peak picking - peaks must be at least this far apart

    mode = '500/80%/6dB'
    min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.8  # 80% of the range in this spectrum
    min_prom = 6  # dB
    (peaks,prop) = signal.find_peaks(spec[bottom_freq:top_freq], height=min_height,
                                  distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):  # if we didn't find any peaks, relax the constraints
        mode = '500/70%/6dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.7
        min_prom = 6  # dB
        (peaks,prop) = signal.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '500/65%/4dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.65
        min_prom = 4  # dB
        (peaks,prop) = signal.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '500/33%/3dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.33
        min_prom = 3  # dB
        (peaks,prop) = signal.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = 'no peak found'
        peaks = np.append(peaks,[0])

    index_of_main_peak = bottom_freq + peaks[0]  # this is selecting the first peak in an array of peaks
    Fm = freq[index_of_main_peak]      # convert to Hz
    FmB = hz2bark(Fm)                   # convert to Bark
    Am = spec[index_of_main_peak]      # get amplitude

    # -------- moments analysis -----------------------
    bottom_freq = int(300/fspace)
    f = freq[bottom_freq:top_freq]  # taking frequencies from 300 to 11000 (or lower if sf < 22000)
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
    
    return(Fm,FmB,Am,mode,COG,SD,Skew,Kurtosis,spec,freq)
