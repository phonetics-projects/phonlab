from scipy.signal import spectrogram
from scipy.signal import windows
import numpy as np
from ..utils.prep_audio_ import prep_audio

def compute_sgram(x,fs,w,s=0.001,order=13):
    """Compute a spectrogram from input waveform array of samples.
    
    Parameters
    ==========
    x : ndarray
        array of audio samples
    fs : integer
        The sampling frequency of the audio samples in `x` 
    w : float
        Length in seconds of the analysis window.  For an effective filter bandwidth of 300 Hz use w = 0.008, and for an effective filter bandwidth of 45 Hz use w = 0.04.
    s : float, default=0.001
        The time (in seconds) between adjacent spectra in the spectrogram. 
    order : integer, default = 13
        This parameter determines the number of points in the FFT analysis that produces the spectrogram.  The number of points will be a power of 2 (2**order) and should be larder than the number of points in the analysis window (which is w*fs).

    Returns
    ======= 
    t : ndarray
        Array of segment times.
    f : ndarray
        Array of sample frequencies.
    Sxx : ndarray
        Spectrogram of the audio. By default, the last axis of Sxx corresponds to the segment times.
        It is the magnitude spectrum on the decibel scale, so 20 * log10(Sxx) of the spectrogram
        returned by scipy.signal.spectrogram.


    """
    step = s  # step size between spectral slices (sec)
     
    # set up parameters for signal.spectrogram()
    noverlap = int((w-step)*fs) # skip forward by step between each frame
    nperseg = int(w*fs)         # number of samples per waveform window
    nfft = np.power(2,order)    # number of points in the fft
    window = windows.blackmanharris(nperseg)

    f,ts,Sxx = spectrogram(x,fs=fs,noverlap = noverlap, window=window, nperseg = nperseg, 
                              nfft = nfft, scaling='spectrum', mode = 'magnitude')
    Sxx = 20 * np.log10(np.maximum(Sxx, np.finfo(Sxx.dtype).tiny))  # put spectrum on decibel scale

    return (ts, f, Sxx)
    

SGRAM_TYPES = ('fft', 'multitaper', 'cepstral', 'lpc', 'mel', 'bark')

# default gray scale floor (see the `min_prop` parameter of sgram()) for each type of spectrogram
DEFAULT_MIN_PROP = {'fft': -0.1, 'multitaper': 0, 'cepstral': 0, 'lpc': 0, 'mel': 0, 'bark': -0.1}


def _typed_sgram(x, fs, type, w, s=None, verbose=False, **kwargs):
    """Compute a spectrogram of the requested `type` and return it in the common layout used by `sgram()`.

    Returns
    =======
    ts : ndarray
        frame times
    f : ndarray
        the frequency axis (Hz, mels, or Bark depending on `type`)
    Sxx : ndarray
        the spectrogram, shape (len(f), len(ts))
    ylabel : string
        label for the frequency axis
    win : float
        the effective length (sec) of the analysis window, used to size the figure
    """
    if type == 'fft':
        ts, f, Sxx = compute_sgram(x, fs, w, s=s or 0.001, **kwargs)
        return ts, f, Sxx, "Frequency (Hz)", w
    elif type == 'multitaper':
        from .multitaper_ import compute_mt_sgram
        kwargs.setdefault('order', 13)  # same FFT size as compute_sgram()
        kwargs.setdefault('NW', 3)      # half-bandwidth NW/w = 150 Hz when w = 0.02
        ts, f, Sxx = compute_mt_sgram(x, fs, w=w, s=s or 0.001, **kwargs)
        return ts, f, Sxx, "Frequency (Hz)", w
    elif type == 'cepstral':
        from .cepstral import cepstral_smooth
        l = kwargs.pop('l', 0.04)
        # the cepstrally smoothed spectrum, Exx, has shape (frequency, time)
        ts, f, Exx, _, _, _, _ = cepstral_smooth(x, fs, tf=int(fs//2), s=s or 0.005, l=l, **kwargs)
        return ts, f, Exx, "Frequency (Hz)", l
    elif type == 'lpc':
        from .rlpc import RLPC_formants
        l = kwargs.pop('l', 0.03)
        order = kwargs.pop('order', -1)
        nfft = 2**kwargs.pop('nfft_order', 13)
        # The audio has already been resampled and preemphasized by sgram(), so don't do it again.
        fmtdf, A = RLPC_formants(x, fs, tf=int(fs//2), s=s or 0.005, l=l, preemphasis=0.0, 
                                 order=order, verbose=verbose, **kwargs)
        # LPC spectra are not windowed, so (as in nasality()) a correction puts them on the
        # same dB scale as a Blackman-Harris windowed FFT spectrum.
        window = windows.blackmanharris(int(l*fs))
        amp_correction = 2.5 + 20 * np.log10(np.sum(window)/np.sqrt(np.sum(np.square(window))))
        tiny = np.finfo(float).tiny
        gain = fmtdf.gain.to_numpy()
        lpc_mag = np.abs(np.fft.rfft(A, n=nfft)).T      # shape (frequency, time)
        Lxx = 20 * np.log10(np.maximum(np.sqrt(gain), tiny)/np.maximum(lpc_mag, tiny)) - amp_correction
        return fmtdf.sec.to_numpy(), np.fft.rfftfreq(nfft, 1/fs), Lxx, "Frequency (Hz)", l
    elif type == 'mel':
        from ..auditory.mel_sgram import compute_mel_sgram
        # compute_mel_sgram() uses a fixed 64 ms window and 10 ms step unless step_sec is given,
        # and returns an array with shape (time, mel)
        ts, f, Sxx = compute_mel_sgram(x, fs, **kwargs)
        return ts, f, Sxx.T, "Frequency (mel)", 0.064
    elif type == 'bark':
        from ..auditory.audspec import Audspec
        step = kwargs.pop('step_size', max(s or 0.005, 0.005))
        aud = Audspec(fs=fs, step_size=step)
        aud.make_zgram(x, fs, preemph=0, **kwargs)   # preemphasis was already applied by sgram()
        return aud.time_axis, aud.zfreqs, aud.zgram.T, "Frequency (Bark)", 0.05
    raise ValueError(f"Unknown sgram type {type!r}. Choose one of {SGRAM_TYPES}.")


def sgram(x,fs, start=0, end=-1, tf=8000, band='wb', preemph = 0.94, font_size = 14,
    min_prop = None, save_name='', slice_time=-1, cmap='Greys', resize=True, ax=None,
    type='fft', verbose=False, **kwargs):
    """Make pretty good looking spectrograms

    * By default this function calls `compute_sgram()` (which calls scipy.signal.spectrogram) to calculate a magnitude spectrogram, which is then transformed to decibels, and passed to plt.imshow for plotting.  Other kinds of spectrogram can be chosen with the `type` parameter.
    
    * It mainly is used to produce nice looking figures with features like readable time and frequency axes, scaling so that the time axis is 6.5 inches per second for spectrograms of less than 2 seconds.
    
    * The function also returns arrays that you can use to create your own figures.  
    
    * The function uses one of two window lengths - 40 msec for narrow band spectrograms, or 8 msec for wideband spectrograms.  

    * One option is to add a "spectral slice" to the display - the amplitude/frequency spectrum at a particular point in time.

    Parameters
    ==========
    x : ndarray
        a one-dimensional array of audio samples.
    fs : numeric
        The sampling rate of the audio in **x**
    start : float, default = 0
        starting time (in seconds) of the waveform chunk to plot -- default plot whole file
    end : float, default = -1
        ending time (in seconds) of the waveform chunk to plot (-1 means go to the end)
    tf : integer, default = 8000
        the top frequency (in Hz) to show in the spectrogram
    band : string, {'wb','nb'}
        effective filter bandwidth of the analysis filter ('wb' = 300 Hz, 'nb' = 45 Hz)
    preemph : float, default = 0.94
        add high frequency preemphasis before making the spectrogram, a value between 0 and 1
    font_size : integer, default = 14
        the font size to use for the axis labels and tick labels.
    min_prop : float, default = None
        set the 'floor' of the gray scale, as a proportion of the range between the 20th and 80th
        percentiles of the values in the spectrogram (so a value of 0.5 puts the floor halfway
        from the 20th to the 80th percentile, and 0 puts it at the 20th percentile).  If `None`, a default value
        is chosen that suits the spectrogram `type`: -0.1 for 'fft' and 'bark', and 0 for all other types.
    save_name : Path, default = ''
        name of a file to save the figure pyplot.savefig(), by default no file is saved.
    slice_time : float, default = -1
        location (in seconds) of an optional spectral slice.
    resize : boolean, default = True
        if a matplotlib axes object is passed in (the `ax` parameter), resize it to 
        classical spectrogram dimensions.
    ax : a matplotlib axes object, default = None
        the user may provide a matplotlib object to which the spectrogram will be plotted.
    cmap : string, default = "Grays"
        name of a matplotlib colormap for the spectrogram
    type : string, {'fft','multitaper','cepstral','lpc','mel','bark'}, default = 'fft'
        the kind of spectrogram to compute and display.  
        
        * 'fft' - a Blackman-Harris windowed FFT magnitude spectrogram in dB, from `compute_sgram()`.
        * 'multitaper' - a multitaper power spectrogram in dB, from `compute_mt_sgram()`.  `band` is ignored: the effective bandwidth is determined by the analysis window length and the time-bandwidth product, so the defaults are a 20 ms window (`w=0.02`) and `NW=3` (a half-bandwidth of NW/w = 150 Hz).  Pass `w` and `NW` to change them.
        * 'cepstral' - a cepstrally smoothed spectrogram (the spectral envelope without the harmonics), from `cepstral_smooth()`.  `band` is ignored: the analysis window is 40 ms (pass `l` to change it), the default step is 5 ms, and the amount of smoothing is set by `lift`.
        * 'lpc' - a spectrogram of the spectral envelope found by robust linear prediction, from `RLPC_formants()`, computed as in `nasality()`.  `band` is ignored: the analysis window is 30 ms (pass `l` to change it), the default step is 5 ms, and the LPC order is chosen automatically unless `order` is passed.  The LPC spectrum is evaluated with a 2**13 point FFT (pass `nfft_order` to change it).
        * 'mel' - a log mel-frequency spectrogram from `compute_mel_sgram()`.  The window length (64 ms) and frequency range (80-7600 Hz) are fixed by that function, so `band` is ignored, `tf` is treated as at least 8000, and the frequency axis is in mels. The default step size is 10 ms; pass `step_sec` to change it.
        * 'bark' - an auditory (critical band) spectrogram from `Audspec.make_zgram()`.  `band` is ignored and the frequency axis is in Bark.  The default step size is 5 ms; pass `step_size` to change it.
    verbose : boolean, default = False
        print diagnostic messages.  This is passed to `prep_audio()` (resampling and preemphasis of the audio) and, for `type='lpc'`, to `RLPC_formants()`.
    **kwargs :
        additional keyword arguments are passed to the function that computes the spectrogram.  The exception is `w`, the length (in seconds) of the analysis window, which overrides the window length implied by `band` for the 'fft' and 'multitaper' types (and is the same as `l` for the 'cepstral' and 'lpc' types; it is an error for 'mel' and 'bark').  For example, `NW=4` for `type='multitaper'`, `s=0.002` or `order=12` for `type='fft'`, `lift=0.002` for `type='cepstral'`, `order=14` for `type='lpc'`, or `step_sec=0.005` for `type='mel'`.

    Returns
    ======= 
    ax : a matplotlib axes object
        The plot axes is returned
    f : ndarray
        Array of sample frequencies (in Hz, mels or Bark depending on `type`).
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of the audio. The last axis of Sxx corresponds to the segment times. 
        For the default `type='fft'` it is the magnitude spectrum on the decibel scale, so 
        20 * log10(Sxx) of the spectrogram returned by scipy.signal.spectrogram.

    Examples
    ========

    Plot a spectrogram of a portion of the sound file from 1.5 to 2 seconds.  
    Then add a vertical red line at time 1.71

    .. code-block:: Python

        import matplotlib.pyplot as plt

        audio_dir = importlib.resources.files('phonlab') / 'data' / 'example_audio'
        example_file = audio_dir / 'sf3_cln.wav'
        
        x,fs = phon.loadsig(example_file,chansel=[0])
        phon.sgram(x,fs,start=1.5, end=2.0)
        plt.axvline(1.71,color="red")

    .. figure:: images/burst.png
       :scale: 50 %
       :alt: a spectrogram with a red line marking the location of the burst
       :align: center

       Marking the burst found by `phon.burst()`

    """
    import matplotlib.pyplot as plt

    target_fs = tf*2    # top frequency is the Nyquist frequency for the analysis

    if type not in SGRAM_TYPES:
        raise ValueError(f"Unknown sgram type {type!r}. Choose one of {SGRAM_TYPES}.")
    if type == 'mel':
        tf = max(tf, 8000)   # the mel filterbank extends to 7600 Hz

    if band=='nb':
        w = 0.04    # analysis window size for narrow band spectrogram (sec)
    else:
        w = 0.008   # analysis window size for wide band spectrogram
    if type == 'multitaper':
        w = 0.02    # the effective bandwidth of a multitaper spectrogram depends on both w and NW, not on `band`

    # an explicit window length (sec) overrides the one implied by `band`
    w_user = kwargs.pop('w', None)
    if w_user is not None:
        if type in ('mel', 'bark'):
            raise ValueError(f"The analysis window length of a '{type}' spectrogram cannot be set with `w`.")
        elif type in ('cepstral', 'lpc'):
            kwargs.setdefault('l', w_user)   # these functions call the window length `l`
        else:
            w = w_user

    # set up parameters for the spectrogram window
    figheight = 4.5  # height in inches
    max_figwidth = 12 # maximum figure width in inches
    inches_per_sec = 6.5 # desired width scaling of printed spectrogram
    slice_width = 1.5  # how much space to give to the spectral slice
    cmap = plt.get_cmap(cmap)
    
    # ----------- condition waveform -----------------------
    x2, fs = prep_audio(x,fs, target_fs = target_fs, 
                        pre = preemph, verbose=verbose)

    i1 = int(start * fs)   # index of starting time: seconds to samples
    i2 = int(end * fs)     # index of ending time
    if i2<0 or i2>len(x2):  # stop at the end of the waveform
        i2 = len(x2)
    if i1>i2:              # don't let start follow end
        i1=0
    
    
    # ----------- compute the spectrogram ---------------------------------
    ts,f,Sxx,ylabel,win = _typed_sgram(x2[i1:i2],fs,type,w,verbose=verbose,**kwargs)
    
    # ------------ display in a matplotlib figure --------------------
    ts = np.add(ts,start)  # increment the spectrogram times by the start value
    dur = max(ts)-min(ts) + win   # scale figure size
    figwidth = np.min([(dur * inches_per_sec), max_figwidth])
    if ax is None:
        if slice_time>0: # if spectral slice is desired, add an axes for it
            fig = plt.figure(figsize=(figwidth+slice_width, figheight),dpi=72)
            gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[figwidth/slice_width, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
        else:
            fig = plt.figure(figsize=(figwidth, figheight),dpi=72)
            ax1 = fig.add_subplot(111)
    else:
        if resize:
            fig = plt.gcf()  # get the current figure
            fig.set_size_inches(figwidth, figheight) # resize it by the values here
        ax1 = ax

    if min_prop is None:
        min_prop = DEFAULT_MIN_PROP[type]
    lo, hi = np.percentile(Sxx, [20, 80])
    vmin = lo + (hi-lo)*min_prop
    extent = (min(ts),max(ts),min(f),max(f))  # get the time and frequency values for indices.
    im = ax1.imshow(Sxx, aspect='auto', interpolation='nearest', cmap=cmap, vmin = vmin, 
                extent = extent, origin='lower')
    ax1.grid(which='major', axis='y', linestyle='-')  # add grid lines
    ax1.set_xlabel("Time (sec)", size=font_size)
    ax1.set_ylabel(ylabel, size=font_size)
    ax1.tick_params(labelsize=font_size)
    ax1.locator_params(axis='y', prune="upper")  # for stacking sgram with other axes
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.subplots_adjust(left=0.1, bottom=0.148, right=0.99, top=0.99, wspace=0, hspace=0)

    if slice_time > 0:  # if spectral slice is desired, plot the spectrum
        i = np.argmin(np.abs(ts-slice_time))  # find the index of the spectral slice
        ax1.axvline(x=slice_time,color='black',linestyle="--")
        spectrum = Sxx.T[i]  
        ax2.plot(spectrum,f,color='black') 
        ax2.grid(which='major', axis='y', linestyle=':')  # add grid lines
        ax2.set_ymargin(0)    # put y-axis at bottom and top of axis (as in spectrogram)
        ax2.tick_params(labelleft=False,labelsize=font_size)  # do not write the frequency axis labels
    
    if len(save_name)>0:
        print(f'Saving file: {save_name}')
        plt.savefig(save_name,dpi=300,bbox_inches='tight')
        
    return (ax1, f,ts,Sxx)
