from scipy.signal import spectrogram
from scipy.signal import windows
import numpy as np
import matplotlib.pyplot as plt
from ..utils.prep_audio_ import prep_audio

def compute_sgram(x,fs,w):
    """Compute a spectrogram from input waveform array of samples.
    
    Parameters
    ==========
    x : ndarray
        array of audio samples
    fs : integer
        The sampling frequency of the audio samples in `x` 
    w : float
        Length in seconds of the analysis window.  For an effective filter bandwidth of 300 Hz use w = 0.008, and for an effective filter bandwidth of 45 Hz use w = 0.04.

    Returns
    ======= 
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of the audio. By default, the last axis of Sxx corresponds to the segment times.
        It is the magnitude spectrum on the decibel scale, so 20 * log10(Sxx) of the spectrogram
        returned by scipy.signal.spectrogram.


    """
    x2 = np.rint(32000 * (x/max(x))).astype(np.intc)  # scale the signal

    step = 0.001  # step size between spectral slices (sec)
    order = 13    # FFT size = 2 ^ order
    
    # set up parameters for signal.spectrogram()
    noverlap = int((w-step)*fs) # skip forward by step between each frame
    nperseg = int(w*fs)         # number of samples per waveform window
    nfft = np.power(2,order)    # number of points in the fft
    window = windows.blackmanharris(nperseg)

    f,ts,Sxx = spectrogram(x2,fs=fs,noverlap = noverlap, window=window, nperseg = nperseg, 
                              nfft = nfft, scaling='spectrum', mode = 'magnitude', detrend = 'linear')
    Sxx = 20 * np.log10(Sxx+1)  # put spectrum on decibel scale

    return (f,ts, Sxx)
    

def sgram(x,fs, start=0, end=-1, tf=8000, band='wb', preemph = 0.94, font_size = 14,
    min_prop = 0.2, save_name='', slice_time=-1, cmap='Greys', ax=None):
    """Make pretty good looking spectrograms

    * This function calls scipy.signal.spectrogram to calculate a magnitude spectrogram, which is then transformed to decibels, and passed to plt.imshow for plotting.  
    
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
    min_prop : float, default = 0.2
        set the 'floor' of the gray scale.  The default value specifies that the floor will be 
        at 20% of the range between min and max amplitudes. 
    save_name : Path, default = ''
        name of a file to save the figure pyplot.savefig(), by default no file is saved.
    slice_time : float, default = -1
        location (in seconds) of an optional spectral slice.
    cmap : string, default = "Grays"
        name of a matplotlib colormap for the spectrogram

    Returns
    ======= 
    ax : a matplotlib axes object
        The plot axes is returned
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of the audio. By default, the last axis of Sxx corresponds to the segment times.
        It is the magnitude spectrum on the decibel scale, so 20 * log10(Sxx) of the spectrogram
        returned by scipy.signal.spectrogram.

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

    Read a file into an array `x`, track the formant frequencies in the file, use them to produce
    sine wave speech, and then plot a spectrogram of the resulting signal.

    .. code-block:: Python

        example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'sf3_cln.wav'

        x,fs = phon.loadsig(example_file, chansel=[0]) 
        fmtsdf = phon.track_formants(x,fs)    # track the formants
        x2,fs2 = phon.sine_synth(fmtsdf)     # use the formants to produce sinewave synthesis
        ax1,f,t,Sxx = phon.sgram(x2,fs2, band="nb", preemph=0)  # plot a spectrogram of it

    .. figure:: images/sine_synth.png
       :scale: 40 %
       :alt: a spectrogram of sine-wave synthesis
       :align: center

       Showing the spectrogram of sine-wave synthesis.

    """
    target_fs = tf*2    # top frequency is the Nyquist frequency for the analysis

    if band=='nb':
        w = 0.04    # analysis window size for narrow band spectrogram (sec)
    else:
        w = 0.008   # analysis window size for wide band spectrogram

    # set up parameters for the spectrogram window
    figheight = 4.5  # height in inches
    max_figwidth = 12 # maximum figure width in inches
    inches_per_sec = 6.5 # desired width scaling of printed spectrogram
    slice_width = 1.5  # how much space to give to the spectral slice
    cmap = plt.get_cmap(cmap)
    
    # ----------- condition waveform -----------------------
    x2, fs = prep_audio(x,fs, target_fs = target_fs, pre = preemph,quiet=True)

    i1 = int(start * fs)   # index of starting time: seconds to samples
    i2 = int(end * fs)     # index of ending time
    if i2<0 or i2>len(x2):  # stop at the end of the waveform
        i2 = len(x2)
    if i1>i2:              # don't let start follow end
        i1=0
    
    
    # ----------- compute the spectrogram ---------------------------------
    f,ts,Sxx = compute_sgram(x2[i1:i2],fs,w)
    
    # ------------ display in a matplotlib figure --------------------
    ts = np.add(ts,start)  # increment the spectrogram time by the start value
    dur = max(ts)-min(ts) + w   # scale figure size
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
        fig = plt.gcf()  # get the current figure
        fig.set_size_inches(figwidth, figheight) # resize it by the values here
        ax1 = ax

    vmin = np.min(Sxx) + (np.max(Sxx)-np.min(Sxx))*min_prop
    extent = (min(ts),max(ts),min(f),max(f))  # get the time and frequency values for indices.
    im = ax1.imshow(Sxx, aspect='auto', interpolation='nearest', cmap=cmap, vmin = vmin, 
                extent = extent, origin='lower')
    ax1.grid(which='major', axis='y', linestyle=':')  # add grid lines
    ax1.set_xlabel("Time (sec)", size=font_size)
    ax1.set_ylabel("Frequency (Hz)", size=font_size)
    ax1.tick_params(labelsize=font_size)

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
