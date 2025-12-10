from scipy import fft
from scipy.signal import windows
from librosa import util
from pandas import DataFrame
import numpy as np
from ..utils.prep_audio_ import prep_audio

def formantPeakPick(spec,n=6):  # maybe replace with scipy.pickpeaks ??
    '''
    spec is a one dimensional array, and we look for peaks that are at least (n-1)*2 points wide

    return an array of indices - the locations of the peaks
    '''
    peakIndex=np.array([])
    for i in range(n,len(spec)-n):
        is_peak = True
        for k in range(1,n):  is_peak = is_peak and spec[i]>=spec[i-k] and spec[i]>=spec[i+k]
        if is_peak:  peakIndex = np.append(peakIndex,i)
            
    return peakIndex

    
def track_formants_DPPT(x,fs, pre=0, l=0.03, s=0.01, deltaF=1100, n=6):
    '''Bozkurt et al.'s (2004) Differential-Phase Peak Tracking (DPPT) method of vowel formant 
    tracking is implemented in this function. The implementation here is a translation and 
    interpretation of the matlab routine `formant_CGDZP()` written by Bozkurt and Drugman and 
    published in the Covarep repository.

    This algorithm tracks formants as peaks in the smoothed spectrum, therefore tends to 
    be more likely than the LPC or IFC methods to track the harmonics of the voice 
    fundamental frequency when pitch is high, and to report a single peak when 
    formants have similar frequencies - reporting the same formant frequency for 
    the close formants, such as when F1 and F2 are close in [u] or F2 and F3 are close
    in [i]. 

    It may be that measuring the peaks in a smoothed spectrum is a more 
    perceptually valid representation of vowel acoustics than a model that focusses on 
    determining the resonances of the vocal tract (see Chistovich & Lublinskaya, 1979).  
    Thus, it is appropriate to consider the formant frequencies reported by this DPPT 
    algorithm to be `perceptual formants`.
    

    Parameters
    ----------
    x : array
        a one-dimensional array of audio samples

    fs : int
        the sampling rate of the audio in **x** 

    deltaF : int, default = 1100
        The expected average interval between formants, used to sort peaks into formant numbers.

    preemphasis : float, default = 0
        value of the preemphasis factor (0-1).  It almost never helps to change this.

    l : float, default = 0.03
        Length of the pitch analysis window in seconds. The default is 30 milliseconds.  

    s : float, default = 0.01
        "Hop" interval between successive analysis windows. The default is 10 milliseconds

    n : int, default = 6
        The required width of a peak, in terms of the number of steps in the spectrum.  The default as in Bozkurt et al. is about 80 Hz wide.


    Returns
    -------
    df : dataframe
        a pandas dataframe with formant measurements at 10 ms intervals (by default).

    Note
    ----

    The columns in the output dataframe are:
        * sec - measurement time
        * F1-4 - the lowest four vowel formants - peaks in the smoothed spectrum.
        

    References
    ----------

    B. Bozkurt, B. Doval, C.D. Alessandro, T. Dutoit (2004) Improved differential phase spectrum processing for formant tracking. InterSpeech ICSLP, 8th International Conference on Spoken Language Processing, Jeju Island, Korea.
    
    L. A. Chistovich, V.V Lublinskaya (1979) The 'center of gravity' effect in vowel spectra and critical distance between the formants: Psychoacoustical study of the perception of vowel-like stimuli. `Hearing Research`, 1, 185-195. 

    Examples
    --------

    This code block shows a call to phon.track_formants_DPPT() and also shows a call to LPC formant tracking, the
    default method in phon.track_formants(), and uses phon.sgram, and seaborn pointplot() to plot the estimated
    formant frequences on the spectrogram.

    .. code-block:: Python
    
        example_file = importlib.resources.files('phonlab') / 'data/example_audio/sf3_cln.wav'
        x,fs = phon.loadsig(example_file,chansel=[0])

        ret = phon.sgram(x,fs, tf=6000, cmap="Reds")  # plot the spectrogram

        dflpc = phon.track_formants(x,fs)

        dot_color = 'dodgerblue'
        sns.pointplot(dflpc,x='sec',y='F1',linestyle='none',native_scale=True,marker=".",color=dot_color)
        sns.pointplot(dflpc,x='sec',y='F2',linestyle='none',native_scale=True,marker=".",color=dot_color)
        sns.pointplot(dflpc,x='sec',y='F3',linestyle='none',native_scale=True,marker=".",color=dot_color)
        sns.pointplot(dflpc,x='sec',y='F4',linestyle='none',native_scale=True,marker=".",color=dot_color)

        df = phon.track_formants_DPPT(x,fs,deltaF=1100)

        dot_color = "darkblue"
        sns.pointplot(df,x='sec',y='F1',linestyle='none',native_scale=True,marker=".",color=dot_color)
        sns.pointplot(df,x='sec',y='F2',linestyle='none',native_scale=True,marker=".",color=dot_color)
        sns.pointplot(df,x='sec',y='F3',linestyle='none',native_scale=True,marker="x",color=dot_color)
        sns.pointplot(df,x='sec',y='F4',linestyle='none',native_scale=True,marker=".",color=dot_color)

    

    .. figure:: images/DPPT.png
       :scale: 33%
       :alt: a spectrogram with the estimated vowel formants marked with light and dark blue dots 
       :align: center

       Plotting the formants found by `track_formants()` (light blue dots) and those found by 
       `track_formants_DPPT()` (dark blue dots) on the spectrogram of the utterance.


    '''
    x,fs = prep_audio(x, fs, target_fs=16000, pre=pre, quiet=True)  # resample to 16kHz

    numFormants = 5
    fsLR = 2048
    viewRange = round(fsLR/3.2)

    frame_length = int(l*fs)
    step = int(s*fs)
    half_frame = round(frame_length/2)
    frame_length = half_frame * 2 + 1    # odd number in frame
    NFFT = int(2**(np.ceil(np.log(frame_length)/np.log(2))))
    n_list = range(NFFT)  # 0,1,2,3,...NFFT-1
    #ScalingFreqAxis = np.arange(0.5,1.5,1/(viewRange-1))

    frames = util.frame(x,frame_length=frame_length, hop_length=step,axis=0)

    nb = frames.shape[0]
    f = frames.shape[1]
    w = windows.gaussian(frame_length,half_frame)  

    # calculate the zero-phase spectrogram
    zero_phase = np.real(fft.ifft(np.abs(fft.fft(np.diff(w*frames,append=0),NFFT))))

    Rfix = meanR = 1.12
    Rlist = np.array([])  # history of found R values

    ts = (np.array(range(nb)) * step + half_frame)/fs
    formants = np.ones((nb,numFormants))*np.nan   # fill with NaN

    for kk in range(nb):
        numPeaks = 0
        R = meanR  # using past values of R seems to speed up processing dramatically
        # -- adjust R:  the radius of the analysis circle in Z plane -----
        while (numPeaks!=numFormants and R>1.01 and R<1.25):  
            expEnv = np.exp(np.log(1/R)*n_list)  # consider precomputing these and use table look up here
            fourierTrans = fft.fft(zero_phase[kk]*expEnv,fsLR)
            angFFT = np.angle(fourierTrans[1:viewRange])
            chirpGroupDelay = -np.diff(angFFT)  # the chirp group delay spectrum

            peakIndex = formantPeakPick(chirpGroupDelay,n)
            numPeaks = len(peakIndex)
            if numPeaks>numFormants and R>=Rfix:
                peakIndex = peakIndex[0:numFormants]
                R += 0.01
            elif numPeaks<numFormants and R<=Rfix:
                peakIndex = np.append(peakIndex,np.zeros(numFormants-len(peakIndex)))
                R -= 0.01
            else:
                break
        if numPeaks>numFormants:
            peakIndex = peakIndex[0:numFormants]
        elif numPeaks<numFormants:
            peakIndex = np.append(peakIndex,np.zeros(numFormants-len(peakIndex)))
        Rlist = np.append(Rlist,R)
        meanR = np.mean(Rlist)  # use past values in future frames
        #print(f"{kk}, {ts[kk]}, {meanR:0.3f},{peakIndex*fs/fsLR}")
        formants[kk,:] = peakIndex

    formants = formants*fs/fsLR

    # assign peaks to formants 1-4 based on expected formant values
    Fs_expected = np.zeros(5)
    Fs_expected[0] = deltaF/2  # set expectations for fornant frequencies based on deltaF
    for i in range(1,5): Fs_expected[i] = Fs_expected[i-1] + deltaF

    # tricky line - gives index number for each peak based on which expected formant it is closest to
    idxs = np.reshape(np.concatenate([np.nanargmin(np.fabs(formants-Fs_expected[i]),axis=-1) for i in range(numFormants)]),(numFormants,nb)).T
    newFs = [formants[i,idxs[i,:]] for i in range(nb)]

    list = ['sec', ] + [f"F{i}" for i in range(1,numFormants+1)]
    return DataFrame(np.concatenate((ts.reshape(nb,1), newFs), axis=1),columns=(list))

