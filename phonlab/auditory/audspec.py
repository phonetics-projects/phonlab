import librosa
import numpy as np
from numpy.polynomial import Polynomial
from scipy.fft import rfft
import scipy.stats as stats
from ..utils.prep_audio_ import prep_audio

class Audspec(object):
    """ Create an an Audspec object; analysis parameters and routines for creating auditory spectrograms.

    Parameters
    ==========
    fs : int, default=16000
        The desired sampling rate of audio analysis, determines the frequency range of the auditory spectrogram.  Note that if the value given here exceeds the sampling rate of the file passed, there can be 'empty' space at the top of the auditory spectrogram.
    step_size : float, default=0.03
        The interval (in seconds) between successive analysis frames.

    Returns
    =======
    object: Audspec
        The object returned by the constructor function is ready for calls to functions like object.make_zgram(), object.make_sharpgram(), etc. to compute auditory representations of sound.

    Examples
    ========
    The examples here show the use of the `Audspec` class to create auditory representations of sounds, roughly based on the properties of the human cochlea and aspects of auditory processing in the brainstem.  The most basic representation is the auditory spectrogram (internally called the `zgram` and sometimes referred to as the `audiogram`), which is simply a critical band filtered spectrogram.

    .. code-block:: Python

         x,fs = phon.loadsig("sf3_cln.wav",chansel=[0])
         aud = phon.Audspec()
         aud.make_zgram(x,fs)
        
         # ---- the rest is to make a nice plot ----
         
         fig,ax = plt.subplots(2,figsize=(6,5))
         
         Hz_extent = (min(aud.time_axis), max(aud.time_axis),
               min(aud.fft_freqs), max(aud.fft_freqs))  # time and frequency values for sgram.
         ax[0].imshow(20*np.log10(aud.sgram.T),origin='lower', aspect='auto', interpolotion="spline36",
                  extent=Hz_extent, cmap = plt.cm.Greys)
         ax[0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
         ax[1].imshow(aud.zgram.T,origin='lower', aspect='auto', interpolotion="spline36",
                  extent=aud.extent, cmap = plt.cm.Purples)
         ax[1].set(xlabel="Time (sec)", ylabel="Frequency (Bark)")
         fig.tight_layout()

    .. figure:: images/make_zgram.png
       :scale: 50 %
       :alt: An acoustic narrow band spectrogram and the auditory spectrogram of the same utterance.
       :align: center

       (top) Acoustic narrow band spectrogram, and (bottom) Auditory spectrogram of the same utterance.

    One type of secondary auditory processing sharpens the frequencies of the audiogram, simulating frequency selectivity through lateral inhibition.
    
    .. code-block:: Python

        aud.make_sharpgram()
        fig,ax = plt.subplots(1,figsize=(8,3))

        ax.imshow(aud.sharpgram.T,origin='lower', aspect='auto',
                 extent=aud.extent, interpolation="spline36", cmap = plt.cm.Reds)
        ax.set(xlabel="Time (sec)", ylabel="Frequency (Bark)")

    .. figure:: images/make_sharpgram.png
       :scale: 50 %
       :alt: The "sharpgram" auditory representation
       :align: center

       (top) The "sharpgram" auditory representation.

    Another auditory representation emphasizes temporal changes in the audiogram.  In the `Audspec` class this is referred to as a `tgram`.

    .. code-block:: Python

        aud.make_tgram()
        fig,ax = plt.subplots(1,figsize=(8,3))

        ax.imshow(aud.tgram.T,origin='lower', aspect='auto',
                 extent=aud.extent, interpolation="spline36", cmap = plt.cm.afmhot)
        ax.set(xlabel="Time (sec)", ylabel="Frequency (Bark)")

    .. figure:: images/make_tgram.png
       :scale: 50 %
       :alt: The "tgram" auditory representation, showing the gradient of change in critical bands
       :align: center

       (top) The "tgram" auditory representation, showing the gradient of change in critical bands.


    Methods and Attributes in an Audspec object
    ===========================================
    
    """

        
    def hz2bark(self, hz):
        return 7 * np.arcsinh(hz/650)

    def bark2hz(self, bark):
        return 650 * np.sinh(bark/7)

    def _make_cb_filters(self):
        """        
        Create and return 2d array of Patterson filters for DFT spectra based
        on attribute values in `self`.

        The returned filters are stored in an 2d array in which the
        rows represent the filter frequency bands in ascending order. The
        columns contain symmetrical filter coefficients as determined by the
        Patterson (1976) formula and centered at the filter frequency in the
        DFT spectrum. Filter coefficients outside the frequency band are set
        to 0.0.

        The one-sided length of filter coefficients for each band is stored
        in the `cbfiltn` attribute. The number of coefficients in the
        symmetrical filter for band `j` is therefore
        `(self.cbfiltn[j] * 2) - 1`. In a few bands this calculation might not
        be correct since the coefficients may not fit when the center frequency
        is near the left or right edge of the DFT spectrum. In such cases the
        coefficients are truncated, and the actual number of coefficients for
        the band `j` can be found with `np.sum(self.cbfilts[j] != 0.0)`.

        References
        ----------
        R. D. Patterson (1976) Auditory filter shapes derived with noise stimuli. `JASA` **59** , 640-54.
        """
        
        cbfilts = np.zeros([len(self.freqs), len(self.sgram)])
        dfreq = np.arange(self.maxcbfiltn) * self.inc
        cbfiltn = np.searchsorted(dfreq, self.freqs / 5)
        cbfiltn[cbfiltn > self.maxcbfiltn] = self.maxcbfiltn
        self.cbfiltn = cbfiltn
        for j, iidx in enumerate(cbfiltn):
            cbfilt = np.zeros(self.maxcbfiltn)
            bw = 10.0 ** ( (8.3 * np.log10(self.freqs[j]) - 2.3) / 10.0 )
            hsq = np.exp(-np.pi * ((dfreq[:iidx] / bw) ** 2))
            cbfilt[:iidx] = np.sqrt(hsq)
            cbfilt /= cbfilt[0] + np.sum(cbfilt[1:] * 2)

            # Make a symmetrical array of coefficients centered at loc.
            # [n, n-1, ..., 2, 1, 0, 1, 2, ... n-1, n]
            loc = (self.freqs[j] / self.inc).astype(int) # get location in dft spectrum
            left_n = iidx if iidx <= loc else loc
            right_n = iidx if loc + iidx < (self.dft_n / 2) else int(self.dft_n / 2) - loc
            coef = np.append(np.flip(cbfilt[:left_n])[:-1], cbfilt[:right_n])
            startidx = loc - left_n + 1
            endidx = loc + right_n
            cbfilts[j, startidx:endidx] = coef
        return cbfilts

    def _make_sgram(self, data, *args, **kwargs):
        '''
        Private function to make an acoustic spectrogram via rfft().
        And add equal loudness contour.

        Parameters
        ----------

        data: 1d array
        Audio data.

        kwargs: dict, optional
        Keyword arguments will be passed to the scipy.fft.rfft() function.

        Returns
        -------
        A tuple, consisting of:

        sgram: 2D array
            The acoustic spectrogram of shape (times, frequency bins).

        spect_times: 1D array
            The times of each spectral slice in `spect`.
        '''
        data = np.pad(data, [int(self.dft_n/2), int(self.dft_n/2)], 'constant')
        if (np.max(data) < 1):   # floating point but we need 16bit int range
            data = (data*(2**15)) #.astype(np.int32)

        hop = int(self.step_size * self.fs)
        frames = librosa.util.frame(data,frame_length=self.dft_n,hop_length=hop
            ).transpose()
        t = librosa.frames_to_time(np.arange(frames.shape[0]),
            sr=self.fs,hop_length=hop,n_fft=self.dft_n)
        # Add some noise, then scale frames by the window.
        frames = (frames + np.random.normal(0, 1, frames.shape)) * np.hamming(self.dft_n)
        A = 2/self.dft_n * rfft(frames, **kwargs)
        sgram = (np.abs(A)).astype(self.sgram.dtype)
        return (sgram, t)



    def make_zgram(self, x, fs, target_fs=16000, preemph = 0, **kwargs):
        '''Make an auditory spectrogram by creating an acoustic spectrogram and then applying critical-band filters to it, using the filter shapes described by Patterson (1976).  The function creates the auditory spectrogram and stores it in `self.zgram`.

Parameters
----------

x : ndarray
    Audio data as a one dimensional numpy array 

fs : int, default = 16000
    the sampling rate of the audio in **x**.

preemp : float, default = 1.0
    The amount of preemphasis to apply before filtering.

**kwargs: dict, optional
    Keyword arguments to be passed to scipy.fft.rfft() 


References
----------

R. D. Patterson (1976) Auditory filter shapes derived with noise stimuli. `J. Acoust. Soc. Am.` **59** , 640-54.
        '''
        
        x, fs = prep_audio(x, fs = fs, target_fs = self.fs, pre = preemph)
        (sgram, self.time_axis) = self._make_sgram(x, kwargs)
        self.sgram = sgram + self.loud
        zgram = self.sgram[:, np.newaxis, :] * self.cbfilts[np.newaxis, :, :]
        self.zgram = 20 * np.log10(zgram.sum(axis=2)+1)
        self.extent = (min(self.time_axis),max(self.time_axis),min(self.zfreqs),max(self.zfreqs)) 
        
    def create_sharp_filter(self, span=4, mult=3, dimension="frequency"):
        if (dimension=="frequency"):  # default value
            steps = int(span / self._zinc)
        else:  # otherwise assume temporal sharpening
            steps = int(span / self.step_size)
            
        if steps % 2 == 0:
            steps += 1
        sharp = np.full(steps, -1.0)
        mid = int(steps / 2)
        sharp[mid] = steps * mult
        sharp /= sharp.sum()
        return sharp 
     

    def create_blur_filter(self, span=3, sigma=1.5):
        steps = int(span / self._zinc)
        if steps % 2 == 0:
            steps += 1
        mid = int(steps / 2)
        blur = 1 / (np.sqrt(np.pi*2) * sigma) * \
            np.exp(((np.arange(-mid, mid+1) ** 2) * -1) / (2 * sigma**2))
        blur /= blur.sum()
        return blur

    def apply_filt(self, gram, filt, axis=0, half_rectify=True):
        # Make the axis to act on the first dimension if required.
        if axis == 1:
            gram = gram.transpose()

        # Do convolution along the first dimension.
        agram = np.zeros_like(gram)
        mid = (len(filt) - 1) / 2
        for j in np.arange(gram.shape[0]):
            agram[j] = np.convolve(
                np.pad(gram[j], int(mid), mode='edge'),
                filt,
                mode='valid'
            )

        # Do half-wave rectification if requested.
        if half_rectify is True:
            agram[agram < 0] = 0

        # Rescale spectrogram values as relative magnitude.
        agram = (agram - np.min(agram)) / (np.max(agram) - np.min(agram))

        if axis == 1:
            return agram.transpose()
        else:
            return agram

    def make_sharpgram(self,span=6, mult=1, dimension = "frequency"):
        '''Sharpens the frequency distinctions or temporal dimension in the auditory spectrogram and stores the resulting sharpened spectrogram in the class property `self.sharpgram`. Note that `make_zgram()` must be called before calling this function.

        Parameters
        ----------

        span : scalar, default = 6
            The time (in seconds) or frequency (in Bark) range of the filter

        mult : scalar, default = 1
            The degree of sharpening, larger value gives more contrast

        dimension : string, default = "frequency"
            For sharpening in the "frequency" domain or the "time" domain.


        '''
        if len(self.zgram.shape)==1:
            print("Call make_zgram() before calling make_sharpgram()")
            return
            
        sharpen = self.create_sharp_filter(span, mult,dimension)
        self.sharpgram = self.apply_filt(self.zgram, sharpen, axis=0, half_rectify=True)  # frequency sharpening

    def make_blurgram(self,span=3, sigma=1.5):
        '''Blur the frequency contrasts in the auditory spectrogram using a 1d Gaussian blur filter.  The resulting blurred auditory spectrogram is stored in the class property `self.blurgram`. Note that `make_zgram()` must be called before calling this function.

        Parameters
        ----------

        span: scalar, default = 3
            Frequency range, in Bark, over which the filter blurs

        sigma: scalar, default = 1.5
            The variance of the Gaussian function

        '''
        
        if len(self.zgram.shape)==1:
            print("Call make_zgram() before calling make_blurgram()")
            return
            
        blur = self.create_blur_filter(span, sigma)
        self.blurgram = self.apply_filt(self.zgram, blur, axis=0, half_rectify=True)  # frequency blurring

    def make_tgram(self):
        '''Compute the change in energy in each critical band in the auditory spectrogram.  The tgram is positive when the amplitude is increasing, and negative when the amplitude in a critical band is decreasing.  The resulting temporal contrast auditory spectrogram is stored in the class property `self.tgram`.  Note that `make_zgram()` must be called before calling this function.

        '''

        if len(self.zgram.shape)==1:
            print("Call make_zgram() before calling make_tgram()")
            return
        self.tgram = stats.zscore(np.gradient(self.zgram,axis=0),axis=0)  # temporal change


    def savez(self, fname, **kwargs):
        '''Calls numpy.savez to save all of the properties of the audspec object.

        Parameters
        ----------

        fname : string
            Name of the file in which to save the data.  Should end in ".npz"

        '''
        
        np.savez(
            fname,
            **self.__dict__,
            **kwargs,
            **{'custom_vars': list(kwargs.keys())}
        )
    
    def __init__(self, fs=16000, step_size=0.03):
        float_t=np.float32
        int_t=np.int32
        super(Audspec, self).__init__()
        self.fs = fs 
        self.dft_n = 2**(int_t(np.log2(0.05*fs)))  # choose fft size based on fs
        spect_points = int_t(self.dft_n/2) + 1
        self._topbark = self.hz2bark(self.fs/2.0)
        self.ncoef = int_t(self._topbark * 3.5)  # number of points in the auditory spectrum
        self._zinc = self._topbark/(self.ncoef+1)
        self.inc = self.fs/self.dft_n;   # get hz stepsize in fft
        self.fft_freqs = np.arange(1, spect_points+1) * self.inc

        self.sgram = np.zeros(spect_points, dtype=float_t) #: ndarray - 2d acoustic narrow band spectrogram
        self.zgram = np.zeros(self.ncoef, dtype=float_t) #: ndarray - 2d auditory spectrogram
        self.sharpgram = np.zeros(self.ncoef,dtype=float_t)  #: ndarray - frequency sharpened auditory spectrogram
        self.blurgram = np.zeros(self.ncoef,dtype=float_t) #: ndarray - blurred auditory spectrogram
        self.tgram = np.zeros(self.ncoef,dtype=float_t) #: ndarray - temporal contrast auditory spectrogram
        
        self.zfreqs = np.arange(1, self.ncoef+1) * self._zinc #: ndarray - Center frequencies of the critical bands in Bark
        self.freqs = self.bark2hz(self.zfreqs) #: ndarray - Center frequencies of the critical bands in Hz
        self.step_size = float_t(step_size)  #: temporal interval between frames in sec
        self.time_axis = np.zeros(0, dtype=float_t)  #: ndarray - time axis for auditory spectrogram
        self.extent = [0,0,0,0] #: ndarray - [xmin,xmax,ymin,ymax] plotting limits of the auditory spectrogram for imshow()

        
        self.maxcbfiltn = int_t(100)  # number of points in biggest CB filter
        self.cbfilts = self._make_cb_filters().astype(float_t)
        loudpoly = Polynomial([22.57, -11.46, -52.58, 226.97, 41.05, -1415.86, 
                 925.53, 5216.88, -5157.93, -10245.93, 11386.57, 
                 9702.65, -11213.73, -3484.65, 4079.037], domain=[20,20000])
        self.loud = (10.0**(-loudpoly(self.fft_freqs)/10.0)).astype(float_t)
