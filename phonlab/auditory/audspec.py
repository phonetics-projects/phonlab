__all__=[ 'Audspec' ]

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

    Examples
    ========

    .. code-block:: Python
    
         aud = phon.Audspec()
         aud.make_zgram("sf3_cln.wav")
         aud.savez('sf3_cln.aud.npz')  # save the Audspec object.
        
         # ---- the rest is to make a nice plot ----
         
         fig,ax = plt.subplots(2,figsize=(6,5))
         
         Hz_extent = (min(aud.time_axis), max(aud.time_axis),
               min(aud.fft_freqs), max(aud.fft_freqs))  # time and frequency values for sgram.
         ax[0].imshow(20*np.log10(aud.sgram.T),origin='lower', aspect='auto', 
                  extent=Hz_extent, cmap = plt.cm.Greys)
         ax[0].set(xlabel="Time (sec)", ylabel="Frequency (Hz)")
         ax[1].imshow(aud.zgram.T,origin='lower', aspect='auto', 
                  extent=aud.extent, cmap = plt.cm.Purples)
         ax[1].set(xlabel="Time (sec)", ylabel="Frequency (Bark)")

    """

    """

    .. figure:: images/aud_spec.png
       :scale: 50 %
       :alt: An acoustic narrow band spectrogram and the auditory spectrogram of the same utterance.
       :align: center

       (top) Acoustic spectrogram, and (bottom) Auditory spectrogram of the same utterance.

       ..
       
    """
    
    def __init__(self, fs=16000, step_size=0.03):
        float_t=np.float32
        int_t=np.int32
        super(Audspec, self).__init__()
        self.fs = float_t(fs)
        self.dft_n = 2**(int_t(np.log2(0.05*fs)))  # choose fft size based on fs
        spect_points = int_t(self.dft_n/2) + 1
        self.sgram = np.zeros(spect_points, dtype=float_t) #: ndarray - 2d acoustic spectrogram

        self.time_axis = np.zeros(0, dtype=float_t)  #: ndarray - time axis for auditory spectrogram
        self.inc = self.fs/self.dft_n;   # get hz stepsize in fft
        self.fft_freqs = np.arange(1, spect_points+1) * self.inc

        self._topbark = self.hz2bark(self.fs/2.0)
        self.ncoef = int_t(self._topbark * 3.5)  # number of points in the auditory spectrum
        self._zinc = self._topbark/(self.ncoef+1)

        self.zgram = np.zeros(self.ncoef, dtype=float_t) #: ndarray - 2d auditory spectrogram
        
        self.sharpgram = np.zeros(self.ncoef,dtype=float_t) 
        '''ndarray - frequency sharpened auditory spectrogram

        '''

        '''
        .. figure:: images/sharpgram.png
           :scale: 50 %
           :alt: The figure shows a frequency 'sharpened' auditory spectrogram.
           :align: center

           Auditory spectrogram after frequency sharpening.

           ..
        
        '''
        
        self.blurgram = np.zeros(self.ncoef,dtype=float_t) #: ndarray - blurred auditory spectrogram
        
        self.tgram = np.zeros(self.ncoef,dtype=float_t) 
        '''ndarray - temporal contrast auditory spectrogram

        '''

        '''
        .. figure:: images/tgram.png
           :scale: 50 %
           :alt: The figure shows temporal contrast in an auditory spectrogram.
           :align: center

           Temporal constrast in an auditory spectrogram.

           ..
       
        '''
       
        #: ndarray - Center frequencies of the critical bands in Bark
        self.zfreqs = np.arange(1, self.ncoef+1) * self._zinc 
        self.freqs = self.bark2hz(self.zfreqs) #: ndarray - Center frequencies of the critical bands in Hz

        self.step_size = float_t(step_size)  # temporal intervals in sec
        self.maxcbfiltn = int_t(100)  # number of points in biggest CB filter

        self.cbfilts = self.make_cb_filters().astype(float_t)

        # see loudness_function.ipynb   
        loudpoly = Polynomial([22.57, -11.46, -52.58, 226.97, 41.05, -1415.86, 
                 925.53, 5216.88, -5157.93, -10245.93, 11386.57, 
                 9702.65, -11213.73, -3484.65, 4079.037], domain=[20,20000])
        self.loud = (10.0**(-loudpoly(self.fft_freqs)/10.0)).astype(float_t)

        #: ndarray - [xmin,xmax,ymin,ymax] plotting limits of the auditory spectrogram for imshow()
        self.extent = [0,0,0,0] 
        
    def hz2bark(self, hz):
        """Convert frequency in Hz to Bark using the Schroeder 1977 formula.

        Parameters
        ==========

        hz : scalar or array
            Frequency in Hz.

        Returns
        =======
        
        scalar or array
            Frequency in Bark, in an array the same size as `hz`, or a scalar if `hz` is a scalar.
        """
        
        return 7 * np.arcsinh(hz/650)

    def bark2hz(self, bark):
        '''
        Convert frequency in Hz to Bark using the Schroeder 1977 formula.

        Parameters
        ----------

        bark: scalar or array
        Frequency in Bark.

        Returns
        -------
        scalar or array
        Frequency in Hz, in an array the same size as `bark`, or a scalar if
        `bark` is a scalar.
        '''
        return 650 * np.sinh(bark/7)

    def make_cb_filters(self):
        """        
        Create and return 2d array of Patterson filters for DFT spectra based
        on attribute values in `self`.

        Patterson, R.D. (1976) Auditory filter shapes derived with noise stimuli. 
                               JASA 59, 640-54.

        The returned filters are stored in an 2d array in which the
        rows represent the filter frequency bands in ascending order. The
        columns contain symmetrical filter coefficients as determined by the
        Patterson formula and centered at the filter frequency in the
        DFT spectrum. Filter coefficients outside the frequency band are set
        to 0.0. To view the filter coefficients for band `j` do
        `self.cbfilts[j][self.cbfilts[j] != 0.0]`.

        The one-sided length of filter coefficients for each band is stored
        in the `cbfiltn` attribute. The number of coefficients in the
        symmetrical filter for band `j` is therefore
        `(self.cbfiltn[j] * 2) - 1`. In a few bands this calculation might not
        be correct since the coefficients may not fit when the center frequency
        is near the left or right edge of the DFT spectrum. In such cases the
        coefficients are truncated, and the actual number of coefficients for
        the band `j` can be found with `np.sum(self.cbfilts[j] != 0.0)`.
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

    def create_sharp_filter(self, span=4, mult=3, dimension="frequency"):
        """Create and return a 1d sharpening filter symmetric in frequency, or time.

        Parameters
        ----------

        span : scalar, default = 4
            The time (in seconds) or frequency (in Bark) range of the filter

        mult : scalar, default = 3
            The degree of sharpening, larger value gives more contrast

        dimension : string, default = "frequency"
            For sharpening in the "frequency" domain or the "time" domain.

        Returns
        -------
        ndarray, a sharpening filter to be passed to `apply_filt()`
        """
        
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
        '''Create and return a 1d Gaussian blur filter.

        Parameters
        ----------

        span: scalar, default = 3
            Frequency range, in Bark, over which the filter blurs

        sigma: scalar, default = 1.5
            The variance of the Gaussian function

        Returns
        -------
        ndarray - a frequency blurring filter to be passed to `apply_filt()`
        '''
        
        steps = int(span / self._zinc)
        if steps % 2 == 0:
            steps += 1
        mid = int(steps / 2)
        blur = 1 / (np.sqrt(np.pi*2) * sigma) * \
            np.exp(((np.arange(-mid, mid+1) ** 2) * -1) / (2 * sigma**2))
        blur /= blur.sum()
        return blur

    def apply_filt(self, gram, filt, axis=0, half_rectify=True):
        '''
        Apply a filter along the axis of an auditory spectrogram. Spectrogram
        values are also rescaled after filtering.

        Parameters
        ----------

        gram : ndarray
            A two-dimensional (time, frequency) array containing an auditory spectrogram, as produced by `make_zgram()`.

        filt : ndarray
            A one-dimensional array containing filter coefficients (as produced by `create_blur_filter()` or `create_sharp_filter()`.

        axis : 0 or 1, default = 0
            The axis to iterate over and apply the filter.

        half_rectify : bool, default = True
            If True, apply half-wave rectification to filtered spectrogram.

        Returns
        -------
        ndarray
            The auditory spectrogram after the filter has been applied. The shape is the same as the input argument `gram`.
        '''
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

    def make_zgram(self, sig, chan=0, fs_in=16000, preemph = 0, **kwargs):
        '''Make an auditory spectrogram by creating an acoustic spectrogram and then applying critical-band filters to it, using the filter shapes described by Patterson (1976).  The function creates the auditory spectrogram and stores it in `self.zgram`.

Parameters
----------

sig : ndarray or str
    Audio data, either as a one dimensional numpy array or the name of a wav file.

chan : {0,1}, default = 0
    The channel of a two channel signal to analyze. Use 0 for the left channel, 1 for the right.  This is ignored if the signal is mono.

fs_in : int, default = 16000
    If sig is an array of audio samples, the sampling rate of the audio must be provided in fs_in.

preemp : float, default = 1.0
    The amount of preemphasis to apply before filtering.

**kwargs: dict, optional
    Keyword arguments to be passed to scipy.fft.rfft() 


References
----------
    Patterson, R.D. (1976) Auditory filter shapes derived with noise stimuli. J. Acoust. Soc. Am. 59, 640-54.
        '''
        
        x, fs = prep_audio(sig,chan = chan, fs = self.fs, fs_in = fs_in, pre = preemph)

        (sgram, self.time_axis) = self._make_sgram(x, kwargs)
        self.sgram = sgram + self.loud
        zgram = self.sgram[:, np.newaxis, :] * self.cbfilts[np.newaxis, :, :]
        self.zgram = 20 * np.log10(zgram.sum(axis=2)+1)
        self.extent = (min(self.time_axis),max(self.time_axis),min(self.zfreqs),max(self.zfreqs)) 

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
