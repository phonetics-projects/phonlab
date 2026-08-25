import numpy as np
from librosa import util, lpc, frames_to_time
from scipy.signal import windows, fftconvolve, lfilter
from scipy.linalg import inv
from pandas import DataFrame
from numba import jit

# prep_audio, choose_order
from ..utils.prep_audio_ import prep_audio
from ..acoustic.choose_order_ import choose_order

@jit(nopython=True, cache=True)
def weighted_cov(x,weight,npoles):
    # weighted covariance matrix, down-weighting samples with high LPC error
    C1 = np.zeros((npoles+1,npoles+1))
    for i in range(1,npoles+1):
        for j in range(1,npoles+1):
            for n in range(npoles+1,len(x)):
                C1[i][j] += x[n-i] * x[n-j] * weight[n]
    return C1[1:npoles+1,1:npoles+1]

@jit(nopython=True, cache=True)
def weighted_cor(x,weight,npoles):
    # weighted correlation matrix, down-weighting samples with high LPC error
    c1 = np.zeros((npoles+1))
    for j in range(1,npoles+1):
        for n in range(npoles+1,len(x)):
            c1[j] += x[n] * x[n-j] * weight[n]
    return c1[1:npoles+1]
    
def RLPC_formants(x, fs, tf = 6000, preemphasis = 0.94,order = -1, base = "BIC", s = 0.01, 
                 l = 0.03, window = None, numiter=1, verbose=False):
    """An implementation of Lee's (1988) method for Robust Linear Prediction Coding. The method computes the residual signal from a conventional LPC analysis and then uses it to down-weight samples that produce large prediction error. It is possible to iterate the residual reweighting process more than once, though not much improvement is gained after the first iteration.  Formant frequencies are found from the LPC coefficients using polynomial root solving.  A couple of functions called by `RLPC_formants()` are accelerated by the use of numba just in time (JIT) compilation into machine code.  Therefore, the first call to `RLPC_formants()` will be slower (by about 10 ms) than subsequent calls.

    Note that unlike other formant analysis functions, this one returns both a dataframe with analysis results and also a 2D matrix with time-varying LPC coefficients.
    
    Parameters
    ==========
    x : ndarray
        a 1D array of audio samples
    fs : int
        the sampling rate of samples in x.  This will be resampled to 12,000 Hz for analysis, so choose the order paramter with this in mind.
    tf : int (default = 6000)
        The top frequency of the analysis band.  The input signal will be resampled to a rate of tf*2.  
    preemphasis : float (default = 0.94)
        value of a high frequency preemphasis (spectral tilt) factor (0-1).
    order : int (default = -1)
        The LPC 'order' (i.e. the number of coefficients in the LPC filter.  A value of -1 uses the function phon.choose_order() to determine the best value for this parameter. 
    base : string (default = 'BIC')
        If `order` is -1, then use this base ('BIC' or 'coef') in phon.choose_order().
    s : float (default = 0.01)
        the interval (in seconds) between successive formant measurements.
    l : float (default = 0.05)
        duration (in seconds) of analysis frames
    window : string (default = 'cosine')
        The name of a window shape for scipy.signal.windows.get_window()
    numiter : integer (default = 1)
        Number of robust LPC iterations to perform.  If this value is 0 the results of the conventional LPC analysis are returned with no robust recalculation of the linear prediction.
    verbose : boolean (default = False)
        Print diagnostic messages
            
    Returns
    =======
    df - a pandas dataframe with columns as described in the note below.
    A - a two-dimensional numpy matrix with the LPC coefficients for each analysis frame.

    Note
    ====
    The columns in the output dataframe are:
        * sec - midpoint time of the frame 
        * gain - the LPC gain (variance of the residual) for each frame.
        * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.
        * BW1-4 - bandwidths of the vowel formants

    References
    ==========
    
    C.H. Lee (1988) On robust prediction of speech. `IEEE Transactions on Acoustics, Speech and Signal Processing`, 36,5: 642-650.
    
    """
    target_fs = tf*2
    
    x, fs = prep_audio(x, fs, target_fs = target_fs, pre = preemphasis, quiet = not verbose)
    
    frame_length = int(fs*l)
    step = int(fs*s)
       
    if order<0:  # guess the correct LPC order
        order,time = choose_order(x,fs,l, base='BIC', verbose=verbose)
        if verbose:
            print(f"in RLPC_formants(), order is {order}, taken at time {time:.3f}")
    
    frames = util.frame(x, frame_length=frame_length, 
                                hop_length=step, axis=0) 
    t = frames_to_time(range(frames.shape[0]),sr=fs, 
                               hop_length=step, n_fft=frame_length)
    resid = np.zeros_like(frames)

    window = window = 'boxcar' if window is None else window
    w = windows.get_window(window, Nx=frames.shape[1])
    frames = frames * w

    A = lpc(frames, order=order)  # get conventional lpc coefficients
    nb = A.shape[0]  # the number of frames in the LPC analysis

    # begin robust LPC process
    for iter in range(numiter):  # iterate this loop numiter times
        for i in range(nb):
            resid[i] = lfilter(A[i], [1.0], frames[i]) # prediction error
        sd = np.std(resid,axis=1)
        temp = np.clip(resid/sd[:,None],a_min=-1.5,a_max=1.5)  # min/max clipping
        weight = (sd[:,None] * temp)/resid   # down weight samples with high error
        for i in range(nb):  # recompute A frame by frame
            C = weighted_cov(frames[i],weight[i],order)
            c = weighted_cor(frames[i],weight[i],order)
            A[i,1:order+1] = np.dot(-inv(C), c)

    # Now we have the refined LPC coefficients 
    if (order-2)/2 < 4: 
        nf = int((order-2)/2)  # the number of formants that will be computed
    else:
        nf = 4  # the number of formants that will be returned

    formants = np.empty((nb,nf))  # arrays to be filled by analysis
    bandwidths = np.empty((nb,nf))

    for i in range(nb):  # find formants from LPC coefficients (A) using root solving
        roots = np.roots(A[i,:])  # solve for the roots of the polynomial
        roots = roots[np.imag(roots)>=0]  # only keep the positive ones
        fr = np.angle(roots) *  fs/(2 * np.pi)  # calculate formant frequencies from them
        bw = -(fs/(np.pi)) * np.log(abs(roots)) # calculate bandwidths from them
        resid[i] = lfilter(A[i], [1.0], frames[i]) # prediction error
        
        # A wide bandwidth criterion (about 800 Hz) is needed to capture fast moving formants.
        fr[bw>800] = np.nan  # reject formants with too wide bandwidth, NaN sorts to the end of the array
        s = np.argsort(fr)  # get the sort order so you can apply it more than one array
        fr = fr[s]  # sort the formants from lowest to highest
        bw = bw[s]  # use the same sort order with the bandwidth array

        formants[i,:] = fr[:nf]  
        bandwidths[i,:] = bw[:nf]

    gain = np.std(resid,axis=1,ddof=order)**2
    list = ['sec',] + [f"F{i}" for i in range(1,nf+1)] + [f"BW{i}" for i in range(1,nf+1)] + ['gain']
    df = DataFrame(np.concatenate((t.reshape(nb,1), formants, 
                                   bandwidths, gain.reshape(nb,1)), axis=1),columns=(list))

    return df,A
