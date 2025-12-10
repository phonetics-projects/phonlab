import numpy as np
from librosa import util, lpc
from scipy.signal import windows, fftconvolve

from ..utils.prep_audio_ import prep_audio

def overlap_add(frames, hop_length, window='cosine'):
    """
Perform Overlap-Add reconstruction from 2D frames to a 1D signal.

Parameters
==========
    frames : ndarray
        A 2D array of shape (n_frames, frame_length)

    hop_length : int
        Number of samples to shift for each frame

    window : str, tuple, numeric, callable, list-like (default 'cosine')
        Window to apply to each frame before adding. This value is passed
        as the `window` parameter to `scipy.signal.windows.get_window`). 
        The default value 'cosine' matches the behavior of the overlap_and_add()
        function in the tensor flow library. If None, add raw 'boxcar' frames.

Returns
=======
    y : ndarray
        The reconstructed 1D signal.
    """
    n_frames, frame_length = frames.shape
    
    # Preallocate the output buffer with zeros
    final_length = (n_frames - 1) * hop_length + frame_length
    y = np.zeros(final_length)
    
    # If a window is provided, we multiply it against the frames
    # to taper the edges and prevent clicking.
    window = 'boxcar' if window is None else window
    win = windows.get_window(window, Nx=frames.shape[1], fftbins=False)
    frames = frames * win

    # Overlap-Add Loop using slice assignment, which is highly optimized in numpy.
    for i in range(n_frames):
        # Calculate start and end indices for this frame and add to the buffer
        start = i * hop_length
        end = start + frame_length
        y[start:end] += frames[i, :]
        
    return y

def lpcresidual(y, fs, target_fs=16000, order = 18, l=0.04, s=0.005, window='cosine'):    
    """Compute the residual signal which results from filtering the input array **y** 
using LPC inverse filtering. This signal is useful in voice quality and periodicity routines.

The LPC order is equal to (target_fs/1000) + 2, which is by default is 18.

Parameters
==========
    y : ndarray
        A one-dimensional array of audio samples
    fs : int
        Sampling rate of **y**
    target_fs: int, default = 16000
        Algorithms from the covarep library of voice analysis routines require target_fs=16000
    order: integer, default = 18
        The "order" of the LPC analysis.  The number of coefficients to use in the LPC analysis.  
        The default value is that recommended by Drugman, Kane, and Gobl (2013) for voice quality
        analysis (fs/1000 + 2), with the caveat that a smaller number may be more appropriate for 
        voices with higher fundamental frequency.
    l: float, default = 0.04
        The duration of the LPC analysis window, 40 milliseconds
    s: float, default = 0.005
        The interval between successive frames in the LPC analysis, 5 milliseconds
    window : str, tuple, numeric, callable, list-like (default 'hann')
        Window function applied to each frame. This parameter is passed as the
        `window` parameter to scipy.signal's `get_window` function.

Returns
=======
    lpc_residual : ndarray
        A one-dimensional array -- the residual derived by inverse filtering the input 
        audio signal. It has the same number of samples as the input **y** array.
    fs : int
        The sampling rate of **lpc_residual**.  It will be the same as **target_fs**, which
        by default is 16000 Hz.

    
    """
    x, fs = prep_audio(y, fs, target_fs=target_fs, pre = 0, quiet=True)  # resample
    
    frame_length = int(fs * l) # number of samples in a frame
    step = int(fs * s)  # number of samples between frames, hop

    frames = util.frame(x, frame_length=frame_length, hop_length=step,axis=0)   # view as frames
    window = window = 'boxcar' if window is None else window
    win = windows.get_window(window, Nx=frames.shape[1])
    frames = frames * win

    A = lpc(frames, order=order)  # get lpc coefficients
    inv = fftconvolve(frames,A,mode="same",axes=1) # inverse filter, 
    inv = inv * np.sum(np.square(frames))/np.sum(np.square(inv))

    lpc_resid = overlap_add(inv, step, window)  # put frames into waveform with overlap and add

    lpc_resid = lpc_resid/np.max(np.fabs(lpc_resid))

    # pad the lpc_residual to be the same length as the input
    npad = len(x)-len(lpc_resid)

    lpc_resid = np.pad(lpc_resid,(0,npad),mode='edge')  ## repeat the last sample npad times
    
    return lpc_resid,fs
