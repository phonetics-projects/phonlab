import numpy as np
from librosa import util, lpc
from scipy.signal import windows, lfilter

from ..utils.prep_audio_ import prep_audio
from numba import njit

@njit(cache=True)
def overlap_add(frames, hop_length, window_vec):
    """
    Perform Overlap-Add reconstruction from 2D frames to a 1D signal.

Parameters
==========
    frames : ndarray
        A 2D array of shape (n_frames, frame_length)

    hop_length : int
        Number of samples to shift for each frame

    window_vec : 1D array (frame_length) 
        use scipy.signal.windows to get this vector with a statement like 
        `win = windows.get_window('cosine', Nx=frames.shape[1], fftbins=False)`

Returns
=======
    y : ndarray
        The reconstructed 1D signal.
    """
    n_frames, frame_length = frames.shape
    final_length = (n_frames - 1) * hop_length + frame_length

    y = np.zeros(final_length, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        for j in range(frame_length):
            # Windowing and Adding performed in a single pass 
            # to maximize CPU cache efficiency
            y[start + j] += frames[i, j] * window_vec[j]
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
        
    # 1. Resample and Pre-process
    x, fs = prep_audio(y, fs, target_fs=target_fs, pre=0, add_tiny_noise=False, quiet=True)
    x = x.astype(np.float32) # Use float32 for speed and memory efficiency
    
    frame_length = int(fs * l)
    step = int(fs * s)

    # 2. Frame the signal
    # util.frame creates a 'view', which is O(1) memory and time.
    frames = util.frame(x, frame_length=frame_length, hop_length=step, axis=0)
    
    # 3. Compute LPC Coefficients
    # librosa.lpc is generally efficient, we process frames in one go.
    A = lpc(frames, order=order) 

    # 4. Inverse Filtering (The Bottleneck)    

    # Pre-allocate output for frames
    inv = np.zeros_like(frames)
    
    # Applying the filter A[i] to frames[i]
    # A coefficients from librosa are [1, a1, a2, ...] 
    # The residual is calculated by filtering the frame with A
    for i in range(frames.shape[0]):
        # lfilter is significantly faster than fftconvolve for short FIR filters
        inv[i] = lfilter(A[i], [1.0], frames[i])

    # 5. Energy Normalization (Vectorized)
    # Compute energy per frame to match original logic
    inv = inv * np.sum(np.square(frames))/np.sum(np.square(inv))

    #frame_energy = np.sum(frames**2, axis=1, keepdims=True)
    #inv_energy = np.sum(inv**2, axis=1, keepdims=True)
    # Avoid division by zero
    #scale = np.sqrt(frame_energy / (inv_energy + 1e-12))
    #inv *= scale
    #print(f"scale.shape = {scale.shape}")

    # 6. Overlap-Add
    window = window = 'boxcar' if window is None else window
    w = windows.get_window(window, Nx=frames.shape[1], fftbins=False)
    lpc_resid = overlap_add(inv, step, w)

    # 7. Final Normalization and Padding
    #lpc_resid = lpc_resid/np.max(np.fabs(lpc_resid))
    
    max_val = np.max(np.fabs(lpc_resid))
    if max_val > 0:
        lpc_resid /= max_val

    # Pad to match original length
    if len(lpc_resid) < len(x):
        lpc_resid = np.pad(lpc_resid, (0, len(x) - len(lpc_resid)), mode='edge')
    else:
        lpc_resid = lpc_resid[:len(x)]
    
    return lpc_resid, fs
