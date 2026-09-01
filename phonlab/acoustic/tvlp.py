import numpy as np
import pandas as pd
import time
from numba import jit, prange
from ..utils.prep_audio_ import prep_audio
from ..acoustic.gci import gci_sedreams
from ..acoustic.choose_order_ import choose_order

# This set of functions was written interactively with the AI model Claude, translating
# and adapting Gowda's ftrack matlab code.  Keith Johnson, Feb 2026

@jit(nopython=True, cache=True)
def _build_matrices(x, p, q, w):
    """Build the Ypu matrix and Yn vector using Numba for speed."""
    n_samples = len(x)
    m = n_samples - p
    n_cols = p * (q + 1)
    
    Ypu = np.zeros((m, n_cols), dtype=np.float32)
    Yn = np.zeros(m, dtype=np.float32)
    
    for idx in range(m):
        n = idx + p
        Yn[idx] = w[n] * x[n]
        
        for i in range(1, p + 1):
            for j in range(q + 1):
                col = (i - 1) * (q + 1) + j
                Ypu[idx, col] = w[n] * (n - p) ** j * x[n - i]
    
    return Ypu, Yn


@jit(nopython=True, cache=True)
def _solve_lstsq_numba(A, b):
    """Solve least squares using QR decomposition."""
    # Ensure arrays are contiguous for optimal performance
    A = np.ascontiguousarray(A)
    b = np.ascontiguousarray(b)
    
    Q, R = np.linalg.qr(A)
    # Ensure Q.T is contiguous before matrix multiplication
    Qt = np.ascontiguousarray(Q.T)
    return np.linalg.solve(R, Qt @ b)


@jit(nopython=True, cache=True)
def _irls_iteration(Ypu, Yn, max_iter=10, tol=1e-4, epsilon=1e-8):
    """IRLS iterations - Numba compatible."""
    m, n = Ypu.shape
    
    # Ensure input arrays are contiguous
    Ypu = np.ascontiguousarray(Ypu)
    Yn = np.ascontiguousarray(Yn)
    
    x = _solve_lstsq_numba(Ypu, Yn)
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Compute residuals
        residuals = Ypu @ x - Yn
        weights = 1.0 / (np.abs(residuals) + epsilon)
        weights_sqrt = np.sqrt(weights)
        
        # Create weighted matrices
        Ypu_weighted = np.zeros((m, n), dtype=np.float32)
        Yn_weighted = np.zeros(m, dtype=np.float32)
        
        for i in range(m):
            Yn_weighted[i] = Yn[i] * weights_sqrt[i]
            for j in range(n):
                Ypu_weighted[i, j] = Ypu[i, j] * weights_sqrt[i]
        
        # These are already contiguous from creation
        x = _solve_lstsq_numba(Ypu_weighted, Yn_weighted)
        
        diff_norm = np.linalg.norm(x - x_old)
        x_norm = np.linalg.norm(x)
        if diff_norm / (x_norm + 1e-10) < tol:
            break
    
    return x


@jit(nopython=True, cache=True)
def _compute_time_varying_coeffs(aki, frame_length):
    """
    Compute time-varying LP coefficients for each sample in a frame.
    
    Parameters:
    -----------
    aki : ndarray, shape (q+1, p)
        Coefficient matrix
    frame_length : int
        Number of samples in the frame
    
    Returns:
    --------
    ak : ndarray, shape (p+1, frame_length)
        Time-varying LP coefficients for each sample (includes leading 1)
    """
    q_plus_1, p = aki.shape
    q = q_plus_1 - 1
    
    # Compute time-varying coefficients for each sample
    akn = np.zeros((p, frame_length), dtype=np.float32)
    for t in range(frame_length):
        for k in range(p):
            for i in range(q + 1):
                akn[k, t] += aki[i, k] * (t ** i)
    
    # Add leading 1 for LP polynomial
    ak = np.zeros((p + 1, frame_length), dtype=np.float32)
    ak[0, :] = 1.0
    ak[1:, :] = -akn  # Negative sign is critical!
    
    return ak


@jit(nopython=True, parallel=True, cache=True)
def _process_frames_l2_parallel(frames, weights, p, q):
    """
    Process multiple frames in parallel using L2 norm.
    Computes time-varying coefficients directly, discarding aki matrices.
    
    Returns:
    --------
    all_ak : ndarray, shape (n_frames, p+1, frame_length)
        Time-varying LP coefficients for all samples in all frames
    """
    n_frames, frame_length = frames.shape
    #n_frames = len(frames)
    #frame_length = len(frames[0])
    
    all_ak = np.zeros((n_frames, p + 1, frame_length), dtype=np.float32)
    
    for frame_idx in prange(n_frames):
        # Ensure frame is contiguous
        x = np.ascontiguousarray(frames[frame_idx])
        w = np.ascontiguousarray(weights[frame_idx])
        
        Ypu, Yn = _build_matrices(x, p, q, w)
        x_opt = _solve_lstsq_numba(Ypu, Yn)
        
        # Reshape to aki matrix (temporary, only for this frame)
        aki = np.zeros((q + 1, p), dtype=np.float32)
        for i in range(q + 1):
            for j in range(p):
                aki[i, j] = x_opt[j * (q + 1) + i]
        
        # Compute time-varying coefficients for all samples
        # aki is discarded after this, only all_ak is kept
        all_ak[frame_idx] = _compute_time_varying_coeffs(aki, frame_length)
    
    return all_ak


@jit(nopython=True, parallel=True, cache=True)
def _process_frames_l1_parallel(frames, weights, p, q, max_iter=10):
    """
    Process multiple frames in parallel using L1 norm (IRLS).
    Computes time-varying coefficients directly, discarding aki matrices.
    
    Returns:
    --------
    all_ak : ndarray, shape (n_frames, p+1, frame_length)
        Time-varying LP coefficients for all samples in all frames
    """
    n_frames, frame_length = frames.shape
    #n_frames = len(frames)
    #frame_length = len(frames[0])
    
    all_ak = np.zeros((n_frames, p + 1, frame_length), dtype=np.float32)
    
    for frame_idx in prange(n_frames):
        # Ensure frame is contiguous
        x = np.ascontiguousarray(frames[frame_idx])
        w = np.ascontiguousarray(weights[frame_idx])
        
        Ypu, Yn = _build_matrices(x, p, q, w)
        x_opt = _irls_iteration(Ypu, Yn, max_iter=max_iter)
        
        # Reshape to aki matrix (temporary, only for this frame)
        aki = np.zeros((q + 1, p), dtype=np.float32)
        for i in range(q + 1):
            for j in range(p):
                aki[i, j] = x_opt[j * (q + 1) + i]
        
        # Compute time-varying coefficients for all samples
        # aki is discarded after this, only all_ak is kept
        all_ak[frame_idx] = _compute_time_varying_coeffs(aki, frame_length)
    
    return all_ak


def _extract_frames(x, frame_length, hop_length):
    """Extract overlapping frames from signal."""
    n_samples = len(x)
    n_frames = 1 + (n_samples - frame_length) // hop_length
    
    frames = np.zeros((n_frames, frame_length), dtype=np.float32)
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        if end_idx > n_samples:
            break
        frames[i] = x[start_idx:end_idx]
    
    return frames


def extract_formants_single_timepoint(ak_t, fs, max_bandwidth, min_freq, max_freq):
    """
    Extract formants from LP coefficients at a single time point.
    Pure NumPy - handles np.roots properly.
    """
    try:
        # Ensure contiguous for np.roots
        ak_t = np.ascontiguousarray(ak_t, dtype=np.float32)
        # Find roots of LP polynomial
        roots = np.roots(ak_t)
    except (np.linalg.LinAlgError, ValueError):
        # If root finding fails, return empty
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    
    valid_freqs = []
    valid_bws = []
    
    # Convert roots to frequencies and bandwidths
    for root in roots:
        # Get angle and radius
        angle = np.angle(root)
        radius = np.abs(root)
        
        # Convert to frequency and bandwidth
        freq = angle * fs / (2.0 * np.pi)
        bw = -np.log(max(radius, 1e-10)) * fs / np.pi
        
        # Validate
        if (freq > min_freq and freq < max_freq and 
            bw < max_bandwidth and bw > 0):
            valid_freqs.append(freq)
            valid_bws.append(bw)
    
    # Convert to arrays and sort
    if len(valid_freqs) > 0:
        freqs = np.array(valid_freqs, dtype=np.float32)
        bandwidths = np.array(valid_bws, dtype=np.float32)
        
        # Sort by frequency
        sort_idx = np.argsort(freqs)
        freqs = freqs[sort_idx]
        bandwidths = bandwidths[sort_idx]
    else:
        freqs = np.array([], dtype=np.float32)
        bandwidths = np.array([], dtype=np.float32)
    
    return freqs, bandwidths


def extract_formants_single_frame_sequential(ak, sample_indices, npeaks, fs, 
                                             max_bandwidth, min_freq, max_freq):
    """
    Extract formants at specified sample indices within a frame.
    Sequential version (no multiprocessing).
    
    Parameters:
    -----------
    ak : ndarray, shape (p+1, frame_length)
        Time-varying LP coefficients
    sample_indices : ndarray
        Sample indices where formants should be extracted
    npeaks : int
        Number of formant peaks to extract
    fs : float
        Sampling frequency
    max_bandwidth : float
        Maximum bandwidth
    min_freq : float
        Minimum frequency
    max_freq : float
        Maximum frequency
    
    Returns:
    --------
    fi : ndarray, shape (len(sample_indices), npeaks)
        Formant frequencies
    bw : ndarray, shape (len(sample_indices), npeaks)
        Formant bandwidths
    """
    # Ensure contiguous for efficient column access
    ak = np.ascontiguousarray(ak, dtype=np.float32)
    
    n_samples = len(sample_indices)
    fi = np.zeros((n_samples, npeaks), dtype=np.float32)
    bw = np.zeros((n_samples, npeaks), dtype=np.float32)
    
    # Extract formants only at specified sample indices
    for idx, t in enumerate(sample_indices):
        if t >= ak.shape[1]:
            continue
        
        # Extract column - make it contiguous
        ak_t = np.ascontiguousarray(ak[:, t])
        
        freqs, bandwidths = extract_formants_single_timepoint(
            ak_t, fs, max_bandwidth, min_freq, max_freq
        )
        
        # Store top npeaks
        n_store = min(npeaks, len(freqs))
        if n_store > 0:
            fi[idx, :n_store] = freqs[:n_store]
            bw[idx, :n_store] = bandwidths[:n_store]
    
    return fi, bw


def tvlptoformants(all_ak, frame_length_samples, step, fs, 
                        npeaks=5, max_bandwidth=800, min_freq=0, max_freq=None, 
                        verbose=False, use_parallel=False):
    """
    Extract formants from multiple frames using precomputed time-varying coefficients.
    
    Parameters:
    -----------
    all_ak : ndarray, shape (n_frames, p+1, frame_length_samples)
        Time-varying LP coefficients for all samples in all frames
    frame_length_samples : int
        Length of each frame in samples
    step : float
        Time interval between formant measurements in seconds (e.g., 0.01 for 10ms)
    fs : float
        Sampling frequency in Hz
    npeaks : int
        Number of formant peaks to extract (default: 5)
    max_bandwidth : float
        Maximum bandwidth in Hz (default: 800)
    min_freq : float
        Minimum frequency in Hz (default: 0)
    max_freq : float or None
        Maximum frequency in Hz (default: None, uses fs/2)
    verbose : bool
        Print progress (default: False)
    use_parallel : bool
        Attempt to use parallel processing (default: False, not recommended for notebooks)
    
    Returns:
    --------
    all_fi : ndarray, shape (n_frames, n_timepoints, npeaks)
        Formant frequencies
    all_bw : ndarray, shape (n_frames, n_timepoints, npeaks)
        Formant bandwidths
    sample_times : ndarray
        Sample indices where formants were extracted (relative to frame start)
    """
    start_time = time.time()
    
    if max_freq is None:
        max_freq = fs / 2.0
    
    # Ensure contiguous
    all_ak = np.ascontiguousarray(all_ak, dtype=np.float32)
    n_frames = len(all_ak)
    
    # Calculate sample indices for formant extraction
    # Start from the middle of the first interval
    interval_samples = int(step * fs)
    start_offset = interval_samples // 2
    sample_times = np.arange(start_offset, frame_length_samples, interval_samples, dtype=np.int64)
    n_timepoints = len(sample_times)
    
    if verbose:
        print(f"Extracting formants from {n_frames} frames")
        print(f"Frame length: {frame_length_samples} samples ({frame_length_samples/fs*1000:.1f} ms)")
        print(f"Formant extraction interval: {step} sec ({interval_samples} samples)")
        print(f"Starting at sample {start_offset} (middle of first interval)")
        print(f"Number of extraction points per frame: {n_timepoints}")
    
    # Pre-allocate results
    all_fi = np.zeros((n_frames, n_timepoints, npeaks), dtype=np.float32)
    all_bw = np.zeros((n_frames, n_timepoints, npeaks), dtype=np.float32)
    
    if use_parallel:
        # Try concurrent.futures for better notebook compatibility
        try:
            from concurrent.futures import ProcessPoolExecutor
            import os
            
            n_workers = min(os.cpu_count() or 4, n_frames)
            
            if verbose:
                print(f"Attempting parallel processing with {n_workers} workers...")
            
            # This may still fail in notebooks, so we wrap in try-except
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                futures = []
                for i in range(n_frames):
                    future = executor.submit(
                        extract_formants_single_frame_sequential,
                        all_ak[i], sample_times, npeaks, fs, 
                        max_bandwidth, min_freq, max_freq
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    fi, bw = future.result()
                    all_fi[i] = fi
                    all_bw[i] = bw
            
            if verbose:
                print("Parallel processing succeeded!")
                
        except Exception as e:
            if verbose:
                print(f"Parallel processing failed ({e}), using sequential processing...")
            use_parallel = False
    
    if not use_parallel:
        # Sequential processing (recommended for Jupyter notebooks)
        if verbose:
            print("Using sequential processing...")
        
        for frame_idx in range(n_frames):
            if verbose and frame_idx % 50 == 0 and frame_idx > 0:
                print(f"  Processing frame {frame_idx}/{n_frames}")
            
            fi, bw = extract_formants_single_frame_sequential(
                all_ak[frame_idx], sample_times, npeaks, 
                fs, max_bandwidth, min_freq, max_freq
            )
            all_fi[frame_idx] = fi
            all_bw[frame_idx] = bw
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Total extraction time: {elapsed:.2f}s")
        print(f"Time per frame: {elapsed / n_frames * 1000:.2f}ms")
    
    return all_fi, all_bw, sample_times


def tvlp_batch(x, fs, frame_length, p, q, norm, qcp=True, f0median=200, max_iter=10, verbose=False):
    """
    Process long audio files in parallel using Numba.
    Returns only time-varying coefficients (aki matrices are not stored).
    
    Returns:
    --------
    all_ak : ndarray, shape (n_frames, p+1, frame_length)
        Time-varying LP coefficients for all samples in all frames
    """
    start_time = time.time()
    
    # Convert to float32 and ensure contiguous
    x = np.ascontiguousarray(x, dtype=np.float32)
    if qcp: 
        gci,mbs,resid = gci_sedreams(x,fs,f0median, target_fs=fs,order=None,cthresh=0.0)
        w = qcp_wt(x, gci['sec'], fs)
    else:
        w = np.ones(len(x), dtype=np.float32)
        
    if verbose:
        print(f"Processing {len(x)} samples")
        print(f"Frame length: {frame_length}")
    
    # Reshape - non-overlapping frames
    frames = x.reshape((-1,frame_length))
    weights = w.reshape((-1,frame_length))    
    
    if verbose:
        print(f"Extracted {len(frames)} frames")
        print(f"Weight shape is {weights.shape}")
        print(f"Processing in parallel with Numba ({norm.upper()})...")
    
    # Process frames in parallel
    if norm.lower() == 'l2':
        all_ak = _process_frames_l2_parallel(frames, weights, p, q)
    elif norm.lower() == 'l1':
        all_ak = _process_frames_l1_parallel(frames, weights, p, q, max_iter=max_iter)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
    
    if verbose:
        elapsed = time.time() - start_time
        memory_mb = all_ak.nbytes / (1024 * 1024)
        print(f"Total processing time: {elapsed:.2f}s")
        print(f"Time per frame: {elapsed / len(frames) * 1000:.2f}ms")
        print(f"Memory for time-varying coefficients: {memory_mb:.1f} MB")
        if len(x) > 8000:
            print(f"Real-time factor: {len(x) / 8000 / elapsed:.1f}x")
    
    return all_ak


def qcp_wt(x, gci, fs, DQ=0.7, PQ=0.05, d=0.00001, Nramp=4):
    """
    Create a pitch-synchronous weight function for x
    
    Parameters:
    -----------
    x : array-like
        Input signal of length N
    gci : array-like
        Glottal Closure Instants in seconds
    fs : int
        Sampling rate in Hz
    DQ : float, default = 0.7
        Duration quotient (from 0 to 1)
    PQ : float, default = 0.05
        Position Quotient (from 0 to 1)
    d : float, default = 0.00001
        Minimum value of the weight function
    Nramp : int, default = 4
        Length of the linear ramp (in samples)
   
    Returns:
    --------
    w : ndarray
        Weight function of length N
    """
    N = len(x)
    
    # Convert GCI from time (seconds) to sample indices
    gci_ins = np.round(np.array(gci) * fs).astype(int)
    
    # Create ramps if needed
    if Nramp > 0:
        UPramp = np.linspace(d, 1, 2 + Nramp)[1:-1]
        DOWNramp = UPramp[::-1]
    
    # Adjust DQ if necessary
    if DQ + PQ > 1:
        DQ = 1 - PQ
    
    # Initialize weight array
    w = np.full(N, d, dtype=np.float32)
    
    # Process all GCI pairs
    num_gci = len(gci_ins)
    for i in range(num_gci - 1):
        T = gci_ins[i + 1] - gci_ins[i]
        T1 = int(np.round(DQ * T))
        T2 = int(np.round(PQ * T))
        
        # Adjust T1 if needed
        while T1 + T2 > T:
            T1 -= 1
        
        # Set main weight region
        start_idx = gci_ins[i] + T2
        end_idx = gci_ins[i] + T2 + T1
        
        if start_idx >= 0 and end_idx <= len(w):
            w[start_idx:end_idx] = 1
            
            # Apply ramps
            if Nramp > 0:
                # Up ramp
                ramp_end = min(start_idx + Nramp, end_idx)
                ramp_len = ramp_end - start_idx
                if ramp_len > 0:
                    w[start_idx:ramp_end] = UPramp[:ramp_len]
                
                # Down ramp
                down_start = end_idx - Nramp
                if down_start > 0 and down_start < end_idx:
                    w[down_start:end_idx] = DOWNramp
    
    # Handle the last period
    if num_gci >= 2:
        i = num_gci - 2  # Last valid i from the loop
        Nend = N - (T2 + gci_ins[i + 1])
        
        if T2 + gci_ins[i + 1] < N:
            if T1 + T2 < Nend:
                start_idx = gci_ins[i + 1] + T2
                end_idx = gci_ins[i + 1] + T2 + T1
                
                if start_idx >= 0 and end_idx <= len(w):
                    w[start_idx:end_idx] = 1
                    
                    if Nramp > 0:
                        # Up ramp
                        ramp_end = min(start_idx + Nramp, end_idx)
                        ramp_len = ramp_end - start_idx
                        if ramp_len > 0:
                            w[start_idx:ramp_end] = UPramp[:ramp_len]
                        
                        # Down ramp
                        down_start = end_idx - Nramp
                        if down_start > 0 and down_start < end_idx:
                            w[down_start:end_idx] = DOWNramp
            else:
                T1_new = Nend - T2
                start_idx = gci_ins[i + 1] + T2
                end_idx = gci_ins[i + 1] + T2 + T1_new
                
                if start_idx >= 0 and end_idx <= len(w):
                    w[start_idx:end_idx] = 1
                    
                    if Nramp > 0:
                        # Up ramp only
                        ramp_end = min(start_idx + Nramp, end_idx)
                        ramp_len = ramp_end - start_idx
                        if ramp_len > 0:
                            w[start_idx:ramp_end] = UPramp[:ramp_len]
    
    return w



def tvlp_warmup_numba(verbose=True):
    """
    Pre-compile Numba functions for TVLP_formants() with minimal data.

    Parameters
    ==========
    verbose: string, default='True'
    
    """
    if verbose:
        print("Warming up Numba (compiling TVLP analysis functions)...",end="",flush=True)
    
    start = time.time()
    
    # Minimal test data
    frames = np.random.randn(2400).astype(np.float32).reshape((3,800))
    weights = np.ones(2400).astype(np.float32).reshape((3,800))
    
    _ = _process_frames_l2_parallel(frames[:2], weights[:2], 10, 3)
    _ = _process_frames_l1_parallel(frames[:2], weights[:2], 10, 3, max_iter=1)
    
    if verbose:
        print(f" Done! ({time.time() - start:.1f}s)")

def TVLP_formants(x, fs, tf = 6000, order=-1, frame_duration_sec=0.1, q=3, 
                  norm='l2', qcp=True, step=0.01, nformants=4, f0median=200, 
                  max_bandwidth=800, use_parallel=False, verbose=False):
    """
    This formant tracking function implements the time-varying, pitch synchronous, method that was described by Gowda et al. (2020).  The most important innovation of this method is that it smooths the Linear Prediction coefficients over time, providing a continuity constraint at the level of the LP coefficients. The second most important method is that (like Robust LPC) the method can optionally weight the audio samples pitch synchronously so the LP coefficients are caluculated from the glottal closed phase where the assumptions of LPC analysis are better met.  The third innovation, which also seems to be both the most computationally costly and least important for good formant tracking, is that the LP coefficients can be calculated minimizing the L1 norm (city-block) norm rather than the default L2 norm (Euclidian distance).  The L1 norm is 10 times slower than the L2 norm and results in almost no change in the derived formant frequencies.   

    This Python implementation is a translation (done with extensive help from the AI model Claude) of Gowda's ftrack matlab code.  It uses numba for just-in-time (JIT) compilation of machine code, and parallelization of some loops which dramatically increases the speed of processing.  However, the first call to the function includes the compilation of the JIT code (on a typical laptop this can be 10 seconds!). Subsequent calls to `TVLP_formants()` will use the compiled JIT code and be much faster than the first call.  However, in some situations you may want to separate the compilation from the first call to `TVLP_formants()`. Therefore a helper function `tvlp_warmup_numba()` is provided.  The helper function can be called at the beginning of a session to compile the JIT code, and then all subsequent calls to TVLP_formants() will be accelerated by the numba functions.
    
    Parameters
    ==========
    x : array_like
        a one-dimensional array of audio samples
    fs : int
        Sampling frequency of **x** in Hz 
    tf : int, default = 6000
        Top frequency.  The signal will be resampled to tf*2.
    order : int, default = 10
        The number of LPC coefficients to use -- the 'order' of the LPC filter.  If this parameter is -1 then we will use `phon.choose_order(base='BIC')` to determine the best value for this parameter.
    frame_duration_sec : float, default 0.1 = 100ms
        Duration of each frame in seconds.  Gowda et al.'s tests found that 100ms provides the most accurate formant measurements.
    q : int, default = 3
        Time-varying polynomial order
    norm : str, default = 'l2'
        Choose whether to minimize the 'l1' or 'l2' norm in finding the LP and polynomial coefficients.  The L1 norm requires an iterative reweighting algorithm which is much slower than minimizing the L2 norm.
    qcp : bool, default = True
        Choose whether or not to do a pitch synchronous analysis, limiting the estimation function to the glottal closed phase (the Quasi-Closed-Phase).  If True the analysis will be pitch synchronous.
    step : float, default = 0.01
        Time interval between formant measurements in seconds. Formants are taken from the middle of each interval
    nformants : int, default=3
        Number of formants to extract
    f0median : float, default=200.0
        This is an estimate of the median F0 value to expect in the signal being analyzed.  This parameter is used in the pitch synchronous method (i.e. when `qcp` is **True**) to guide the algorithm's expectation for finding glottal closure intervals.  It is important to get this approximately correct.
    max_bandwidth : float, default=800
        Maximum formant bandwidth in Hz, formants with bandwidth greater than this are rejected.
    use_parallel : bool, default=True
        Try to use parallel processing for formant extraction.
        Note: Usually doesn't work in Jupyter notebooks, set to False for notebooks
    verbose : bool, default=False
        Print progress
    
    Returns
    =======
    df : a Pandas DataFrame
         See the note.  The number of formant columns depends on the nformants parameter.

    Note
    ====
    The columns in the output dataframe are:
        * sec - midpoint time of the frame 
        * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.
        * BW1-4 - bandwidths of the vowel formants

    References
    ==========

    D. Gowda, R. Kadiri, B. Story, P. Alku (2020) `Time-varying quasi-closed-phase analysis for accurate formant tracking in speech signals.` in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1901-1914, doi: 10.1109/TASLP.2020.3000037.
    """
    overall_start = time.time()

    target_fs = tf * 2
    x, fs = prep_audio(x, fs, target_fs = target_fs, pre = 0, pad_to=frame_duration_sec, add_tiny_noise=False, quiet = not verbose)
    frame_length_samples = int(frame_duration_sec * fs)

    p=order
    if p<0:  # guess the correct LPC order
        p,order_time = choose_order(x,fs,verbose=verbose)
        if verbose:
            print(f"in TVLP_formants(), order is {p}, taken at time {order_time:.3f}")

    max_freq = fs / 2.0
    min_freq = f0median


    if verbose:
        duration_sec = len(x) / fs
        print("=" * 60)
        print(f"TRACKING FORMANTS WITH TIME-VARYING LP")
        print("=" * 60)
        print(f"Audio: {len(x)} samples ({duration_sec:.2f}s at {fs} Hz)")
        print(f"Frame duration: {frame_duration_sec}s ({frame_length_samples} samples)")
        print(f"Formant interval: {step} sec")
        print(f"LP order: p={p}, q={q}, norm={norm}")
        if qcp:
            print(f"QCP: Window the residual to fit LP coefficients on closed phase samples")
        else:
            print(f"No pitch synchronous windowing - no QCP")
        print(f"Parallel processing: {'Enabled (may fail in notebooks)' if use_parallel else 'Disabled (recommended for notebooks)'}")
        print("=" * 60)
        print()
    
    # Step 1: Extract LP coefficients and compute time-varying coefficients (Numba parallel)
    if verbose:
        print("Step 1: Computing time-varying LP coefficients...")
    
    all_ak = tvlp_batch(
        x, fs, frame_length=frame_length_samples, p=p, q=q, 
        norm=norm, qcp=qcp, f0median=f0median, verbose=verbose
    )
    
    if verbose:
        print()
    
    # Step 2: Extract formants
    if verbose:
        print("Step 2: Extracting formants...")
    
    formant_freqs, formant_bws, sample_times = tvlptoformants(
        all_ak, frame_length_samples=frame_length_samples, 
        step=step, fs=fs,
        npeaks=nformants, max_bandwidth=max_bandwidth, 
        min_freq=min_freq, max_freq=max_freq, 
        use_parallel=use_parallel, verbose=verbose
    )
    
    if verbose:
        print()
        print("Step 3: Creating DataFrame...")
    
    # Get dimensions
    n_frames, n_timepoints, n_formants = formant_freqs.shape
    
    # Flatten the first two dimensions
    # Shape: (n_frames, n_timepoints, n_formants) -> (n_frames * n_timepoints, n_formants)
    formant_freqs_flat = formant_freqs.reshape(-1, n_formants)
    formant_bws_flat = formant_bws.reshape(-1, n_formants)
    
    # Create time array in seconds
    frame_starts = np.arange(n_frames) * frame_duration_sec
    sample_offsets = sample_times / fs
    ta = frame_starts[:, np.newaxis] + sample_offsets[np.newaxis, :]
    time_array = ta.ravel()
    
    time_array = np.array(time_array, dtype=np.float64)  # Keep time in float64 for precision
    
    # Create DataFrame
    data_dict = {'sec': time_array}
    
    # Add formant frequency columns
    for i in range(n_formants):
        data_dict[f'F{i+1}'] = formant_freqs_flat[:, i]
    
    # Add bandwidth columns
    for i in range(n_formants):
        data_dict[f'BW{i+1}'] = formant_bws_flat[:, i]
    
    df = pd.DataFrame(data_dict)
    
    # Replace zeros with NaN for cleaner representation
    formant_cols = [f'F{i+1}' for i in range(n_formants)]
    bandwidth_cols = [f'BW{i+1}' for i in range(n_formants)]
    df[formant_cols] = df[formant_cols].replace(0, np.nan)
    df[bandwidth_cols] = df[bandwidth_cols].replace(0, np.nan)
    
    overall_time = time.time() - overall_start
    
    if verbose:
        print()
        print("=" * 60)
        print(f"COMPLETE!")
        print(f"Total processing time: {overall_time:.2f}s")
        print(f"DataFrame shape: {df.shape}")
        print(f"Time range: {df['sec'].min():.4f}s to {df['sec'].max():.4f}s")
        if len(x) > fs:
            rtf = (len(x) / fs) / overall_time
            print(f"Real-time factor: {rtf:.1f}x")
        print("=" * 60)
    
    return df
