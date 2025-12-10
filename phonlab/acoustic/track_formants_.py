# coding: utf-8

# Author: Keith Johnson

import numpy as np
from scipy import signal
from scipy import fft
import librosa
from pandas import DataFrame
from ..utils.prep_audio_ import prep_audio

# constants and global variables
SR = 12000
FMAX = 6000
FMIN = 200
FFT_PTS = 4096

freq_axis = fft.rfftfreq(FFT_PTS,1/SR)  # frequency axis for FFT  -- only need to calc this once
frame_length = int(SR * 0.02) # frame length 20 ms
half_frame = frame_length//2
step = int(SR * 0.01)  # number of samples between frames, 10 ms       
g_method = 'lpc'       # formant tracking method

# speaker parameters used in IFC tracking.  EG params[spkr]["fr"] are formant expectations for male if spkr == 0
params = [
    {
        'fr':np.array([ 120, 500, 1500, 2500, 3500]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5500,4500,4500]),
        'upper_bws': np.array([80,80,200]),
        'bands':np.array([[200,1100], [700, 2200], [2000,3500], [3000,4300]]),
        'spec_bounds':np.array([120,5500]),
        'du1':300, 'du2':600, 'dh1':200, 'dh2':500
    },
    {
        'fr':np.array([ 60, 620, 1860, 3100, 4340]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([5580,5580]),
        'upper_bws':np.array([3200,200]),
        'bands':np.array([[300,1300], [1000, 2500], [3500,4800], [3000,4300]]),
        'spec_bounds':np.array([250,5500]),
        'du1':400, 'du2':700, 'dh1':300, 'dh2':600
    },
    {
        'fr':np.array([ 60, 200, 1800, 3000, 4450]),
        'bws':np.array([200,50,70,90,130]),
        'upper_fs':np.array([6000,5500]),
        'upper_bws':np.array([2700,2700]),
        'bands':np.array([[400,1600], [1300, 3000], [3000,4500], [3800,5500]]),
        'spec_bounds':np.array([300,5800]),
        'du1':500, 'du2':800, 'dh1':400, 'dh2':700
    }
]


def abc_coefs(f,b,fs):
    '''abc_coefs()  -- compute coefficients for an inverse filter
    Input
        f - center frequency
        b - bandwidth
        fs - sampling frequency

    Return
        a, b c - filter coefficients
    '''
    
    exp1 = np.exp(-(np.pi/fs) * b)
    a = exp1*exp1
    b = -2.0 * exp1 * np.cos((np.pi/fs * 2) * f)
    c = 1.0/(1.0+a+b)
    return (a,b,c)

def inv_filter(x, fs, fr, bw):
    ''' inv_filter()  inverse filter x using a list of frequencies and bandwidths for the filters
    Input
        x -  a one-dimensional numpy array
        fs - the sampling frequency of x
        fr - a list of filter frequencies - the order doesn't seem to  matter
        bw - a list of filter bandwidths

    Return
        y - a one-dimensinal numpy array: the inverse filtered verion of x
    '''
    
    y = np.empty_like(x)

    # First inverse filter
    a,b,c = abc_coefs(fr[0],bw[0],fs)
    y[0] = c * x[0]
    y[1] = c * (b * x[0] + x[1])
    for n in range(2,len(y)):
        y[n] = c*(a*x[n-2] + b*x[n-1] + x[n])

    # Remaining inverse filters
    for nn in range(1,len(fr)):
        a,b,c = abc_coefs(fr[nn],bw[nn],fs)
        ay2 = ay3 = 0
        for n in range(0,len(y)):
            ay1 = ay2
            ay2 = ay3
            ay3 = y[n]
            y[n] = c*(a*ay1 + b*ay2 + ay3)
    return y

def order(n1, n2, n3, n4):
    ''' order() takes four numbers (say formant frequencies), and return them in order from lowest to highest.
    
        e.g. (f1,f2,f3,f4) = order(5000,1200,3600,500)
            will return f1=500, f2 = 1200, ....
    
    '''
    buf = np.array([n1,n2,n3,n4]);
    buf.sort()
    return(buf[0],buf[1],buf[2],buf[3])

def dominant_frequency(y):
    '''dominant_frequency() return the dominant frequency in the inverse filtered waveform.
    This function uses fft, where Watenabe's IFC tracker used zero crossing
    rate to estimate the dominant frequency in the waveform (after filtering
    out all of the other formants).

    Input
        y - a one-dimensional numpy array of one frame of inverse filtered audio
    Result
        f - the dominant frequency
    '''
    
    Z = fft.rfft(y*signal.windows.hamming(len(y)),FFT_PTS)
    f = freq_axis[np.argmax(Z)]  # return the frequency of the peak in the FFT
    
    if f>FMAX:
        f = FMAX
    if f<FMIN:
        f = FMIN
    return f

def zc_frequency2(y, fs, loop, f_no, in_freq, spkr):
    ''' zc_frequency2() is an implementation of Ueda et al.'s 2008 C code using array processing 
    techniques that are available with numpy arrays. One difference from Ueda is that the weight
    function used to calculate the weighted sum of frequency estimates from zero crossing periods
    is symmetrical.  Ueda et al. had an unsymmetrical window for F1 (favoring higher frequency estimates).
    * note that the input frequency (in_freq) is not used if loop==0.  This means that at no point
    does the estimate from a prior frame come into play for the estimate of this frame.  This 
    implementation of the algorithm is almost twice as fast as the direct python transaltion of
    Ueda's code, but it is still painfully slow.
        
    Input
        y - a one dimensional numpy array of 240 audio waveform samples
        fs - the waveform sampling frequency
        loop - which iteration in IFCBLOCK() are we in? - 0,1,2
        f_no - which formant are we measuring - 1,2,3,4
        in_freq - the previously found frequency of this formant (for this frame)
        spkr - speaker type [0=male, 1=female, 2=child]
    Result
        freq -- the dominant frequency in y
    '''

    par = params[spkr]  # use the global params[] buffer.

    if loop==0:  # first loop in IFCBLOCK, we calculate mean freq as a spectral parameter
        ca=0.2
        dd1 = 400
        dd2 = 4000

        wv = np.sum(y[1:] * y[:-1])  # product of successive samples
        sv = np.sum(y * y)            # squares of samples
        
        ww = wv/sv
        if ww > 1: ww = 1
        if ww < -1: ww = -1
            
        freq = np.arccos(ww) * (1/(2*np.pi/fs))
       
        in_freq = freq

    freq = in_freq  # loops 2 and 3 we are passed freq
    if loop > 0:
        ca=0.3 
        dd1 = 300 
        dd2 = 3000
    if loop > 1: 
        ca = 0.4
        
    aa = 0.5 * (0.3+ca)

    y = y - np.mean(y)
    wv = np.clip(y[:-1] * y[1:], a_min=-1, a_max=1)  # detect zero crossings
    ab = np.abs(y)
    rrr = ab[:-1]/(ab[:-1]+ab[1:])  # fine tune by amplitude ratio
    zci = np.where(wv<=0)[0]  # indeces of zero crossings
    zcp = (zci[1:-1] - zci[:-2]) - rrr[zci[:-2]] + rrr[zci[1:-1]]
    z2 = fs/(zcp[:-1]+zcp[1:])  # frequency of each zero crossing period
    
    if f_no<=1: # F1
        al1 =(par["du2"]-par["du1"])/1000 # these variables are different for different vocal tract lengths
        b1 = par["du1"] - al1*200  
    if f_no==2 or f_no==3:  # F2 and F3
        al1 = (dd2-dd1)/4300
        b1 = dd1 - al1*200
    if f_no>=4:   # F4
        al1 = 0.83721  # (4000 - 400) / 4300
        b1 = 400 - al1 * 200     # 400 - al1*200
    
    dep = freq * al1 + b1
    for i in range(3):
        deviation = np.fabs(z2-freq)
        dep = dep - (i * aa * dep) 
        w = np.clip(1-deviation/dep,a_min=0,a_max=1)
        freq = np.sum(z2*w)/(np.sum(w) + 0.001)
    
    if freq < FMIN: freq=FMIN
    if freq > FMAX: freq=FMAX
        
    return freq

def zc_frequency(x, fs, loop, f_no, in_freq, spkr):
    ''' zc_frequency - a direct copy of Watenabe/Ueda's zero crossing c code to determine 
    the dominant frequency in y.  This method calculates the weighted mean of the zero-crossing
    period distribution in the frame.

    Input
        y - a one dimensional numpy array of 240 audio waveform samples
        fs - the waveform sampling frequency
        loop - which iteration in IFCBLOCK() are we in? - 0,1,2
        f_no - which formant are we measuring - 1,2,3,4
        in_freq - the previously found frequency of this formant
        spkr - speaker type [0=male, 1=female, 2=child]
    Result
        freq -- the dominant frequency in y
    '''
    par = params[spkr]  # use the global params[] buffer.
    
    z = np.empty(frame_length)
    z2 = np.empty(frame_length)
    y = x-np.mean(x)
    
    if loop==0:  # first loop in IFCBLOCK, we calculate mean freq as a spectral parameter
        wv = np.sum(y[1:] * y[:-1])  # product of successive samples
        sv = np.sum(y * y)            # squares of samples
        
        ww = wv/sv
        if ww > 1: ww = 1
        if ww < -1: ww = -1
            
        freq = np.arccos(ww) * (1/(2*np.pi/fs))
        if freq < FMIN: freq = FMIN
        if freq > FMAX: freq = FMAX
            
        in_freq = freq
        ca=0.2
        dd1 = 400
        dd2 = 4000

    freq = in_freq  # loops 2 and 3 we are passed freq
    if loop > 0:
        ca=0.3 
        dd1 = 300 
        dd2 = 3000
    if loop > 1: 
        ca = 0.4
        
    aa = 0.5 * (0.3+ca)
    noc = zcp = 0
    wv2 = y[0]
    for n in range(1,frame_length):
        wv1 = wv2  # two samples separated by one time step
        wv2 = abn = y[n]
        if (abn<0): abn = -abn
        if (wv1==0 and wv2==0): zcp=0
        if (wv1*wv2 > 0): zcp = zcp + 1  # both samples are positive  
        if wv1==0 and wv2!=0:  zcp = zcp + 1  # the first is zero and second is not
        if wv1*wv2 < 0 or (wv1!=0 and wv2==0):  # one is negative and one is positive
            abb=wv1  # amplitude of the first
            if abb<0: abb = -abb
            rrr=abb/(abb+abn)  # interpolate between waveform points, for accurate period length
            z[noc] = zcp+rrr # length of a half period
            noc = noc+1  # number of zero crossings
            wmax = abn
            zcp=1-rrr  
            
    nz=noc-1  # number of zero-crossings detected
    if f_no ==1:
        z2 = (fs*0.5)/z[:nz]  # estimate frequency from 1/2 period length for F1
    else:
        z2 = fs/(z[:nz-1] + z[1:nz])  #estimate frequency from period length for F2-4
        nz = nz-1

    if f_no<=1: # F1
        al1 =(par["du2"]-par["du1"])/1000 # these variables are different for different vocal tract lengths
        al2 = (par["dh2"]-par["dh1"])/1000
        b1 = par["du1"] - al1*200  
        b2 = par["dh1"] - al2*200  
    if f_no==2 or f_no==3:  # F2 and F3
        al1 = al2 = (dd2-dd1)/4300
        b1 = b2 = dd1 - al1*200
    if f_no>=4:   # F4
        al1 = al2 = 0.83721  # (4000 - 400) / 4300
        b1 = b2 = 400 - al1 * 200     # 400 - al1*200

    ifl=True
    for m in range(3):
        dep = freq*al1 + b1  # size of search window is frequency dependent
        dem = freq*al2 + b2
        if ifl:
            dep = dep - (m * aa * dep)  # narrow the search window, positive side
            dem = dem - (m * aa * dem)   #   and negative size
            if dep <= 100 or dem <= 100: ifl = False

        sw=wv=0
        for n in range(nz):
            if z2[n] >= freq: 
                w = 1 - (z2[n] - freq)/dep
            if z2[n] < freq: 
                w = 1 - (freq - z2[n])/dem
            if w>0:
                sw = sw + w
                wv = wv + w*z2[n]
                
        if sw != 0 and wv != 0:
            freq = wv/sw
            
    if freq < FMIN: freq=FMIN
    if freq > FMAX: freq=FMAX

    return freq
    
        
def IFC(x,fs,n,f,b,f0,b0):
    ''' IFC - this is basically just a call to inv_filter() with one frequency to 
    inverse filter out of x.  However, F1 inverse filtering gets an assist from
    an f0 inverse filter.  So if f0 is greater than zero we will use it and b0 
    in an extra step of inverse filtering 

    Input
        x - a one-dimensional numpy array with audio waveform files
        fs - the sampling frequency
        n - the formant number of the formant being removed
        f - the frequency (Hz) of the formant being removed
        b - the bandwidth (Hz) of the formant being removed
        f0 - the f0 frequency (used if n==1) to be removed
        b0 - the bandwidth of f0
    Result
        y - a numpy array after inverse filtering, with the same shape of x
    '''
    
    if (n==1):  # for F1 filter out both at F1 and F0
        fr = np.array([f,f0])
        bw = np.array([b,b0])
    else:
        fr = np.array([f])
        bw = np.array([b])

    y = inv_filter(x,fs,fr,bw)
    return y

def IFCBLOCK(x,fs, nc,fc,bc,nd,fd,bd,f0,b0,spkr):
    '''IFCBLOCK() performs three loops of inverse filtering and frequency estimation
    of two formants (fc and fd).  First one (fc) is filtered out and then the other (fd) is estimated. 
    Then fd is filted out and fc is estimated.
    
    Input
        x - a one-dimensional numpy array with audio waveform files
        fs - sampling frequency
        fc,bc - center frequency and bandwidth of a formant to be estimated
        fd,bd - center frequency and bandwidth of the other
        f0,b0 - used to help estimate F1
        spkr - speaker type [0=male, 1=female, 2=child]
    Result
        fc, fd - new estimates of the formants
    '''
    
    if g_method == "ifc" or g_method == "ifc_old":
        loop = 3
    else:
        loop = 1
        
    for i in range(loop):  # Ueda et al do this 3 times
        y = IFC(x,fs,nc,fc,bc,f0,b0)  # inverse filter fc
        if g_method=="ifc_old":
            fd = zc_frequency(y,fs,i,nd,fd,spkr)
        elif g_method=="ifc":
            fd = zc_frequency2(y,fs,i,nd,fd,spkr)
        else:
            fd = dominant_frequency(y)
            
        y = IFC(x,fs,nd,fd,bd,f0,b0)  # inverse filter fd
        if g_method=="ifc_old":
            fc = zc_frequency(y,fs,i,nc,fc,spkr)
        elif g_method=="ifc":
            fc = zc_frequency2(y,fs,i,nc,fc,spkr) 
        else:
            fc = dominant_frequency(y) 
            
    return fc, fd   # our two estimates of the target formants

def design_filter(bounds, fs, order=4):
    ''' Use scipy.signal.butter() to design a Butterworth bandpass filter.

        inputs:
            bounds - the lower bound, upper bound of a bandpass filter
            fs - sampling frequency of the audio
            order - the filter order of the filter, default = 4
        
        output:
            coefs - sos filter coefficients of the bandpass filters

    '''
    
    return signal.butter(order, bounds, fs=fs, btype='bandpass', output='sos')
    
def band_limit(x, fs, bounds):
    '''band_limit(x,bounds,fs) -- bandpass the input array by the upper and lower
    frequency bounts.

    Input
        x - a one-dimensional numpy array with audio waveform samples
        fs - sampling frequency
        bounds - a two-element array with the lower and upper bounds of the filter
   Result
        y - the result of passing x through the bandpass filter
    '''
    
    coefs = design_filter(bounds,fs,8)
    return signal.sosfiltfilt(coefs,x)

def get_amplitude_ratios(x, fs, filterbank):
    '''get_amplitude_ratios() - returns the relative amplitude ratios in 
    formant spectral regions.  If we wanted to keep multiple copies of the 
    waveform, we could do this step once for the entire file.

    Input
        x - a one-dimensional numpy array with audio waveform samples
        fs - sampling frequency
        filterbank - a two dimensional array - upper and lower bounds for each formant
    Result
        r12 - ratio of engery in F2 and F1 in dB:  20*log10(a2/a1)
        r23 - ratio of energy in F3 and F2 regions
        r34 - the same for F4 versus F3
    '''
        
    n_channels = len(filterbank)   # must be 4 - one for each formant
    rms = np.zeros(n_channels)
    n_samples = len(x)
    
    y = np.zeros((n_channels, n_samples))
    for idx, coefs in enumerate(filterbank):
        y[idx] = signal.sosfiltfilt(coefs, x)  # mean square amp in each band
        rms[idx] = np.sqrt(np.sum(y[idx]**2)/len(y))
        
    r12 = 20*np.log10(rms[1]/rms[0])
    r23 = 20*np.log10(rms[2]/rms[1])
    r34 = 20*np.log10(rms[3]/rms[2])
    
    return (r12,r23,r34)
    
def track_pitch(x,fs, fr,bw,f0_range):
    '''track_pitch() find f0 in a frame of audio, using inverse filtering of formants and bandwidths 
    to derive a glottal waveform.  Uses autocorrelation from the numpy.correlate() function.

    Input
        x - a one-dimensional numpy array with one frame (20ms) of audio waveform samples 
        fr - an array of formant frequencies to remove from x
        bw - an array of the bandwidths for fr
        pitch_range - List of [low,high] defining the pitch range to be considered for f0 (in Hz)
        fs - sampling frequency
    Result
        f0 - the pitch of x
    '''
    
    th = fs//f0_range[1]
    tl = fs//f0_range[0]
    y = inv_filter(x,fs,fr,bw)  # get the inverse filtered waveform - an approximation of glottal flow
    result = np.correlate(y, y, mode='full') # autocorrelation 
    ac = result[result.size//2:] # the autocorrelation is in the last half of the result
    i = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
    f0 = 1/(i/fs)      # converted to Hz
    
    return f0, np.sqrt(ac[i])/np.sqrt(ac[0])

def track_pitch_lpc(x, fs, a, pitch_range):
    '''track_pitch() find f0 in a frame of audio, using inverse filtering from LPC coefficients
    to derive a glottal waveform.  Uses autocorrelation from the numpy.correlate() function.

    Input
        x - a one-dimensional numpy array with audio waveform samples
        fs - sampling frequency
        a - an array of LPC coefficients [1,-A] from get_LPC()
        pitch_range - List of [low, high] defining the pitch range to be considered for f0 (in Hz)
    Result
        f0 - the pich of x
        c - peak of the autocorrelation
    '''
    
    th = fs//pitch_range[1]
    tl = fs//pitch_range[0]
    
    y = np.convolve(x,a)  #inverse filter
    result = np.correlate(y, y, mode='full') # autocorrelation 
    ac = result[result.size//2:] # the autocorrelation is in the last half of the result
    i = np.argmax(ac[th:tl]) + th # index of peak correlation (in range lowest to highest)
    f0 = 1/(i/fs)      # converted to Hz
    
    return f0, np.sqrt(ac[i])/np.sqrt(ac[0])


def get_rms_amplitude(y):
    '''get_rms_amplitude() -- calculate RMS amplidude over a frame of an auditory waveform.  
    You can also use this function with numpy.apply_along_axis()
    
        rms = np.apply_along_axis(get_rms_amplitude, 1, y)

    Input
        y -- a 1-dimensional array of audio samples (one frame of audio)

    Result
        amp - an array of RMS amplitude measurements, one per frame of y
    '''
    
    return np.sqrt(np.mean(y**2))  # RMS amplitude
     
def IFC_process_frame(x,fs, spkr,f0_range,filterbank):
    '''IFC_tracking() find formant frequencys, RMS amplitude, and f0 in a 20ms chunk of audio waveform.
    Formants are found using the Inverse Filter Control (IFC) method of Watenabe (2001), Ueda et al. (2007). 
    RMS amplitude is measured after removing energy in the region of F5 and F6. Pitch (f0) is found by 
    autocorrelation of the inverse filtered signal after we have found the formants.

    Input
        x - a one-dimensional numpy array with one 20ms frame of audio waveform samples (must be sampled at 12kHz)
        fs - the sampling frequency of x
        spkr -- a flag relating to vocal tract length (1=long/male; 2=medium/female; 3=short/child)
        f0_range - the lowest and highest values to consider in pitch tracking.  Default is [63,400]
        filterbank - filter bank coefficients for determing amplitude ratios in formant bands

    Result
        amp,f0,f1,f2,f3,f4 - acoustic measurements for this frame of audio
    '''   
    y = inv_filter(x,fs, params[spkr]["upper_fs"],params[spkr]["upper_bws"])  # remove upper formants

    # unpack array of starting expected formant frequencies
    f0,f1,f2,f3,f4 = params[spkr]["fr"][[0,1,2,3,4]]
    b0,b1,b2,b3,b4 = params[spkr]["bws"][[0,1,2,3,4]]
    r12, r23, r34 = get_amplitude_ratios(y,fs, filterbank)  

    if g_method == "ifc_old" or g_method == "ifc":
        loop = 3
    else:
        loop = 1

    for i in range(loop):  # loop over main IFC blocks to find formant frequencies
        b0 = 200
        
        # estimate F2 and F3 first
        y2 = inv_filter(y,fs,[f4,f0,f1],[b4,b0,b1])  # filter out all but f2 and f3
        if (r23 >= -20): b0 = 100*r23 + 2200
        f2,f3 = IFCBLOCK(y2,fs,3,f3,b3,2,f2,b2,f0,b0,spkr)
        (f1,f2,f3,f4) = order(f1, f2, f3, f4)

        # estimate F1 and F2 next
        if spkr==1:  # longer vocal tract - remove f6 again
            y2 = inv_filter(y,fs, [5500,f3,f4],[200,b3,b4])
        else:
            y2 = inv_filter(y,fs, [f3,f4],[b3,b4]) # filter out all but f1 and f2
        
        if (r12 >= -20): b0 = 100*r12 + 2200
        f1,f2 = IFCBLOCK(y2,fs,1,f1,b1,2,f2,b2,f0,b0,spkr)
        (f1,f2,f3,f4) = order(f1, f2, f3, f4)

        # estimate F3 and F4
        
        y2 = inv_filter(y,fs,[f1,f0,f2],[b1,b0,b2])  # filter out all but f3 and f4
        if (r34 >= -20): b0 = 100*r34 + 2200
        f3,f4 = IFCBLOCK(y2,fs,3,f3,b3,4,f4,b4,f0,b0,spkr)
        (f1,f2,f3,f4) = order(f1, f2, f3, f4)
    
        oldFs = np.array([f1, f2, f3, f4])    
        
    # track pitch
    f0,c = track_pitch(y,fs,oldFs,params[spkr]["bws"],f0_range)  # use final estimate formants in pitch tracking

    return np.round([f1,f2,f3,f4,f0,c],3)


# ------------ functions for LPC analysis -----------------------
#
#  The functions create_frames(), solve_lpc(), and get_LPC()  are lightly adapted from those provided by 
#  Guilherme Kunigami, in a 2021 blog post:
#  https://www.kuniga.me/blog/2021/05/13/lpc-in-python.html
#
#  Then, get_LPC_lr() uses Librosa vectorized functions to do the same thing at twice the speed.  

def create_frames(x, w, R = 0.5):
    '''create_frames() - create a 2D array that has windowed waveform frames,
    ready to be passed, one at a time, to an analysis routine (like solve_lpc().
    The number of samples in each frame is determined by the length of the window (w),
    and the frames may overlap by a proportion R.  The time-point (in samples) of 
    the center of each frame is returned in an array t.  Convert t into seconds by 
    dividing by the sampling frequency.

    Input
        x - a one-dimensional numpy array with audio waveform samples 
        w - a one-dimensional numpy array with a window function (hamming, boxcar, etc.)
        R - the proportion of overlap between successive frames (>0, and <=1), default is 0.5
    Result
        B - a two-dimensional numpy array of windowed audio samples 
            the shape is:  (number of frames, number of samples in a frame)
        t - an array of time points (in samples) marking the midpoint of each frame
    '''
    
    n = len(x)
    nw = len(w)
    step = int(nw * (1 - R))
    nb = int((n - nw) / step) + 1

    B = np.zeros((nb, nw))
    t = np.zeros(nb)

    for i in range(nb):
        offset = i * step  # offset is the location of the start of the window
        t[i] = (offset + nw/2)
        B[i, :] = w * x[offset : nw + offset]

    return B,t

def solve_lpc(x, p):
    '''solve_lpc() - get LPC coefficients and error variance for one frame of audio.

    Input
        x - a one-dimensional numpy array with audio waveform samples 
        p - the number of coefficients (order) for the LPC polynomial
    Result
        a - the fitted LPC coefficients
        g - the error variance 
    '''
    
    b = x[1:]

    # ----- make matrix X ---------
    n = len(x)
    # [x_n, ..., x_1, 0, ..., 0]
    xz = np.concatenate([x[::-1], np.zeros(p)])

    X = np.zeros((n - 1, p))
    for i in range(n - 1):
        offset = n - 1 - i
        X[i, :] = xz[offset : offset + p]
    # ------------------
    a = np.linalg.lstsq(X, b.T,rcond=None)[0]  # order coefficients
    
    return [a]

def get_LPC(x,fs, frame_length,p):  # add a flag for computing RMS
    '''get_LPC() -- compute LPC coefficients for a waveform array.  You can,
    and usually do, pass an array of the waveform of a whole file.

    Input
        x - a one-dimensional numpy array with audio waveform samples 
            (must be sampled at 12kHz, and at least 'frame_length' long)
        fs - the sampling frequency of x
        frame_length - the number of samples in a frame
        p - the LPC order (see choose_order())
    Result
        A - a two dimensional array of p coefficients for each frame of the audio file
        G - the variance of the LPC residual
        t - the time points, in seconds, of each frame in A and G
    '''
    
    w = signal.windows.hamming(frame_length)
    B,t = create_frames(x,w)
    t = t/fs
    nb = B.shape[0]

    rms = np.apply_along_axis(get_rms_amplitude, 1, B)

    # I also tried "apply_along_axis() for solving the lpc.  It is a tad slower than this loop
    A = np.zeros((nb,p))
    for i in range(nb):
        [a] = solve_lpc(B[i, :], p)
        A[i,:] = a
        
    A = np.insert(-A, 0, 1,axis=1)  # inverse filter is a[0] = 1, a[1-p] = -a
    return (A,rms,t)
# ------------ functions for LPC analysis -----------------------


def get_LPC_lr(x,fs, frame_length,p):
    '''get_LPC_lr() -- uses librosa functions to compute LPC coefficients for a waveform array.  
    You can, and usually do, pass an array of the waveform of a whole file.  This routine is more than 
    twice as fast as get_LPC() which uses 'solve_lpc() in a for loop.

    Input
        x - a one-dimensional numpy array with audio waveform samples 
            (must be sampled at 12kHz, and at least 'frame_length' long)
        fs - the sampling frequency of x
        frame_length - the number of samples in a frame
        p - the LPC order (see choose_order())

    Result
        A - a two dimensional array of p coefficients for each frame of the audio file
        rms - a one dimensional array of RMS amplitude values.
        t - the time points, in seconds, of each frame in A and rms (a 1D array)
    '''
    
    step = frame_length//2
    
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=step,axis=0) 
    w = signal.windows.hamming(frame_length)
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame
    t = librosa.frames_to_time(range(frames.shape[0]),sr=fs, hop_length=step,n_fft=frame_length)
    A = librosa.lpc(frames, order=p,axis= -1)
    return A,t
    

def choose_order(x,frame_length,fs):
    '''choose_order() -- The most impactful parameter in LPC analysis is the 'order',
    the number of LPC coefficients to use in calculating a fit to the spectrum.  Many
    authors propose a rule of thumb having to do with the expected number of formants
    in the analyzed frequency range.  This function takes an array of waveform samples,
    and calculates LPC coeffients
    at each of several orders [8,10,12,14,16] -- assuming that the fs is 12,0000 Hz.
    It then returns the order that produced the lowest A[1] over the span.  

    Input
        x - a one-dimensional numpy arry with audio waveform samples (must be sampled at 12kHz)
        frame_length - the number of samples in a frame
        fs - the sampling frequency of x
    Result
        lpc_order - the best fitting choice to use as the order term in LPC analysis
    '''
    
    min = 1e10 
    os = (fs//1000) - 4  # 12-4 = 8
    oe = (fs//1000) + 3  # 12+3 = 15

    frames = librosa.util.frame(x, frame_length=frame_length, 
                                hop_length=step,axis=0)  # use librosa to make frames  
    w = signal.windows.hamming(frame_length)
    frames = np.multiply(frames,w)   # apply a Hamming window to each frame

    for order in range(os,oe,2):  # try different order values
        A = librosa.lpc(frames,order = order, axis=1)
        A1 = np.linalg.norm(A[:,1])
        if A1 < min:
            min = A1
            lpc_order = order        
    return lpc_order

def LPC_tracking(x, fs, f0_range = [63,400], order = -1, preemphasis = 1.0, quiet = False):
    '''LPC_tracking() uses the Librosa implemenation of linear predictive coding 
(Markel & Gray) to find the vowel formant frequencies in a sound file, or array of audio samples.  Formant freqquencies are found from the LPC coefficients using polynomial root solving.  The function also uses the LPC coefficients to inverse filter the waveform and then calculate f0 (voice pitch) from the quasi glottal waveform using autocorrelation. The peak autocorrelation values are normalized and returned as a voicing score between 0 and 1.  Finally, the RMS amplitude of the waveform in each frame of audio is also returned.  A dataframe of measurements is returned with data at intervals of 10ms.  
 
    Input
        sig - the name of a sound file, or an array of audio samples
        chan - if the audio in sound is stereo, which channel should be analyzed? Default: 0
        preemphasis - factor of a preemphasis factor (0-1). Default: 1.0
        sample_rate - the sampling rate of sound if it is an array - will be resampled to 12,000 Hz
        f0_range - the lowest and highest values to consider in pitch tracking.  Default is [63,400]
        lpc_order - pass a value of -1 to have the function call choose_order() to determine the best
            value for this parameter.  Or pass a positive integer.

    Result
        df - a pandas dataframe with formant, f0, amplitude, and voicing score measurements at 0.01 sec intervals.

        The columns in the output dataframe are:
            sec - midpoint time of the frame 
            rms - the rms amplitude 
            F1-4 - the lowest four vowel 'formants' - vocal tract resonances.
            f0 - the fundamental frequency of voicing
            voicing - a voicing score [0,1], higher values indicate stronger evidence of voicing
    '''

    if not quiet: print(f"LPC_tracking(), with order set to {order}, and pitch range {f0_range}")

    x, fs = prep_audio(x, fs, target_fs = SR, pre = 0,quiet = quiet)  # downsample, no preemphasis
    rms = librosa.feature.rms(y=x,frame_length=frame_length, 
                              hop_length=step,center=False) # get rms amplitude
    rms = 20*np.log10(rms[0])
   
    if (preemphasis > 0): y = np.append(x[0], x[1:] - preemphasis * x[:-1])  # now apply pre-emphasis
        
    if order<0:  # guess the correct LPC order
        if len(x) < fs*2:
            test_x = y   # choose order using the whole file
        else:
            m = np.argmax(rms)*step  # limit the sample to 2 seconds (centered on the peak RMS amplitude)
            s = m - fs  # back a second
            e = m + fs  # forward a second
            if s<0:  # don't foll off the front or back of the buffer
                s=0; e=fs*2
            if e > len(x)-1:
                e = len(x)-1
                s = e-fs*2
            test_x = y[s:e]
        order = choose_order(test_x,frame_length,fs)
        if not quiet: print(f"Selected LPC order is: {order}")

    (A,t) = get_LPC_lr(y,fs, frame_length,order)  # LPC coefs for whole file using this order

    nb = A.shape[0]  # the number of frames (or blocks) in the LPC analysis
    nf = int((order-2)/2)  # the number of formants that will be computed

    formants = np.empty((nb,nf))  # arrays to be filled by analysis
    #bandwidths = np.empty((nb,nf))
    f0 = np.empty((nb))
    c = np.empty((nb))

    for i in range(nb):  # find formants from LPC coefficients (A) using root solving
        start = int(t[i]*fs-half_frame)
        end = int(t[i]*fs+half_frame)

        # measure F0 using inverse filtering with the LPC filter A[z]
        f0[i],c[i] = track_pitch_lpc(y[start:end],fs,A[i,:],f0_range) 

        roots = np.roots(A[i,:])  # solve for the roots of the polynomial
        roots = roots[np.imag(roots)>=0]  # only keep the positive ones
        fr = np.angle(roots) *  fs/(2 * np.pi)  # calculate formant frequencies from them
        bw = -(fs/(np.pi)) * np.log(abs(roots)) # calculate bandwidths from them

        # A wide bandwidth criterion (about 800 Hz) is needed to capture fast moving formants.
        fr[bw>800] = np.nan  # reject formants with too wide bandwidth, NaN sorts to the end of the array
        s = np.argsort(fr)  # get the sort order so you can apply it more than one array
        fr = fr[s]  # sort the formants from lowest to highest
        bw = bw[s]  # use the same sort order with the bandwidth array

        formants[i,:] = fr[:nf]  
        #bandwidths[i,:] = bw[:nf]

    list = ['sec', 'rms', ] + [f"F{i}" for i in range(1,nf+1)] + ['f0', 'c']
    df = DataFrame(np.concatenate((t.reshape(nb,1), rms.reshape(nb,1), formants, f0.reshape(nb,1),c.reshape(nb,1)), axis=1),columns=(list))

    return df


def IFC_tracking(x, fs, preemphasis = 0.94, f0_range = [63,400], speaker=0, quiet = False):
    
    if not quiet: 
        print(f"IFC_tracking(), using method {g_method}, with speaker set to {speaker}, and pitch range {f0_range}")

    x, fs = prep_audio(x, fs, target_fs = SR, pre = 0,quiet=quiet)  # downsample waveform, no preemphasis

    rms = librosa.feature.rms(y=x,frame_length=frame_length, 
                              hop_length=step,center=False) # get rms amplitude
    rms = 20*np.log10(rms[0])

    # apply preemphasis
    y, fs = prep_audio(x,fs,pre=preemphasis,target_fs=fs,quiet=True) 
    
    filterbank = [design_filter(b, fs,order=12) for b in params[speaker]["bands"]]  
    time_axis = np.arange(len(y))/fs

    formants = np.empty((0,8))
    frame_count = 0
    for index in range(half_frame,len(y)-frame_length,step):
        t = index/fs
        x_win = y[index-half_frame:index+half_frame+1]

        row = IFC_process_frame(x_win,fs,speaker,f0_range, filterbank)
        formants = np.append(formants,[np.concatenate(([t],[rms[frame_count]],row))],axis=0)
        if not quiet:   # count time though the file
            if (t % 0.02) < 0.001:  print(f"\r {t:.2f} sec.", end='')
        frame_count += 1
        
    if not quiet: print(f"\r done         ")
        
    df = DataFrame(formants,columns=("sec","rms","F1","F2","F3","F4","f0","c"))

    return df

def track_formants(x,fs, method='lpc', preemphasis = 1.0, f0_range = [63,400], speaker = 0, order= -1, quiet=False):
    """Computes the vowel formant values in audio of speech.
    
The function uses either LPC analysis or the IFC method to calculate vowel formants (the resonant frequencies of the vocal tract) and then returns a dataframe of measurements (formants, f0, voicing, and amplitude) at 10ms intervals for the duration of sound.

The IFC option uses the Watenabe/Ueda Inverse Filter Control method of vowel formant tracking (Watenabe, 2001; Ueda et al., 2007).  This method can be quite a bit more accurate than LPC, but is also substantially slower. In addition to calculating formants, the function uses the estimated formant frequencies to inverse filter the waveform and then calculate f0 (voice pitch) from the quasi glottal waveform using autocorrelation. The peak autocorrelation values are normalized and returned as a voicing score between 0 and 1.  Finally, the RMS amplitude of the preemphasised waveform is also returned.  

The LPC option uses the Librosa implemenation of Burg's (1975) autocorrelation method for linear predictive coding and then polynomial root solving to find the vowel formant frequencies. The function also uses the LPC coefficients to inverse filter the waveform and then calculate f0 (voice pitch) from the quasi glottal waveform using autocorrelation. The peak autocorrelation value of each frame is also returned.  Finally, the RMS amplitude of the waveform is also returned.  

Parameters
==========
x : array
    a one-dimensional array of audio samples

fs : int
    the sampling rate of the audio in **x** 

preemphasis : float, default = 1.0
    factor of a preemphasis factor (0-1).

f0_range : list, default = [63,400]
    the lowest and highest values to consider in pitch tracking.
    
method : string, default = "lpc"
    a string indicating which formant tracking method to use.

        * lpc - Linear predictive coding (the default value)
        * ifc - Inverse filter control
        * ifc_old - a slower direct python implementation of Ueda et al.'s code
        * ifc_fast - a much faster but less accurate approach to IFC

speaker : int, default = 0
    set formant expectations for IFC. This parameter is only used in IFC analysis.
    
      * 0 = long vocal tract, male speaker
      * 1 = medium, female speaker
      * 2 = small, a child.
    
order : int, default = -1
    the default value, -1 tries several values of lpc_order and determines the best value for this audio file.  You can also specify the order by setting this parameter to a positive integer (12 or 14 for a male speaker, 10 or 12 for a female speaker, 8 or 10 for a child).  This parameter is only used in LPC analysis.

quiet : boolean, default = False
    False means print progress or warning messages

Returns
=======
df : dataframe
    a pandas dataframe with formant, f0, amplitude, and voicing score measurements at 0.01 sec intervals.

Note
====

The columns in the output dataframe are:
    * sec - midpoint time of the frame 
    * rms - the rms amplitude 
    * F1-4 - the lowest four vowel 'formants' - vocal tract resonances.
    * f0 - the fundamental frequency of voicing
    * c - the autocorrelation value, higher indicates stronger evidence of voicing

References
==========

J. P. Burg (1975) `Maximum entropy spectral analysis`, Ph.D. dissertation, Dep. Geophys., Stanford Univ., Stanford, CA.

A. Watenabe (2001) `Formant estimation method using inverse-filter control`. IEEE Trans. Speech Audio Processing, **9**, 317-326.

Y. Ueda, T. Hamakawa, T. Sakata, S. Hario, & A. Watanabe (2007) `A real-time formant tracker based on the inverse filter control method`. Acoustical Science and Technology,  **28** (4), 271–274. https://doi.org/10.1250/ast.28.271



Examples
========

Use Inverse Filter Control to track formants in a file

.. code-block:: Python

    df = phon.track_formants(x,fs,method='ifc',speaker=1)

The next example uses LPC analysis, which by default will try to pick the best `lpc_order` for the speaker.
An array of samples is loaded by `phon.loadsig()` and then passed to `phon.track_formants()`.  Then `phon.sgram()` plots
the spectrogram of `x`, and the seaborn graphics package is used to add the formants to the spectrogram.

.. code-block:: Python

    example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'sf3_cln.wav'
    x,fs = phon.loadsig(example_file, chansel=[0])
    fmtsdf = phon.track_formants(x, fs)

    ret = phon.sgram(x,fs, tf=6000, cmap="Reds")  # plot the spectrogram

    dot_color = "dodgerblue"
    sns.pointplot(fmtsdf,x='sec',y='F1',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F2',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F3',linestyle='none',native_scale=True,marker=".",color=dot_color)
    sns.pointplot(fmtsdf,x='sec',y='F4',linestyle='none',native_scale=True,marker=".",color=dot_color)

    fmtsdf.to_csv("sf3_cln.csv",index=False)

.. figure:: images/track_formants.png
   :scale: 33%
   :alt: a spectrogram with the estimated vowel formants marked with blue dots 
   :align: center

   Plotting the formants found by `track_formants(... method='ifc')`, on the spectrogram of the utterance.

"""

    globals()['g_method'] = method
    
    if method == 'lpc':
        df = LPC_tracking(x, fs, preemphasis = preemphasis, f0_range=f0_range, order=order, quiet = quiet)
    else: 
        df = IFC_tracking(x, fs, preemphasis = preemphasis, f0_range=f0_range, speaker=speaker,quiet = quiet)

    return df


