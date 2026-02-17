import numpy as np
from scipy import signal

def test_signal(type = "sine", fs = 10000, dur=0.5, freq=500, amp = 1) :
    '''Generate audio test signals of various types.

Parameters
==========
    type : string, default="sine"  
        The type of waveform to be generated.  May be one of:
        
            * "sine", 
            * "sawtooth", 
            * "triangle", 
            * "square", or 
            * "impulse".

    fs : integer, default = 10000
        The sampling rate of the returned audio test signal.
    dur : float, default = 0.5
        The duration (in seconds) of the returned audio test signal
    freq : numeric, default = 500
        The frequency (in Hz) of the returned audio test signal.
    amp : float, default = 1.0
        The amplitude of the returned signal, ranging from 0 (silence) to 1 (maximum amplitude)

Returns
=======
    out: ndarry 
        a one-dimensional numpy array of length `dur` * `fs`

Raises
======
    ValueError: 
        if the type parameter is not one of the allowed types.


.. code-block:: Python

    import matplotlib.pyplot as plt

    x = phon.test_signal('sawtooth',dur=1,freq=1000,amp=0.5)
    plt.plot(x[0:50])

    '''

    samples = np.arange(dur * fs) / fs
    if type == "sine":
        outwave = np.sin(2 * np.pi * freq * samples)
    elif type == "sawtooth": 
        outwave = signal.sawtooth(2 * np.pi * freq * samples, width=0.0001)
    elif type == "triangle":
        outwave = signal.sawtooth(2 * np.pi * freq * samples, width=0.5)
    elif type == "square":
        outwave = signal.square(2 * np.pi * freq * samples, duty = 0.5)
    elif type == "impulse":
        temp = np.sin(2 * np.pi * freq * samples)
        outwave = temp==1
    else:
        raise ValueError(f"type '{type}' is not valid")

    return outwave * amp