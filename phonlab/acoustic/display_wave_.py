import numpy as np
import matplotlib.pyplot as plt

def display_wave(x,fs,start=0, end=-1, ax = None, 
                 ylabel = "Amplitude", font_size=14, **kwargs):
    '''Use the matplotlib.pyplot.plot() function to plot an acoustic 
waveform or other time series.

Parameters
==========
    x : ndarray
        a one-dimensional array of audio samples, or other time series data.

    fs : numeric
        Sampling rate of the data in `x`
    
    start : float (default 0.0)
        start time of the plot's x axis (in seconds)

    end : float (default -1)
        end time of the plot's x axis (in seconds), default value of -1 means plot to the end of the data.

    ax : Matplotlib axes object (default None)
        by default (ax=None) a new matplotlib figure will be created.

    ylabel : string (default "Amplitude")
        The function is used to plot an audio waveform, so the default y-axis label is "Amplitude".  However, this function can be use to plot any time series.

    font_size : integer (default 14)
        Size of the text in the figure's labels and tick labels.
        
    **kwargs : keyword arguments
        key word arguments will be passed to pyplot.plot()

Returns
=======
    ax : Axes 
        The matplotlib axes 


Examples
========

Plot a portion of the sound file from 0.2 to 0.9 seconds.  

.. code-block:: Python

    audio_dir = importlib.resources.files('phonlab') / 'data' / 'example_audio'
    example_file = audio_dir / 'im_twelve.wav'

    x,fs = phon.loadsig(example_file, chansel=[0])

    ax = phon.display_wave(x,fs,0.2,0.9,color='teal')

.. figure:: images/display_wave.png
       :scale: 50 %
       :alt: a waveform produced by display_wave()
       :align: center

    '''

    if "color" not in kwargs.keys():
        kwargs["color"] = "k"

    s = int(start * fs)
    if end==-1:
        end = len(x)
    e = int(end * fs)

    sec = (np.array(range(len(x[s:e]))).astype(int)/fs) + start

    if ax==None:
        fig = plt.figure(figsize=(12, 2.5),dpi=72)
        ax = fig.add_subplot(111)

    ax.plot(sec,x[s:e], **kwargs)
    ax.set_ylabel(ylabel, size=font_size)
    ax.set_xlabel("Time (sec)", size=font_size)
    ax.tick_params(labelsize=font_size)
 
    return ax
