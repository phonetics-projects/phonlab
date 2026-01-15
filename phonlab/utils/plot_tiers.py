import matplotlib.pyplot as plt
import pandas as pd

def plot_tier(df, start=0.0, end=-1, ax=None, mark_in_plot = [], 
              span_time=None, vertical_place = 0.5, **kwargs):
    """
    Plot the labels from a textgrid tier in a matplotlib plot axes.  See the example for an illustration of how this is used to add textgrid labels to a spectrogram.  This function is a combination of functions plot_tier_times() and plot_tier_spans() that were written by Martin Oberg at UBC.


Parameters
==========

    df : a Pandas dataframe
        Textgrid data as produced by phon.tg_to_df().  There must be three columns - 't1', 't2' and a column of labels.

    start : float (default 0.0)
        start time of the plot's x axis (in seconds)

    end : float (default -1)
        end time of the plot's x axis (in seconds), default value of -1 means plot to the end of the dataframe.

    ax : axes (default None)
        a matplotlib axes in which to plot the tier.  If none is given the function uses the matplotlib function `gca()`  to find the current axes.  

    mark_in_plot : list of axes (default [])
        a list of matplotlib axes where vertical black lines at label boundaries (t1 and t2) will be marked.  

    span_time: float (default None)
        a time value (in seconds) used to choose a label interval that will be highlighted by color shading overlaid on the axes in `mark_in_plot`.  By default the color of the shading is blue, and the alpha of the shading is 0.2.  These defaults can be changed by keyword arguments that will be passed to the pyplot functin `axvspan()`.

    vertical_place: float (default 0.5)
        relative vertical location of the label in the axes.  0 = centered at the bottom of the axes, and 1 = centered at the top.

Returns
=======

    there is no return value.

Raises
======

    TypeError 
        if the first argument is not a Pandas DataFrame

    ValueError 
        if the dataframe does not have at least three columns.

Example
=======
The first example illustrates the use of `phon.make_figure()`, `phon.sgram()` and `phon.plot_tier()` to produce a figure.

The same start and end times are passed to sgram() and plot_tier(), and the phone tier is plotted with segment boundaries 
in the spectrogram, and with one particular phone highlighed (the one that includes time 1.5 seconds)


.. code-block:: Python

    audio_dir = importlib.resources.files('phonlab') / 'data' / 'example_audio'
    example_tg = audio_dir / 'im_twelve.TextGrid'
    example_file = audio_dir / 'im_twelve.wav'

    # read the audio file
    x,fs = phon.loadsig(example_file,chansel=[0])

    # read the text grid file
    tiers = ['phone', 'word']
    phdf, wddf = phon.tg_to_df(example_tg, tiersel=tiers)

    start = 0.175
    end = 0.85

    height_ratios = [1, 1, 1.5, 10]

    # create a figure that has four plot axes in it
    fig,[wrd,phn,wav,specgrm] = make_figure(height_ratios)

    # fill the figure with textgrid and acoustic data
    phon.plot_tier(phdf, start, end, ax=phn, mark_in_plot=[wav,specgrm],span_time=0.5)
    phon.plot_tier(wddf, start, end, ax=wrd) # word tier at top
    wav = phon.display_wave(x,fs,start, end, ylabel="", font_size=8, ax=wav)
    ret = phon.sgram(x,fs,start, end, ax=specgrm)
    
.. figure:: images/plot_tier.png
    :scale: 40 %
    :alt: Plotting textgrid information with a spectrogram
    :align: center

    Plotting textgrid information with a spectrogram.


The second example shows that plot_tier() can be used to add lables directly to a spectrogram (or any other matplotlib axes).

.. code-block:: Python

    phon.sgram(y, fs, cmap='Purples')
    phon.plot_tier(wddf, vertical_place = 0.75)

.. figure:: images/plot_tier2.png
    :scale: 40 %
    :alt: Adding textgrid labels to a spectrogram
    :align: center

    Adding textgrid labels to a spectrogram.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a dataframe")
    if len(df.columns)<3:
        raise ValueError("Dataframe must have at least three columns")
    if "alpha" not in kwargs.keys():
        kwargs["alpha"] = 0.2
    if "color" not in kwargs.keys():
        kwargs["color"] = "b"

    if ax is None:
        ax = plt.gca()
    ax.axhline(0,color="k")  # horizontal line under the tier
    yrange = ax.get_ylim()
    y_loc = vertical_place * (yrange[0] + yrange[1])

    if end == -1:
        end = float("inf")
    
    for row in df.itertuples():   # look at each row in the file
        if row.t1 < start or row.t2 > end:
            continue
        ax.axvline(row.t1, color="k")
        ax.axvline(row.t2, color="k")        
        x_loc = 0.5 * (row.t1 + row.t2)
        ax.text(x_loc, y_loc, row[3], size=16,
                verticalalignment='center',
                horizontalalignment='center')
        for other_axes in mark_in_plot:
            if isinstance(other_axes,plt.Axes):
                other_axes.axvline(row.t1,color='k')
                other_axes.axvline(row.t2,color='k')
        if not span_time is None:
            if row.t1<span_time and row.t2>span_time:
                ax.axvspan(row.t1, row.t2, color='g', alpha=0.2) 
                for other_axes in mark_in_plot:
                    if isinstance(other_axes,plt.Axes):
                        other_axes.axvspan(row.t1,row.t2,**kwargs)

def make_figure(height_ratios = [1,1,5]):
    """
Create a matplotlib figure with axes to be filled with calls to plotting functions like `phon.sgram()` `phon.displacy_wave()`, and `phon.plot_tier()`.  The first version of this function was written by Martin Oberg at UBC.  

Parameters
==========
    height_ratios: array of numbers, default = [1,1,5]
        This list determines the number of axes that will be included in the figure, and determines their relative heights.  The default list prepares a figure that will have two narrow axes at the top (like textgrid tiers) and one five-time taller axes for plotting a spectrogram.  Note that when height_ratios is "1" the axis labeling is turned off for the axes.
    
Returns
=======

    Figure
        a Matplotlib Figure object.

    list
        a list of Matplotlib Axes objects, as defined by the height_ratios.
        
    """
    fig = plt.figure(figsize=(5, 2), dpi=72)
    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=1, height_ratios=height_ratios
    )
    ax = [fig.add_subplot(x) for x in gs]
    
    for i in range(len(height_ratios)): 
        if height_ratios[i] == 1: # hide axes for tiers
            ax[i].set_axis_off()  
        if i != (len(height_ratios)-1):  # hide x axis for all but the bottom
            ax[i].get_xaxis().set_visible(False)
    for i in range(1, len(height_ratios)): # share x axis across all plots
        ax[i].sharex(ax[0])   

    return fig, ax

