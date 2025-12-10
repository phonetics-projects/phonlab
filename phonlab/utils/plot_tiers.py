import matplotlib.pyplot as plt
import pandas as pd

def plot_tier(df, start=0.0, end=-1, ax=None, mark_in_plot = None, 
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

    mark_in_plot : axes (default None)
        a matplotlib axes where vertical black lines at t1 and t2 will be marked.  

    span_time: float (default None)
        a time value (in seconds) used to choose a label interval that will be highlighted by color shading overlaid on the spectrogram.  By default the color of the shading is blue, and the alpha of the shading is 0.2.  These defaults can be changed by keyword arguments that will be passed to the pyplot functin `axvspan()`.

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

    example_file = importlib.resources.files('phonlab') / 'data/example_audio/im_twelve.wav'
    x,fs = phon.loadsig(example_file,chansel=[0])
    y,fs = phon.prep_audio(x, fs, target_fs=16000)

    example_tg = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'im_twelve.TextGrid'
    phdf, wddf = phon.tg_to_df(example_tg, tiersel=['phone', 'word'])
      
    start = 0.9
    end = 2

    fig, ax = phon.make_figure(n_plots=1, n_tiers=2)  # prepare figure for content
    ret = phon.sgram(y, fs, start=start, end=end, ax=ax[-1]) # spectrogram at bottom
    phon.plot_tier(phdf,start=start, end=end, ax=ax[1],mark_in_plot=ax[-1],span_time=1.5)
    phon.plot_tier(wddf,start=start, end=end, ax=ax[0]) # word tier at top

    
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
        if isinstance(mark_in_plot,plt.Axes):
            mark_in_plot.axvline(row.t1,color='k')
            mark_in_plot.axvline(row.t2,color='k')
        if not span_time is None:
            if row.t1<span_time and row.t2>span_time:
                ax.axvspan(row.t1, row.t2, color='g', alpha=0.2) 
                if isinstance(mark_in_plot,plt.Axes):
                    mark_in_plot.axvspan(row.t1,row.t2,**kwargs)

def make_figure(n_plots=1, n_tiers=1):
    """
Set up a matplotlib figure to be filled with calls to `phon.sgram()` and `phon.plot_tier()`.  This
function was written by Martin Oberg at UBC.  The function determines height ratios for multiple
plotting axes in the figure based on the number of data plots and text label plots that are anticipated.

Parameters
==========
    n_plots : integer (default 1)
        The number of axes to be added to the figure for plotting phonetic data (e.g. axes for adding spectrograms, waveform plots, articulatory data, etc.)
        
    n_tiers : integer (default 1)
        The number of axes to be added to the figure for plotting text labels.

Returns
=======

    Figure
        a Matplotlib Figure object.

    list
        a list of Matplotlib Axes objects. The number of axes will be equal to n_plots + n_tiers.  The figure will have n_tiers axes at the top of the figure (ax[0]..ax[n_tiers-1]) and n_plots axes at the bottom of the figure (ax[n_tiers]...ax[-1].  The textgrid axes will be 1/10th as tall as the data axes.
    """
    fig = plt.figure(figsize=(5, 2), dpi=72)
    height_ratios = [1] * n_tiers + [10] * n_plots
    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=1, height_ratios=height_ratios
    )
    ax = [fig.add_subplot(x) for x in gs]
    for i in range(n_tiers):
        ax[i].set_axis_off()  # hide axes for tiers
    for i in range(1, len(height_ratios)):
        ax[i].sharex(ax[0])   # share x axis across all plots

    return fig, ax
