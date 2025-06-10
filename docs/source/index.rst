Welcome to phonlab!
=====================================

This is a collection of python functions for doing phonetics.

:doc:`acoustphon`: Functions to calculate acoustic measures of vowels, stops and
fricatives, tone and intonation.

:doc:`articphon`: Functions for working with electro-glottography (EGG) data. 

:doc:`audphon`: Functions related to auditory phonetics, including several that modify speech for speech perception
experiments, including speech in noise, noise vocoding, and sinewave synthesis.

:doc:`util`:   Functions for opening and preparing audio files, reading and writing Praat TextGrid files, manipulating the dataframes that are read from TextGrids,  reading
directories into dataframes, converting subtitles files into textgrids, and more.


Installation
------------

>>> pip install phonlab


Examples
--------

The docstring examples assume that `phonlab` has been imported as ``phon``::

  >>> import phonlab as phon


Table of Contents
-----------------
.. toctree::
    :maxdepth: 2

    acoustphon
    articphon
    audphon
    util



Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
