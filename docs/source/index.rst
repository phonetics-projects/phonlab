Welcome to phonlab!
=====================================

This is a collection of python functions for doing phonetics.

The package is divided into functions for :doc:`acoustphon`, to calculate acoustic measures of vowels, stops and
fricatives, tone and intonation.  In addition there are functions relating to :doc:`articphon` and :doc:`audphon`, including several that modify speech for speech perception experiments, including speech in noise, noise vocoding, and sinewave synthesis.

The package also includes functions to :doc:`prep_audio` (read sound files, extract channels, extract clips, scale amplitude, change sampling rate, and apply preemphasis), and functions for :doc:`textgrids` (reading textgrids into Pandas dataframes, writing dataframes into textgrids, merging hierarchical dataframes, and adding context columns), functions for :doc:`corpora` (including reading directories into dataframes, and working with subtitle files), and some miscellaneous :doc:`util` functions.

Keith Johnson and Ronald Sprouse


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
    prep_audio
    textgrids
    corpora
    util



Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
