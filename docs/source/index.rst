Phonlab 
=======
A collection of python functions for doing Phonetics.

The package is divided into functions for :doc:`acoustphon`, to calculate acoustic measures of vowels, stops and
fricatives, tone and intonation.  In addition there are functions relating to :doc:`articphon` and :doc:`audphon`, including several that modify speech for speech perception experiments, including speech in noise, noise vocoding, and sinewave synthesis.

The package also includes functions to :doc:`prep_audio` (read sound files, extract channels, extract clips, scale amplitude, change sampling rate, and apply preemphasis), and functions for :doc:`textgrids` (reading textgrids into Pandas dataframes, writing dataframes into textgrids, merging hierarchical dataframes, and adding context columns), functions for :doc:`corpora` (including reading directories into dataframes, and working with subtitle files), and some miscellaneous :doc:`util` functions.

Keith Johnson and Ronald Sprouse


Installation
------------

>>> pip install phonlab


How to Cite
-----------

If you use this software in published research please cite it in a way that directs readers to this documentation page. 


    Phonlab Development Team (2025). `Phonlab: A collection of python functions for doing Phonetics`.  https://phonlab.readthedocs.io/ 


Examples
--------

The docstring examples assume that `phonlab` has been imported as ``phon``::

  >>> import phonlab as phon

See the examples folder in the project's `github repository <https://github.com/phonetics-projects/phonlab.git>`_.

Feedback
--------

The developers welcome your feedback and suggestions.  Please visit the project's `github repository <https://github.com/phonetics-projects/phonlab.git>`_ and use the `Issues` tab to report an issue.  See also the project readme file for some instructions on how to contribute a change or addition to the package.


Table of Contents
-----------------
.. toctree::
    :maxdepth: 2

    prep_audio
    acoustphon
    articphon
    audphon
    textgrids
    corpora
    util



Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
