Changelog
=========

All notable changes to phonlab are listed here, newest first.  Each release has a
**Breaking changes** section first, which lists changes that can alter your results or
make existing code fail.  Read it before upgrading.

Version numbers
---------------

phonlab is still in beta (version 0.x), and versions are numbered ``0.MINOR.PATCH``.

* A **MINOR** release (0.1.0 to 0.2.0) may contain breaking changes: changes to function
  parameters or defaults, to the columns or units of what a function returns, or to
  the numerical values it computes.
* A **PATCH** release (0.1.0 to 0.1.1) contains bug fixes and backward-compatible additions
  only.  Fixes that change numerical results are listed in the notes.

When practical, a change that would break existing code is announced with a warning for
at least one MINOR release before it takes effect.

To avoid surprises, pin the version you have tested your analysis with, for example in a
``requirements.txt`` file::

    phonlab>=0.1,<0.2

Older releases stay available on PyPI (``pip install phonlab==0.0.70``) and their
documentation stays on Read the Docs, e.g. https://phonlab.readthedocs.io/en/v0.0.70/ .
The docs for the newest release are at https://phonlab.readthedocs.io/en/stable/ .


Unreleased
----------

Breaking changes
~~~~~~~~~~~~~~~~

* **sgram(): a step of zero or less is now an error.**  ``step`` (formerly ``s``) must be
  a positive number of seconds.  Before, ``s=0`` was quietly replaced by the default.
* **sgram(type='mel') and sgram(type='bark') use ``step``.**  Before, ``s`` was ignored
  for ``type='mel'`` (it needed ``step_sec``), and ``s`` was raised to at least 5 ms for
  ``type='bark'``.  Now ``step`` sets the step for every type, and the default step of
  each type is unchanged.

Deprecated
~~~~~~~~~~

* **The step between analysis frames is always called** ``step`` **and the length of an
  analysis frame is always called** ``length``, **both in seconds.**  The functions had
  used ``s``, ``step_sec``, ``step_size`` and ``hop_dur`` for the step, and ``l``, ``w``
  and ``frame_duration_sec`` for the length.  The old names still work by keyword, but
  give a ``DeprecationWarning``, and will be removed in a later release.  Positional
  calls are not affected, because the order of the parameters did not change.  The
  results are the same.

  - ``step`` (formerly ``s``): ``CPP()``, ``DPPT_formants()``, ``HNR()``,
    ``LPC_formants()``, ``RLPC_formants()``, ``VAD()``, ``cepstral_smooth()``,
    ``compute_cepstrogram()``, ``compute_mt_sgram()``, ``compute_sgram()``, ``get_f0()``,
    ``get_f0_acd()``, ``get_f0_shs()``, ``get_f0_srh()``, ``get_rms()`` and
    ``lpcresidual()``.
  - ``step`` (formerly ``step_sec``): ``compute_mel_sgram()``.  ``step`` (formerly
    ``step_size``): ``Audspec()``, whose attribute is now ``Audspec.step``.  ``step``
    (formerly ``hop_dur``): ``egg_to_oq()``.
  - ``length`` (formerly ``l``): ``CPP()``, ``DPPT_formants()``, ``HNR()``,
    ``LPC_formants()``, ``RLPC_formants()``, ``cepstral_smooth()``, ``choose_order()``,
    ``compute_cepstrogram()``, ``get_f0_acd()``, ``get_f0_shs()``, ``get_f0_srh()``,
    ``get_rms()`` and ``lpcresidual()``.
  - ``length`` (formerly ``w``): ``compute_mt_sgram()``, ``compute_sgram()``,
    ``fricative()`` and ``tandem_fric()``.  ``length`` (formerly ``frame_duration_sec``):
    ``TVLP_formants()``.
  - ``sgram()`` accepts ``step`` and ``length`` for every type of spectrogram.  The old
    names ``s``, ``step_sec``, ``step_size``, ``w`` and ``l`` still work.
  - ``smooth1d()`` and ``smoothn()`` keep their ``s``, which is the amount of smoothing and
    not a step.

Fixed
~~~~~

* ``get_f0_srh()`` no longer changes its ``f0_range`` argument.  It lowered the top of
  the list that was passed in, and so of its default, which made later calls in the same
  session use a narrower range than the first call.

Added
~~~~~

* ``get_rms()`` documents its ``step`` and ``length`` parameters.


0.1.0 (2026-11-09)
------------------

Version 0.0.70 was the last release in the 0.0.x series.

Breaking changes
~~~~~~~~~~~~~~~~

* **Spectra are now in true decibels.**  ``compute_sgram()`` and the other spectral
  functions computed ``10*log10(|X|)``, which is half the decibel value of an amplitude
  spectrum.  They now compute ``20*log10(|X|)``, so every dB value is **twice** as large as
  before.  This affects ``compute_sgram()`` and ``sgram()``, ``compute_cepstrogram()``,
  ``cepstral_smooth()``, ``CPP()`` and ``get_HNR()``.  Any
  threshold, floor, or reference value that you set in dB for these outputs must be
  doubled.
* **nasality() was rewritten.**

  - The parameters ``acoustic_spectrum``, ``smoothing_factor`` and ``target_fs`` were
    removed (analysis is always at 12000 Hz with a cepstrally smoothed spectrum).
  - Defaults changed: ``order`` 18 to -1 (chosen by ``choose_order()``), ``analysis_band``
    (80, 2500) to (100, 1200), ``length`` 0.04 to 0.03, and ``lift`` 0.0025 to 0.002.
  - The ``SCD`` (spectral cosine distance) column was replaced by ``SD`` (a spectral
    distance in dB).  Because of the new definition, **SD is lower for more nasal frames**.
  - There are new parameters ``index_smoothing``, ``calibration``, ``voicing_threshold``
    and ``reference``, and a new ``nas_index`` column that combines SD and a1h1 into a
    single index (larger means more nasal).
  - With no peak above the height limit for h1, ``a1h1`` is now NaN rather than an error.

* **RLPC_formants() gain changed.**  The first ``order`` residual samples of each frame
  are produced by the filter start-up and are not prediction errors.  They are now
  left out of the gain and of the residual scale used for the robust reweighting.  The
  ``gain`` column, and in some frames the formants, differ from 0.0.70.  A frame that
  started at a large sample value could have its gain inflated by about 8 dB,
  which showed as vertical lines in LPC spectrograms.  ``nasality()`` and
  ``sgram(type='lpc')`` use this function.
* ``TVLP_formants()`` has a new ``pre`` parameter placed after ``f0median``.  Code that
  passed ``max_bandwidth``, ``use_parallel`` or ``verbose`` by position must use
  keywords instead.
* ``cepstral_smooth()`` now raises ``ValueError`` if ``lift`` is shorter than one sample
  or longer than the cepstrum.

Added
~~~~~

* ``sgram()`` has a ``type`` parameter: ``'fft'`` (default, as before), ``'multitaper'``,
  ``'cepstral'``, ``'lpc'``, ``'mel'`` or ``'bark'``.  The grey-scale floor of the
  display is now taken from the 20th to 80th percentile of the values, and the default
  ``min_prop`` depends on the type.
* ``nasalance()``, which computes nasalance from a two-channel nasometer recording.
* ``TVLP_formants()`` has ``pre`` (pre-emphasis, default 0.95 as before) and ``min_freq``
  (lowest reported formant frequency, default ``None`` which keeps the old behavior of
  using ``f0median``).
* ``sine_synth()`` has a ``voicing_threshold`` parameter (default 0.8) that is passed to
  ``VAD()``.

Fixed
~~~~~

* ``gci_sedreams()`` and ``TVLP_formants()`` now find glottal closures when the F0 is very
  low (below about 30 Hz).  Results are unchanged when ``f0median`` is 100 Hz or more.
* ``lpcresidual()`` no longer produces an all-NaN residual, or a residual crushed to zero,
  when ``librosa.lpc`` returns unusable coefficients for a frame.  Valid frames are
  unchanged.
* ``choose_order()`` no longer fails when the sampling rate is 6000 Hz or less, and no
  longer analyzes a short or empty frame when the loudest point is near the start or end
  of the signal.
* ``TVLP_formants()`` no longer drops F1 when ``f0median`` is above F1 (use ``min_freq``).
* ``nasality()``: the ``nas_index`` sign is now as documented, and ``nasalance()`` no
  longer raises ``NameError``.

Changed
~~~~~~~

* ``cepstral_smooth()`` is 4-6 times faster, and ``nasality()`` is 3-4 times faster and
  uses about a quarter of the memory.  The numerical results of ``cepstral_smooth()`` are
  unchanged apart from float32 rounding.


0.0.70 and earlier
------------------

Releases before 0.1.0 are not described here.  See the commit history at
https://github.com/phonetics-projects/phonlab/commits/v0.0.70 .
