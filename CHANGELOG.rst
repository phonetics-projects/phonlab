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

* **gci_sedreams() is now gci().**  The new name is shorter.  ``gci()`` is based on the ``sedreams``
  function of COVAREP, as before, and the documentation now says so.  The old name still works, with a
  ``DeprecationWarning``.
* **TVLP_formants() makes sure that a frame holds two glottal pulses.**  With ``qcp=True`` (the default) a
  ``length`` shorter than two periods of ``f0median`` (``2/f0median`` seconds) now gives a warning, and
  ``length`` is set to ``2/f0median``.  Nothing changes with the defaults (``length=0.1``, ``f0median=200``).
* **A ``step`` of zero or less is now an error in every function that has one.**  ``step``
  (formerly ``s``) must be a positive number of seconds, and a ``ValueError`` is raised for
  zero, a negative number, NaN or a value that is not a number.  Before, ``sgram()`` quietly
  replaced ``s=0`` by the default, ``compute_mt_sgram()``, ``CPP()`` (with smoothing),
  ``Audspec()``, ``lpcresidual()`` and ``tandem_fric()`` accepted it, and the other functions
  failed with an obscure error from ``librosa`` or ``scipy``.
* **lpcresidual() scales each frame of the residual to the energy of the audio frame.**  It had a line
  for this, but the energy sums were over all the frames at once, which gave one number for the whole
  file that the final peak normalization then removed, so it did nothing, and the residual was the plain
  prediction error.  Now (``gain='speech'``, the default) each frame is scaled as in COVAREP's
  ``lpcresidual.m``, so the level of the residual follows the level of the speech.  The residual
  changes (correlation 0.8 with the old one), and so do the functions that use it:
  ``gci()`` (the times of the glottal closures hardly change, but ``soe`` and ``shimmer`` do),
  ``get_f0_srh()`` (the f0 of most frames is the same, a few frames change) and the ``h2h1()`` of a
  residual made by ``lpcresidual()``.  ``lpcresidual(y, fs, gain=None)`` and
  ``gci(x, fs, gain=None)`` give the old behavior.  ``TVLP_formants()`` uses ``gain=None`` for the glottal
  closure instants of its pitch-synchronous analysis, so its results do not change.
* **cepstral_smooth(): the quefrency axis is in ms and the frequency axis of the smoothed spectrum is
  corrected.**  ``quef`` was in seconds, although the documentation (and ``compute_cepstrogram()``) say
  ms, so it is now 1000 times larger.  ``freqE`` was shifted up by one bin: the smoothed spectrum was
  evaluated at ``k*fs/(2*nbins-1)`` but labeled ``(k+1)*fs/(2*nbins)``, an error of up to 1 Hz.  The axis now starts at 0 Hz.
  The values of the spectra are unchanged.  (``sgram(type='cepstral')`` uses this axis.)
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
  - **The pre-emphasis factor is always called** ``preemph``.  The functions had used ``pre``,
    ``preemph`` and ``preemphasis``.  ``preemph`` (formerly ``pre``): ``prep_audio()``,
    ``DPPT_formants()`` and ``TVLP_formants()``.  ``preemph`` (formerly ``preemphasis``):
    ``IFC_formants()``, ``LPC_formants()``, ``RLPC_formants()``, ``cepstral_smooth()``,
    ``nasality()`` and ``track_formants()``.  ``sgram()`` and ``Audspec.make_zgram()`` already used
    ``preemph``.  The old names still work by keyword with a ``DeprecationWarning``, and positional
    calls are not affected.
  - ``sgram()`` accepts ``step`` and ``length`` for every type of spectrogram.  The old
    names ``s``, ``step_sec``, ``step_size``, ``w`` and ``l`` still work.
  - ``smooth1d()`` and ``smoothn()`` keep their ``s``, which is the amount of smoothing and
    not a step.

Fixed
~~~~~

* ``compute_mel_sgram()`` now has the step and window that it documents at every sampling rate.
  At rates where 64 ms is not a power of two number of samples (22050, 44100 and 48000 Hz, but
  not 16000 or 32000) it used a window of the next power of two and a step that was too long (39 ms
  for a requested 10 ms at 44100 Hz).  Results at 16000 Hz are unchanged, and ``sgram(type='mel')``
  was not affected because it resamples to 16000 Hz.
* **The step between frames is the same number of samples in every function.**  The functions
  truncated ``step*fs`` to a whole number of samples (``int(0.29*100)`` is 28), and ``compute_sgram()``
  truncated a difference of times, which gave a step one sample too long in about a quarter of the
  combinations of sampling rate, ``length`` and ``step`` that I tried (for example ``length=0.03``,
  ``step=0.01`` at 16000 Hz).  All of them now use the nearest whole number of samples.  Results change
  by one sample in the frame spacing only where ``step*fs`` is not already a whole number.  A step
  shorter than half a sample is now an error.  ``nasality()`` failed with "analysis frames do not
  line up" at some steps, such as ``step=0.01`` at 12000 Hz.  ``egg_to_oq()`` now gives frame times
  from the hop in samples.
* **phonlab no longer changes numpy's error handling.**  Importing ``get_f0()`` (which happens when you first use any
  of the ``get_f0`` functions) called ``np.seterr(divide="ignore")``, which silenced divide-by-zero warnings in
  all of your code after that.  The values that phonlab computes do not change.  With speech the functions do not give
  any floating point warnings without it.  The places where silence gave one (``choose_order()``, ``egg_to_oq()``
  and ``rhythmogram()``) now handle it themselves, and ``get_f0_srh()`` of a pure tone, which
  failed with ``cannot convert float NaN to integer``, works.
* Docstrings that had not been updated: ``LPC_formants()`` (``l`` and a ``quiet`` parameter), ``tandem_fric()`` and
  ``compute_mt_sgram()`` (``w``), ``TVLP_formants()`` (the defaults of ``nformants`` and ``use_parallel``), ``CPP()`` (a
  ``return_Sxx`` parameter that does not exist, the default ``f0_range``, and the step of 0.002 s that is
  used when ``smooth`` is not 0), ``egg_to_oq()`` (``f0_range``), ``get_f0_srh()`` (the band of ``rms``),
  ``DPPT_formants()`` and ``cepstral_smooth()``.
* ``h2h1()`` gave wrong values unless the residual was sampled at 16000 Hz.  It resampled the
  residual to 16000 Hz but then filtered the original array, with a filter designed for 16000 Hz, so
  the narrow-band filter was centered on the wrong frequency (the h2h1 values were off by about 17 dB on
  average at 22050 and 44100 Hz).  ``lpcresidual()`` returns 16000 Hz audio by default, so
  results of the usual ``lpcresidual()`` then ``h2h1()`` sequence are unchanged.
* ``h2h1()`` no longer assumes that ``f0df`` is a complete, evenly spaced f0 analysis with the default index.
  It took the window length from ``f0df['sec'][0]`` and the step from ``f0df['sec'][1] - f0df['sec'][0]``
  rounded to the ms, so a dataframe that had been filtered (for example to the voiced frames), had another
  index, or had a step such as 7.5 ms gave a ``KeyError`` or measurements in the wrong windows.  It now makes
  one measurement in a window centered on the ``sec`` of each row.  It has a new parameter ``length``
  (the window length, by default twice the first ``sec`` as before) for dataframes that do not start at
  the first frame.  A row with no spectral peak is NaN.  The window now has exactly the length of the f0
  analysis window and is centered on its time.  Before, the length was rounded to the ms (47 ms for a
  window of 47.4 ms) and the window was centered a few samples early, so values for the ordinary
  output of an f0 function change slightly (typically by about 0.25 dB, but by more in the few frames
  where the choice of spectral peaks changes).
* ``burst()`` found no burst (it returned ``(-1, 0)``) unless the audio had a sampling rate of 16000 Hz,
  because it searched the original waveform with indices for the resampled one.
* ``sgram()`` with both ``ax`` and ``slice_time`` raised ``UnboundLocalError``.  It now warns that the
  slice is not drawn when an axes is passed in.
* ``deltaF_norm()`` with both ``column`` and ``deltaF`` raised ``NameError``.  The ``deltaF`` is now used for
  every talker.  The docstring called ``column`` ``groupby`` and said that the function changes the
  dataframe in place and returns nothing, but it returns the dataframe (in place only without ``column``).
* ``get_f0_acd(test_time=...)`` raised ``NameError`` (``plt`` and ``best_c`` were not defined).
* ``split_speaker_df()`` raised ``NameError`` instead of warning when it found overlapping times.
* ``get_f0_srh()`` no longer changes its ``f0_range`` argument.  It lowered the top of
  the list that was passed in, and so of its default, which made later calls in the same
  session use a narrower range than the first call.

Added
~~~~~

* ``get_rms()`` documents its ``step`` and ``length`` parameters.
* ``lpcresidual()`` has a ``gain`` parameter (``'speech'``, the default, or ``None`` for the plain
  prediction error, which the LPC filters turn back into the audio, for use as the source in a
  source-filter resynthesis), and a ``normalize`` parameter (``False`` keeps the level of the residual
  instead of scaling its peak to 1).  ``gci()`` has the same ``gain`` parameter.



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
