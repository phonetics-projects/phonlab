Auditory Phonetics
==================

Signal Manipulation
-------------------
.. autofunction:: phonlab.add_noise
.. autofunction:: phonlab.sigcor_noise
.. autofunction:: phonlab.vocode
.. autofunction:: phonlab.sine_synth
.. autofunction:: phonlab.shannon_bands
.. autofunction:: phonlab.third_octave_bands
.. autofunction:: phonlab.apply_filterbank		  

Auditory Representations
------------------------
.. autofunction:: phonlab.compute_mel_sgram
.. autofunction:: phonlab.mfcc_to_df
.. autoclass:: phonlab.Audspec
    :members:
    :member-order: bysource

Helper Functions
----------------
.. autofunction:: phonlab.peak_rms
.. autofunction:: phonlab.hz2bark
.. autofunction:: phonlab.bark2hz
.. autofunction:: phonlab.Hz_to_mel
.. autofunction:: phonlab.mel_to_Hz
.. autofunction:: phonlab.linear_to_mel_weight_matrix