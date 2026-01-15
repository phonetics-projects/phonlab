''' In this __init__ stub we are incrementing the definition of __all__ with each import statement.  This is to 
make it easier to ensure that each imported function is added to __all__.

To add a new function to the package it should be listed here, and should be added to the documentation
by having a complete doc string and then added to one of the .rst files in the phonlab/docs/source directory.
'''

__all__ = []

from .acoustic.display_wave_ import display_wave
__all__ += ["display_wave"]

from .acoustic.DPPT import track_formants_DPPT
__all__ += ["track_formants_DPPT"]

from .acoustic.amp_env import amplitude_envelope
__all__ += ["amplitude_envelope"]

from .acoustic.burst_detect import burst
__all__ += ["burst"]

from .acoustic.cepstral import compute_cepstrogram, CPP
__all__ += ["compute_cepstrogram", 
            "CPP"]

from .acoustic.fric_meas import hz2bark, bark2hz, fricative
__all__ += ["hz2bark", 
            "bark2hz", 
            "fricative"]

from .acoustic.gci import gci_sedreams
__all__ += ["gci_sedreams"]

from .acoustic.get_HNR import HNR
__all__ += ["HNR"]

from .acoustic.get_f0_ import get_rms, get_f0, get_f0_srh, get_f0_acd
__all__ += ["get_rms", 
            "get_f0", 
            "get_f0_srh", 
            "get_f0_acd"]

from .acoustic.shs import get_f0_shs
__all__ += ["get_f0_shs"]

from .acoustic.h2h1_ import h2h1
__all__ += ["h2h1"]

from .acoustic.lpc_residual import lpcresidual, overlap_add
__all__ += ["lpcresidual", 
            "overlap_add"]

from .acoustic.rhythm import get_rhythm_spectrum,rhythmogram
__all__ += ["get_rhythm_spectrum", 
            "rhythmogram"]

from .acoustic.sgram_ import sgram, compute_sgram
__all__ += ["sgram", 
            "compute_sgram"]

from .acoustic.tidypraat import formant_to_df, pitch_to_df, intensity_to_df, mfcc_to_df
__all__ += ["formant_to_df", 
            "pitch_to_df", 
            "intensity_to_df", 
            "mfcc_to_df"]

from .acoustic.track_formants_ import track_formants
__all__ += ["track_formants"]

from .acoustic.vowel_norm import get_deltaF, deltaF_norm, resize_vt
__all__ += ["get_deltaF", 
            "deltaF_norm", 
            "resize_vt"]

from .artic.egg2oq_ import egg_to_oq
__all__ += ["egg_to_oq"]

from .auditory.add_noise_ import peak_rms, add_noise
__all__ += ["peak_rms", 
            "add_noise"]

from .auditory.audspec import Audspec
__all__ += ["Audspec"]

from .auditory.mel_sgram import compute_mel_sgram, linear_to_mel_weight_matrix, mel_to_Hz, Hz_to_mel
__all__ += ["compute_mel_sgram", 
            "mel_to_Hz", 
            "Hz_to_mel", 
            "linear_to_mel_weight_matrix"]

from .auditory.noise_vocoder import shannon_bands, third_octave_bands, vocode, apply_filterbank
__all__ += ["shannon_bands", 
            "third_octave_bands", 
            "vocode", 
            "apply_filterbank"]

from .auditory.sigcor import sigcor_noise
__all__ += ["sigcor_noise"]

from .auditory.sinewave_synth import sine_synth
__all__ += ["sine_synth"]

from .utils.prep_audio_ import prep_audio
__all__ += ["prep_audio"]

from .utils.tidy import df_to_tg, tg_to_df, add_context, merge_tiers, adjust_boundaries, explode_intervals, interpolate_measures, srt_to_df, split_speaker_df
__all__ += ["df_to_tg", 
            "tg_to_df", 
            "add_context", 
            "merge_tiers", 
            "adjust_boundaries", 
            "explode_intervals", 
            "interpolate_measures", 
            "srt_to_df", 
            "split_speaker_df"]

from .utils.signal import loadsig
__all__ += ["loadsig"]

from .utils.plot_tiers import plot_tier, make_figure
__all__ += ["plot_tier", 
            "make_figure"]

from .third_party.robustsmoothing import smoothn
__all__ += ["smoothn"]


