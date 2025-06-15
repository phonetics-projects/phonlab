__name__="phonlab"
__version__="0.0.20"

# -------- functions in the "acoustic" portion of the package -------
from .acoustic.sgram_ import *
from .acoustic.burst_detect import *
from .acoustic.fric_meas import *
from .acoustic.track_formants_ import *
from .acoustic.get_f0_ import *
from .acoustic.tidypraat import *
from .acoustic.amp_env import *
from .acoustic.rhythm import *
from .acoustic.vowel_norm import *

__all__ = acoustic.sgram_.__all__.copy()
__all__ += acoustic.burst_detect.__all__
__all__ += acoustic.fric_meas.__all__
__all__ += acoustic.track_formants_.__all__
__all__ += acoustic.get_f0_.__all__
__all__ += acoustic.tidypraat.__all__
__all__ += acoustic.amp_env.__all__
__all__ += acoustic.rhythm.__all__
__all__ += acoustic.vowel_norm.__all__

# -------- functions in the "auditory" portion of the package ---------
from .auditory.sigcor import *
from .auditory.noise_vocoder import *
from .auditory.add_noise_ import *
from .auditory.sinewave_synth import *
from .auditory.audspec import *

__all__ += auditory.sigcor.__all__
__all__ += auditory.noise_vocoder.__all__
__all__ += auditory.add_noise_.__all__
__all__ += auditory.sinewave_synth.__all__
__all__ += auditory.audspec.__all__

# -------- functions in the 'articulatory' portion of the package -------
from .artic.egg2oq_ import *

__all__ += artic.egg2oq_.__all__


#--------- functions in the 'load Audio' portion of the package ---------
from .utils.signal import *
from .utils.prep_audio_ import *

__all__ += utils.signal.__all__
__all__ += utils.prep_audio_.__all__

#---------- functions in the 'working with textgrids and working with corpora' portions ------
from .utils.tidy import *

__all__ += utils.tidy.__all__


#--------  functions in the 'Utility' portion of the package
from .third_party.robustsmoothing import *
__all__ += third_party.robustsmoothing.__all__
