from pathlib import Path
import numpy as np
import phonlab as phon

AUDIO = Path(phon.__file__).parent / "data" / "example_audio"


def test_rlpc_gain_ignores_filter_startup_samples():
    # In "the_soviet_union.wav" the analysis frames centered at 0.675 and 0.690 s (s=0.005, l=0.03)
    # begin at a large sample value.  The zero-initial-condition lfilter() residual of the first
    # `order` samples of such a frame is large, and used to inflate the gain by 8 dB for that frame.
    x, fs = phon.loadsig(str(AUDIO / "the_soviet_union.wav"), chansel=[0])
    df, A = phon.RLPC_formants(x, fs, tf=8000, step=0.005, length=0.03, order=14)
    g = df.set_index("sec").gain
    window = g[(g.index > 0.6) & (g.index < 0.8)]
    local_median = window.rolling(7, center=True, min_periods=3).median()
    assert (window / local_median).max() < 2.5
