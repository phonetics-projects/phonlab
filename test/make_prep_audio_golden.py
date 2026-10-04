"""Regenerate test/golden/prep_audio_golden.npz, the golden output used by test_prep_audio.py.

Run this only when prep_audio()'s output is meant to change, and review the diff of
the resulting npz before committing it.
"""

import sys
from pathlib import Path

import numpy as np

# make sure this checkout's phonlab is used, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonlab.utils.prep_audio_ import prep_audio

FS = 22050

# name -> keyword arguments for prep_audio(). add_tiny_noise is off so the output is deterministic.
CASES = {
    "default": dict(),
    "scale_0dB": dict(scale=0),
    "scale_-6dB": dict(scale=-6),
    "scale_False": dict(scale=False),
    "native_fs": dict(target_fs=None),
    "pre_0.97": dict(target_fs=16000, pre=0.97, scale=-6),
    "padded": dict(target_fs=16000, pad_to=0.1),
    "int": dict(target_fs=16000, scale=-3, outtype="int"),
}


def make_input():
    """A deterministic test signal whose negative peak is larger than its positive peak."""
    rng = np.random.default_rng(12345)
    t = np.arange(int(0.5 * FS)) / FS
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1800 * t)
    x += 0.01 * rng.standard_normal(len(t))
    x[1000] = -0.9   # an asymmetric negative peak
    return x


def main():
    x = make_input()
    out = {"input": x}
    for name, kwargs in CASES.items():
        y, fs = prep_audio(x, FS, add_tiny_noise=False, **kwargs)
        out[name] = y
        out[name + "_fs"] = np.array(fs)
    path = Path(__file__).parent / "golden" / "prep_audio_golden.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
