"""Regenerate test/data/smooth1d_golden.npz, the golden output used by test_smooth1d.py.

The golden output comes from phonlab.smoothn (Garcia's original N-D implementation), so
test_smooth1d.py checks that smooth1d gives the same smooths. Only regenerate this if the test
signals change, and check that smoothn is unchanged first.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

# make sure this checkout's phonlab is used, not an installed copy
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonlab.third_party.robustsmoothing import smoothn


def make_inputs():
    """name -> (y, extra keyword arguments that depend on the data, like weights)."""
    rng = np.random.default_rng(20240601)

    # the example curve in the smoothn docstring
    x = np.linspace(0, 100, 2 ** 5)
    curve = np.cos(x / 10) + (x / 50) ** 2 + rng.random(len(x)) / 2
    curve[[14, 17, 20]] = [2, 2.5, 3]

    # a longer curve with outliers
    n = 500
    long = np.cos(np.linspace(0, 10, n)) + rng.normal(0, .3, n)
    long[::37] += 3

    # the same, with missing data: scattered NaNs and a gap
    gappy = long.copy()
    gappy[rng.random(n) < .15] = np.nan
    gappy[150:175] = np.nan

    weights = rng.random(n) + .1
    sd = rng.random(n) + .5
    return {
        "curve": (curve, {}),
        "long": (long, {}),
        "gappy": (gappy, {}),
        "long_W": (long, {"W": weights}),
        "long_sd": (long, {"sd": sd}),
    }


# name -> (input name, keyword arguments for smoothn)
CASES = {
    "curve_auto": ("curve", {}),
    "curve_robust": ("curve", {"isrobust": True}),
    "curve_s5": ("curve", {"s": 5}),
    "curve_robust_s5": ("curve", {"isrobust": True, "s": 5}),
    "long_auto": ("long", {}),
    "long_robust": ("long", {"isrobust": True}),
    "long_cauchy": ("long", {"isrobust": True, "weightstr": "cauchy"}),
    "long_talworth": ("long", {"isrobust": True, "weightstr": "talworth"}),
    "gappy_auto": ("gappy", {}),
    "gappy_robust": ("gappy", {"isrobust": True}),
    "gappy_robust_s300": ("gappy", {"isrobust": True, "s": 300}),
    "long_W_robust": ("long_W", {"isrobust": True}),
    "long_sd": ("long_sd", {}),
}


def main():
    inputs = make_inputs()
    out = {}
    for name, (y, extra) in inputs.items():
        out[f"in_{name}"] = y
        for k, v in extra.items():
            out[f"in_{name}_{k}"] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (inp, kwargs) in CASES.items():
            y, extra = inputs[inp]
            z, s, _ = smoothn(y.copy(), **extra, **kwargs)
            out[f"z_{name}"] = z
            out[f"s_{name}"] = np.array(s)
    path = Path(__file__).parent / "data" / "smooth1d_golden.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
