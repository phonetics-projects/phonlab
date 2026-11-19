#!/usr/bin/env python3
"""Speed and equivalence check for the speaker-params changes to phonlab/acoustic/ifc_formant.py.

Loads a reference ifc_formant.py from git (default: HEAD) and the working-tree one, compiles
both, and times IFC_formants() on the example audio for each preset, serial and parallel, plus a
dict equal to preset 0.  Reports numba compile time, best-of-N wall time, the ratio new/old, and
the largest difference between the two outputs (expected 0 for the presets).  Run from the
repository root in the environment where phonlab is installed:

    python test/bench_ifc_params.py                     # HEAD vs working tree, test/s09003.wav
    python test/bench_ifc_params.py --reps 7 my.wav     # your own audio
    python test/bench_ifc_params.py --ref-rev 95ec2ac

Nothing in the repository is modified: both modules are loaded from temporary copies.
"""
import argparse, importlib, importlib.util, os, platform, subprocess, sys, tempfile, time, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp(prefix='ifc_bench_')


def load_module(name, source):
    import phonlab.acoustic  # noqa: F401  (so the relative imports in the module resolve)
    path = os.path.join(TMP, name + '.py')
    with open(path, 'wb') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(f'phonlab.acoustic.{name}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def timed(fn, reps):
    best = np.inf
    for _ in range(reps):
        np.random.seed(0)                      # prep_audio jitters exact-zero samples with np.random
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', default=[os.path.join(HERE, 's09003.wav')])
    ap.add_argument('--ref-rev', default='HEAD')
    ap.add_argument('--reps', type=int, default=5)
    a = ap.parse_args()

    import librosa
    ref = load_module('ifc_ref', subprocess.check_output(['git', '-C', ROOT, 'show', f'{a.ref_rev}:phonlab/acoustic/ifc_formant.py']))
    new = load_module('ifc_new', open(os.path.join(ROOT, 'phonlab/acoustic/ifc_formant.py'), 'rb').read())
    print(f"{platform.platform()}  python {platform.python_version()}  numpy {np.__version__}")
    print(f"reference: {a.ref_rev}   candidate: working tree\n")

    for path in a.files:
        x, fs = librosa.load(path, sr=None)
        print(f"== {os.path.basename(path)}: {len(x)/fs:.1f} s at {fs} Hz")
        # compile (first call) time, serial and parallel
        for label, mod in (('old', ref), ('new', new)):
            t0 = time.perf_counter(); mod.IFC_formants(x[:fs], fs, parallel=False); ts = time.perf_counter() - t0
            t0 = time.perf_counter(); mod.IFC_formants(x[:fs], fs, parallel=True); tp = time.perf_counter() - t0
            print(f"  {label}: first-call (compile) serial {ts:5.1f} s, parallel {tp:5.1f} s")
        print(f"  {'case':28s} {'old':>8s} {'new':>8s} {'new/old':>8s}  max|diff|")
        cases = [(f"preset {s}, {'parallel' if par else 'serial'}", dict(speaker=s, parallel=par)) for par in (False, True) for s in (0, 1, 2)]
        for name, kw in cases:
            to, ro = timed(lambda: ref.IFC_formants(x, fs, **kw), a.reps)
            tn, rn = timed(lambda: new.IFC_formants(x, fs, **kw), a.reps)
            d = np.nanmax(np.abs(ro.values - rn.values)) if ro.shape == rn.shape else float('nan')
            print(f"  {name:28s} {to*1000:7.0f}ms {tn*1000:7.0f}ms {tn/to:8.3f}  {d:g}")
        # a dict equal to preset 0: the new code path only
        if hasattr(new, 'ifc_speaker_params'):
            p = new.ifc_speaker_params(0)
            to, ro = timed(lambda: ref.IFC_formants(x, fs, speaker=0, parallel=True), a.reps)
            tn, rn = timed(lambda: new.IFC_formants(x, fs, speaker=p, parallel=True), a.reps)
            d = np.nanmax(np.abs(ro.values - rn.values))
            print(f"  {'dict == preset 0, parallel':28s} {to*1000:7.0f}ms {tn*1000:7.0f}ms {tn/to:8.3f}  {d:g}")
        print()


if __name__ == '__main__':
    main()
