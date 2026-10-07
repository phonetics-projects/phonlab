#!/usr/bin/env python3
"""Benchmark and equivalence check for the ifc* paths of track_formants().

Compares a reference phonlab/acoustic/track_formants_.py, read from git (default: the `master`
revision), with a candidate file (default: the working tree's phonlab/acoustic/track_formants_.py)
on the example audio: numba compile time, run time per method, and whether the two give the same
numbers.  Run from the repository root, in the environment where phonlab is installed (editable
install, or the working tree on sys.path):

    python test/bench_track_formants_ifc.py                                # master vs working tree
    python test/bench_track_formants_ifc.py --variants parallel,serial,perframe   # the three code paths
    python test/bench_track_formants_ifc.py --ref-rev b532cc4                # against the pre-numba version
    python test/bench_track_formants_ifc.py --fastmath-ab                    # also time the reference with fastmath off
    python test/bench_track_formants_ifc.py my.wav other.wav                 # your own files

Candidate variants (only meaningful for a module that has the IFC_JIT / IFC_PARALLEL switches):
    parallel  - compiled whole-frame analysis on all cores (the module's default)
    serial    - compiled whole-frame analysis on one core
    perframe  - the per-frame python path (IFC_process_frame), i.e. the pre-branch structure

Nothing in the repository is modified: both modules are loaded from temporary copies.
"""
import argparse, importlib, importlib.util, os, platform, subprocess, sys, tempfile, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp(prefix='ifc_bench_')


def load_module(name, source):
    """Exec `source` as module phonlab.acoustic.<name> (so its relative imports resolve)."""
    import phonlab.acoustic  # noqa: F401
    path = os.path.join(TMP, name + '.py')
    with open(path, 'wb') as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(f'phonlab.acoustic.{name}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_source(rev):
    return subprocess.check_output(['git', '-C', ROOT, 'show', f'{rev}:phonlab/acoustic/track_formants_.py'])


def set_variant(mod, variant):
    if variant is None:
        return
    if not hasattr(mod, 'IFC_JIT'):
        raise RuntimeError('this candidate has no IFC_JIT / IFC_PARALLEL switches; drop --variants')
    mod.IFC_JIT = variant != 'perframe'
    mod.IFC_PARALLEL = variant == 'parallel'


def run(mod, x, fs, method, speaker, reps, variant=None):
    """Best-of-`reps` wall time and the result array; np.random is reseeded before every run
    because prep_audio() jitters exact-zero samples with np.random."""
    set_variant(mod, variant)
    best, df = float('inf'), None
    for _ in range(reps):
        np.random.seed(12345)
        t0 = time.perf_counter()
        df = mod.track_formants(x, fs, method=method, speaker=speaker, quiet=True)
        best = min(best, time.perf_counter() - t0)
    return best, df.to_numpy()


def compare(a, b):
    if a.shape != b.shape:
        return f'SHAPES DIFFER {a.shape} vs {b.shape}'
    d = np.abs(a - b)
    if np.array_equal(a, b, equal_nan=True):
        return 'identical'
    n_diff = int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum())
    n_big = int((d > 0.0015).sum())      # more than one unit in the last reported decimal
    return f'max|diff| {np.nanmax(d):.3g}, {n_diff} of {a.size} cells differ ({n_big} by more than 0.001)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--candidate', default=os.path.join(ROOT, 'phonlab', 'acoustic', 'track_formants_.py'),
                    help='candidate track_formants_.py (default: the working tree copy)')
    ap.add_argument('--ref-rev', default='master', help='git revision of the reference (default: master)')
    ap.add_argument('--old-rev', default=None, help='also run this (e.g. slow, pre-numba) revision, once per case')
    ap.add_argument('--fastmath-ab', action='store_true', help='also time the reference with fastmath=False')
    ap.add_argument('--variants', default='parallel',
                    help='comma list of candidate code paths to time: parallel, serial, perframe (default: parallel)')
    ap.add_argument('--methods', default='ifc,ifc_fast,ifc_old')
    ap.add_argument('--speakers', default='0')
    ap.add_argument('--reps', type=int, default=3)
    args = ap.parse_args()

    import phonlab as phon
    import numba, scipy
    print(f'{platform.platform()} | python {platform.python_version()} | numpy {np.__version__} | '
          f'scipy {scipy.__version__} | numba {numba.__version__} | {os.cpu_count()} cpus')
    print(f'reference: git {args.ref_rev}   candidate: {os.path.relpath(args.candidate, ROOT)}\n')

    ref = load_module('track_formants_reference_', git_source(args.ref_rev))
    with open(args.candidate, 'rb') as f:
        cand = load_module('track_formants_candidate_', f.read())
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    if not hasattr(cand, 'IFC_JIT'):
        variants = [None]
    extra = {}
    if args.fastmath_ab:
        src = git_source(args.ref_rev)
        assert b'fastmath=True' in src, 'reference has no fastmath=True to toggle'
        extra['ref-nofastmath'] = load_module('track_formants_refnofm_', src.replace(b', fastmath=True', b''))
    if args.old_rev:
        extra['old'] = load_module('track_formants_old_', git_source(args.old_rev))

    files = args.files
    if not files:
        import importlib.resources as ir
        d = ir.files('phonlab') / 'data' / 'example_audio'
        files = [str(d / f) for f in ['sf3_cln.wav', 'three_plus_five.wav', 'im_twelve.wav',
                                     'the_soviet_union.wav', 'might_have_a_class.wav']]
    methods = args.methods.split(',')
    speakers = [int(s) for s in args.speakers.split(',')]

    # warm-up: numba compiles on the first call of each method; report that separately.  A
    # variant that fails to compile is reported and dropped rather than aborting the run.
    xw, fsw = phon.loadsig(files[0], chansel=[0])
    xw = xw[:int(0.5 * fsw)]
    jobs = [('reference', ref, None)] + [(f'candidate[{v}]' if v else 'candidate', cand, v) for v in variants]
    jobs += [(k, m, None) for k, m in extra.items() if k != 'old']
    ok_variants = []
    for label, mod, variant in jobs:
        try:
            t0 = time.perf_counter()
            for m in methods:
                run(mod, xw, fsw, m, speakers[0], 1, variant)
            t1 = time.perf_counter()
            for m in methods:
                run(mod, xw, fsw, m, speakers[0], 1, variant)
            print(f'{label:22s} first call of all methods (incl. numba compile / cache load): {t1-t0:6.2f} s; '
                  f'second call: {time.perf_counter()-t1:6.3f} s')
            if mod is cand:
                ok_variants.append(variant)
        except Exception:
            print(f'{label:22s} FAILED:\n' + ''.join(traceback.format_exception(*sys.exc_info())[-6:]))
            if mod is not cand:
                raise
    print()

    tot = {}
    for path in files:
        x, fs = phon.loadsig(path, chansel=[0])
        for method in methods:
            for spk in speakers:
                tr, ar = run(ref, x, fs, method, spk, args.reps)
                tot['reference'] = tot.get('reference', 0) + tr
                line = f'{os.path.basename(path):22s} {method:9s} spk={spk} frames={len(ar):5d} | reference {tr:7.3f}s'
                for v in ok_variants:
                    tc, ac = run(cand, x, fs, method, spk, args.reps, v)
                    key = f'candidate[{v}]' if v else 'candidate'
                    tot[key] = tot.get(key, 0) + tc
                    line += f' | {key} {tc:7.3f}s ({tr/tc:5.1f}x) {compare(ar, ac)}'
                if 'ref-nofastmath' in extra:
                    tn, an = run(extra['ref-nofastmath'], x, fs, method, spk, args.reps)
                    tot['ref-nofastmath'] = tot.get('ref-nofastmath', 0) + tn
                    line += f' | ref without fastmath {tn:7.3f}s, vs ref: {compare(ar, an)}'
                if 'old' in extra:
                    to_, ao = run(extra['old'], x, fs, method, spk, 1)
                    tot['old'] = tot.get('old', 0) + to_
                    line += f' | old {to_:7.2f}s, ref vs old: {compare(ao, ar)}'
                print(line)
    print('\ntotal seconds: ' + ', '.join(f'{k} {v:.2f}' for k, v in tot.items()))


if __name__ == '__main__':
    main()
