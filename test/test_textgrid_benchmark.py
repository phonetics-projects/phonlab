"""Timing and memory comparison of the 'python' and 'praat' textgrid parsers.

These benchmarks are not part of the regular test run. Enable them with the
PHONLAB_BENCHMARK environment variable:

    PHONLAB_BENCHMARK=1 python -m pytest test/test_textgrid_benchmark.py

The timings are printed whether or not pytest's output capture is on. The size
of the generated textgrid and the number of timed repeats can be set with:

    PHONLAB_BENCHMARK_LABELS   phone intervals in the textgrid (default 20000);
                               the textgrid also gets a word tier with one
                               interval per 4 phones and a point tier with one
                               point per 2 phones
    PHONLAB_BENCHMARK_REPEATS  timed runs per parser (default 3); the best
                               time is reported

Each benchmark times both parsers with the '.only' suffix, so that a failure
cannot be hidden by a fallback, and asserts that both return the same result,
so that the timings compare equivalent work. They do not assert which parser is
faster, since that depends on the machine and the file.

The memory benchmarks report two figures for each parser:

    peak RSS    the growth in the process's peak resident memory over a
                baseline process that imports the same modules but reads
                nothing. This includes memory allocated inside Praat, which
                is compiled code, so it is the figure to compare the parsers
                by. Peak RSS cannot be reset within a process, so each
                measurement runs in a fresh subprocess, and the smallest of
                PHONLAB_BENCHMARK_REPEATS runs is reported.
    Python heap the peak of memory allocated by Python code, as measured by
                `tracemalloc` in a separate run. It does not see Praat's own
                allocations, so for the 'praat' parser it covers only the
                Python side of the work.

Both figures include the result, which is held until the measurement ends.
Peak RSS is coarse, since the operating system allocates memory in pages and
Python's allocator does not always return freed memory, so differences of a
few megabytes are noise.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from phonlab.utils.textgrid import tg_tiernames
from phonlab.utils.tidy import df_to_tg, tg_to_df

pytestmark = pytest.mark.skipif(
    not os.environ.get('PHONLAB_BENCHMARK'),
    reason='set PHONLAB_BENCHMARK=1 to run the parser benchmarks'
)

NLABELS = int(os.environ.get('PHONLAB_BENCHMARK_LABELS', 20000))
REPEATS = int(os.environ.get('PHONLAB_BENCHMARK_REPEATS', 3))

# Label content includes empty labels, non-ASCII text, and escaped quotes.
WORDS = ['', 'the', 'cat', 'sat', 'on', 'a', 'mat', 'ʃip', 'na\xefve', '"quoted"']
PHONES = ['', '\xf0', 'ə', 'k', '\xe6', 't', 's', 'ʃ', 'ɪ', 'p']

@pytest.fixture(scope='module', params=['short', 'long'])
def large_tg(request, tmp_path_factory):
    """A large, well-formed textgrid in short or long format, with two
    interval tiers and a point tier. Returns (path, tgtype, total labels)."""
    tgtype = request.param
    nph = max(4, NLABELS - (NLABELS % 4))   # a whole number of words
    phdf = pd.DataFrame({
        't1': [i / 100 for i in range(nph)],
        't2': [(i + 1) / 100 for i in range(nph)],
        'phone': [PHONES[i % len(PHONES)] for i in range(nph)],
    })
    nwd = nph // 4
    wddf = pd.DataFrame({
        't1': [(4 * i) / 100 for i in range(nwd)],
        't2': [(4 * (i + 1)) / 100 for i in range(nwd)],
        'word': [WORDS[i % len(WORDS)] for i in range(nwd)],
    })
    npt = nph // 2
    ptdf = pd.DataFrame({
        't1': [(2 * i + 0.5) / 100 for i in range(npt)],
        'point': [PHONES[i % len(PHONES)] or 'x' for i in range(npt)],
    })
    path = tmp_path_factory.mktemp('benchmark') / f'large.{tgtype}.TextGrid'
    df_to_tg(
        [wddf, phdf, ptdf],
        tiercols=['word', 'phone', 'point'],
        ts=[['t1', 't2'], ['t1', 't2'], ['t1', None]],
        start=0.0, end=nph / 100, tgtype=tgtype, outfile=path
    )
    return (path, tgtype, nph + nwd + npt)

def best_time(fn, repeats=REPEATS):
    """Return (best elapsed seconds, result of the last call) over `repeats`
    calls of `fn`."""
    times = []
    result = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return (min(times), result)

def report(capsys, func, tgfile, tgtype, nlabels, tpython, tpraat):
    """Print one line of timings, bypassing pytest's output capture."""
    if tpython <= tpraat:
        verdict = f'python {tpraat / tpython:.1f}x faster'
    else:
        verdict = f'praat {tpython / tpraat:.1f}x faster'
    size = os.path.getsize(tgfile) / 1e6
    with capsys.disabled():
        print(
            f'\n{func:<12} {tgtype:<5} {nlabels:>7} labels {size:6.1f} MB | '
            f'python {tpython:8.4f}s | praat {tpraat:8.4f}s | {verdict} '
            f'(best of {max(1, REPEATS)})'
        )

def test_benchmark_tg_to_df(large_tg, capsys):
    """Time `tg_to_df` with each parser on the same large textgrid."""
    pytest.importorskip('parselmouth')  # imported here, outside the timings
    path, tgtype, nlabels = large_tg
    tpython, pydfs = best_time(lambda: tg_to_df(path, parser='python.only'))
    tpraat, prdfs = best_time(lambda: tg_to_df(path, parser='praat.only'))
    assert [len(df) for df in pydfs] == [len(df) for df in prdfs]
    for pydf, prdf in zip(pydfs, prdfs):
        pd.testing.assert_frame_equal(pydf, prdf)
    report(capsys, 'tg_to_df', path, tgtype, nlabels, tpython, tpraat)

def test_benchmark_tg_tiernames(large_tg, capsys):
    """Time `tg_tiernames` with each parser on the same large textgrid."""
    pytest.importorskip('parselmouth')
    path, tgtype, nlabels = large_tg
    tpython, pynames = best_time(lambda: tg_tiernames(path, parser='python.only'))
    tpraat, prnames = best_time(lambda: tg_tiernames(path, parser='praat.only'))
    assert pynames == prnames == ('word', 'phone', 'point')
    report(capsys, 'tg_tiernames', path, tgtype, nlabels, tpython, tpraat)


#### Memory ####

REPO = Path(__file__).resolve().parent.parent

# Run in a fresh interpreter by the memory benchmarks. It imports the same
# modules whatever it is asked to do, so that the baseline process differs
# from a measured one only in the read itself.
_MEMORY_CHILD = r"""
import json, resource, sys, tracemalloc
mode, func, path, parser = sys.argv[1:5]
import numpy, pandas
try:
    import parselmouth
except ImportError:
    pass
from phonlab.utils.textgrid import tg_tiernames
from phonlab.utils.tidy import tg_to_df
reader = {'tg_to_df': tg_to_df, 'tg_tiernames': tg_tiernames}[func]
out = {}
if mode == 'traced':
    tracemalloc.start()
    result = reader(path, parser=parser)
    out['traced'] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
elif mode == 'rss':
    result = reader(path, parser=parser)
else:
    result = None
if result is not None:
    out['summary'] = (
        [[list(df.columns), len(df)] for df in result]
        if func == 'tg_to_df' else list(result)
    )
maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# ru_maxrss is in bytes on macOS and in kilobytes on Linux.
out['rss'] = maxrss if sys.platform == 'darwin' else maxrss * 1024
print(json.dumps(out))
"""

def _run_memory_child(mode, func='tg_to_df', path='', parser=''):
    """Run the measurement script in a fresh interpreter using the phonlab in
    this repository, and return what it reports."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(REPO)] + [p for p in [env.get('PYTHONPATH')] if p]
    )
    proc = subprocess.run(
        [sys.executable, '-c', _MEMORY_CHILD, mode, func, str(path), parser],
        capture_output=True, text=True, cwd=REPO, env=env
    )
    if proc.returncode != 0:
        pytest.fail(
            f'Memory measurement ({mode} {func} {parser}) failed:\n{proc.stderr}'
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])

@pytest.fixture(scope='module')
def baseline_rss():
    """Peak RSS of a process that imports everything but reads nothing."""
    pytest.importorskip('parselmouth')
    return min(_run_memory_child('baseline')['rss'] for _ in range(max(1, REPEATS)))

def _measure(func, path, parser, baseline):
    """Return (peak RSS over baseline, Python heap peak, result summary)."""
    rss_runs = [_run_memory_child('rss', func, path, parser) for _ in range(max(1, REPEATS))]
    traced = _run_memory_child('traced', func, path, parser)
    summaries = {json.dumps(r['summary']) for r in rss_runs + [traced]}
    assert len(summaries) == 1, 'repeated runs returned different results'
    rss = min(r['rss'] for r in rss_runs) - baseline
    return (max(rss, 0), traced['traced'], traced['summary'])

def _mb(nbytes):
    return nbytes / 1e6

def report_memory(capsys, func, tgfile, tgtype, nlabels, py, pr):
    """Print one line of memory figures, bypassing pytest's output capture."""
    size = os.path.getsize(tgfile) / 1e6
    with capsys.disabled():
        print(
            f'\n{func:<12} {tgtype:<5} {nlabels:>7} labels {size:6.1f} MB | '
            f'peak RSS python {_mb(py[0]):7.1f} MB, praat {_mb(pr[0]):7.1f} MB | '
            f'Python heap python {_mb(py[1]):7.1f} MB, praat {_mb(pr[1]):7.1f} MB '
            f'(RSS best of {max(1, REPEATS)})'
        )

@pytest.mark.parametrize('func', ['tg_to_df', 'tg_tiernames'])
def test_memory(func, large_tg, baseline_rss, capsys):
    """Measure the memory each parser uses to read the same large textgrid."""
    path, tgtype, nlabels = large_tg
    py = _measure(func, path, 'python.only', baseline_rss)
    pr = _measure(func, path, 'praat.only', baseline_rss)
    assert py[2] == pr[2], 'the parsers returned different results'
    report_memory(capsys, func, path, tgtype, nlabels, py, pr)
