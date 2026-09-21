"""Timing comparison of the 'python' and 'praat' textgrid parsers.

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
"""

import os
import time

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
