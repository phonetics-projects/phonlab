"""Does smoothing the cepstrogram in CPP() help the voicing decision, and what does it cost?

CPP(smooth=n) smooths the cepstrogram with a Gaussian filter (sigma = n frames, in both time and quefrency)
before finding the cepstral peak.  Any nonzero smooth also makes the frame step 2 ms instead of 5 ms.  The cost of
smoothing is taken as a whole (the extra frames and the filtering), and each smoothed run is compared with the
unsmoothed run at the 5 ms default step:
    off (5 ms)   smooth=0
    smooth=n     for each n in --smooth
(--split-cost adds an unsmoothed run at 2 ms, to show how much of the cost is the extra frames.)

For each ASC file (channel 0 = audio, channel 1 = EGG) this
  * times each CPP() call (not including reading the file),
  * measures cpp (interpolated to the EGG frame times) and amp, and gets the EGG voicing from egg_to_oq(),
and then, for each variant, fits the VAD model (logistic regression on amp and cpp) and reports held-out
AUC and balanced accuracy by cross-validation grouped by file.  The folds are the same for every variant, so
the differences can be compared fold by fold.  'cpp alone' is the AUC of cpp by itself.

Usage:
    python compare_cpp_smooth.py /Volumes/WDBook/Corpora/asc/corpus [--frac 0.1] [--smooth 1 2 3 4] [--cache cpp.csv]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fileorganize import dir_to_df
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phonlab as phon
from phonlab.acoustic.vad import AMP_BAND, band_db
from train_vad import EGG_OPTIONS, F0_RANGE

warnings.filterwarnings("ignore")


def variants(smooths, split_cost=False):
    v = {'off (5 ms)': dict(smooth=0)}
    if split_cost:
        v['off (2 ms)'] = dict(smooth=0, s=0.002)
    v.update({f'smooth={n}': dict(smooth=n) for n in smooths})
    return v


def collect(corpus, frac, seed, var):
    rng = np.random.default_rng(seed)
    tgdf = dir_to_df(Path(corpus), fnpat=r"(?P<gender>[mf]).+.TextGrid$", addcols=['barename'])
    print(f'Found {len(tgdf)} TextGrid files.')
    frames, cost, dur = [], dict.fromkeys(var, 0.0), 0.0
    for row in tgdf.itertuples():
        if rng.random() > frac:
            continue
        audio, egg, fs = phon.loadsig(str(Path(corpus) / row.relpath / f'{row.barename}.wav'), chansel=[0, 1])
        dur += len(audio) / fs
        eggdf = phon.egg_to_oq(egg, fs, f0_range=F0_RANGE, **EGG_OPTIONS)
        sec = eggdf['sec'].to_numpy()
        df = pd.DataFrame({'sec': sec, 'amp': band_db(audio, fs, sec, AMP_BAND),
                           'vuv': np.where(eggdf['voiced'].to_numpy(), 'v', 'uv'), 'file': row.barename})
        for name, kw in var.items():
            t = time.perf_counter()
            c = phon.CPP(audio, fs, norm=False, f0_range=F0_RANGE, **kw)
            cost[name] += time.perf_counter() - t
            df[name] = np.interp(sec, c['sec'].to_numpy(), c['cpp'].to_numpy())
        frames.append(df)
        print(f'{row.relpath}/{row.barename}', flush=True)
    return pd.concat(frames, ignore_index=True), {k: v / dur for k, v in cost.items()}


def evaluate(df, names, folds):
    y, groups = (df['vuv'] == 'v').to_numpy(), df['file'].to_numpy()
    splits = list(GroupKFold(n_splits=folds).split(df, y, groups))
    out = {}
    for name in names:
        X = df[['amp', name]].to_numpy()
        auc, bacc = [], []
        for tr, te in splits:
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')).fit(X[tr], y[tr])
            auc.append(roc_auc_score(y[te], m.decision_function(X[te])))
            bacc.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        out[name] = (np.array(auc), np.array(bacc))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('corpus', help='root directory of the ASC corpus')
    ap.add_argument('--frac', type=float, default=0.1, help='fraction of files to use (default 0.1)')
    ap.add_argument('--smooth', type=int, nargs='+', default=[1, 2, 3, 4], help='smooth values to compare')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cache', help='csv file for the measurements (timings are kept in <cache>.json)')
    ap.add_argument('--split-cost', action='store_true', help='also time and score an unsmoothed run at the 2 ms step')
    ap.add_argument('--reference', default='off (5 ms)', help='variant that the others are compared with')
    a = ap.parse_args()
    var = variants(a.smooth, a.split_cost)

    cache = Path(a.cache) if a.cache else None
    if cache and cache.exists():
        df, cost = pd.read_csv(cache), json.loads(Path(f'{cache}.json').read_text())
    else:
        df, cost = collect(a.corpus, a.frac, a.seed, var)
        if cache:
            df.to_csv(cache, index=False)
            Path(f'{cache}.json').write_text(json.dumps(cost))

    names = [n for n in var if n in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['amp'] + names)
    print(f'\nN = {len(df)} frames, {df["file"].nunique()} files, {(df["vuv"] == "v").mean():.1%} EGG voiced')

    res = evaluate(df, names, a.folds)
    ref_name = a.reference if a.reference in res else names[0]
    ref = res[ref_name]
    y = (df['vuv'] == 'v').to_numpy()
    rows = []
    for n in names:
        d = res[n][0] - ref[0]
        rows.append({'variant': n, 'CPP s per s audio': cost[n], 'cost x': cost[n] / cost[ref_name], 'cpp alone AUC': roc_auc_score(y, df[n]),
                     'AUC (amp+cpp)': res[n][0].mean(), 'bal.acc': res[n][1].mean(),
                     f'dAUC vs {ref_name}': d.mean(), 'folds better': f'{np.sum(d > 0)}/{len(d)}'})
    print('\nHeld-out (by file) performance of the amp + cpp model, and the cost of CPP():')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
