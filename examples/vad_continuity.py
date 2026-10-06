"""Does a minimum-duration rule improve VAD voicing decisions, and how reliable are the EGG labels?

Reads the frames.csv made by train_vad.py (columns sec, f0, cpp, amp, vuv, file).  The predictions are
made by a logistic regression on amp and cpp, fit by cross-validation grouped by file, so that no file
is predicted by a model that was trained on it.

It reports
  1. how many very short voiced and unvoiced runs there are in the predictions and in the EGG labels,
  2. the accuracy after coercing runs shorter than a minimum duration to the value that surrounds them,
  3. where the prediction errors are, relative to the nearest EGG voicing transition, and
  4. the long stretches in which the audio and the EGG disagree well away from any transition, along with
     how steady the audio f0 is in them, to help decide whether it is the predictor or the EGG that is wrong.

Usage:
    python vad_continuity.py frames.csv [--min-dur 0.01 0.015 0.02 0.025 0.03 0.04 0.05]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PREDICTORS = ['amp', 'cpp']


def runs(v):
    """Run-length encode a boolean array: returns lists of (value, length)."""
    change = np.flatnonzero(np.diff(v.astype(np.int8))) + 1
    starts = np.r_[0, change]
    lens = np.diff(np.r_[starts, len(v)])
    return v[starts], lens


def merge_short_runs(v, min_frames):
    """Coerce every run shorter than `min_frames` into the value of the runs around it.

    A run in a two-valued sequence always has the same value on both sides, so flipping it merges three
    runs into one.  The shortest run is flipped first, and runs at the start and end of the file (which have
    only one neighbour) are left alone.
    """
    vals, lens = [list(a) for a in runs(v)]
    while len(lens) > 2:
        interior = lens[1:-1]
        i = int(np.argmin(interior)) + 1
        if lens[i] >= min_frames:
            break
        lens[i - 1:i + 2] = [lens[i - 1] + lens[i] + lens[i + 1]]
        vals[i - 1:i + 2] = [vals[i - 1]]
    return np.repeat(vals, lens).astype(bool)


def smooth(df, pred, min_frames):
    out = pred.copy()
    if min_frames <= 1:
        return out
    for idx in df.groupby('file').indices.values():
        out[idx] = merge_short_runs(pred[idx], min_frames)
    return out


def bacc(truth, pred):
    return balanced_accuracy_score(truth, pred)


def run_table(df, labels, name, step):
    """Counts of interior runs by duration, for voiced and unvoiced."""
    rows = {}
    for val, label in ((True, 'voiced'), (False, 'unvoiced')):
        lens = []
        for idx in df.groupby('file').indices.values():
            v, l = runs(labels[idx])
            lens += list(l[1:-1][v[1:-1] == val])            # interior runs only
        ms = np.array(lens) * step * 1000
        rows[label] = {'n runs': len(ms), '<15 ms': np.sum(ms < 15), '<25 ms': np.sum(ms < 25),
                       '<50 ms': np.sum(ms < 50), 'median ms': np.median(ms) if len(ms) else np.nan}
    print(f'\n{name}: interior runs')
    print(pd.DataFrame(rows).T.round(0).astype(int).to_string())


def distance_to_transition(df, truth):
    """Frames from each frame to the nearest change in the EGG voicing state (within its file)."""
    dist = np.full(len(df), np.iinfo(np.int32).max, dtype=np.int64)
    for idx in df.groupby('file').indices.values():
        t = np.flatnonzero(np.diff(truth[idx].astype(np.int8))) + 0.5     # transitions between frames
        if len(t):
            pos = np.arange(len(idx))
            j = np.clip(np.searchsorted(t, pos), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(pos), int)
            near = np.minimum(np.abs(pos - t[j]), np.abs(pos - t[np.maximum(j - 1, 0)]))
            dist[idx] = np.ceil(near).astype(np.int64)
    return dist


def steadiness(f0, idx):
    """Fraction of adjacent frames in idx (consecutive) whose f0 differs by less than 10%."""
    r = f0[idx][1:] / f0[idx][:-1]
    return np.mean((r > 0.9) & (r < 1.1)) if len(r) else np.nan


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='the frames.csv file written by train_vad.py')
    ap.add_argument('--min-dur', type=float, nargs='+', default=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.frames).replace([np.inf, -np.inf], np.nan).dropna(subset=PREDICTORS)
    df = df.sort_values(['file', 'sec'], kind='stable').reset_index(drop=True)
    step = float(np.median(np.diff(df['sec'])[np.diff(df['sec']) > 0]))
    truth = (df['vuv'] == 'v').to_numpy()
    print(f'{len(df)} frames, {df["file"].nunique()} files, {step * 1000:.1f} ms step, {truth.mean():.1%} EGG voiced')

    X = df[PREDICTORS].to_numpy()
    pred = np.zeros(len(df), bool)
    for tr, te in GroupKFold(a.folds).split(X, truth, df['file']):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')).fit(X[tr], truth[tr])
        pred[te] = m.predict(X[te])

    # ---- 1. run lengths
    run_table(df, pred, 'Predicted', step)
    run_table(df, truth, 'EGG', step)

    # ---- 2. minimum duration rule
    print('\nMinimum duration rule (runs shorter than this are coerced to the surrounding value):')
    print(f'  {"min dur":>8s} {"frames":>6s}  {"bal.acc":>8s} {"change":>8s}   {"vs EGG smoothed too":>20s}')
    base = bacc(truth, pred)
    print(f'  {"none":>8s} {"":>6s}  {base:8.4f} {"":>8s}')
    for d in a.min_dur:
        n = max(int(round(d / step)), 1)
        sp = smooth(df, pred, n)
        st = smooth(df, truth, n)
        print(f'  {d * 1000:6.0f}ms {n:6d}  {bacc(truth, sp):8.4f} {bacc(truth, sp) - base:+8.4f}   {bacc(st, sp):20.4f}')
    print('  (the last column scores the smoothed predictions against EGG labels given the same rule)')

    # ---- 3. where are the errors?
    dist = distance_to_transition(df, truth)
    err = pred != truth
    print('\nPrediction errors by distance from the nearest EGG voicing transition:')
    bins = [(0, 1), (2, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 10**9)]
    rows = []
    for lo, hi in bins:
        sel = (dist >= lo) & (dist <= hi)
        rows.append({'frames away': f'{lo}-{hi}' if hi < 10**9 else f'{lo}+',
                     'ms': f'{lo * step * 1000:.0f}+' if hi >= 10**9 else f'{lo * step * 1000:.0f}-{hi * step * 1000:.0f}',
                     'frames': sel.sum(), 'error rate': err[sel].mean() if sel.any() else np.nan,
                     'share of errors': err[sel].sum() / err.sum()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- 3b. by speaker (the speaker is the first 4 characters of the ASC file names)
    sp = df['file'].str[:4]
    far_ = dist >= 6
    tab = pd.DataFrame({'speaker': sp, 'truth': truth, 'pred': pred, 'far': far_})
    rows = []
    for spk, g in tab.groupby('speaker'):
        gf = g[g.far]
        rows.append({'speaker': spk, 'frames': len(g), 'EGG voiced': g.truth.mean(),
                     'bal.acc': bacc(g.truth, g.pred) if g.truth.nunique() > 1 else np.nan,
                     'far errors: EGG uv/pred v': np.mean(~gf.truth & gf.pred), 'far errors: EGG v/pred uv': np.mean(gf.truth & ~gf.pred)})
    bysp = pd.DataFrame(rows).sort_values('bal.acc')
    print('\nBy speaker (far = at least 30 ms from an EGG transition; error columns are proportions of far frames).')
    print(f'Lowest five of {len(bysp)} speakers by balanced accuracy, and the median over speakers:')
    print(pd.concat([bysp.head(5), bysp.median(numeric_only=True).to_frame().T.assign(speaker='median')]).round(3).to_string(index=False))

    # ---- 4. long disagreements away from any transition
    far = dist >= 6                                          # at least 30 ms from an EGG transition
    mis = err & far
    f0 = df['f0'].to_numpy()
    print(f'\nErrors at least 30 ms from an EGG transition: {mis.sum()} frames ({mis.sum() / err.sum():.1%} of all errors)')
    out = []
    for fname, idx in df.groupby('file').indices.items():
        v, l = runs(mis[idx])
        starts = np.r_[0, np.flatnonzero(np.diff(mis[idx].astype(np.int8))) + 1]
        for s0, val, ln in zip(starts, v, l):
            if val and ln >= 8:                               # disagreement of 40 ms or more
                seg = idx[s0:s0 + ln]
                out.append({'file': fname, 'sec': df['sec'][seg[0]], 'ms': ln * step * 1000,
                            'EGG': 'v' if truth[seg[0]] else 'uv', 'amp': df['amp'][seg].mean(),
                            'cpp': df['cpp'][seg].mean(), 'f0 steady': steadiness(f0, seg)})
    out = pd.DataFrame(out)
    if len(out):
        agree_v = steadiness(f0, np.flatnonzero(truth & ~err)) if (truth & ~err).any() else np.nan
        agree_u = steadiness(f0, np.flatnonzero(~truth & ~err)) if (~truth & ~err).any() else np.nan
        print(f'{len(out)} stretches of 40 ms or more.  For reference, the audio f0 is steady (adjacent frames within 10%) '
              f'in {agree_v:.2f} of frames where both say voiced, and {agree_u:.2f} where both say unvoiced.')
        print(out.groupby('EGG').agg(stretches=('ms', 'size'), median_ms=('ms', 'median'), amp=('amp', 'mean'),
                                     cpp=('cpp', 'mean'), f0_steady=('f0 steady', 'mean')).round(2).to_string())
        print('\nThe 15 longest (to look at in the waveform and EGG):')
        print(out.sort_values('ms', ascending=False).head(15).round(2).to_string(index=False))
